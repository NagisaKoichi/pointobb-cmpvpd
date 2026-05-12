# Copyright (c) OpenMMLab. All rights reserved.
"""VPD-enhanced Pseudo-Label Head.

Extends PseudoLabelHead with learned mu (center offset) and sigma
(uncertainty) from CPMVPDHead. During pseudo-label generation:

1. **Mu-based center refinement**: Aggregates mu predictions near each
   GT center (weighted by cls_prob) to produce a refined center point.
   This corrects annotation click noise and improves PCA orientation
   and boundary walk symmetry.

2. **Sigma-modulated boundary walk**: Uses directional sigma projected
   onto the walk direction to raise cls_prob threshold at uncertain
   locations, producing tighter boxes between nearby objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32

from ..builder import ROTATED_HEADS
from .pseudo_label_head import PseudoLabelHead
from .rotated_anchor_free_head import RotatedAnchorFreeHead

INF = 1e8


@ROTATED_HEADS.register_module()
class VPDPseudoLabelHead(PseudoLabelHead):
    """PseudoLabelHead with learned mu + sigma for better pseudo-labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        train_cfg = kwargs.get('train_cfg', {})
        # Sigma modulates cls_prob threshold:
        #   effective_thresh = thresh3 * (sigma_dir / sigma_center_dir) ^ sigma_power
        self.sigma_power = train_cfg.get('sigma_power', 1.0)
        # Neutral point multiplier: ratio reference = sigma_center * this
        # ratio < 1 → expand, ratio > 1 → shrink
        # e.g., 2.0 means "at 2x center sigma, threshold is unchanged"
        self.sigma_neutral = train_cfg.get('sigma_neutral', 2.0)
        # Sigma spike threshold: stop walk if sigma_ratio > this (independent criterion)
        self.sigma_spike_thresh = train_cfg.get('sigma_spike_thresh', 2.5)
        # Whether to use sigma to weight PCA orientation estimation
        self.sigma_pca = train_cfg.get('sigma_pca', True)
        # Mu refinement radius (in stride-0 feature pixels)
        self.mu_refine_radius = train_cfg.get('mu_refine_radius', 6)

    def _init_layers(self):
        """Add mu head + sigma tower (matching CPMVPDHead architecture)."""
        super()._init_layers()

        # Mu head: 2-channel, takes reg_feat (output of parent's reg_convs)
        # Named conv_mu to avoid conflict with parent's conv_reg (4-ch ltrb)
        self.conv_mu = nn.Conv2d(self.feat_channels, 2, 3, padding=1)
        self.scale_mu = Scale(1.0)

        # 2-layer independent sigma tower — same as CPMVPDHead._init_predictor
        self.sigma_convs = nn.ModuleList()
        for i in range(2):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.sigma_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                           norm_cfg=self.norm_cfg, bias=self.conv_bias))
        self.conv_sigma = nn.Conv2d(self.feat_channels, 2, 3, padding=1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Map checkpoint conv_reg → conv_mu for weight loading."""
        # CPMVPDHead saves mu as conv_reg; we renamed to conv_mu
        reg_w = prefix + 'conv_reg.weight'
        reg_b = prefix + 'conv_reg.bias'
        mu_w = prefix + 'conv_mu.weight'
        mu_b = prefix + 'conv_mu.bias'
        if reg_w in state_dict and mu_w not in state_dict:
            state_dict[mu_w] = state_dict.pop(reg_w)
        if reg_b in state_dict and mu_b not in state_dict:
            state_dict[mu_b] = state_dict.pop(reg_b)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward_single(self, x, scale, stride):
        """Override to compute and cache mu + sigma per level."""
        cls_score, bbox_pred, cls_feat, reg_feat = \
            super(RotatedAnchorFreeHead, self).forward_single(x)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)

        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        angle_pred = self.conv_angle(reg_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()

        # Compute mu from reg_feat (same path as CPMVPDHead)
        mu_pred = self.scale_mu(self.conv_mu(reg_feat).float())  # (B, 2, H, W)

        # Compute sigma from independent tower (raw FPN features)
        sigma_feat = x
        for conv in self.sigma_convs:
            sigma_feat = conv(sigma_feat)
        log_sigma = self.conv_sigma(sigma_feat).float()  # (B, 2, H, W)

        # Cache for pseudo-label generation
        if not hasattr(self, '_sigma_per_level'):
            self._sigma_per_level = {}
        if not hasattr(self, '_mu_per_level'):
            self._mu_per_level = {}
        self._sigma_per_level[stride] = log_sigma.detach()
        self._mu_per_level[stride] = mu_pred.detach()

        return cls_score, bbox_pred, angle_pred, centerness

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def loss(self, cls_scores, bbox_preds, angle_preds, centernesses,
             gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Cache mu + sigma then delegate to parent."""
        stride0 = self.strides[0]
        self._sigma_batch = None
        self._mu_batch = None
        self._sigma_img_idx = 0

        if hasattr(self, '_sigma_per_level') and stride0 in self._sigma_per_level:
            log_sigma = self._sigma_per_level[stride0]  # (B, 2, H, W)
            self._sigma_batch = log_sigma.exp()

        if hasattr(self, '_mu_per_level') and stride0 in self._mu_per_level:
            self._mu_batch = self._mu_per_level[stride0]  # (B, 2, H, W)

        return super().loss(cls_scores, bbox_preds, angle_preds, centernesses,
                            gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)

    def _refine_centers_with_mu(self, center_point_gt, gt_labels,
                                cls_scores_sigmoid, mu_map, stride):
        """Refine GT center points using mu + cls_prob weighted centroid.

        Two complementary signals:
        1. Mu consensus: mu predictions near the GT center vote on where the
           true center is. Weighted by cls_prob to suppress background.
        2. Cls centroid: cls_prob-weighted centroid of the activation blob
           provides a geometry-based center estimate.

        The final refinement is a blend of both, clipped to max_shift pixels
        to prevent catastrophic drift.

        Args:
            center_point_gt: (N, 2) GT center points in image coords.
            gt_labels: (N,) class labels per GT.
            cls_scores_sigmoid: (C, H, W) sigmoid class scores at stride-0.
            mu_map: (2, H, W) mu predictions at stride-0.
            stride: stride of level 0 (typically 4).

        Returns:
            refined_centers: (N, 2) refined center points in image coords.
        """
        num_gts = center_point_gt.shape[0]
        H, W = cls_scores_sigmoid.shape[1], cls_scores_sigmoid.shape[2]
        refined = center_point_gt.clone()
        radius = self.mu_refine_radius  # in feature pixels
        max_shift = stride * 2.0  # max allowed shift in pixels (conservative)

        for i in range(num_gts):
            cx_feat = center_point_gt[i, 0] / stride
            cy_feat = center_point_gt[i, 1] / stride

            # Define local window
            x_min = max(int(cx_feat - radius), 0)
            x_max = min(int(cx_feat + radius) + 1, W)
            y_min = max(int(cy_feat - radius), 0)
            y_max = min(int(cy_feat + radius) + 1, H)

            if x_max <= x_min or y_max <= y_min:
                continue

            # Class probability as weight
            cls_prob = cls_scores_sigmoid[gt_labels[i],
                                         y_min:y_max, x_min:x_max]  # (h, w)

            # ── Signal 1: Mu consensus voting ──
            mu_local = mu_map[:, y_min:y_max, x_min:x_max]  # (2, h, w)

            # Grid of anchor points in image coords
            yy = torch.arange(y_min, y_max, device=mu_map.device).float()
            xx = torch.arange(x_min, x_max, device=mu_map.device).float()
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
            anchor_x = grid_x * stride
            anchor_y = grid_y * stride

            # Each anchor predicts where the center is: anchor + mu * stride
            pred_cx = anchor_x + mu_local[0] * stride
            pred_cy = anchor_y + mu_local[1] * stride

            # ── Signal 2: Cls centroid ──
            # Anchor positions weighted by cls_prob = geometric center
            # (no mu needed, pure classifier geometry)

            # Combined: use cls_prob * proximity weight
            dist_from_gt = torch.sqrt(
                (anchor_x - center_point_gt[i, 0]) ** 2 +
                (anchor_y - center_point_gt[i, 1]) ** 2)
            prox_weight = (1.0 - dist_from_gt / (radius * stride)).clamp(min=0)
            weight = cls_prob * prox_weight
            weight_sum = weight.sum()

            if weight_sum < 1e-6:
                continue

            # Mu consensus: weighted average of predicted centers
            mu_cx = (pred_cx * weight).sum() / weight_sum
            mu_cy = (pred_cy * weight).sum() / weight_sum

            # Cls centroid: weighted average of anchor positions
            cls_cx = (anchor_x * weight).sum() / weight_sum
            cls_cy = (anchor_y * weight).sum() / weight_sum

            # Blend: 50% mu consensus + 50% cls centroid
            ref_x = 0.5 * mu_cx + 0.5 * cls_cx
            ref_y = 0.5 * mu_cy + 0.5 * cls_cy

            # Clip shift to max_shift
            dx = ref_x - center_point_gt[i, 0]
            dy = ref_y - center_point_gt[i, 1]
            shift = torch.sqrt(dx ** 2 + dy ** 2)
            if shift > max_shift:
                scale = max_shift / shift
                dx = dx * scale
                dy = dy * scale

            refined[i, 0] = center_point_gt[i, 0] + dx
            refined[i, 1] = center_point_gt[i, 1] + dy

        return refined

    def _get_target_single(self, gt_bboxes, gt_labels, cls_scores_all,
                           img_metas, points, regress_ranges,
                           num_points_per_lvl, alpha=1, thresh1=8,
                           thresh2_bg=4, thresh3=0.1, pca_length=28,
                           default_max_length=128, mode='near'):
        """Override to use mu-refined centers and sigma for pseudo-labels."""
        alpha = alpha
        thresh3 = self.thresh3
        pca_length = self.pca_length
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        center_point_gt = gt_bboxes[:, :2]

        if num_gts == 0:
            filename_raw = img_metas['filename'].split('/')[-1].split('.')[0]
            with open(self.store_ann_dir + filename_raw + '.txt', 'w') as w:
                w.write('\n')
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))

        # ── Mu-based center refinement ──────────────────────────
        stride0 = self.strides[0]
        mu_map_refine = None
        if self._mu_batch is not None and self.mu_refine_radius > 1:
            idx = min(self._sigma_img_idx, self._mu_batch.shape[0] - 1)
            mu_map_refine = self._mu_batch[idx]  # (2, H, W)

        if mu_map_refine is not None:
            refined_center = self._refine_centers_with_mu(
                center_point_gt, gt_labels,
                cls_scores_all[0].sigmoid(), mu_map_refine, stride0)
        else:
            refined_center = center_point_gt

        # Use refined_center for PCA and boundary walk,
        # but keep original center_point_gt for label assignment
        # (label assignment is based on GT annotation, not refinement)

        dist_sample_and_gt = torch.cdist(points, center_point_gt)
        dist_gt_and_gt = (torch.cdist(center_point_gt, center_point_gt)
                          + torch.eye(num_gts).to(dist_sample_and_gt.device) * INF)
        dist_min_gt_and_gt, dist_min_gt_and_gt_index = dist_gt_and_gt.min(dim=1)
        labels = -1 * torch.ones(num_points, dtype=gt_labels.dtype,
                                 device=gt_labels.device)

        index_neg = ((alpha * dist_sample_and_gt) > dist_min_gt_and_gt).all(
            dim=1).nonzero().squeeze(-1)
        if len(index_neg) > 0:
            labels[index_neg] = self.num_classes

        thresh1_tensor = thresh1 * torch.ones_like(dist_min_gt_and_gt)
        dist_min_thresh1_gt = torch.min(dist_min_gt_and_gt / 2, thresh1_tensor)
        index_pos = (dist_sample_and_gt < dist_min_thresh1_gt).nonzero()
        if len(index_pos) > 0:
            labels[index_pos[:, 0]] = gt_labels[index_pos[:, 1]]

        is_nearest_same_class = (gt_labels[dist_min_gt_and_gt_index] == gt_labels)
        valid_middle_point = (
            center_point_gt[is_nearest_same_class]
            + center_point_gt[dist_min_gt_and_gt_index][is_nearest_same_class]) / 2
        dist_sample_and_mid = torch.cdist(points, valid_middle_point)
        index_neg_additional = (dist_sample_and_mid < thresh2_bg).any(
            dim=1).nonzero().squeeze(-1)
        if len(index_neg_additional) > 0:
            labels[index_neg_additional] = self.num_classes

        # ── PCA on refined centers ──────────────────────────────
        center_factor = self.get_center_factor(
            refined_center, gt_labels, cls_scores_all[0])
        gt_ctr_rect = self.get_rectangle_cls_prob(
            cls_scores_all[0].sigmoid(), self.strides[0], center_factor,
            refined_center, pca_length, mode='near')
        gt_ctr_rect_label = gt_ctr_rect[torch.arange(num_gts), gt_labels, :, :]

        # ── Sigma-weighted PCA: de-weight uncertain boundary points ──
        if self._sigma_batch is not None and self.sigma_pca:
            sigma_img_idx = min(self._sigma_img_idx,
                                self._sigma_batch.shape[0] - 1)
            sigma_map_pca = self._sigma_batch[sigma_img_idx]  # (2, H, W)
            sigma_mag = torch.sqrt(sigma_map_pca[0] ** 2 + sigma_map_pca[1] ** 2)
            # Extract sigma rectangle same way as cls_prob
            H_s, W_s = sigma_mag.shape
            stride0 = self.strides[0]
            length_lvl = pca_length / stride0
            rect_len = 2 * int((length_lvl - 1) / 2) + 1
            pad = 10
            padded_sigma = F_nn.pad(sigma_mag[None, None], (pad, pad, pad, pad),
                                    mode='constant', value=10.0)[0, 0]
            gt_ctr_lvl = (refined_center / stride0 + pad).round().long()
            half = int((length_lvl - 1) / 2)
            for gi in range(num_gts):
                x0 = int(gt_ctr_lvl[gi, 0]) - half
                y0 = int(gt_ctr_lvl[gi, 1]) - half
                sigma_rect_i = padded_sigma[y0:y0 + rect_len, x0:x0 + rect_len]
                if sigma_rect_i.shape[0] == rect_len and sigma_rect_i.shape[1] == rect_len:
                    # confidence = 1 / (1 + sigma), high sigma → low weight
                    sigma_conf = 1.0 / (1.0 + sigma_rect_i)
                    gt_ctr_rect_label[gi] = gt_ctr_rect_label[gi] * sigma_conf

        gt_rect_ctr2edge = gt_ctr_rect_label.shape[-1] // 2
        points_rect_x = torch.arange(
            -gt_rect_ctr2edge, gt_rect_ctr2edge + 1, 1).to(gt_ctr_rect.device)
        points_rect_y = torch.arange(
            -gt_rect_ctr2edge, gt_rect_ctr2edge + 1, 1).to(gt_ctr_rect.device)
        points_rect_xy = torch.stack(
            torch.meshgrid(points_rect_x, points_rect_y), -1).reshape(-1, 2)
        gt_ctr_rect_label = gt_ctr_rect_label.transpose(1, 2).contiguous().view(
            num_gts, -1)
        points_rect_xy_adapt = (
            points_rect_xy.unsqueeze(0).repeat(num_gts, 1, 1)
            * torch.sqrt(gt_ctr_rect_label).unsqueeze(-1))
        points_cov_matrix = (
            torch.matmul(points_rect_xy_adapt.transpose(1, 2),
                         points_rect_xy_adapt)
            / (gt_ctr_rect_label.shape[-1] ** 2 - 1))
        eigvals, eigvecs = torch.linalg.eigh(points_cov_matrix, UPLO='L')

        larger_eigvals_index = (eigvals[:, 1] > eigvals[:, 0]).int()
        eigvec_first = (
            eigvecs[:, 0, :] * (1 - larger_eigvals_index).unsqueeze(1).repeat(1, 2)
            + eigvecs[:, 1, :] * larger_eigvals_index.unsqueeze(1).repeat(1, 2))
        mask_eigvec = (eigvec_first[:, 1] > 0).int()
        epsilon = mask_eigvec * 1e-6 + (1 - mask_eigvec) * -1e-6

        angle_targets = -torch.atan(
            eigvec_first[:, 0] / (eigvec_first[:, 1] + epsilon)).unsqueeze(-1)
        eigvec_second = torch.stack(
            [-eigvec_first[:, 1], eigvec_first[:, 0]], -1)

        first_axis_range = self.get_closest_gt_first_axis(
            gt_labels, eigvec_first, refined_center, angle_threshold=0.866)

        # Get sigma map for this image
        sigma_map = None
        if self._sigma_batch is not None:
            idx = min(self._sigma_img_idx, self._sigma_batch.shape[0] - 1)
            sigma_map = self._sigma_batch[idx]  # (2, H, W)
            self._sigma_img_idx += 1

        # Get mu_map for boundary walk mu-consistency criterion
        mu_map_walk = None
        if self._mu_batch is not None and self.mu_refine_radius > 0:
            mu_idx = min(self._sigma_img_idx - 1 if self._sigma_img_idx > 0
                         else 0, self._mu_batch.shape[0] - 1)
            mu_map_walk = self._mu_batch[mu_idx]  # (2, H, W)

        # ── Boundary walk from refined centers ──────────────────
        top_simple, bottom_simple = self.get_edge_boundary_simple(
            gt_labels, eigvec_first, refined_center, cls_scores_all,
            thresh3, default_max_length, mode='simple',
            is_secondary=False,
            is_nearest_same_class=is_nearest_same_class,
            nearest_gt_point=center_point_gt[dist_min_gt_and_gt_index],
            first_axis_range=first_axis_range,
            sigma_map=sigma_map,
            mu_map=mu_map_walk,
            original_center=center_point_gt)
        left_simple, right_simple = self.get_edge_boundary_simple(
            gt_labels, eigvec_second, refined_center, cls_scores_all,
            thresh3, default_max_length, mode='simple',
            is_secondary=True,
            is_nearest_same_class=is_nearest_same_class,
            nearest_gt_point=center_point_gt[dist_min_gt_and_gt_index],
            sigma_map=sigma_map,
            mu_map=mu_map_walk,
            original_center=center_point_gt)

        top_simple = top_simple * self.strides[0] + 1
        bottom_simple = bottom_simple * self.strides[0] + 1
        left_simple = left_simple * self.strides[0] + 1
        right_simple = right_simple * self.strides[0] + 1

        # Use REFINED center as the pseudo-label center
        pseudo_gt_bboxes = torch.cat([
            refined_center,
            (left_simple + right_simple).unsqueeze(-1),
            (top_simple + bottom_simple).unsqueeze(-1),
            angle_targets], -1)
        pseudo_gt_bboxes = pseudo_gt_bboxes.detach()

        self.generate_labels(gt_labels, pseudo_gt_bboxes, angle_targets, img_metas)

        if num_gts == 1:
            index_pos = (dist_sample_and_gt < 8).nonzero().reshape(-1)
            index_neg = (dist_sample_and_gt > 128).nonzero().reshape(-1)
            labels[index_pos] = gt_labels[0]
            labels[index_neg] = self.num_classes
            return labels, gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))

        return labels, None, None

    def get_edge_boundary_simple(self, gt_labels, eigvec, center_point_gt,
                                 cls_scores_all, thresh3=0.1,
                                 default_max_length=128, mode='near',
                                 is_secondary=False,
                                 is_nearest_same_class=None,
                                 nearest_gt_point=None,
                                 first_axis_range=None,
                                 sigma_map=None,
                                 mu_map=None,
                                 original_center=None):
        """Boundary walk with directional sigma + mu consistency.

        Sigma: Projects 2D sigma onto walk direction for direction-aware
        threshold modulation.

        Mu (optional): At each walk step, mu predicts where the center is.
        If the predicted center diverges from the original GT center, the
        walk has entered another object's territory → stop. This provides
        object-identity awareness that sigma alone cannot.

        effective_thresh = thresh3 * (sigma_dir / sigma_center_dir) ^ sigma_power
        mu_stop: |pred_center - gt_center| > mu_diverge_thresh * stride
        """
        num_gts = center_point_gt.shape[0]
        stride = self.strides[0]
        center_point_gt_feat = center_point_gt / stride

        eigvec_norm = eigvec / torch.norm(eigvec, dim=1, keepdim=True)
        top_bottom = torch.zeros(num_gts, 2).to(center_point_gt.device)
        cls_scores_sigmoid = cls_scores_all[0].sigmoid()
        H, W = cls_scores_all[0].shape[1], cls_scores_all[0].shape[2]

        # Precompute sigma components
        sigma_dx = None
        sigma_dy = None
        if sigma_map is not None:
            sigma_dx = sigma_map[0]  # (H_s, W_s) - x uncertainty
            sigma_dy = sigma_map[1]  # (H_s, W_s) - y uncertainty
            if sigma_dx.shape[0] != H or sigma_dx.shape[1] != W:
                sigma_dx = F_nn.interpolate(
                    sigma_dx[None, None], size=(H, W),
                    mode='bilinear', align_corners=False)[0, 0]
                sigma_dy = F_nn.interpolate(
                    sigma_dy[None, None], size=(H, W),
                    mode='bilinear', align_corners=False)[0, 0]

        # Precompute mu components for consistency check
        mu_dx = None
        mu_dy = None
        if mu_map is not None:
            mu_dx = mu_map[0]  # (H_m, W_m)
            mu_dy = mu_map[1]
            if mu_dx.shape[0] != H or mu_dx.shape[1] != W:
                mu_dx = F_nn.interpolate(
                    mu_dx[None, None], size=(H, W),
                    mode='bilinear', align_corners=False)[0, 0]
                mu_dy = F_nn.interpolate(
                    mu_dy[None, None], size=(H, W),
                    mode='bilinear', align_corners=False)[0, 0]
        # Mu divergence threshold: cosine similarity between mu prediction
        # and direction from walk point to GT center. Below this = stop.
        # cos(45°)=0.707, cos(60°)=0.5, cos(90°)=0
        mu_cos_thresh = 0.5  # stop if mu points >60° away from GT center

        # Use original_center (annotation) for mu consistency check,
        # not the potentially refined center_point_gt
        if original_center is not None:
            orig_ctr_feat = original_center / stride
        else:
            orig_ctr_feat = center_point_gt_feat

        if first_axis_range is not None:
            first_axis_range = first_axis_range / stride

        for i in range(num_gts):
            ctr = center_point_gt_feat[i]
            eigvec_i = eigvec_norm[i]
            is_same_class = is_nearest_same_class[i]
            nearest_gt_point_i = nearest_gt_point[i] / stride
            direction = nearest_gt_point_i - ctr
            direction_norm = direction / torch.norm(direction)
            distance = torch.abs((direction * eigvec_i).sum())
            if not is_secondary:
                valid_dup_remove = torch.abs(
                    (direction_norm * eigvec_i).sum()) > 0.866
            else:
                valid_dup_remove = torch.abs(
                    (direction_norm * eigvec_i).sum()) > 0.5

            # Directional sigma at GT center for threshold modulation
            # Project sigma vector onto walk direction:
            # sigma_dir = |eigvec_x| * sigma_dx + |eigvec_y| * sigma_dy
            sigma_center_dir = None
            abs_ex = abs(eigvec_i[0].item())
            abs_ey = abs(eigvec_i[1].item())
            if sigma_dx is not None:
                cy = int(ctr[1].clamp(0, H - 1).item())
                cx = int(ctr[0].clamp(0, W - 1).item())
                sigma_center_dir = max(
                    abs_ex * sigma_dx[cy, cx].item() +
                    abs_ey * sigma_dy[cy, cx].item(), 1e-6)

            base_thresh = self.thresh3[gt_labels[i]]

            # Original GT center in feature coords for mu consistency
            orig_ctr_i = orig_ctr_feat[i]

            # Walk forward
            for j in range(default_max_length):
                point = (ctr + j * eigvec_i).round().long()
                if point[0] < 0 or point[0] >= W or point[1] < 0 or point[1] >= H:
                    top_bottom[i, 0] = j
                    break
                # Bidirectional sigma-modulated threshold
                # neutral_sigma = sigma_center * sigma_neutral (a "typical boundary" level)
                # ratio = sigma_walk / neutral_sigma
                #   ratio < 1 (clearer than neutral) → lower thresh → expand
                #   ratio > 1 (more uncertain)       → raise thresh → shrink
                if sigma_center_dir is not None:
                    s_dir = (abs_ex * sigma_dx[point[1], point[0]].item() +
                             abs_ey * sigma_dy[point[1], point[0]].item())
                    neutral_sigma = sigma_center_dir * self.sigma_neutral
                    ratio = s_dir / neutral_sigma
                    eff_thresh = base_thresh * (ratio ** self.sigma_power)
                    eff_thresh = max(min(eff_thresh, 0.9), base_thresh * 0.3)
                else:
                    eff_thresh = base_thresh
                if cls_scores_sigmoid[gt_labels[i], point[1], point[0]] < eff_thresh:
                    top_bottom[i, 0] = j
                    break
                # Sigma spike stopping: sudden uncertainty increase = boundary
                if sigma_center_dir is not None and j >= 3:
                    if ratio > self.sigma_spike_thresh:
                        top_bottom[i, 0] = j
                        break
                # Mu consistency: check if mu at walk point still points
                # toward our GT center (cosine similarity with direction to GT)
                if mu_dx is not None and j > 2:
                    mu_x = mu_dx[point[1], point[0]]
                    mu_y = mu_dy[point[1], point[0]]
                    # Direction from walk point to GT center
                    dir_x = orig_ctr_i[0] - point[0].float()
                    dir_y = orig_ctr_i[1] - point[1].float()
                    dir_norm = (dir_x ** 2 + dir_y ** 2).sqrt().clamp(min=1e-6)
                    mu_norm = (mu_x ** 2 + mu_y ** 2).sqrt().clamp(min=1e-6)
                    cos_sim = (mu_x * dir_x + mu_y * dir_y) / (mu_norm * dir_norm)
                    if cos_sim < mu_cos_thresh:
                        top_bottom[i, 0] = j
                        break
                if valid_dup_remove:
                    if is_same_class and j > 0.5 * distance:
                        top_bottom[i, 0] = j
                        break
                if first_axis_range is not None:
                    if j > 0.6 * first_axis_range[i]:
                        top_bottom[i, 0] = j
                        break

            # Walk backward
            for j in range(default_max_length):
                point = (ctr - j * eigvec_i).round().long()
                if point[0] < 0 or point[0] >= W or point[1] < 0 or point[1] >= H:
                    top_bottom[i, 1] = j
                    break
                if sigma_center_dir is not None:
                    s_dir = (abs_ex * sigma_dx[point[1], point[0]].item() +
                             abs_ey * sigma_dy[point[1], point[0]].item())
                    neutral_sigma = sigma_center_dir * self.sigma_neutral
                    ratio = s_dir / neutral_sigma
                    eff_thresh = base_thresh * (ratio ** self.sigma_power)
                    eff_thresh = max(min(eff_thresh, 0.9), base_thresh * 0.3)
                else:
                    eff_thresh = base_thresh
                if cls_scores_sigmoid[gt_labels[i], point[1], point[0]] < eff_thresh:
                    top_bottom[i, 1] = j
                    break
                # Sigma spike stopping (backward)
                if sigma_center_dir is not None and j >= 3:
                    if ratio > self.sigma_spike_thresh:
                        top_bottom[i, 1] = j
                        break
                # Mu consistency check (backward) — cosine direction
                if mu_dx is not None and j > 2:
                    mu_x = mu_dx[point[1], point[0]]
                    mu_y = mu_dy[point[1], point[0]]
                    dir_x = orig_ctr_i[0] - point[0].float()
                    dir_y = orig_ctr_i[1] - point[1].float()
                    dir_norm = (dir_x ** 2 + dir_y ** 2).sqrt().clamp(min=1e-6)
                    mu_norm = (mu_x ** 2 + mu_y ** 2).sqrt().clamp(min=1e-6)
                    cos_sim = (mu_x * dir_x + mu_y * dir_y) / (mu_norm * dir_norm)
                    if cos_sim < mu_cos_thresh:
                        top_bottom[i, 1] = j
                        break
                if is_secondary and valid_dup_remove:
                    if is_same_class and j > 0.5 * distance:
                        top_bottom[i, 0] = j - 1
                        break
                if first_axis_range is not None:
                    if j > 0.6 * first_axis_range[i]:
                        top_bottom[i, 0] = j
                        break

        return top_bottom[:, 0], top_bottom[:, 1]
