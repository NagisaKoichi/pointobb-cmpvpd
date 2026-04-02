# Copyright (c) OpenMMLab. All rights reserved.
"""CPMVPDHead: CPM Head with Point-Supervised Variational Inference.

Latent variable z = (delta_x, delta_y, log_w, log_h) per anchor point.
Posterior: q_phi(z|f,p) = N(mu_phi, diag(sigma_phi^2))
Prior:     p_psi(z|p_i, N_i) = N(mu_psi, diag(sigma_psi^2))  [point-conditioned]

Network output (conv_reg, 8 channels):
    [0:4] = (delta_x, delta_y, log_w, log_h)  -- posterior mean mu_phi
    [4:8] = (log_sx, log_sy, log_sw, log_sh)  -- posterior log-std

Inference: center = anchor + (delta_x, delta_y),
           box = (center_x, center_y, exp(log_w), exp(log_h))
"""

import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean

from mmrotate.core import multiclass_nms_rotated
from ..builder import ROTATED_HEADS, build_loss
from .cpm_head import CPMHead
from .rotated_anchor_free_head import RotatedAnchorFreeHead

INF = 1e8


@ROTATED_HEADS.register_module()
class CPMVPDHead(CPMHead):
    """CPM Head with Point-Supervised Variational Inference.

    Predicts a Gaussian posterior q_phi(z|f,p) over the latent proposal state
    z = (delta_x, delta_y, log_w, log_h), regularized by a point-conditioned
    prior p_psi(z|p_i, N_i) constructed from GT point annotation geometry.

    Args:
        warmup_iters (int): Iterations for KL warm-up. Default: 2000.
        num_samples (int): Number of samples for inference refinement. Default: 10.
        use_refinement (bool): Use sampling-based bbox refinement at test time.
    """

    def __init__(self, *args,
                 warmup_iters=2000,
                 num_samples=10,
                 use_refinement=False,
                 **kwargs):
        self.warmup_iters = warmup_iters
        self.num_samples = num_samples
        self.use_refinement = use_refinement
        super().__init__(*args, **kwargs)

        train_cfg = kwargs.get('train_cfg', {})
        test_cfg = kwargs.get('test_cfg', {})
        if 'warmup_iters' in train_cfg:
            self.warmup_iters = train_cfg['warmup_iters']
        if 'num_samples' in test_cfg:
            self.num_samples = test_cfg['num_samples']
        if 'use_refinement' in test_cfg:
            self.use_refinement = test_cfg['use_refinement']

        self.loss_vpd = build_loss(dict(
            type='PointSupervisedVPDLoss',
            lambda_center=1.0,
            lambda_kl=0.1,
            lambda_kl_warmup=0.02,
            lambda_var=0.01,
            warmup_iters=self.warmup_iters,
        ))

    def _init_predictor(self):
        """Override predictor: conv_reg outputs 8 channels (4 mu + 4 log_sigma).

        _init_predictor is called last in _init_layers, after _init_reg_convs,
        so overriding here prevents the base class from re-creating a 4-ch conv_reg.
        """
        super()._init_predictor()
        # Replace conv_reg with 8-channel version:
        # (delta_x, delta_y, log_w, log_h, log_sx, log_sy, log_sw, log_sh)
        self.conv_reg = nn.Conv2d(self.feat_channels, 8, 3, padding=1)

    def forward_single(self, x, scale, stride):
        """Forward for a single FPN level. Returns (cls_score, bbox_pred, centerness).

        bbox_pred: (N, 8, H, W)
            [:, 0:4] = posterior mean (delta_x, delta_y, log_w, log_h)
            [:, 4:8] = posterior log-std
        """
        cls_score, _, cls_feat, reg_feat = \
            super(RotatedAnchorFreeHead, self).forward_single(x)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)

        bbox_dist = self.conv_reg(reg_feat).float()  # (N, 8, H, W)

        # Mean: scale delta_x, delta_y; keep log_w, log_h as-is from network
        # (Scale layer applied to all 4 channels, then we split semantics)
        bbox_mu = scale(bbox_dist[:, :4])   # (N, 4, H, W)
        bbox_log_sigma = bbox_dist[:, 4:]   # (N, 4, H, W)

        bbox_pred_full = torch.cat([bbox_mu, bbox_log_sigma], dim=1)  # (N, 8, H, W)
        return cls_score, bbox_pred_full, centerness

    # ------------------------------------------------------------------
    # Label assignment: override _get_target_single to also return gt_ids
    # ------------------------------------------------------------------

    def _get_target_single_vpd(self, gt_bboxes, gt_labels, points,
                                regress_ranges, num_points_per_lvl):
        """Like CPMHead._get_target_single but also returns gt_instance_ids.

        gt_instance_ids[i] = index into gt_bboxes for positive point i,
                             -1 for ignored/negative points.
        """
        alpha = self.alpha
        thresh1 = self.thresh1
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        center_point_gt = gt_bboxes[:, :2]

        labels = -1 * torch.ones(num_points, dtype=gt_labels.dtype,
                                  device=gt_labels.device)
        gt_ids = torch.full((num_points,), -1, dtype=torch.long,
                             device=gt_labels.device)

        if num_gts == 0:
            return (gt_labels.new_full((num_points,), self.num_classes),
                    gt_ids)

        dist_sample_and_gt = torch.cdist(points, center_point_gt)  # (P, G)
        dist_gt_and_gt = (torch.cdist(center_point_gt, center_point_gt)
                          + torch.eye(num_gts, device=dist_sample_and_gt.device) * INF)
        dist_min_gt_and_gt, dist_min_gt_and_gt_index = dist_gt_and_gt.min(dim=1)

        if num_gts == 1:
            index_pos = (dist_sample_and_gt < 8).nonzero().reshape(-1)
            index_neg = (dist_sample_and_gt > 128).nonzero().reshape(-1)
            labels[index_pos] = gt_labels[0]
            gt_ids[index_pos] = 0
            labels[index_neg] = self.num_classes
            return labels, gt_ids

        # Negative labels
        index_neg = ((alpha * dist_sample_and_gt) > dist_min_gt_and_gt).all(
            dim=1).nonzero().squeeze(-1)
        if len(index_neg) > 0:
            labels[index_neg] = self.num_classes

        # Positive labels
        thresh1_tensor = thresh1 * torch.ones_like(dist_min_gt_and_gt)
        dist_min_thresh1_gt = torch.min(dist_min_gt_and_gt / 2, thresh1_tensor)
        index_pos = (dist_sample_and_gt < dist_min_thresh1_gt).nonzero()
        if len(index_pos) > 0:
            labels[index_pos[:, 0]] = gt_labels[index_pos[:, 1]]
            gt_ids[index_pos[:, 0]] = index_pos[:, 1].long()

        # Additional background labels (midpoints between same-class neighbors)
        is_nearest_same_class = (gt_labels[dist_min_gt_and_gt_index] == gt_labels)
        valid_middle_point = (
            center_point_gt[is_nearest_same_class]
            + center_point_gt[dist_min_gt_and_gt_index][is_nearest_same_class]) / 2
        dist_sample_and_mid = torch.cdist(points, valid_middle_point)
        index_neg_additional = (dist_sample_and_mid < 4).any(
            dim=1).nonzero().squeeze(-1)
        if len(index_neg_additional) > 0:
            labels[index_neg_additional] = self.num_classes

        return labels, gt_ids

    def get_targets_vpd(self, points, gt_bboxes_list, gt_labels_list):
        """Like CPMHead.get_targets but also returns per-point gt_instance_ids.

        Returns:
            concat_lvl_labels (list[Tensor]): Per-level class labels.
            concat_lvl_gt_ids (list[Tensor]): Per-level GT instance index (-1=ignore/neg).
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i])
            for i in range(num_levels)]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]

        labels_list, gt_ids_list = multi_apply(
            self._get_target_single_vpd,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # Split per-level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        gt_ids_list = [gt_ids.split(num_points, 0) for gt_ids in gt_ids_list]

        concat_lvl_labels = []
        concat_lvl_gt_ids = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_gt_ids.append(
                torch.cat([gt_ids[i] for gt_ids in gt_ids_list]))

        return concat_lvl_labels, concat_lvl_gt_ids

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self, cls_scores, bbox_preds, centernesses,
             gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute ELBO loss for point-supervised VPD."""
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        # Label assignment with GT instance IDs
        labels, gt_ids = self.get_targets_vpd(
            all_level_points, gt_bboxes, gt_labels)

        if self.visualize and self.store_dir and self.iter % self.train_duration == 0:
            self.draw_image(img_metas[0]['filename'],
                            img_metas[0].get('flip_direction'),
                            cls_scores[0][0].sigmoid())
        self.iter += 1

        num_imgs = cls_scores[0].size(0)

        # Flatten predictions, also build per-point stride for coord conversion
        flatten_cls_scores = torch.cat([
            cs.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cs in cls_scores])
        flatten_bbox_preds = torch.cat([
            bp.permute(0, 2, 3, 1).reshape(-1, 8)
            for bp in bbox_preds])
        flatten_labels = torch.cat(labels)
        flatten_gt_ids = torch.cat(gt_ids)
        flatten_points = torch.cat(
            [pts.repeat(num_imgs, 1) for pts in all_level_points])

        # Per-point stride (for converting norm_on_bbox predictions to pixel space)
        flatten_strides = torch.cat([
            bbox_preds[0].new_full((num_imgs * pts.shape[0],), s)
            for pts, s in zip(all_level_points, self.strides)])

        # Which image does each flattened sample belong to?
        # flatten order: for each level, all images are stacked
        # i.e. [img0_lvl0..., img1_lvl0..., img0_lvl1..., img1_lvl1...]
        num_pts_per_lvl = [pts.shape[0] for pts in all_level_points]
        img_ids = torch.cat([
            torch.arange(num_imgs, dtype=torch.long,
                         device=bbox_preds[0].device).repeat_interleave(n_pts)
            for n_pts in num_pts_per_lvl])

        bg_class_ind = self.num_classes
        avail_inds = (flatten_labels >= 0).nonzero().reshape(-1)
        pos_inds = ((flatten_labels >= 0) &
                    (flatten_labels < bg_class_ind)).nonzero().reshape(-1)

        num_avail = max(reduce_mean(torch.tensor(
            len(avail_inds), dtype=torch.float,
            device=bbox_preds[0].device)), 1.0)

        # Classification loss
        loss_cls = self.loss_cls(
            flatten_cls_scores[avail_inds],
            flatten_labels[avail_inds],
            avg_factor=num_avail)

        # VPD loss on positive samples
        if len(pos_inds) == 0:
            zero = flatten_bbox_preds.sum() * 0.0
            return dict(loss_cls=loss_cls,
                        loss_center=zero, loss_kl=zero, loss_var=zero)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]   # (Np, 8)
        pos_points = flatten_points[pos_inds]           # (Np, 2)
        pos_img_ids = img_ids[pos_inds]                 # (Np,)
        pos_gt_ids = flatten_gt_ids[pos_inds]           # (Np,) -- index into per-img gt
        pos_strides = flatten_strides[pos_inds]         # (Np,)

        bbox_mu = pos_bbox_preds[:, :4]        # (Np, 4)
        bbox_log_sigma = pos_bbox_preds[:, 4:] # (Np, 4)

        # Build matched gt_centers per positive sample (image coords)
        gt_centers_per_pos = torch.zeros(
            len(pos_inds), 2, device=bbox_preds[0].device)
        for img_id in range(num_imgs):
            mask = (pos_img_ids == img_id)
            if not mask.any():
                continue
            gt_center_this = gt_bboxes[img_id][:, :2]  # (num_gt_i, 2)
            ids_this = pos_gt_ids[mask]
            gt_centers_per_pos[mask] = gt_center_this[ids_this]

        # GT centers list for kNN prior (image coords, one entry per image)
        gt_centers_list = [gt_bbox[:, :2] for gt_bbox in gt_bboxes]

        vpd_losses = self.loss_vpd(
            bbox_mu=bbox_mu,
            bbox_log_sigma=bbox_log_sigma,
            pos_points=pos_points,
            pos_strides=pos_strides,
            gt_centers=gt_centers_per_pos,
            gt_centers_list=gt_centers_list,
            cur_iter=self.iter,
        )

        return dict(
            loss_cls=loss_cls,
            loss_center=vpd_losses['loss_center'],
            loss_kl=vpd_losses['loss_kl'],
            loss_var=vpd_losses['loss_var'],
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _decode_bbox_from_mu(self, points, bbox_mu, stride):
        """Decode posterior mean to absolute box (cx, cy, w, h, angle=0).

        bbox_mu = (delta_x, delta_y, log_w, log_h) in feature-map units.
        With norm_on_bbox=True the network learns delta/stride, so we multiply.

        Returns:
            Tensor: (N, 5) rotated box (cx, cy, w, h, angle) with angle=0.
        """
        dx = bbox_mu[:, 0]
        dy = bbox_mu[:, 1]
        log_w = bbox_mu[:, 2]
        log_h = bbox_mu[:, 3]

        if self.norm_on_bbox:
            cx = points[:, 0] + dx * stride
            cy = points[:, 1] + dy * stride
            w = log_w.exp() * stride
            h = log_h.exp() * stride
        else:
            cx = points[:, 0] + dx
            cy = points[:, 1] + dy
            w = log_w.exp()
            h = log_h.exp()

        angle = torch.zeros_like(cx)
        return torch.stack([cx, cy, w, h, angle], dim=1)  # (N, 5)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self, cls_scores, bbox_preds, centernesses,
                   img_metas, cfg=None, rescale=None):
        """Inference: decode posterior mean, apply NMS."""
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        result_list = []
        for img_id in range(len(img_metas)):
            mlvl_bboxes = []
            mlvl_scores = []

            for lvl_idx in range(num_levels):
                stride = self.strides[lvl_idx]
                cls_score = cls_scores[lvl_idx][img_id]  # (C, H, W)
                bbox_pred = bbox_preds[lvl_idx][img_id]  # (8, H, W)
                centerness = centernesses[lvl_idx][img_id]  # (1, H, W)
                points = mlvl_points[lvl_idx]              # (H*W, 2)

                # Flatten
                cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 8)
                centerness = centerness.reshape(-1).sigmoid()

                scores = cls_score.sigmoid()
                scores = scores * centerness[:, None]

                # Top-k selection
                nms_pre = cfg.get('nms_pre', 2000) if cfg else 2000
                max_scores, _ = scores.max(dim=1)
                nms_pre = min(nms_pre, scores.shape[0])
                _, topk_inds = max_scores.topk(nms_pre)

                points = points[topk_inds]
                bbox_mu = bbox_pred[topk_inds, :4]  # use mean only
                scores = scores[topk_inds]

                # Optionally sample multiple times for refinement
                if self.use_refinement:
                    bbox_log_sigma = bbox_pred[topk_inds, 4:]
                    bbox_std = bbox_log_sigma.exp()
                    samples = [bbox_mu + bbox_std * torch.randn_like(bbox_mu)
                               for _ in range(self.num_samples)]
                    bbox_mu = torch.stack(samples, dim=0).mean(dim=0)

                decoded = self._decode_bbox_from_mu(points, bbox_mu, stride)
                mlvl_bboxes.append(decoded)
                mlvl_scores.append(scores)

            mlvl_bboxes = torch.cat(mlvl_bboxes)
            mlvl_scores = torch.cat(mlvl_scores)

            if rescale:
                scale_factor = mlvl_bboxes.new_tensor(
                    img_metas[img_id]['scale_factor'][:2]).repeat(2)
                mlvl_bboxes[:, :4] /= scale_factor

            det_bboxes, det_labels = multiclass_nms_rotated(
                mlvl_bboxes, mlvl_scores,
                cfg.score_thr, cfg.nms, cfg.max_per_img)
            result_list.append((det_bboxes, det_labels))

        return result_list

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Delegate to parent CPMHead.get_targets (used by base class calls)."""
        return super(CPMVPDHead, self).get_targets(
            points, gt_bboxes_list, gt_labels_list)
