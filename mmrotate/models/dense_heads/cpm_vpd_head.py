# Copyright (c) OpenMMLab. All rights reserved.
"""CPMVPDHead: CPM Head with Point-Supervised Variational Inference.

Center-only latent: z = (delta_x, delta_y) per anchor point.
No w/h prediction — point supervision has no box size signal.

Posterior: q_phi(z|f,p) = N(mu_phi, diag(sigma_phi^2))   (2-dim)
Prior:     p_psi(z|p_i, N_i) = N(0, sigma_c^2 I)         (point-conditioned)

Network output (two separate conv heads):
    conv_reg   (2 channels): (delta_x, delta_y)   -- posterior mean mu_phi
    conv_sigma (2 channels): (log_sx, log_sy)      -- posterior log-std

Inference: center = anchor + (delta_x, delta_y) * stride
           w = h = kNN_distance (placeholder), angle = 0
"""

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import numpy as np
import os
from PIL import Image
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean

from mmrotate.core import multiclass_nms_rotated
from ..builder import ROTATED_HEADS
from .cpm_head import CPMHead
from .rotated_anchor_free_head import RotatedAnchorFreeHead

INF = 1e8


@ROTATED_HEADS.register_module()
class CPMVPDHead(CPMHead):
    """CPM Head with Point-Supervised VPD (center-only, no w/h).

    Args:
        warmup_iters (int): Iterations for KL warm-up. Default: 2000.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Dense spatial supervision for mu and sigma (configurable via train_cfg)
        train_cfg = kwargs.get('train_cfg', {})
        self.dense_radius = train_cfg.get('dense_radius', 8) if train_cfg else 8
        self.lambda_mu_dense = train_cfg.get('lambda_mu_dense', 1.0) if train_cfg else 1.0
        self.lambda_sigma_dense = train_cfg.get('lambda_sigma_dense', 1.0) if train_cfg else 1.0
        self.sigma_min_target = train_cfg.get('sigma_min_target', 1.0) if train_cfg else 1.0
        self.sigma_max_target = train_cfg.get('sigma_max_target', 4.0) if train_cfg else 4.0
        self.ambiguity_weight = train_cfg.get('ambiguity_weight', 2.0) if train_cfg else 2.0

        # Freeze base model (backbone/FPN/cls/reg_convs), only train mu+sigma
        self.freeze_base = train_cfg.get('freeze_base', False) if train_cfg else False

    def _init_predictor(self):
        """Separate heads for mu (conv_reg) and sigma (sigma_convs + conv_sigma).

        sigma has its own 2-layer conv tower so its gradients don't interfere
        with the shared reg_convs used by mu.
        """
        super()._init_predictor()
        self.conv_reg = nn.Conv2d(self.feat_channels, 2, 3, padding=1)

        # Independent sigma feature tower (2 layers, lighter than 4-layer reg)
        self.sigma_convs = nn.ModuleList()
        for i in range(2):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.sigma_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                           norm_cfg=self.norm_cfg, bias=self.conv_bias))
        self.conv_sigma = nn.Conv2d(self.feat_channels, 2, 3, padding=1)

    def forward_single(self, x, scale, stride):
        """Forward for a single FPN level.

        Returns (cls_score, bbox_pred, centerness).
        bbox_pred: (B, 4, H, W)  — [0:2]=mu, [2:4]=log_sigma
        """
        cls_score, _, cls_feat, reg_feat = \
            super(RotatedAnchorFreeHead, self).forward_single(x)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)

        bbox_mu = scale(self.conv_reg(reg_feat).float())    # (B, 2, H, W)

        # Sigma uses its own tower from raw FPN features (x), not reg_feat
        sigma_feat = x
        for conv in self.sigma_convs:
            sigma_feat = conv(sigma_feat)
        bbox_log_sigma = self.conv_sigma(sigma_feat).float()  # (B, 2, H, W)

        bbox_pred = torch.cat([bbox_mu, bbox_log_sigma], dim=1)  # (B, 4, H, W)
        return cls_score, bbox_pred, centerness

    # ------------------------------------------------------------------
    # Label assignment
    # ------------------------------------------------------------------

    def _get_target_single_vpd(self, gt_bboxes, gt_labels, points,
                                regress_ranges, num_points_per_lvl):
        """Assign labels + gt_ids per feature point."""
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

        dist_sample_and_gt = torch.cdist(points, center_point_gt)
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

        index_neg = ((alpha * dist_sample_and_gt) > dist_min_gt_and_gt).all(
            dim=1).nonzero().squeeze(-1)
        if len(index_neg) > 0:
            labels[index_neg] = self.num_classes

        thresh1_tensor = thresh1 * torch.ones_like(dist_min_gt_and_gt)
        dist_min_thresh1_gt = torch.min(dist_min_gt_and_gt / 2, thresh1_tensor)
        index_pos = (dist_sample_and_gt < dist_min_thresh1_gt).nonzero()
        if len(index_pos) > 0:
            labels[index_pos[:, 0]] = gt_labels[index_pos[:, 1]]
            gt_ids[index_pos[:, 0]] = index_pos[:, 1].long()

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
        """Return per-level labels and gt_ids."""
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
            gt_bboxes_list, gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

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
    # Visualization
    # ------------------------------------------------------------------

    def _draw_image(self, max_probs, max_indices, thr, img_flip_direction,
                    img_A, num_width):
        """Override: overlay class mask on image instead of side-by-side."""
        mask_rgb = self.get_mask_image(max_probs, max_indices, thr, num_width)

        # Undo augmentation flip on mask
        mask_pil = Image.fromarray(mask_rgb)
        if img_flip_direction == 'horizontal':
            mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
        elif img_flip_direction == 'vertical':
            mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)
        elif img_flip_direction == 'diagonal':
            mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM) \
                               .transpose(Image.FLIP_LEFT_RIGHT)
        mask_np = np.array(mask_pil)

        img_np = np.array(img_A).astype(np.float32)
        mask_f = mask_np.astype(np.float32)

        # White pixels (255,255,255) = no detection → show original image
        is_active = ~np.all(mask_np == 255, axis=-1)   # (H, W) bool
        alpha = np.where(is_active, 0.6, 0.0)[..., None]
        blended = img_np * (1.0 - alpha) + mask_f * alpha
        return Image.fromarray(blended.astype(np.uint8))

    def draw_image(self, img_path, flip, score_probs, bbox_pred):
        """Visualize CPM class mask + mu + sigma during training.

        Outputs per iteration directory:
            {thr}.jpg        — CPM class mask
            mu_dx.jpg        — predicted x-offset heatmap (diverging)
            mu_dy.jpg        — predicted y-offset heatmap (diverging)
            sigma_dx.jpg     — x uncertainty heatmap
            sigma_dy.jpg     — y uncertainty heatmap
            sigma_center.jpg — combined center uncertainty

        Args:
            img_path (str): Original image path.
            flip (str or None): Flip direction.
            score_probs (Tensor): (C, H, W) sigmoid class scores.
            bbox_pred (Tensor): (4, H, W) — [0:2]=mu, [2:4]=log_sigma.
        """
        H, W = score_probs.shape[1:]
        out_dir = os.path.join(self.store_dir, 'visualize', str(self.iter))
        os.makedirs(out_dir, exist_ok=True)

        img_A = Image.open(img_path).convert('RGB').resize((W, H))
        img_np = np.array(img_A)

        # ── CPM class mask ──
        max_probs, max_indices = torch.max(score_probs, dim=0)
        for thr in [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]:
            combined = self._draw_image(
                max_probs, max_indices, thr, flip, img_A, H)
            combined.save(os.path.join(out_dir, f'{thr}.jpg'))

        # ── Mu maps (signed offset, diverging colormap) ──
        mu = bbox_pred[0:2].detach().cpu().numpy()  # (2, H, W)
        for ch, name in enumerate(['mu_dx', 'mu_dy']):
            hm = self._mu_to_heatmap(mu[ch], img_np, flip)
            hm.save(os.path.join(out_dir, f'{name}.jpg'))

        # ── Sigma maps ──
        log_sigma = bbox_pred[2:4]
        sigma = log_sigma.exp().detach().cpu().numpy()  # (2, H, W)
        for ch, name in enumerate(['sigma_dx', 'sigma_dy']):
            hm = self._sigma_to_heatmap(sigma[ch], img_np, flip)
            hm.save(os.path.join(out_dir, f'{name}.jpg'))

        center_unc = np.sqrt(sigma[0] ** 2 + sigma[1] ** 2)
        hm = self._sigma_to_heatmap(center_unc, img_np, flip)
        hm.save(os.path.join(out_dir, 'sigma_center.jpg'))

        # ── Center uncertainty (clean, no background image) ──
        import cv2
        center_unc_flipped = self._undo_flip(center_unc, flip)
        vmin = float(np.percentile(center_unc_flipped, 2))
        vmax = float(np.percentile(center_unc_flipped, 98))
        vmax = max(vmax, vmin + 1e-6)
        norm = np.clip((center_unc_flipped - vmin) / (vmax - vmin), 0, 1)
        norm_u8 = (norm * 255).astype(np.uint8)
        hm_bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_MAGMA)
        cv2.imwrite(os.path.join(out_dir, 'sigma_center_clean.jpg'), hm_bgr)

        # ── Dump statistics for debugging ──
        stats = (
            f"mu_dx:  min={mu[0].min():.4f} max={mu[0].max():.4f} "
            f"mean={mu[0].mean():.4f} std={mu[0].std():.4f}\n"
            f"mu_dy:  min={mu[1].min():.4f} max={mu[1].max():.4f} "
            f"mean={mu[1].mean():.4f} std={mu[1].std():.4f}\n"
            f"sigma_dx: min={sigma[0].min():.4f} max={sigma[0].max():.4f} "
            f"mean={sigma[0].mean():.4f} std={sigma[0].std():.4f}\n"
            f"sigma_dy: min={sigma[1].min():.4f} max={sigma[1].max():.4f} "
            f"mean={sigma[1].mean():.4f} std={sigma[1].std():.4f}\n"
            f"log_sigma_dx: min={bbox_pred[2].min().item():.4f} "
            f"max={bbox_pred[2].max().item():.4f} "
            f"std={bbox_pred[2].std().item():.4f}\n"
            f"log_sigma_dy: min={bbox_pred[3].min().item():.4f} "
            f"max={bbox_pred[3].max().item():.4f} "
            f"std={bbox_pred[3].std().item():.4f}\n"
        )
        with open(os.path.join(out_dir, 'stats.txt'), 'w') as f:
            f.write(stats)

    @staticmethod
    def _undo_flip(arr, flip):
        """Undo data-augmentation flip on a (H, W) numpy array."""
        if flip == 'horizontal':
            return arr[:, ::-1].copy()
        elif flip == 'vertical':
            return arr[::-1, :].copy()
        elif flip == 'diagonal':
            return arr[::-1, ::-1].copy()
        return arr

    @staticmethod
    def _mu_to_heatmap(mu_hw, img_np, flip):
        """(H, W) signed mu → overlay heatmap (blue=negative, red=positive)."""
        import cv2
        mu_hw = CPMVPDHead._undo_flip(mu_hw, flip)

        # Symmetric normalisation around 0
        amax = max(float(np.abs(mu_hw).max()), 1e-6)
        norm = np.clip((mu_hw + amax) / (2 * amax), 0, 1)  # [0, 1], 0.5 = zero
        norm_u8 = (norm * 255).astype(np.uint8)
        hm_bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_COOL)
        hm_rgb = cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)
        blended = (img_np.astype(np.float32) * 0.4
                   + hm_rgb.astype(np.float32) * 0.6).astype(np.uint8)
        return Image.fromarray(blended)

    @staticmethod
    def _sigma_to_heatmap(sigma_hw, img_np, flip):
        """(H, W) sigma → overlay heatmap on original image."""
        import cv2
        sigma_hw = CPMVPDHead._undo_flip(sigma_hw, flip)

        vmin = float(np.percentile(sigma_hw, 2))
        vmax = float(np.percentile(sigma_hw, 98))
        vmax = max(vmax, vmin + 1e-6)
        norm = np.clip((sigma_hw - vmin) / (vmax - vmin), 0, 1)
        norm_u8 = (norm * 255).astype(np.uint8)
        hm_bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_MAGMA)
        hm_rgb = cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)
        blended = (img_np.astype(np.float32) * 0.4
                   + hm_rgb.astype(np.float32) * 0.6).astype(np.uint8)
        return Image.fromarray(blended)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self, cls_scores, bbox_preds, centernesses,
             gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss: cls + dense mu/sigma."""
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        import torch.nn.functional as F

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        labels, _ = self.get_targets_vpd(
            all_level_points, gt_bboxes, gt_labels)

        if self.visualize and self.store_dir and self.iter % self.train_duration == 0:
            self.draw_image(img_metas[0]['filename'],
                            img_metas[0].get('flip_direction'),
                            cls_scores[0][0].sigmoid(),
                            bbox_preds[0][0])
        self.iter += 1

        num_imgs = cls_scores[0].size(0)

        flatten_cls_scores = torch.cat([
            cs.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cs in cls_scores])
        flatten_bbox_preds = torch.cat([
            bp.permute(0, 2, 3, 1).reshape(-1, 4)
            for bp in bbox_preds])
        flatten_labels = torch.cat(labels)
        flatten_points = torch.cat(
            [pts.repeat(num_imgs, 1) for pts in all_level_points])
        flatten_strides = torch.cat([
            bbox_preds[0].new_full((num_imgs * pts.shape[0],), s)
            for pts, s in zip(all_level_points, self.strides)])

        num_pts_per_lvl = [pts.shape[0] for pts in all_level_points]
        img_ids = torch.cat([
            torch.arange(num_imgs, dtype=torch.long,
                         device=bbox_preds[0].device).repeat_interleave(n_pts)
            for n_pts in num_pts_per_lvl])

        # ── Classification loss (unchanged) ──────────────────────────
        bg_class_ind = self.num_classes
        avail_inds = (flatten_labels >= 0).nonzero().reshape(-1)
        pos_inds = ((flatten_labels >= 0) &
                    (flatten_labels < bg_class_ind)).nonzero().reshape(-1)

        num_avail = max(reduce_mean(torch.tensor(
            len(avail_inds), dtype=torch.float,
            device=bbox_preds[0].device)), 1.0)

        loss_cls = self.loss_cls(
            flatten_cls_scores[avail_inds],
            flatten_labels[avail_inds],
            avg_factor=num_avail)

        # ── Dense mu & sigma losses (all points, per-image) ─────────
        P = flatten_points.shape[0]
        all_mu = flatten_bbox_preds[:, :2]           # (P, 2)
        all_log_sigma = flatten_bbox_preds[:, 2:]    # (P, 2)

        # Per-point: nearest GT center, distance, and offset target
        # Computed in no_grad to save memory (these are fixed targets)
        d_nearest = torch.full((P,), float('inf'), device=flatten_points.device)
        d_second = torch.full((P,), float('inf'), device=flatten_points.device)
        gt_delta_dense = torch.zeros(P, 2, device=flatten_points.device)

        with torch.no_grad():
            for img_id in range(num_imgs):
                img_mask = (img_ids == img_id)
                gt_c = gt_bboxes[img_id][:, :2]
                if gt_c.shape[0] == 0:
                    continue
                pts = flatten_points[img_mask]                     # (P_i, 2)
                strides_i = flatten_strides[img_mask].unsqueeze(1) # (P_i, 1)

                # Chunked cdist to limit peak memory
                chunk_size = 4096
                pts_chunks = pts.split(chunk_size, dim=0)
                strides_chunks = strides_i.split(chunk_size, dim=0)
                idx_start = 0
                inds = img_mask.nonzero(as_tuple=False).squeeze(1)

                for pts_c, strides_c in zip(pts_chunks, strides_chunks):
                    n = pts_c.shape[0]
                    dists_c = torch.cdist(pts_c, gt_c)        # (n, M_i)
                    sl = inds[idx_start:idx_start + n]

                    if gt_c.shape[0] >= 2:
                        top2, _ = dists_c.topk(2, dim=1, largest=False)
                        d_nearest[sl] = top2[:, 0]
                        d_second[sl] = top2[:, 1]
                    else:
                        d_nearest[sl] = dists_c[:, 0]

                    min_idx = dists_c.argmin(dim=1)
                    nearest_gt = gt_c[min_idx]
                    gt_delta_dense[sl] = (nearest_gt - pts_c) / strides_c

                    idx_start += n
                    del dists_c  # free immediately

        # Mu: weighted SmoothL1 within dense_radius (stride units)
        d_norm = d_nearest / flatten_strides                   # (P,)
        mu_weight = (1.0 - d_norm / self.dense_radius).clamp(min=0)  # (P,)
        # Weighted element-wise SmoothL1
        mu_loss_per_pt = F.smooth_l1_loss(
            all_mu, gt_delta_dense.detach(),
            reduction='none', beta=0.5)                        # (P, 2)
        mu_loss_weighted = (mu_loss_per_pt * mu_weight.unsqueeze(1)).sum() \
            / max(mu_weight.sum(), 1.0)
        loss_mu_dense = self.lambda_mu_dense * mu_loss_weighted

        # Sigma: distance-based target with boundary ambiguity boost
        # ambiguity = d_nearest / d_second ∈ [0, 1];  ≈1 at boundary, ≈0 near GT
        # When d_second is inf (single GT), ambiguity = 0 (no boost)
        ambiguity = torch.where(
            d_second.isinf(),
            torch.zeros_like(d_nearest),
            (d_nearest / d_second.clamp(min=1e-6)).clamp(max=1.0))
        sigma_target = d_norm.clamp(min=self.sigma_min_target) \
            * (1.0 + self.ambiguity_weight * ambiguity)
        sigma_target = sigma_target.clamp(max=self.sigma_max_target)
        log_sigma_target = sigma_target.log().unsqueeze(1).expand(-1, 2)

        # Weight sigma loss: linear decay like mu, but wider radius to cover
        # the full transition from GT center (sigma_target=0.5) to background
        # (sigma_target=10). This ensures both low and high sigma targets
        # contribute proportionally, rather than extremes dominating.
        sigma_radius = self.dense_radius * 1.5  # 12 stride units
        sigma_weight = (1.0 - d_norm / sigma_radius).clamp(min=0)  # (P,)
        sigma_loss_per_pt = F.smooth_l1_loss(
            all_log_sigma, log_sigma_target.detach(),
            reduction='none', beta=1.0)                        # (P, 2)
        loss_sigma_dense = self.lambda_sigma_dense * \
            (sigma_loss_per_pt * sigma_weight.unsqueeze(1)).sum() \
            / max(sigma_weight.sum(), 1.0)

        return dict(
            loss_cls=loss_cls,
            loss_mu_dense=loss_mu_dense,
            loss_sigma_dense=loss_sigma_dense,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self, cls_scores, bbox_preds, centernesses,
                   img_metas, cfg=None, rescale=None):
        """Inference: decode center, use placeholder w/h, apply NMS."""
        cfg = cfg or self.test_cfg
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
                cls_score = cls_scores[lvl_idx][img_id]
                bbox_pred = bbox_preds[lvl_idx][img_id]  # (4, H, W)
                centerness = centernesses[lvl_idx][img_id]
                points = mlvl_points[lvl_idx]

                cls_score = cls_score.permute(1, 2, 0).reshape(
                    -1, self.cls_out_channels)
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                centerness = centerness.reshape(-1).sigmoid()

                scores = cls_score.sigmoid() * centerness[:, None]

                nms_pre = cfg.get('nms_pre', 2000) if cfg else 2000
                nms_pre = min(nms_pre, scores.shape[0])
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)

                points = points[topk_inds]
                pred_delta = bbox_pred[topk_inds, :2]
                scores = scores[topk_inds]

                # Decode center
                if self.norm_on_bbox:
                    cx = points[:, 0] + pred_delta[:, 0] * stride
                    cy = points[:, 1] + pred_delta[:, 1] * stride
                else:
                    cx = points[:, 0] + pred_delta[:, 0]
                    cy = points[:, 1] + pred_delta[:, 1]

                # Placeholder w/h: use stride as a rough size proxy
                w = torch.full_like(cx, stride * 4.0)
                h = torch.full_like(cy, stride * 4.0)
                angle = torch.zeros_like(cx)

                decoded = torch.stack([cx, cy, w, h, angle], dim=1)
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
        return super(CPMVPDHead, self).get_targets(
            points, gt_bboxes_list, gt_labels_list)
