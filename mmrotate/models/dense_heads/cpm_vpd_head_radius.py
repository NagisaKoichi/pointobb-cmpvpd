# Copyright (c) OpenMMLab. All rights reserved.
"""CPMVPDHead: CPM Head with Point-Supervised VPD-style distribution loss.

Latent variable z = (delta_x, delta_y) per anchor point.
Posterior: q_phi(z|f,p) = N(mu_phi, diag(sigma_phi^2))

Network output (conv_reg, 4 channels):
    [0:2] = (delta_x, delta_y)  -- posterior mean mu_phi
    [2:4] = (log_sx, log_sy)    -- posterior log-std
"""

import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean

from mmrotate.core import multiclass_nms_rotated
from ..builder import ROTATED_HEADS, build_loss
from .cpm_head import CPMHead
from .rotated_anchor_free_head import RotatedAnchorFreeHead

INF = 1e8


@ROTATED_HEADS.register_module()
class CPMVPDHead(CPMHead):
    """CPM Head with Point-Supervised VPD-style XY distribution supervision.

    Predicts a Gaussian posterior q_phi(z|f,p) over the latent proposal state
    z = (delta_x, delta_y), trained with projected-target JS divergence.

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
        self.num_samples_train = 1
        super().__init__(*args, **kwargs)

        train_cfg = kwargs.get('train_cfg') or {}
        test_cfg = kwargs.get('test_cfg') or {}
        if 'warmup_iters' in train_cfg:
            self.warmup_iters = train_cfg['warmup_iters']
        if 'num_samples' in test_cfg:
            self.num_samples = test_cfg['num_samples']
        if 'use_refinement' in test_cfg:
            self.use_refinement = test_cfg['use_refinement']
        if 'num_samples_train' in train_cfg:
            self.num_samples_train = train_cfg['num_samples_train']
        self.use_remap_score = bool(test_cfg.get('use_remap_score', False))
        self.js_weight = float(train_cfg.get('js_weight', 1.0))
        self.js_project_min = float(train_cfg.get('js_project_min', -16.0))
        self.js_project_max = float(train_cfg.get('js_project_max', 16.0))
        self.js_num_bins = int(train_cfg.get('js_num_bins', 33))
        
        # --- 新增：解析动态分配过渡参数 ---
        # 默认前期 2000 iter 纯固定 radius，4000 iter 后完全使用 VPD 校准
        self.assign_transition_start = int(train_cfg.get('assign_transition_start', 2000))
        self.assign_transition_end = int(train_cfg.get('assign_transition_end', 7000))
        # -----------------------------------

        self.visualize_variance_map = bool(
            train_cfg.get('visualize_variance_map', False))
        if self.visualize_variance_map and self.store_dir:
            os.makedirs(os.path.join(self.store_dir, 'variance_map'), exist_ok=True)

        self.loss_vpd = build_loss(dict(
            type='PointSupervisedVPDLoss',
            lambda_center=self.js_weight,
            project_min=self.js_project_min,
            project_max=self.js_project_max,
            num_bins=self.js_num_bins,
        ))

    def _init_predictor(self):
        """Override predictor: conv_reg outputs 4 channels (2 mu + 2 log_sigma).

        _init_predictor is called last in _init_layers, after _init_reg_convs,
        so overriding here prevents the base class from re-creating a 4-ch conv_reg.
        """
        super()._init_predictor()
        # Replace conv_reg with 4-channel version:
        # (delta_x, delta_y, log_sx, log_sy)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def forward_single(self, x, scale, stride):
        """Forward for a single FPN level. Returns (cls_score, bbox_pred, centerness).

        bbox_pred: (N, 4, H, W)
            [:, 0:2] = posterior mean (delta_x, delta_y)
            [:, 2:4] = posterior log-std
        """
        cls_score, _, cls_feat, reg_feat = \
            super(RotatedAnchorFreeHead, self).forward_single(x)

        if self.centerness_on_reg:  # if centerness is computed from reg branch, use reg_feat; else use cls_feat
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)

        bbox_dist = self.conv_reg(reg_feat).float()  # (N, 4, H, W)

        # Mean: scale center offsets only.
        bbox_mu = scale(bbox_dist[:, :2])   # (N, 2, H, W)
        bbox_log_sigma = bbox_dist[:, 2:]   # (N, 2, H, W)

        bbox_pred_full = torch.cat([bbox_mu, bbox_log_sigma], dim=1)  # (N, 4, H, W)
        return cls_score, bbox_pred_full, centerness

    def _save_variance_map(self, img_path, flip_direction, bbox_pred_lvl):
        """Visualize and save center/scale lstd maps from predicted log-std."""
        if bbox_pred_lvl.dim() != 3 or bbox_pred_lvl.shape[0] < 4:
            return

        # Channels map to lstd of (x, y).
        log_sigma = bbox_pred_lvl[2:4]
        lstd = torch.nan_to_num(log_sigma, nan=0.0, posinf=1e4, neginf=-1e4)

        center_lstd = lstd.mean(dim=0)

        def to_heatmap(var_tensor):
            var_np = var_tensor.detach().cpu().numpy().astype(np.float32)
            var_min = float(var_np.min())
            var_max = float(var_np.max())
            if var_max - var_min < 1e-8:
                norm = np.zeros_like(var_np, dtype=np.float32)
            else:
                norm = (var_np - var_min) / (var_max - var_min)

            # Lightweight blue->cyan->yellow->red colormap without extra deps.
            heatmap = np.zeros((norm.shape[0], norm.shape[1], 3), dtype=np.uint8)
            heatmap[..., 0] = np.clip(255.0 * norm, 0, 255).astype(np.uint8)
            heatmap[..., 1] = np.clip(255.0 * (1.0 - np.abs(2.0 * norm - 1.0)), 0, 255).astype(np.uint8)
            heatmap[..., 2] = np.clip(255.0 * (1.0 - norm), 0, 255).astype(np.uint8)

            var_img = Image.fromarray(heatmap, mode='RGB')
            if flip_direction == 'horizontal':
                var_img = var_img.transpose(Image.FLIP_LEFT_RIGHT)
            elif flip_direction == 'vertical':
                var_img = var_img.transpose(Image.FLIP_TOP_BOTTOM)
            elif flip_direction == 'diagonal':
                var_img = var_img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
            return var_img

        center_img = to_heatmap(center_lstd)
        base_img = Image.open(img_path).convert('RGB')
        base_img = base_img.resize((center_img.width, center_img.height))

        center_merged = Image.new('RGB', (base_img.width + center_img.width, base_img.height))
        center_merged.paste(base_img, (0, 0))
        center_merged.paste(center_img, (base_img.width, 0))

        out_dir = os.path.join(self.store_dir, 'variance_map', str(self.iter))
        os.makedirs(out_dir, exist_ok=True)
        center_merged.save(os.path.join(out_dir, 'center_lstd.jpg'))

    # ------------------------------------------------------------------
    # Label assignment: override _get_target_single to also return gt_ids
    # ------------------------------------------------------------------

    def _get_target_single_vpd(self, gt_bboxes, gt_labels, bbox_preds_single, points, regress_ranges, strides, num_points_per_lvl):
        """Like CPMHead._get_target_single but also returns gt_instance_ids, candidate_gt_ids 
        and supports VPD-calibrated distance during later training stages.
        """
        alpha = self.alpha
        thresh1 = self.thresh1
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        center_point_gt = gt_bboxes[:, :2]
        labels = -1 * torch.ones(num_points, dtype=gt_labels.dtype, device=gt_labels.device)
        gt_ids = torch.full((num_points,), -1, dtype=torch.long, device=gt_labels.device)
        
        # --- 新增：初始化 candidate_gt_ids ---
        candidate_gt_ids = torch.full((num_points,), -1, dtype=torch.long, device=gt_labels.device)

        if num_gts == 0:
            return (gt_labels.new_full((num_points,), self.num_classes), gt_ids, candidate_gt_ids)

        dist_sample_and_gt = torch.cdist(points, center_point_gt)  # (P, G) 原始几何距离

        # --- 修改开始: 构建 VPD Loss 候选池 ---
        # 设定候选池阈值，例如 thresh1 * 2.0 范围内的点都参与 VPD Loss
        candidate_thresh = thresh1 * 1.5
        min_geo_dist, min_geo_gt_inds = dist_sample_and_gt.min(dim=1)
        candidate_mask = min_geo_dist < candidate_thresh
        candidate_gt_ids[candidate_mask] = min_geo_gt_inds[candidate_mask]
        # --- 修改结束 ---

        # --- VPD 预测校准距离混合 ---
        t_start = self.assign_transition_start
        t_end = self.assign_transition_end
        if self.iter < t_start:
            w_vpd = 0.0
        elif self.iter > t_end:
            w_vpd = 1.0
        else:
            w_vpd = float(self.iter - t_start) / float(t_end - t_start)

        if w_vpd > 0.0 and bbox_preds_single is not None:
            delta_pred = bbox_preds_single[:, :2] * strides.unsqueeze(1)
            pred_centers = points + delta_pred
            dist_pred_to_gt = torch.cdist(pred_centers, center_point_gt)
            dist_mixed = (1.0 - w_vpd) * dist_sample_and_gt + w_vpd * dist_pred_to_gt
        else:
            dist_mixed = dist_sample_and_gt

        dist_gt_and_gt = (torch.cdist(center_point_gt, center_point_gt) + torch.eye(num_gts, device=dist_sample_and_gt.device) * INF)
        dist_min_gt_and_gt, dist_min_gt_and_gt_index = dist_gt_and_gt.min(dim=1)

        if num_gts == 1:
            index_pos = (dist_mixed < 8).nonzero().reshape(-1)
            index_neg = (dist_mixed > 128).nonzero().reshape(-1)
            labels[index_pos] = gt_labels[0]
            gt_ids[index_pos] = 0
            labels[index_neg] = self.num_classes
            return labels, gt_ids, candidate_gt_ids

        index_neg = ((alpha * dist_mixed) > dist_min_gt_and_gt).all(dim=1).nonzero().squeeze(-1)
        if len(index_neg) > 0:
            labels[index_neg] = self.num_classes

        thresh1_tensor = thresh1 * torch.ones_like(dist_min_gt_and_gt)
        dist_min_thresh1_gt = torch.min(dist_min_gt_and_gt / 2, thresh1_tensor)
        index_pos = (dist_mixed < dist_min_thresh1_gt).nonzero()
        if len(index_pos) > 0:
            labels[index_pos[:, 0]] = gt_labels[index_pos[:, 1]]
            gt_ids[index_pos[:, 0]] = index_pos[:, 1].long()

        is_nearest_same_class = (gt_labels[dist_min_gt_and_gt_index] == gt_labels)
        valid_middle_point = (center_point_gt[is_nearest_same_class] + center_point_gt[dist_min_gt_and_gt_index][is_nearest_same_class]) / 2
        dist_sample_and_mid = torch.cdist(points, valid_middle_point)
        index_neg_additional = (dist_sample_and_mid < 4).any(dim=1).nonzero().squeeze(-1)
        if len(index_neg_additional) > 0:
            labels[index_neg_additional] = self.num_classes

        # 返回三个变量
        return labels, gt_ids, candidate_gt_ids

    def get_targets_vpd(self, points, gt_bboxes_list, gt_labels_list, bbox_preds_list=None):
        """Like CPMHead.get_targets but also returns per-point gt_instance_ids and candidate_gt_ids."""
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i])
            for i in range(num_levels)]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        concat_strides = []
        for i in range(num_levels):
            concat_strides.append(points[i].new_full((points[i].size(0),), self.strides[i]))
        concat_strides = torch.cat(concat_strides)

        num_points = [center.size(0) for center in points]
        if bbox_preds_list is None:
            bbox_preds_list = [None] * len(gt_bboxes_list)

        # multi_apply 现在会接收三个返回值
        labels_list, gt_ids_list, candidate_gt_ids_list = multi_apply(
            self._get_target_single_vpd,
            gt_bboxes_list,
            gt_labels_list,
            bbox_preds_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            strides=concat_strides,
            num_points_per_lvl=num_points)

        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        gt_ids_list = [gt_ids.split(num_points, 0) for gt_ids in gt_ids_list]
        candidate_gt_ids_list = [cand.split(num_points, 0) for cand in candidate_gt_ids_list]

        concat_lvl_labels = []
        concat_lvl_gt_ids = []
        concat_lvl_candidate_gt_ids = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_gt_ids.append(torch.cat([gt_ids[i] for gt_ids in gt_ids_list]))
            concat_lvl_candidate_gt_ids.append(torch.cat([cand[i] for cand in candidate_gt_ids_list]))

        # 返回三个变量
        return concat_lvl_labels, concat_lvl_gt_ids, concat_lvl_candidate_gt_ids

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self, cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute ELBO loss for point-supervised VPD."""
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)

        num_imgs = cls_scores[0].size(0)

        # --- 修改开始: 重组 bbox_preds 用于 label assignment ---
        flatten_bbox_preds_per_lvl = [
            bp.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bp in bbox_preds
        ]
        bbox_preds_per_img = torch.cat(flatten_bbox_preds_per_lvl, dim=1)
        bbox_preds_list = [bbox_preds_per_img[i].detach() for i in range(num_imgs)]

        # 获取 labels, gt_ids 以及新增的 candidate_gt_ids
        labels, gt_ids, candidate_gt_ids = self.get_targets_vpd(
            all_level_points, gt_bboxes, gt_labels, bbox_preds_list)
        # --- 修改结束 ---

        if self.visualize and self.store_dir and self.iter % self.train_duration == 0:
            self.draw_image(img_metas[0]['filename'], img_metas[0].get('flip_direction'), cls_scores[0][0].sigmoid())
        if self.visualize_variance_map and self.store_dir and self.iter % self.train_duration == 0:
            self._save_variance_map(img_metas[0]['filename'], img_metas[0].get('flip_direction'), bbox_preds[0][0])
        self.iter += 1

        flatten_cls_scores = torch.cat([
            cs.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cs in cls_scores])
        flatten_bbox_preds = torch.cat([
            bp.permute(0, 2, 3, 1).reshape(-1, 4) for bp in bbox_preds])
        flatten_labels = torch.cat(labels)
        flatten_gt_ids = torch.cat(gt_ids)
        flatten_candidate_gt_ids = torch.cat(candidate_gt_ids) # 新增
        flatten_points = torch.cat([pts.repeat(num_imgs, 1) for pts in all_level_points])

        flatten_strides = torch.cat([
            bbox_preds[0].new_full((num_imgs * pts.shape[0],), s)
            for pts, s in zip(all_level_points, self.strides)])

        img_ids = torch.cat([
            torch.arange(num_imgs, dtype=torch.long, device=bbox_preds[0].device).repeat_interleave(n_pts)
            for n_pts in [pts.shape[0] for pts in all_level_points]])

        bg_class_ind = self.num_classes
        avail_inds = (flatten_labels >= 0).nonzero().reshape(-1)
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_avail = max(reduce_mean(torch.tensor(len(avail_inds), dtype=torch.float, device=bbox_preds[0].device)), 1.0)

        loss_cls = self.loss_cls(flatten_cls_scores[avail_inds], flatten_labels[avail_inds], avg_factor=num_avail)
        loss_cls = torch.nan_to_num(loss_cls, nan=0.0, posinf=1e4, neginf=-1e4)

        # --- 修改开始: 使用候选池计算 VPD Loss ---
        candidate_inds = (flatten_candidate_gt_ids >= 0).nonzero().reshape(-1)
        if len(candidate_inds) == 0:
            zero = flatten_bbox_preds.sum() * 0.0
            return dict(loss_cls=loss_cls, loss_vpd=zero, vpd_center=zero.detach(), vpd_kl=zero.detach(), vpd_var=zero.detach())

        cand_bbox_preds = flatten_bbox_preds[candidate_inds]
        cand_points = flatten_points[candidate_inds]
        cand_img_ids = img_ids[candidate_inds]
        cand_gt_ids = flatten_candidate_gt_ids[candidate_inds]
        cand_strides = flatten_strides[candidate_inds]

        bbox_mu = cand_bbox_preds[:, :2]
        bbox_log_sigma = cand_bbox_preds[:, 2:]

        gt_centers_per_pos = torch.zeros(len(candidate_inds), 2, device=bbox_preds[0].device)
        for img_id in range(num_imgs):
            mask = (cand_img_ids == img_id)
            if not mask.any():
                continue
            gt_center_this = gt_bboxes[img_id][:, :2]
            ids_this = cand_gt_ids[mask]
            gt_centers_per_pos[mask] = gt_center_this[ids_this]

        gt_centers_list = [gt_bbox[:, :2] for gt_bbox in gt_bboxes]
        vpd_losses = self.loss_vpd(
            bbox_mu=bbox_mu, bbox_log_sigma=bbox_log_sigma, pos_points=cand_points,
            pos_strides=cand_strides, gt_centers=gt_centers_per_pos,
            gt_centers_list=gt_centers_list, cur_iter=self.iter,
            pos_img_ids=cand_img_ids, num_samples=self.num_samples_train)
        # --- 修改结束 ---

        loss_vpd = torch.nan_to_num(vpd_losses['loss_total'], nan=0.0, posinf=1e4, neginf=-1e4)
        return dict(
            loss_cls=loss_cls, loss_vpd=loss_vpd,
            vpd_center=vpd_losses['loss_center'].detach(),
            vpd_kl=vpd_losses['loss_kl'].detach(),
            vpd_var=vpd_losses['loss_var'].detach())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _decode_bbox_from_mu(self, points, bbox_mu, stride):
        """Decode posterior mean to absolute box (cx, cy, w, h, angle=0).

        bbox_mu = (delta_x, delta_y) in feature-map units.
        Width/height are set to one stride as a fallback for xy-only heads.

        Returns:
            Tensor: (N, 5) rotated box (cx, cy, w, h, angle) with angle=0.
        """
        dx = bbox_mu[:, 0]
        dy = bbox_mu[:, 1]

        if self.norm_on_bbox:
            cx = points[:, 0] + dx * stride
            cy = points[:, 1] + dy * stride
        else:
            cx = points[:, 0] + dx
            cy = points[:, 1] + dy

        w = torch.full_like(cx, float(stride))
        h = torch.full_like(cy, float(stride))

        angle = torch.zeros_like(cx)
        return torch.stack([cx, cy, w, h, angle], dim=1)  # (N, 5)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self, cls_scores, bbox_preds, centernesses,
                   img_metas, cfg=None, rescale=None):
        """Inference: decode posterior mean, apply NMS."""
        cfg = self.test_cfg if cfg is None else cfg
        if cfg is None:
            raise ValueError('Test config is missing. Please set model.test_cfg or pass cfg to get_bboxes.')

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
                bbox_pred = bbox_preds[lvl_idx][img_id]  # (4, H, W)
                centerness = centernesses[lvl_idx][img_id]  # (1, H, W)
                points = mlvl_points[lvl_idx]              # (H*W, 2)

                # Flatten
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                if self.use_remap_score:
                    # Remap max class probability by center mean offsets (dx, dy).
                    # Score at (y, x) reads max prob from (y + dy, x + dx).
                    cls_prob_map = cls_score.sigmoid()  # (C, H, W)
                    max_prob_map, _ = cls_prob_map.max(dim=0)  # (H, W)

                    dx_map = bbox_preds[lvl_idx][img_id][0]
                    dy_map = bbox_preds[lvl_idx][img_id][1]
                    h, w = max_prob_map.shape
                    yy, xx = torch.meshgrid(
                        torch.arange(h, device=max_prob_map.device, dtype=dx_map.dtype),
                        torch.arange(w, device=max_prob_map.device, dtype=dx_map.dtype),
                        indexing='ij')
                    new_x = torch.round(xx + dx_map).long().clamp(0, w - 1)
                    new_y = torch.round(yy + dy_map).long().clamp(0, h - 1)
                    remapped_max_prob = max_prob_map[new_y, new_x]

                    remap_scale = remapped_max_prob / (max_prob_map + 1e-6)
                    remap_scale = torch.nan_to_num(remap_scale, nan=0.0, posinf=1e4, neginf=0.0)
                    score_map = cls_prob_map * remap_scale.unsqueeze(0)
                    scores = score_map.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                else:
                    cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                    centerness = centerness.reshape(-1).sigmoid()
                    scores = cls_score.sigmoid()
                    scores = scores * centerness[:, None]

                # Top-k selection
                nms_pre = cfg.get('nms_pre', 2000) if cfg else 2000
                max_scores, _ = scores.max(dim=1)
                nms_pre = min(nms_pre, scores.shape[0])
                _, topk_inds = max_scores.topk(nms_pre)

                points = points[topk_inds]
                bbox_mu = bbox_pred[topk_inds, :2]  # use center mean only
                scores = scores[topk_inds]

                # Optionally sample multiple times for refinement
                if self.use_refinement:
                    bbox_log_sigma = bbox_pred[topk_inds, 2:]
                    bbox_std = bbox_log_sigma.exp()
                    samples = [bbox_mu + bbox_std * torch.randn_like(bbox_mu)
                               for _ in range(self.num_samples)]
                    bbox_mu = torch.stack(samples, dim=0).mean(dim=0)

                decoded = self._decode_bbox_from_mu(points, bbox_mu, stride)
                mlvl_bboxes.append(decoded)
                mlvl_scores.append(scores)

            mlvl_bboxes = torch.cat(mlvl_bboxes)
            mlvl_scores = torch.cat(mlvl_scores)
            padding = mlvl_scores.new_zeros(mlvl_scores.size(0), 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

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
