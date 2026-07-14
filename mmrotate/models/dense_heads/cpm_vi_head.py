# Copyright (c) OpenMMLab. All rights reserved.
"""CPMVIHead: CPM Head with Variational Inference branch for
positive/negative sample probability prediction.

This head extends PointOBB-v2 (CPMHead) with an additional VI branch that
predicts a Beta distribution over P(positive) at each feature location.

Network output (conv_vi, 2 channels):
  [0] = mu_logit  -- logit of Beta mean  →  P(positive) = sigmoid(mu_logit)
  [1] = log_kappa -- log concentration   →  uncertainty (high = confident)

The VI branch is trained with soft-label JS divergence matching, where soft
labels are generated from distance-to-GT-center with Gaussian decay.

At inference, the VI probability can:
  (a) Replace centerness as a quality score: score = cls_prob * vi_prob
  (b) Filter low-quality predictions: only keep vi_prob > threshold
  (c) Be used alongside centerness: score = cls_prob * vi_prob * centerness
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
class CPMVIHead(CPMHead):
    """CPM Head with VI branch for positive/negative probability prediction.

    Args:
        vi_weight (float): Weight for VI matching loss. Default: 1.0.
        vi_kl_weight (float): Weight for KL-to-prior regularization. Default: 0.01.
        vi_num_bins (int): Number of bins for Beta distribution discretization.
        vi_soft_label_sigma (float): Sharpness of soft labels. Default: 0.4.
        use_vi_score (bool): Use VI probability as quality score at inference.
        vi_score_mode (str): How to combine VI score:
            'replace'  -- score = cls_prob * vi_prob
            'multiply' -- score = cls_prob * vi_prob * centerness
            'filter'   -- only keep predictions with vi_prob > vi_thr
            'none'     -- do not use VI score (centerness only)
        vi_thr (float): Threshold for 'filter' mode. Default: 0.5.
    """

    def __init__(self, *args, vi_weight=1.0, vi_num_bins=21,
                 vi_soft_label_sigma=1.28, use_vi_score=False,
                 vi_score_mode='multiply', vi_thr=0.5,
                 warmup_iters=2000, temp_start=1.0, temp_end=0.1, temp_decay_iters=5000,
                 **kwargs):
        super(CPMVIHead, self).__init__(*args, **kwargs)
        train_cfg = kwargs.get('train_cfg') or {}
        test_cfg = kwargs.get('test_cfg') or {}

        self.vi_weight = float(train_cfg.get('vi_weight', vi_weight))
        self.vi_num_bins = int(train_cfg.get('vi_num_bins', vi_num_bins))
        self.vi_soft_label_sigma = float(train_cfg.get('vi_soft_label_sigma', vi_soft_label_sigma))
        
        # 动态分配控制参数
        self.warmup_iters = int(train_cfg.get('warmup_iters', warmup_iters))
        self.temp_start = float(train_cfg.get('temp_start', temp_start))
        self.temp_end = float(train_cfg.get('temp_end', temp_end))
        self.temp_decay_iters = int(train_cfg.get('temp_decay_iters', temp_decay_iters))

        self.use_vi_score = bool(test_cfg.get('use_vi_score', use_vi_score))
        self.vi_score_mode = test_cfg.get('vi_score_mode', vi_score_mode)
        self.vi_thr = float(test_cfg.get('vi_thr', vi_thr))

        self.loss_vi = build_loss(dict(
            type='VIPosNegLoss',
            lambda_vi=self.vi_weight,
            num_bins=self.vi_num_bins,
            soft_label_sigma=self.vi_soft_label_sigma,
        ))
        
    def _init_predictor(self):
        """Override predictor: add conv_vi branch (2 channels).

        _init_predictor is called last in _init_layers, so we first call
        the parent to create cls/reg/angle/centerness convs, then add
        the VI branch.
        """
        super()._init_predictor()
        # VI branch: 2 channels = [mu_logit, log_kappa]
        self.conv_vi = nn.Conv2d(self.feat_channels, 2, 3, padding=1)

    def forward_single(self, x, scale, stride):
        """Forward for a single FPN level.
        
        跳级调用 RotatedAnchorFreeHead.forward_single 以获取 reg_feat 和 cls_feat。
        """
        # 1. 获取基础特征，此时返回 4 个值
        cls_score, _, cls_feat, reg_feat = \
            super(RotatedAnchorFreeHead, self).forward_single(x)
            
        # 2. 计算 centerness
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
            
        # 3. 计算 bbox_pred 和 angle_pred (保持与 CPMHead / RotatedFCOSHead 一致)
        bbox_pred = scale(self.conv_reg(reg_feat)).float()
        if hasattr(self, 'conv_angle'):
            angle_pred = self.conv_angle(reg_feat).float()
        else:
            # 兼容性处理：如果没有分离角度分支
            angle_pred = torch.zeros_like(bbox_pred[:, :1])
            
        # 4. 计算 VI 分支预测
        vi_pred = self.conv_vi(reg_feat).float()  # (N, 2, H, W)
        # vi_pred = self.conv_vi(cls_feat).float()  # (N, 2, H, W)
        
        return cls_score, bbox_pred, angle_pred, centerness, vi_pred

    def forward(self, feats):
        """Forward multi-level features.

        Returns:
            tuple: (cls_scores, bbox_preds, angle_preds, centernesses, vi_preds)
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    # ------------------------------------------------------------------
    # Label assignment: extend to return distance and threshold info
    # ------------------------------------------------------------------
    def _get_target_single_vi(self, gt_bboxes, gt_labels, points,
                              regress_ranges, num_points_per_lvl):
        """Like CPMHead._get_target_single but also returns:
        - dist_to_gt: (P,) distance to nearest GT center (INF if no GT)
        - pos_thresh: (P,) positive assignment threshold per point
        - is_ignored: (P,) bool, True for ignored points (label == -1)

        This information is needed for soft-label generation in VI loss.
        """
        alpha = self.alpha
        thresh1 = self.thresh1
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        center_point_gt = gt_bboxes[:, :2]

        labels = -1 * torch.ones(
            num_points, dtype=gt_labels.dtype, device=gt_labels.device)

        # Default: no GT nearby → large distance, zero threshold
        dist_to_gt = torch.full(
            (num_points,), INF, dtype=torch.float32, device=points.device)
        pos_thresh = torch.zeros(
            num_points, dtype=torch.float32, device=points.device)

        if num_gts == 0:
            labels[:] = self.num_classes
            return labels, dist_to_gt, pos_thresh

        dist_sample_and_gt = torch.cdist(points, center_point_gt)  # (P, G)
        dist_gt_and_gt = (
            torch.cdist(center_point_gt, center_point_gt)
            + torch.eye(num_gts, device=dist_sample_and_gt.device) * INF
        )

        dist_min_gt_and_gt, dist_min_gt_and_gt_index = \
            dist_gt_and_gt.min(dim=1)

        # Per-point minimum distance to any GT
        dist_min_sample_and_gt, min_gt_idx = \
            dist_sample_and_gt.min(dim=1)  # (P,), (P,)

        # Fill distance info for all points (even negatives)
        dist_to_gt = dist_min_sample_and_gt.float()

        # Compute per-GT positive threshold
        thresh1_tensor = thresh1 * torch.ones_like(dist_min_gt_and_gt)
        dist_min_thresh1_gt = torch.min(
            dist_min_gt_and_gt / 2, thresh1_tensor)  # (G,)

        # Per-point positive threshold = threshold of nearest GT
        pos_thresh = dist_min_thresh1_gt[min_gt_idx]

        if num_gts == 1:
            index_pos = (dist_sample_and_gt < 8).nonzero().reshape(-1)
            index_neg = (dist_sample_and_gt > 128).nonzero().reshape(-1)
            labels[index_pos] = gt_labels[0]
            labels[index_neg] = self.num_classes
            # Points between 8 and 128 are ignored (label stays -1)
            return labels, dist_to_gt, pos_thresh

        # Negative labels
        index_neg = (
            (alpha * dist_sample_and_gt) > dist_min_gt_and_gt
        ).all(dim=1).nonzero().squeeze(-1)
        if len(index_neg) > 0:
            labels[index_neg] = self.num_classes

        # Positive labels
        index_pos = (dist_sample_and_gt < dist_min_thresh1_gt).nonzero()
        if len(index_pos) > 0:
            labels[index_pos[:, 0]] = gt_labels[index_pos[:, 1]]

        # Additional background labels (midpoints between same-class neighbors)
        is_nearest_same_class = (
            gt_labels[dist_min_gt_and_gt_index] == gt_labels)
        valid_middle_point = (
            center_point_gt[is_nearest_same_class]
            + center_point_gt[dist_min_gt_and_gt_index][is_nearest_same_class]
        ) / 2
        dist_sample_and_mid = torch.cdist(points, valid_middle_point)
        index_neg_additional = (
            dist_sample_and_mid < 4
        ).any(dim=1).nonzero().squeeze(-1)
        if len(index_neg_additional) > 0:
            labels[index_neg_additional] = self.num_classes

        return labels, dist_to_gt, pos_thresh

    def get_targets_vi(self, points, gt_bboxes_list, gt_labels_list):
        """Like CPMHead.get_targets but also returns VI-specific targets.

        Returns:
            concat_lvl_labels: list[Tensor] -- per-level labels
            concat_lvl_dist: list[Tensor] -- per-level distance to GT
            concat_lvl_thresh: list[Tensor] -- per-level positive threshold
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i])
            for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]

        labels_list, dist_list, thresh_list = multi_apply(
            self._get_target_single_vi,
            gt_bboxes_list, gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        dist_list = [dist.split(num_points, 0) for dist in dist_list]
        thresh_list = [thresh.split(num_points, 0) for thresh in thresh_list]

        concat_lvl_labels = []
        concat_lvl_dist = []
        concat_lvl_thresh = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_dist.append(
                torch.cat([dist[i] for dist in dist_list]))
            concat_lvl_thresh.append(
                torch.cat([thresh[i] for thresh in thresh_list]))

        return concat_lvl_labels, concat_lvl_dist, concat_lvl_thresh
    
    def _get_current_temperature(self, cur_iter):
        if cur_iter < self.warmup_iters:
            return self.temp_start
        elif cur_iter < self.warmup_iters + self.temp_decay_iters:
            # 线性衰减
            alpha = (cur_iter - self.warmup_iters) / self.temp_decay_iters
            return self.temp_start * (1 - alpha) + self.temp_end * alpha
        else:
            return self.temp_end
        
    def _visualize_vi_predictions(self, vi_preds, save_path):
        """Visualize VI predictions.

        Args:
            vi_preds (Tensor): VI predictions (2, H_feat, W_feat).
            save_path (str): Path to save the visualization.
        """
        mu_logit = vi_preds[0].cpu().numpy()
        log_kappa = vi_preds[1].cpu().numpy()
        mu = 1 / (1 + np.exp(-mu_logit))  # Sigmoid
        kappa = np.exp(log_kappa)

        # Create a visualization image
        H, W = mu.shape
        vis_img = np.zeros((H, W*2, 3), dtype=np.uint8)
        # vis_img[..., 0] = (mu * 255).astype(np.uint8)  # Red channel: P(positive)
        # vis_img[..., 1] = (kappa / (kappa.max() + 1e-6) * 255).astype(np.uint8)  # Green: uncertainty
        # vis_img[..., 2] = 0  # Blue channel unused

        # vis_img[..., :W, 0] = (mu * 255).astype(np.uint8)  # Red channel: P(positive)
        # vis_img[..., W:, 1] = (kappa / (kappa.max() + 1e-6) * 255).astype(np.uint8)  # Green: uncertainty
        
        # or use the colormap jet to visualize mu and kappa
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        mu_colormap = cm.jet(mu)[:, :, :3]  # (H, W, 3)
        kappa_colormap = cm.jet(kappa / (kappa.max() + 1e-6))[:, :, :3]
        vis_img[..., :W, :] = (mu_colormap * 255).astype(np.uint8)
        vis_img[..., W:, :] = (kappa_colormap * 255).astype(np.uint8)

        Image.fromarray(vis_img).save(save_path)


    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses', 'vi_preds'))
    def loss(self, cls_scores, bbox_preds, angle_preds, centernesses, vi_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(vi_preds)
        
        # ... (visualize code omitted for brevity) ...
        # self._visualize_vi_predictions(vi_preds[0][0].detach().cpu(), os.path.join(self.store_dir, f"vi_preds/vi_pred_iter_{self.iter}.png"))
        
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        
        # 获取几何信息（距离和阈值）
        labels, dists, threshs = self.get_targets_vi(all_level_points, gt_bboxes, gt_labels)
        
        cur_iter = self.iter
        self.iter += 1
        num_imgs = cls_scores[0].size(0)
        
        flatten_cls_scores = torch.cat([cs.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cs in cls_scores])
        flatten_vi_preds = torch.cat([vp.permute(0, 2, 3, 1).reshape(-1, 2) for vp in vi_preds])
        flatten_labels = torch.cat(labels)
        flatten_dists = torch.cat(dists)
        flatten_threshs = torch.cat(threshs)
        
        bg_class_ind = self.num_classes
        avail_inds = (flatten_labels >= 0).nonzero().reshape(-1)
        
        # 1. Get current Temperature (for VI Loss sampling)
        temperature = self._get_current_temperature(cur_iter)
        
        # 2. Compute VI Loss
        is_ignored = (flatten_labels < 0)
        flatten_dists = torch.where(torch.isinf(flatten_dists), torch.full_like(flatten_dists, 1e4), flatten_dists)
        
        vi_mu_logit = flatten_vi_preds[:, 0] # 预测的 mu_logit (Logit 空间的 Gap)
        vi_log_sigma = flatten_vi_preds[:, 1]
        
        # 计算变分推断 Loss
        vi_losses = self.loss_vi(
            vi_mu_logit=vi_mu_logit,
            vi_log_sigma=vi_log_sigma,
            dist_to_gt=flatten_dists,
            pos_thresh=flatten_threshs.clamp(min=1.0),
            is_ignored=is_ignored,
            temperature=temperature,
            iter=self.iter
        )
        
        loss_vi = torch.nan_to_num(vi_losses['loss_total'], nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 3. 动态计算分类 Loss (Dynamic Assignment)
        num_avail = max(reduce_mean(torch.tensor(len(avail_inds), dtype=torch.float, device=bbox_preds[0].device)), 1.0)
        
        if cur_iter < self.warmup_iters:
            # 前期 Warmup：使用标准硬标签
            loss_cls = self.loss_cls(
                flatten_cls_scores[avail_inds], 
                flatten_labels[avail_inds], 
                avg_factor=num_avail
            )
        else:
            # 后期：基于物理距离的动态分配
            
            # 获取有效区域的预测和几何信息
            vi_mu = vi_mu_logit[avail_inds] # 预测的距离差值 Delta
            dists_geo = flatten_dists[avail_inds] # 实际距离 d
            threshs_geo = flatten_threshs[avail_inds].clamp(min=1.0) # 硬阈值 T_hard
            
            # --- 核心修正：使用修正后的分配距离进行比较 ---
            
            # 1. 计算动态分配距离: T_dynamic = T_hard + mu_pred
            # vi_mu 已经是距离单位（由 VIPosNegLoss 的回归目标决定）
            dynamic_thresh = threshs_geo + vi_mu
            
            # 2. 比较 T_dynamic 与实际距离 d_actual
            # Margin > 0 意味着 动态阈值 覆盖了 实际距离 -> 正样本
            margin = dynamic_thresh - dists_geo
            
            # 3. 定义正负样本 Mask
            pos_mask = (margin > 0)
            neg_mask = (margin <= 0)
            
            # 4. 计算软权重
            # 将 Margin 映射到 [0, 1] 概率空间
            # Margin 越大（点越深），权重趋近 1
            # Margin 越小（点越浅），权重趋近 0
            dynamic_prob = torch.sigmoid(margin)
            
            weight = torch.ones_like(dynamic_prob)
            weight[pos_mask] = dynamic_prob[pos_mask]
            weight[neg_mask] = (1.0 - dynamic_prob[neg_mask])
            
            loss_cls = self.loss_cls(
                flatten_cls_scores[avail_inds],
                flatten_labels[avail_inds],
                weight=weight,
                avg_factor=max(weight.sum(), 1.0)
            )
            
        loss_cls = torch.nan_to_num(loss_cls, nan=0.0, posinf=1e4, neginf=-1e4)
        zero = loss_cls.sum() * 0.0
                
        return dict(
            loss_cls=self.cls_weight * loss_cls,
            loss_bbox=zero,
            loss_centerness=zero,
            loss_vi=loss_vi,
            loss_vi_recon=vi_losses['loss_vi_recon'].detach(),
            loss_vi_kl=vi_losses['loss_vi_kl'].detach(),
            loss_vi_reg=vi_losses['loss_vi_reg'].detach(),
            # 修改日志：分别监控正样本概率和全局概率
            vi_prob_mean=vi_losses['vi_prob'].mean().detach(),
            vi_sigma_mean=vi_losses['vi_sigma'].mean().detach(),
            temperature=torch.tensor(temperature, device=loss_cls.device)
        )
    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'angle_preds',
                          'centernesses', 'vi_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, angle_preds, centernesses,
                   vi_preds, img_metas, cfg=None, rescale=None):
        """Inference: decode bboxes, optionally use VI probability as score.

        The VI probability P(positive) = sigmoid(mu_logit) can:
        - Replace centerness as quality score
        - Multiply with centerness for combined quality
        - Filter low-quality predictions
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(vi_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id] for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id] for i in range(num_levels)]
            angle_pred_list = [
                angle_preds[i][img_id] for i in range(num_levels)]
            centerness_list = [
                centernesses[i][img_id] for i in range(num_levels)]
            vi_pred_list = [vi_preds[i][img_id] for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, angle_pred_list,
                centerness_list, vi_pred_list, mlvl_points,
                img_shape, scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self, cls_scores, bbox_preds, angle_preds,
                           centernesses, vi_preds, mlvl_points,
                           img_shape, scale_factor, cfg, rescale=False):
        """Decode bboxes for a single image with optional VI scoring."""
        cfg = self.test_cfg if cfg is None else cfg
        mlvl_bboxes = []
        mlvl_scores = []

        for cls_score, bbox_pred, angle_pred, centerness, vi_pred, points \
                in zip(cls_scores, bbox_preds, angle_preds,
                       centernesses, vi_preds, mlvl_points):

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)

            # VI probability: P(positive) = sigmoid(mu_logit)
            vi_pred_flat = vi_pred.permute(1, 2, 0).reshape(-1, 2)
            vi_prob = torch.sigmoid(vi_pred_flat[:, 0])  # (P,)

            # Combine scores based on vi_score_mode
            if self.use_vi_score:
                if self.vi_score_mode == 'replace':
                    # Replace centerness entirely
                    score_factor = vi_prob
                elif self.vi_score_mode == 'multiply':
                    # Multiply centerness and VI probability
                    score_factor = centerness * vi_prob
                elif self.vi_score_mode == 'filter':
                    # Hard filter: zero out low VI prob
                    score_factor = centerness
                    score_factor = score_factor * (vi_prob > self.vi_thr).float()
                else:
                    score_factor = centerness
            else:
                score_factor = centerness

            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)

            # Top-k selection before NMS
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * score_factor[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                score_factor = score_factor[topk_inds]

            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        # Use score_factor for NMS
        score_factors = score_factor
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
            cfg.max_per_img, score_factors=score_factors)
        return det_bboxes, det_labels

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Delegate to parent CPMHead.get_targets for base class calls."""
        return super(CPMVIHead, self).get_targets(
            points, gt_bboxes_list, gt_labels_list)

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None,
                      gt_bboxes_ignore=None, proposal_cfg=None, **kwargs):
        """Train forward, returning losses."""
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
