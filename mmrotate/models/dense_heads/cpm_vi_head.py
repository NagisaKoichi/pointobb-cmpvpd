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
                 vi_soft_label_sigma=1.28, use_vi_score=False, vi_thresh_scale=1.0,
                 vi_score_mode='multiply', vi_thr=0.5,
                 warmup_iters=200, temp_start=1.0, temp_end=0.1, temp_decay_iters=5000,
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
        
        self.vi_thresh_scale = float(train_cfg.get('vi_thresh_scale', vi_thresh_scale))

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
        # self.conv_vi = nn.Conv2d(self.feat_channels, 2, 3, padding=1)
        self.conv_vi = nn.Conv2d(self.feat_channels, 2, 3, padding=2, dilation=2)

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
    # VI prior generation: use the gt_centers to assign a prior mu
    # ------------------------------------------------------------------
    def _get_vi_prior_single(self, gt_bboxes, points):
        """Generate VI prior for a single image.

        Args:
            gt_bboxes (Tensor): (G, 5) GT boxes in (x, y, w, h, a).
            points (Tensor): (P, 2) feature points.
        Returns:
            vi_prior (Tensor): (P,) prior for each point: [mu]
        """
        # 

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
    
    # def _get_target_single_vi_dynamic(self, gt_bboxes, gt_labels, points, vi_mu_logit, regress_ranges, num_points_per_lvl):
    def _get_target_single_vi_dynamic(self, gt_bboxes, gt_labels, vi_mu_logit, points, regress_ranges, num_points_per_lvl):
        """
        基于 VI 分支结果的动态样本分配 (单张图片)。
        
        Args:
            gt_bboxes (Tensor): (G, 5) 当前图片的 GT Boxes
            gt_labels (Tensor): (G,) 当前图片的 GT Labels
            points (Tensor): (Total_Points, 2) 当前图片的所有特征点 (按 Level 拼接)
            vi_mu_logit (Tensor): (Total_Points,) 当前图片的 VI 预测 (mu_logit)
            regress_ranges: (保持接口一致性，未使用)
            num_points_per_lvl: List[int], 每个 Level 的点数，用于拆分结果
            
        Returns:
            labels (Tensor): (Total_Points, ) -1=Ignore, [0, num_classes-1]=Pos, num_classes=Neg
            dist_to_gt (Tensor): (Total_Points, )
            pos_thresh (Tensor): (Total_Points, ) 保持接口一致
        """
        thresh1 = self.thresh1
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        center_point_gt = gt_bboxes[:, :2]

        # 初始化
        labels = -1 * torch.ones(num_points, dtype=gt_labels.dtype, device=points.device)
        dist_to_gt = torch.full((num_points,), INF, dtype=torch.float32, device=points.device)
        pos_thresh = torch.zeros(num_points, dtype=torch.float32, device=points.device)

        if num_gts == 0:
            # 没有 GT，全图背景
            labels[:] = self.num_classes
            return labels, dist_to_gt, pos_thresh

        # 1. 几何信息计算 (用于获取最近 GT 的类别和距离)
        dist_sample_and_gt = torch.cdist(points, center_point_gt) # (Total_Points, G)
        
        # 找到每个点最近的 GT
        dist_min_sample_and_gt, min_gt_idx = dist_sample_and_gt.min(dim=1)
        dist_to_gt = dist_min_sample_and_gt.float()

        # 2. 计算 pos_thresh (保留输出接口)
        # 这里的逻辑与原始 CPM 保持一致，虽然分配逻辑变了，但这个阈值仍可能被外部或其他逻辑使用
        dist_gt_and_gt = (
            torch.cdist(center_point_gt, center_point_gt) + 
            torch.eye(num_gts, device=points.device) * INF
        )
        dist_min_gt_and_gt, dist_min_gt_and_gt_index = dist_gt_and_gt.min(dim=1)
        thresh1_tensor = thresh1 * torch.ones_like(dist_min_gt_and_gt)
        dist_min_thresh1_gt = torch.min(dist_min_gt_and_gt / 2, thresh1_tensor)
        # 每个点取其最近 GT 的阈值
        pos_thresh = dist_min_thresh1_gt[min_gt_idx]

        # 3. VI 动态分配逻辑
        # Weight > 0.7 -> 正样本
        # Weight < 0.3 -> 负样本
        # 其他 -> Ignore (-1)
        log_actual_dist = torch.log(dist_min_sample_and_gt + 1.0)
        margin = vi_mu_logit - log_actual_dist
        weight = torch.sigmoid(margin)

        mask_pos = (weight > 0.6)
        mask_neg = (weight < 0.4)

        if mask_pos.any():
            labels[mask_pos] = gt_labels[min_gt_idx[mask_pos]]
        
        if mask_neg.any():
            labels[mask_neg] = self.num_classes

        # 4. 硬约束：相邻同类型样本之间的额外负样本
        # 只有当 GT 数量大于 1 时才有“邻居”的概念
        if num_gts > 1:
            is_nearest_same_class = (gt_labels[dist_min_gt_and_gt_index] == gt_labels)
            if is_nearest_same_class.any():
                # 计算中点
                valid_middle_point = (
                    center_point_gt[is_nearest_same_class] + 
                    center_point_gt[dist_min_gt_and_gt_index][is_nearest_same_class]
                ) / 2
                
                # 计算点到中点的距离
                dist_sample_and_mid = torch.cdist(points, valid_middle_point)
                
                # 距离小于 4 的点强制设为负样本
                index_neg_additional = (dist_sample_and_mid < 4).any(dim=1).nonzero().squeeze(-1)
                if len(index_neg_additional) > 0:
                    labels[index_neg_additional] = self.num_classes

        return labels, dist_to_gt, pos_thresh

    def get_targets_vi_dynamic(self, points, gt_bboxes_list, gt_labels_list, vi_mu_logit_list):
        """
        Wrapper for dynamic assignment.
        
        Args:
            points: List[Tensor], points per level (单张图的全集, shape: (P_level, 2))
            gt_bboxes_list: List[Tensor], GT boxes per image (Batch size N)
            gt_labels_list: List[Tensor], GT labels per image
            vi_mu_logit_list: List[Tensor], vi_mu_logit per image (Batch size N), 
                              每个元素形状为 (Total_Points,)
        
        Returns: Same format as get_targets_vi (List of Tensors per level, concatenated over batch)
        """
        assert len(points) == len(self.regress_ranges)
        assert len(vi_mu_logit_list) == len(gt_bboxes_list)

        num_levels = len(points)
        
        # 拼接 Points，得到 (Total_Points_Single_Image, 2)
        # 注意：points 列表中的每个元素是对应的 Level，它们加起来是单张图的所有点
        concat_points = torch.cat(points, dim=0)
        
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i]) 
            for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        num_points_per_lvl = [center.size(0) for center in points]

        # 调用 multi_apply
        # multi_apply 会遍历 gt_bboxes_list 和 vi_mu_logit_list
        # concat_points 会被复用传给每一张图片的处理函数
        labels_list, dist_list, thresh_list = multi_apply(
            self._get_target_single_vi_dynamic,
            gt_bboxes_list,     # Arg 0
            gt_labels_list,     # Arg 1
            vi_mu_logit_list,   # Arg 2 (位置参数，会被自动拆分)
            points=concat_points,         # Kwarg
            regress_ranges=concat_regress_ranges, # Kwarg
            num_points_per_lvl=num_points_per_lvl # Kwarg
        )

        # 还原结果结构 (按 Level 拆分)
        # labels_list 是 List[Single_Image_Labels]
        # Single_Image_Labels 需要按 num_points_per_lvl 切分回各个 Level
        labels_list = [labels.split(num_points_per_lvl, 0) for labels in labels_list]
        dist_list = [dist.split(num_points_per_lvl, 0) for dist in dist_list]
        thresh_list = [thresh.split(num_points_per_lvl, 0) for thresh in thresh_list]

        # 重新组合成 Level 优先的格式，用于后续计算 Loss
        concat_lvl_labels = []
        concat_lvl_dist = []
        concat_lvl_thresh = []
        
        for i in range(num_levels):
            # 取出所有图片的第 i 层 Label 并拼接
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_dist.append(torch.cat([dist[i] for dist in dist_list]))
            concat_lvl_thresh.append(torch.cat([thresh[i] for thresh in thresh_list]))

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
        flatten_threshs = torch.cat(threshs)  # = min(dist_to_nearest_gt / 2, thresh1) for each point
        
        bg_class_ind = self.num_classes
        avail_inds = (flatten_labels >= 0).nonzero().reshape(-1)
        
        # 1. Get current Temperature
        temperature = self._get_current_temperature(cur_iter)

        # --- 关键修改：扩大 VI 分支的有效阈值 ---
        # 原始的 flatten_threshs 是 CPM 的几何阈值（通常限制在 8 左右）
        # 我们将其扩大 scale 倍，用于生成 VI 的 Soft Label
        vi_threshs = flatten_threshs.clamp(min=1.0) * self.vi_thresh_scale

        # 2. Compute VI Loss (传入放大后的阈值)
        is_ignored = (flatten_labels < 0)
        flatten_dists = torch.where(torch.isinf(flatten_dists), torch.full_like(flatten_dists, 1e4), flatten_dists)
        vi_mu_logit = flatten_vi_preds[:, 0]
        vi_log_sigma = flatten_vi_preds[:, 1]

        vi_losses = self.loss_vi(
            vi_mu_logit=vi_mu_logit,
            vi_log_sigma=vi_log_sigma,
            dist_to_gt=flatten_dists,
            pos_thresh=vi_threshs,  # 使用放大后的阈值
            is_ignored=is_ignored,
            temperature=temperature,
            iter=self.iter
        )
        
        loss_vi = torch.nan_to_num(vi_losses['loss_total'], nan=0.0, posinf=1e4, neginf=-1e4)

        # 3. 动态计算分类 Loss (保持不变，使用 Log-Distance 逻辑)
        num_avail = max(reduce_mean(torch.tensor(len(avail_inds), dtype=torch.float, device=bbox_preds[0].device)), 1.0)
        
        # vi_mu = vi_mu_logit[avail_inds]
        # dists_geo = flatten_dists[avail_inds]
        vi_mu = vi_mu_logit
        dists_geo = flatten_dists
        log_actual_dist = torch.log(dists_geo + 1.0)
        
        margin = vi_mu - log_actual_dist
        dynamic_prob = torch.sigmoid(margin)
        
        if cur_iter < self.warmup_iters:
            loss_cls = self.loss_cls(
                flatten_cls_scores[avail_inds], 
                flatten_labels[avail_inds], 
                avg_factor=num_avail
            )
        else:
            
            num_imgs = cls_scores[0].size(0)
            
            # 1. 构建 vi_mu_logit_list
            # 必须确保输出顺序与 all_level_points 的拼接顺序一致 (Level 优先)
            vi_mu_logit_list = []
            for img_id in range(num_imgs):
                logits_per_img = []
                # 遍历所有 Level
                for pred in vi_preds:
                    # pred: (B, 2, H, W) -> 取出 img_id -> (2, H, W) -> Flatten -> (H*W, 2)
                    logits_per_img.append(pred[img_id].view(-1, 2))
                
                # 拼接所有 Level -> (Total_Points, 2)
                single_img_preds = torch.cat(logits_per_img, dim=0)
                # 提取 mu_logit (通道 0)
                vi_mu_logit_list.append(single_img_preds[:, 0])

            # 2. 调用新的分配函数
            labels, dists, threshs = self.get_targets_vi_dynamic(
                all_level_points, 
                gt_bboxes, 
                gt_labels, 
                vi_mu_logit_list
            )

            # 3. Flatten 结果
            flatten_labels = torch.cat(labels)
            flatten_dists = torch.cat(dists)
            flatten_threshs = torch.cat(threshs)
            
            avail_inds = (flatten_labels >= 0).nonzero().reshape(-1)
            
            pos_mask = (margin > 0)
            neg_mask = (margin <= 0)
            
            weight = torch.ones_like(dynamic_prob)
            weight[pos_mask] = dynamic_prob[pos_mask]
            weight[neg_mask] = (1.0 - dynamic_prob[neg_mask])
            
            # 替换avail_inds为vi分支计算的正负样本
            
            loss_cls = self.loss_cls(
                flatten_cls_scores[avail_inds], 
                flatten_labels[avail_inds], 
                avg_factor=num_avail
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
            vi_prob_mean=dynamic_prob.mean().detach(),
            # vi_prob_mean=vi_losses['vi_prob'].mean().detach(),
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
