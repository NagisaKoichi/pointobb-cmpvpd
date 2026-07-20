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

import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
                 warmup_iters=2400, temp_start=1.0, temp_end=0.1, temp_decay_iters=5000,
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
        # --- 新增：为 VI 分支建立独立的特征提取层 ---
        # 这样 VI 分支就不受 loss_bbox 和 loss_cls 的梯度影响了
        self.vi_convs = nn.ModuleList()
        for i in range(2): # 堆叠两层卷积，增加表达能力
            self.vi_convs.append(
                # nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1, dilation=1)
                # or dilated
                nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=2, dilation=2)
            )
            self.vi_convs.append(
                nn.GroupNorm(32, self.feat_channels) # 或 BN
            )
            self.vi_convs.append(nn.ReLU(inplace=True))

        
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
        # vi_pred = self.conv_vi(reg_feat).float()  # (N, 2, H, W)
        # vi_pred = self.conv_vi(cls_feat).float()  # (N, 2, H, W)
        
        vi_feat = x
        for i in range(0, len(self.vi_convs), 3): # Conv -> Norm -> ReLU
            vi_feat = self.vi_convs[i](vi_feat)
            vi_feat = self.vi_convs[i+1](vi_feat)
            vi_feat = self.vi_convs[i+2](vi_feat)
        
        vi_pred = self.conv_vi(vi_feat).float()

        
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
        assert points.size(0) == vi_mu_logit.size(0), "Points and VI predictions must have the same number of points."
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
        # 每个 GT 取其最近 GT 的阈值
        pos_thresh = dist_min_thresh1_gt[min_gt_idx]

        # 3. VI 动态分配逻辑
        # Weight > 0.7 -> 正样本
        # Weight < 0.3 -> 负样本
        # 其他 -> Ignore (-1)
        # log_actual_dist = torch.log(dist_min_sample_and_gt + 1.0)
        # margin = vi_mu_logit - log_actual_dist
        # weight = torch.sigmoid(margin)
        
        weight = torch.sigmoid(vi_mu_logit)  # 适用于vi_mu_logit直接预测logit空间的分数的情况

        mask_pos = (weight > 0.5)
        # geo_in_range = (dist_min_sample_and_gt < 2.0 * dist_min_thresh1_gt[min_gt_idx])
        # mask_pos = mask_pos & geo_in_range  # 同时满足 VI 正样本条件和几何正样本条件
        # 同时满足 VI 负样本条件和几何负样本条件的点才被标记为负样本
        # 几何负样本条件沿用原来的逻辑：alpha * dist_sample_and_gt > dist_min_gt_and_gt
        # mask_neg = (weight < 0.3) & (self.alpha * dist_min_sample_and_gt > dist_min_gt_and_gt[min_gt_idx])
        mask_neg = self.alpha * dist_min_sample_and_gt > dist_min_gt_and_gt[min_gt_idx]

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
        mu_colormap = cm.jet(mu)[:, :, :3]  # (H, W, 3)
        kappa_colormap = cm.jet(kappa / (kappa.max() + 1e-6))[:, :, :3]
        vis_img[..., :W, :] = (mu_colormap * 255).astype(np.uint8)
        vis_img[..., W:, :] = (kappa_colormap * 255).astype(np.uint8)

        Image.fromarray(vis_img).save(save_path)
        
    def _visualize_labels_cat(self, orig_labels, vi_labels, save_path):
        """Visualize concatenated labels (original vs VI-assigned).
        Args:
            orig_labels (Tensor): (N,) Original labels.
            vi_labels (Tensor): (N,) VI-assigned labels.
            save_path (str): Path to save the visualization.
        """
        orig_labels_np = orig_labels.cpu().numpy()
        vi_labels_np = vi_labels.cpu().numpy()
        
        # 修复：检查维度，如果是 1D 则重塑为 2D 网格以便可视化
        if orig_labels_np.ndim == 1:
            # 计算近似正方形的尺寸
            n_points = orig_labels_np.shape[0]
            h = int(np.sqrt(n_points))
            w = h
            if h * w == 0:
                return # 点数太少无法可视化
            
            # 截取前 h*w 个点进行可视化
            orig_labels_np = orig_labels_np[:h*w].reshape(h, w)
            vi_labels_np = vi_labels_np[:h*w].reshape(h, w)

        # Create a color map for visualization
        num_classes = self.num_classes
        # 使用 jet colormap，并将数值归一化到 0-1
        # 注意：-1 (Ignore) 会被映射到特定颜色，这里简单处理
        
        # 将标签归一化：-1 -> 0, 0 -> 0, num_classes -> 1 (示例映射)
        # 更好的做法是使用离散 colormap
        cmap = plt.get_cmap('jet')
        
        # 映射标签到 0-1 范围以便着色
        # -1 (Ignore) -> 0.0
        # 0..C-1 (FG) -> 0.2..0.8
        # C (BG) -> 1.0
        # 这里简单处理：直接除以 (num_classes + 1)
        norm_orig = (orig_labels_np + 1) / (num_classes + 2)
        norm_vi = (vi_labels_np + 1) / (num_classes + 2)

        H, W = orig_labels_np.shape
        vis_img = np.zeros((H, W*2, 3), dtype=np.uint8)
        
        # 应用 colormap 并转换为 uint8
        vis_img[:, :W, :] = (cmap(norm_orig)[:, :, :3] * 255).astype(np.uint8)
        vis_img[:, W:, :] = (cmap(norm_vi)[:, :, :3] * 255).astype(np.uint8)

        Image.fromarray(vis_img).save(save_path)


    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses', 'vi_preds'))
    def loss(self, cls_scores, bbox_preds, angle_preds, centernesses, vi_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(vi_preds)
        
        if self.iter % 50 == 0 and self.store_dir is not None:
            os.makedirs(os.path.join(self.store_dir, "vi_preds"), exist_ok=True)
            self._visualize_vi_predictions(vi_preds[0][0].detach().cpu(), os.path.join(self.store_dir, f"vi_preds/vi_pred_iter_{self.iter}.png"))
        
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)

        cur_iter = self.iter
        self.iter += 1
        
        # === 步骤 1: 先计算所有层的几何分配 (作为基础和 P1+ 的最终结果) ===
        # 这里的 labels, dists, threshs 是按 Level 组织的 List[Tensor]
        geo_labels, geo_dists, geo_threshs = self.get_targets_vi(
            all_level_points, gt_bboxes, gt_labels)
        
        # === 步骤 2: 初始化最终结果容器 (深拷贝几何结果) ===
        # 这样 P1-P3 默认就是几何分配的结果
        final_labels = [label.clone() for label in geo_labels]
        final_dists = [dist.clone() for dist in geo_dists]
        final_threshs = [thresh.clone() for thresh in geo_threshs]

        # === 步骤 3: 针对 Level 0 进行 VI 动态分配 (仅在 Warmup 后) ===
        if cur_iter >= self.warmup_iters:
            
            # 3.1 准备 Level 0 的数据
            # vi_preds[0]: (B, 2, H, W) -> 取出 Level 0
            # all_level_points[0]: (P0, 2) -> Level 0 的坐标点
            
            lvl = 0
            vi_pred_lvl0 = vi_preds[lvl]  # (B, 2, H, W)
            points_lvl0 = all_level_points[lvl] # (P0, 2)
            
            B, _, H, W = vi_pred_lvl0.shape
            num_points_lvl0 = points_lvl0.size(0)
            
            # 3.2 提取 VI 预测的 mu_logit 并展平
            # (B, 2, H, W) -> (B, H, W) -> (B, P0)
            vi_mu_logit_lvl0 = vi_pred_lvl0[:, 0, :, :].view(B, -1)
            
            # 转换为 List 以便 multi_apply 处理
            vi_logit_list = [vi_mu_logit_lvl0[i] for i in range(B)]
            
            # 准备 regress_ranges (Level 0 的范围)
            # 扩展为 (P0, 2) 格式以匹配函数签名
            regress_range_lvl0 = points_lvl0.new_tensor(self.regress_ranges[lvl])[None].expand_as(points_lvl0)
            
            # 3.3 调用动态分配函数 (仅针对 Level 0)
            # 这里的逻辑与之前类似，但输入限定为 Level 0 的点
            lvl0_labels_list, lvl0_dists_list, lvl0_threshs_list = multi_apply(
                self._get_target_single_vi_dynamic,
                gt_bboxes,          # Batch 遍历
                gt_labels,          # Batch 遍历
                vi_logit_list,      # Batch 遍历
                points=points_lvl0, # 固定 Level 0 的点
                regress_ranges=regress_range_lvl0, # 固定 Level 0 的范围
                num_points_per_lvl=[num_points_lvl0] # 单层
            )
            
            # 3.4 更新 final_labels 中的 Level 0 部分
            # multi_apply 返回的是 List[Tensor]，每个 Tensor 是 (P0,)
            # 我们需要 stack 成 batch 然后替换原来的 Level 0
            # 注意：final_labels[0] 原本是 (B*P0,) 的形状吗？
            # 不，get_targets_vi 返回的是 concat_lvl_labels，是按 Level concat batch 的。
            # 即 final_labels[0] 的形状是 (B*P0,)
            
            # 重新拼接 dynamic 分配的结果
            dynamic_labels_lvl0 = torch.cat(lvl0_labels_list)  # (B*P0,)
            dynamic_dists_lvl0 = torch.cat(lvl0_dists_list)    # (B*P0,)
            dynamic_threshs_lvl0 = torch.cat(lvl0_threshs_list)# (B*P0,)
            
            # 替换 Level 0 的结果
            final_labels[lvl] = dynamic_labels_lvl0
            final_dists[lvl] = dynamic_dists_lvl0
            final_threshs[lvl] = dynamic_threshs_lvl0

            # === 可视化对比 (可选) ===
            if self.iter % 50 == 0 and self.store_dir is not None:
                os.makedirs(os.path.join(self.store_dir, "vi_labels"), exist_ok=True)
                # 对比 Level 0 的几何分配 vs 动态分配
                self._visualize_labels_cat(
                    geo_labels[0].detach(), # 几何分配
                    final_labels[0].detach(), # 动态分配
                    os.path.join(self.store_dir, f"vi_labels/lvl0_cmp_iter_{self.iter}.png")
                )

        # === 步骤 4: 后续 Loss 计算 (使用 final_labels) ===
        # 将 List 展平
        flatten_labels = torch.cat(final_labels)
        flatten_dists = torch.cat(final_dists)
        flatten_threshs = torch.cat(final_threshs)        
        flatten_cls_scores = torch.cat([cs.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cs in cls_scores])
        flatten_vi_preds = torch.cat([vp.permute(0, 2, 3, 1).reshape(-1, 2) for vp in vi_preds])
        # flatten_labels = torch.cat(labels)
        # flatten_dists = torch.cat(dists)
        # flatten_threshs = torch.cat(threshs)
        
        bg_class_ind = self.num_classes
        avail_inds = (flatten_labels >= 0).nonzero().reshape(-1)
        
        # # 安全检查
        # if len(avail_inds) == 0:
        #     zero = flatten_cls_scores.sum() * 0.0
        #     return dict(loss_cls=zero, loss_bbox=zero, loss_centerness=zero, loss_vi=zero) # 简略返回

        
        # 1. Get current Temperature (for VI Loss sampling)
        temperature = self._get_current_temperature(cur_iter)
        
        # 2. Compute VI Loss
        is_ignored = (flatten_labels < 0)
        flatten_dists = torch.where(torch.isinf(flatten_dists), torch.full_like(flatten_dists, 1e4), flatten_dists)
        
        vi_mu_logit = flatten_vi_preds[:, 0] # 预测的 mu_logit (Logit 空间的 Gap)
        vi_log_sigma = flatten_vi_preds[:, 1]
        
        # === 修改部分：在 Warmup 结束时，冻结整个特征提取网络 ===
        if cur_iter == self.warmup_iters:
            # 1. 冻结 VI 分支自身
            for param in self.conv_vi.parameters():
                param.requires_grad = False
            if hasattr(self, 'vi_convs'):
                for param in self.vi_convs.parameters():
                    param.requires_grad = False
                                
            print(f"Iter {cur_iter}: Freezing VI branch, Backbone, and Neck.")        
        # 在 warmup 阶段结束后，冻结 vi 分支（不更新 vi 分支的参数，也就是停止训练）
        
        # 计算变分推断 Loss
        # vi_losses = self.loss_vi(
        #     vi_mu_logit=vi_mu_logit,
        #     vi_log_sigma=vi_log_sigma,
        #     dist_to_gt=flatten_dists,
        #     pos_thresh=flatten_threshs.clamp(min=1.0),
        #     is_ignored=is_ignored,
        #     temperature=temperature,
        #     iter=self.iter
        # )
        
        if cur_iter < self.warmup_iters:
            # Warmup 阶段，使用原始的 VI Loss 计算
            vi_losses = self.loss_vi(
                vi_mu_logit=vi_mu_logit,
                vi_log_sigma=vi_log_sigma,
                dist_to_gt=flatten_dists,
                pos_thresh=flatten_threshs.clamp(min=1.0),
                is_ignored=is_ignored,
                temperature=temperature,
                iter=self.iter
            )
        else:
            # Warmup 阶段结束后，冻结 vi 分支，使用 detach() 避免梯度传播
            with torch.no_grad():
                vi_losses = self.loss_vi(
                    vi_mu_logit=vi_mu_logit.detach(),
                    vi_log_sigma=vi_log_sigma.detach(),
                    dist_to_gt=flatten_dists,
                    pos_thresh=flatten_threshs.clamp(min=1.0),
                    is_ignored=is_ignored,
                    temperature=temperature,
                    iter=self.iter
                )
        
        
        loss_vi = torch.nan_to_num(vi_losses['loss_total'], nan=0.0, posinf=1e4, neginf=-1e4)

        # 3. 动态计算分类 Loss (保持不变，使用 Log-Distance 逻辑)
        num_avail = max(reduce_mean(torch.tensor(len(avail_inds), dtype=torch.float, device=bbox_preds[0].device)), 1.0)
        
        # vi_mu = vi_mu_logit[avail_inds]
        # dists_geo = flatten_dists[avail_inds]
        # vi_mu = vi_mu_logit  # probability in logit space
        dists_geo = flatten_dists
        log_actual_dist = torch.log(dists_geo + 1.0)
        
        margin = vi_mu_logit - log_actual_dist
        dynamic_prob = torch.sigmoid(margin)
        
        # dynamic_prob = torch.sigmoid(vi_mu)
        
        # if cur_iter < self.warmup_iters:
        #     loss_cls = self.loss_cls(
        #         flatten_cls_scores[avail_inds], 
        #         flatten_labels[avail_inds], 
        #         avg_factor=num_avail
        #     )
        # else:
            
        #     num_imgs = cls_scores[0].size(0)
            
        #     # 1. 构建 vi_mu_logit_list
        #     # 必须确保输出顺序与 all_level_points 的拼接顺序一致 (Level 优先)
        #     vi_mu_logit_list = []
        #     for img_id in range(num_imgs):
        #         logits_per_img = []
        #         # 遍历所有 Level
        #         for pred in vi_preds:
        #             # pred: (B, 2, H, W) -> 取出 img_id -> (2, H, W) -> Flatten -> (H*W, 2)
        #             logits_per_img.append(pred[img_id].view(-1, 2))
                
        #         # 拼接所有 Level -> (Total_Points, 2)
        #         single_img_preds = torch.cat(logits_per_img, dim=0)
        #         # 提取 mu_logit (通道 0)
        #         vi_mu_logit_list.append(single_img_preds[:, 0].detach())  # Can this detach avoid loss_cls affecting vi branch gradients? Yes, we only use it for assignment.

        #     # 2. 调用新的分配函数
        #     labels, dists, threshs = self.get_targets_vi_dynamic(
        #         all_level_points, 
        #         gt_bboxes, 
        #         gt_labels, 
        #         vi_mu_logit_list
        #     )
            
        #     if self.iter % 50 == 0 and self.store_dir is not None:
        #         os.makedirs(os.path.join(self.store_dir, "vi_labels"), exist_ok=True)
                
        #         # 建议：对比 "几何分配标签" 与 "VI动态分配标签"
        #         # 我们可以重新计算一份几何标签用于对比
        #         geo_labels, _, _ = self.get_targets_vi(all_level_points, gt_bboxes, gt_labels)
        #         geo_labels_flat = torch.cat(geo_labels)
                
        #         self._visualize_labels_cat(
        #             geo_labels_flat,            # 几何分配 (左)
        #             torch.cat(labels),          # VI动态分配 (右)
        #             os.path.join(self.store_dir, f"vi_labels/vi_labels_iter_{self.iter}.png")
        #         )
            
        #     # 3. Flatten 结果
        #     flatten_labels = torch.cat(labels)
        #     flatten_dists = torch.cat(dists)
        #     flatten_threshs = torch.cat(threshs)
            
        #     avail_inds = (flatten_labels >= 0).nonzero().reshape(-1)
            
        #     pos_mask = (margin > 0)
        #     neg_mask = (margin <= 0)
            
        #     weight = torch.ones_like(dynamic_prob)
        #     weight[pos_mask] = dynamic_prob[pos_mask]
        #     weight[neg_mask] = (1.0 - dynamic_prob[neg_mask])
            
        #     # 替换avail_inds为vi分支计算的正负样本
            
        #     loss_cls = self.loss_cls(
        #         flatten_cls_scores[avail_inds], 
        #         flatten_labels[avail_inds], 
        #         avg_factor=num_avail
        #     )
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
            loss_vi_dist_match=vi_losses['loss_vi_dist_match'].detach(),
            loss_vi_sampled=vi_losses['loss_vi_sampled'].detach(),
            # loss_vi_recon=vi_losses['loss_vi_recon'].detach(),
            # loss_vi_kl=vi_losses['loss_vi_kl'].detach(),
            loss_vi_reg=vi_losses['loss_vi_reg'].detach(),
            # vi_prob_mean=dynamic_prob.mean().detach(),
            vi_prob_mean=vi_losses['vi_prob'].mean().detach(),
            vi_sigma_mean=vi_losses['vi_sigma'].mean().detach(),
            vi_prob_max=vi_losses['vi_prob'].max().detach(),
            vi_prob_min=vi_losses['vi_prob'].min().detach(),
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
        # === 新增逻辑：在 Warmup 结束后，阻断 Backbone/Neck 的梯度 ===
        if self.iter >= self.warmup_iters:
            # 阻断输入特征 x 的梯度，使其不回传到 Backbone 和 Neck
            # 这相当于“冻结”了 Backbone 和 Neck 的权重更新
            if isinstance(x, (list, tuple)):
                x = [feat.detach() for feat in x]
            else:
                x = x.detach()
        # ==========================================================
        
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
