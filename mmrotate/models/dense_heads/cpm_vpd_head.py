# Copyright (c) OpenMMLab. All rights reserved.

import os
import re

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean

from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from mmrotate.utils import dump_vpd_heatmaps
from ..builder import ROTATED_HEADS, build_loss
from .cpm_head import CPMHead
from .rotated_anchor_free_head import RotatedAnchorFreeHead

INF = 1e8


@ROTATED_HEADS.register_module()
class CPMVPDHead(CPMHead):
    """CPM Head with VPD-style Variational Inference (bbox only, no angle).

    This head implements the VPD framework for CPM:
    1. Predicts mean and log-std for bbox regression only (variational inference)
    2. Uses Gaussian reparameterization during training
    3. Adds JS divergence loss or point-supervised loss
    4. Uses CPM label assignment strategy
    5. No angle prediction (horizontal bboxes only)

    Args:
        js_weight (float): Weight for JS divergence loss. Default: 1.0.
        use_point_supervised (bool): Whether to use point supervision. Default: False.
        num_samples (int): Number of samples for inference refinement. Default: 10.
        use_refinement (bool): Whether to use sampling-based refinement. Default: True.
    """

    def __init__(self, *args, js_weight=1.0, use_point_supervised=False,
                 num_samples=10, use_refinement=True, **kwargs):
        self.js_weight = js_weight
        self.use_point_supervised = use_point_supervised
        self.num_samples = num_samples  # Number of samples for inference refinement
        self.use_refinement = use_refinement  # Whether to use sampling-based refinement
        self.save_vpd_intermediates = False
        self.save_vpd_heatmaps = False
        self.vpd_heatmap_cmap = 'turbo'
        self.vpd_save_variance_channels = True
        self.vpd_save_dir = None
        self.log_std_min = -7.0
        self.log_std_max = 5.0
        self._vpd_case_counter = 0
        super().__init__(*args, **kwargs)

        train_cfg = kwargs.get('train_cfg') or {}
        test_cfg = kwargs.get('test_cfg') or {}

        # if 'js_weight' in train_cfg:
        #     self.js_weight = train_cfg['js_weight']
        if 'use_point_supervised' in train_cfg:
            self.use_point_supervised = train_cfg['use_point_supervised']
        if 'num_samples' in test_cfg:
            self.num_samples = test_cfg['num_samples']
        if 'use_refinement' in test_cfg:
            self.use_refinement = test_cfg['use_refinement']
        if 'save_vpd_intermediates' in test_cfg:
            self.save_vpd_intermediates = test_cfg['save_vpd_intermediates']
        if 'vpd_save_dir' in test_cfg:
            self.vpd_save_dir = test_cfg['vpd_save_dir']
        if 'dump_vpd_intermediates' in test_cfg:
            self.save_vpd_intermediates = test_cfg['dump_vpd_intermediates']
        if 'dump_vpd_dir' in test_cfg:
            self.vpd_save_dir = test_cfg['dump_vpd_dir']
        if 'save_vpd_heatmaps' in test_cfg:
            self.save_vpd_heatmaps = test_cfg['save_vpd_heatmaps']
        if 'vpd_heatmap_cmap' in test_cfg:
            self.vpd_heatmap_cmap = test_cfg['vpd_heatmap_cmap']
        if 'vpd_save_variance_channels' in test_cfg:
            self.vpd_save_variance_channels = test_cfg['vpd_save_variance_channels']
        if 'log_std_min' in train_cfg:
            self.log_std_min = train_cfg['log_std_min']
        if 'log_std_max' in train_cfg:
            self.log_std_max = train_cfg['log_std_max']
        if 'log_std_min' in test_cfg:
            self.log_std_min = test_cfg['log_std_min']
        if 'log_std_max' in test_cfg:
            self.log_std_max = test_cfg['log_std_max']

        # Build loss based on supervision type
        if self.use_point_supervised:
            # Point-supervised: use center + uncertainty + KL loss
            self.loss_vpd = build_loss(dict(
                type='PointSupervisedVPDLoss',
                center_weight=1.0,
                uncertainty_weight=0.1,
                kl_weight=0.01))
        else:
            # Full supervision: use JS divergence loss
            self.loss_js = build_loss(dict(type='JSLoss', loss_weight=self.js_weight))

    def _init_predictor(self):
        """Initialize predictors with 8-channel bbox distribution output.

        AnchorFreeHead creates ``conv_reg`` in ``_init_predictor`` with 4 output
        channels by default. VPD needs 8 channels (4 mean + 4 log_std), so the
        override must happen here instead of ``_init_reg_convs``.
        """
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 8, 3, padding=1)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level with VPD."""
        # Get base features
        cls_score, _, cls_feat, reg_feat = \
            super(RotatedAnchorFreeHead, self).forward_single(x)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)

        # Predict distribution: 8 channels (4 mean + 4 log_std) for bbox only
        bbox_dist = self.conv_reg(reg_feat)

        # Split mean and log_std
        bbox_mean = bbox_dist[:, :4]  # First 4: bbox mean
        bbox_lstd = bbox_dist[:, 4:]  # Last 4: bbox log_std
        bbox_lstd = bbox_lstd.clamp(min=self.log_std_min, max=self.log_std_max)

        # Scale the mean
        bbox_mean = scale(bbox_mean).float()
        if self.norm_on_bbox:
            bbox_mean = bbox_mean.clamp(min=0)
            if not self.training:
                bbox_mean *= stride
        else:
            bbox_mean = bbox_mean.exp()

        # Concatenate mean and log_std for output
        bbox_pred_full = torch.cat([bbox_mean, bbox_lstd], dim=1)  # 8 channels
        bbox_pred_full = torch.nan_to_num(bbox_pred_full, nan=0.0, posinf=1e4, neginf=-1e4)

        return cls_score, bbox_pred_full, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self, cls_scores, bbox_preds, centernesses,
             gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss with VPD variational inference."""
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)

        # Get targets (using CPM label assignment)
        labels = self.get_targets(
            all_level_points, gt_bboxes, gt_labels)

        if self.visualize and self.store_dir and self.iter % self.train_duration == 0:
            self.draw_image(img_metas[0]['filename'],
                          img_metas[0].get('flip_direction'),
                          cls_scores[0][0].sigmoid())
        self.iter += 1

        num_imgs = cls_scores[0].size(0)

        # Flatten predictions
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores])
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 8)  # 4 mean + 4 log_std
            for bbox_pred in bbox_preds])
        flatten_bbox_preds = torch.nan_to_num(
            flatten_bbox_preds, nan=0.0, posinf=1e4, neginf=-1e4)
        flatten_labels = torch.cat(labels)
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # Defensive alignment for custom target assignment implementations.
        # Prevents out-of-bounds indexing if any branch length diverges.
        num_valid = min(
            flatten_cls_scores.size(0),
            flatten_bbox_preds.size(0),
            flatten_labels.size(0),
            flatten_points.size(0))
        flatten_cls_scores = flatten_cls_scores[:num_valid]
        flatten_bbox_preds = flatten_bbox_preds[:num_valid]
        flatten_labels = flatten_labels[:num_valid]
        flatten_points = flatten_points[:num_valid]

        # Get positive/negative indices
        bg_class_ind = self.num_classes

        # Keep labels in valid focal-loss index range [-1, num_classes].
        invalid_label_mask = (flatten_labels < -1) | (flatten_labels > bg_class_ind)
        if invalid_label_mask.any():
            flatten_labels = flatten_labels.clone()
            flatten_labels[invalid_label_mask] = bg_class_ind

        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        avail_inds = ((flatten_labels >= 0) & (flatten_labels <= bg_class_ind)).nonzero().reshape(-1)

        # num_pos = max(reduce_mean(torch.tensor(len(pos_inds), dtype=torch.float,
        #                                        device=bbox_preds[0].device)), 1.0)
        num_avail = max(reduce_mean(torch.tensor(len(avail_inds), dtype=torch.float,
                                                 device=bbox_preds[0].device)), 1.0)

        # Classification loss
        if avail_inds.numel() == 0:
            raise ValueError('No valid classification samples after label sanitization.')
        loss_cls = self.loss_cls(flatten_cls_scores[avail_inds],
                                flatten_labels[avail_inds], avg_factor=num_avail)
        loss_cls = torch.nan_to_num(loss_cls, nan=0.0, posinf=1e4, neginf=-1e4)

        # Regression losses (only for positive samples)
        if len(pos_inds) > 0:
            pos_bbox_dist = flatten_bbox_preds[pos_inds]
            # pos_centerness = flatten_centerness[pos_inds]
            pos_points = flatten_points[pos_inds]

            # VPD: Gaussian reparameterization
            # pos_bbox_mean = pos_bbox_dist[:, :4]
            # pos_bbox_lstd = pos_bbox_dist[:, 4:]

            # Sample: mean + std * noise
            # pos_bbox_reg = pos_bbox_mean + pos_bbox_lstd.exp() * torch.randn_like(pos_bbox_mean)

            # VPD loss (JS divergence or point-supervised)
            pos_dist_full = pos_bbox_dist  # 8 channels (bbox only)

            if self.use_point_supervised:
                # Point-supervised: only use GT center points
                # Extract GT centers from bbox targets: each gt_bbox is (num_gt, 5)
                gt_centers_list = [gt_bbox[:, :2] for gt_bbox in gt_bboxes if gt_bbox.numel() > 0]
                if gt_centers_list:
                    gt_centers = torch.cat(gt_centers_list, dim=0)  # (num_all_gt, 2)
                else:
                    gt_centers = pos_points.new_zeros((0, 2))

                loss_vpd = self.loss_vpd(pos_dist_full, pos_points, gt_centers)
            else:
                raise NotImplementedError
        else:
            raise ValueError('len(pos_inds) == 0')

        return dict(
            loss_cls=loss_cls,
            loss_center=loss_vpd['center'],
            loss_uncertainty=loss_vpd['uncertainty'],
            loss_kl=loss_vpd['kl_weight'],
            loss_sum=loss_vpd['sum'])

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Use CPM's label assignment, but also return bbox/angle targets."""
        # Call parent's get_targets which returns labels, bbox_targets, angle_targets
        return super(CPMVPDHead, self).get_targets(points, gt_bboxes_list, gt_labels_list)

    def _per_layer_inference(self, cls_scores, bbox_preds,
                            centernesses, mlvl_points, mlvl_strides, cfg):
        """Per-layer inference with sampling-based bbox refinement.

        Inspired by VPD network's per_layer_inference approach:
        1. For each pyramid level, select top-k candidates based on classification score
        2. Sample multiple bbox predictions from the Gaussian distribution
        3. Aggregate sampled bboxes to get refined predictions

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level with shape
                (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution params for each scale level
                with shape (N, 8, H, W) [4 mean + 4 log_std].
            centernesses (list[Tensor]): Centerness for each scale level with shape
                (N, 1, H, W).
            mlvl_points (list[Tensor]): Points for each pyramid level.
            mlvl_strides (list[Tensor]): Strides for each pyramid level.
            cfg: Test config.

        Returns:
            tuple: Refined bbox predictions, scores, and points.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)

        num_levels = len(cls_scores)
        refined_bbox_preds = []
        refined_scores = []
        refined_points = []
        refined_centernesses = []

        # Process each pyramid level separately
        for lvl_idx in range(num_levels):
            cls_score = cls_scores[lvl_idx][0]  # Remove batch dim: (C, H, W)
            bbox_pred = bbox_preds[lvl_idx][0]  # (8, H, W)
            centerness = centernesses[lvl_idx][0]  # (H, W) or (1, H, W)
            points = mlvl_points[lvl_idx]  # (H*W, 2)
            stride = self.strides[lvl_idx]

            # Flatten spatial dimensions
            num_classes = cls_score.shape[0]
            H, W = cls_score.shape[1:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, num_classes)  # (H*W, C)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 8)  # (H*W, 8)

            if centerness.dim() == 3:
                centerness = centerness.squeeze(0)
            centerness = centerness.reshape(-1)  # (H*W,)

            # Get top-k candidates based on max class score
            max_scores, _ = cls_score.max(dim=1)  # (H*W,)
            max_scores = torch.sigmoid(max_scores) * centerness  # Multiply with centerness

            # Select top-k per layer (similar to VPD's test_layer_topk)
            test_layer_topk = getattr(cfg, 'test_layer_topk', 1000)
            num_candidates = min(test_layer_topk, len(max_scores))

            topk_scores, topk_inds = max_scores.topk(num_candidates)

            # Get selected predictions
            selected_cls_scores = cls_score[topk_inds]  # (K, C)
            selected_bbox_dist = bbox_pred[topk_inds]  # (K, 8)
            selected_points = points[topk_inds]  # (K, 2)
            selected_centerness = centerness[topk_inds]  # (K,)

            # Refine bboxes using sampling (if enabled)
            if self.use_refinement and not self.training:
                # Split mean and log_std
                bbox_mean = selected_bbox_dist[:, :4]  # (K, 4)
                bbox_lstd = selected_bbox_dist[:, 4:]  # (K, 4)
                bbox_lstd = bbox_lstd.clamp(min=self.log_std_min, max=self.log_std_max)

                # Sample multiple times and aggregate
                refined_bbox = self._restore_bbox_with_sampling(
                    selected_points, bbox_mean, bbox_lstd, stride,
                    num_samples=self.num_samples)

                refined_bbox_preds.append(refined_bbox)
            else:
                # Use mean predictions only
                bbox_mean = selected_bbox_dist[:, :4]
                refined_bbox_preds.append(bbox_mean)

            refined_scores.append(selected_cls_scores)
            refined_points.append(selected_points)
            refined_centernesses.append(selected_centerness)

        return (refined_bbox_preds, refined_scores,
                refined_points, refined_centernesses)

    def _restore_bbox_with_sampling(self, points, bbox_mean, bbox_lstd, stride,
                                    num_samples=10):
        """Restore bbox using multiple samples from Gaussian distribution.

        Similar to VPD's restore_bbox, but with sampling-based refinement:
        1. Sample bbox deltas from Gaussian: delta = mean + std * noise
        2. Decode each sample to absolute bbox coordinates
        3. Aggregate all samples (mean) to get refined bbox

        Args:
            points (Tensor): Anchor points with shape (K, 2).
            bbox_mean (Tensor): Mean of bbox deltas with shape (K, 4).
            bbox_lstd (Tensor): Log-std of bbox deltas with shape (K, 4).
            stride (int): Stride for current pyramid level.
            num_samples (int): Number of samples for refinement.

        Returns:
            Tensor: Refined bbox deltas with shape (K, 4).
        """
        K = bbox_mean.shape[0]
        bbox_std = bbox_lstd.exp()

        # Sample multiple bbox predictions
        all_sampled_bboxes = []
        for _ in range(num_samples):
            # Gaussian reparameterization: sample = mean + std * noise
            noise = torch.randn_like(bbox_mean)
            sampled_bbox_delta = bbox_mean + bbox_std * noise

            # Clip to reasonable range to avoid extreme values
            sampled_bbox_delta = sampled_bbox_delta.clamp(min=0, max=stride * 4)

            all_sampled_bboxes.append(sampled_bbox_delta)

        # Aggregate: compute mean of all samples
        all_sampled_bboxes = torch.stack(all_sampled_bboxes, dim=0)  # (num_samples, K, 4)
        refined_bbox_delta = all_sampled_bboxes.mean(dim=0)  # (K, 4)

        return refined_bbox_delta

    def _dump_vpd_intermediates(self, cls_scores, bbox_preds, centernesses,
                                img_metas):
        """Dump VPD intermediate predictions for each test sample.

        For each image in the current test batch, this function creates one
        dedicated directory and stores:
        1. centerness tensors of all FPN levels
        2. variance tensors of all bbox channels (computed from log_std)
        3. class score tensors of all FPN levels
        """
        if not self.save_vpd_intermediates or not self.vpd_save_dir:
            return

        os.makedirs(self.vpd_save_dir, exist_ok=True)

        num_imgs = len(img_metas)
        for img_id in range(num_imgs):
            filename = img_metas[img_id].get('filename', f'img_{img_id}')
            base_name = os.path.splitext(os.path.basename(filename))[0]
            base_name = re.sub(r'[^0-9a-zA-Z._-]+', '_', base_name)

            case_dir_name = f'{self._vpd_case_counter:06d}_{base_name}'
            case_dir = os.path.join(self.vpd_save_dir, case_dir_name)
            os.makedirs(case_dir, exist_ok=True)
            self._vpd_case_counter += 1

            cls_scores_per_level = []
            centerness_per_level = []
            variance_per_level = []

            for lvl_idx in range(len(cls_scores)):
                cls_scores_per_level.append(
                    cls_scores[lvl_idx][img_id].sigmoid().detach().cpu())
                centerness_per_level.append(
                    centernesses[lvl_idx][img_id].detach().cpu())

                bbox_lstd = bbox_preds[lvl_idx][img_id, 4:, ...]
                variance_per_level.append((2 * bbox_lstd).exp().detach().cpu())

            torch.save(cls_scores_per_level, os.path.join(case_dir, 'cls_scores.pt'))
            torch.save(centerness_per_level, os.path.join(case_dir, 'centerness.pt'))
            torch.save(variance_per_level, os.path.join(case_dir, 'variance.pt'))
            torch.save(img_metas[img_id], os.path.join(case_dir, 'img_meta.pt'))

            if self.save_vpd_heatmaps:
                dump_vpd_heatmaps(
                    case_dir=case_dir,
                    cls_scores_per_level=cls_scores_per_level,
                    centerness_per_level=centerness_per_level,
                    variance_per_level=variance_per_level,
                    cmap=self.vpd_heatmap_cmap,
                    save_variance_channels=self.vpd_save_variance_channels)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self, cls_scores, bbox_preds, centernesses,
                   img_metas, cfg=None, rescale=None):
        """Get bboxes with sampling-based refinement during inference.

        This method implements VPD-style inference:
        1. Per-layer top-k selection (similar to per_layer_inference)
        2. Sampling-based bbox refinement (similar to restore_bbox with sampling)
        3. Standard NMS post-processing
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        cfg = self.test_cfg if cfg is None else cfg

        self._dump_vpd_intermediates(cls_scores, bbox_preds, centernesses, img_metas)

        num_levels = len(cls_scores)

        # Get points for each pyramid level
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        mlvl_strides = [
            bbox_preds[0].new_full((featmap_size[0] * featmap_size[1],), stride)
            for featmap_size, stride in zip(featmap_sizes, self.strides)
        ]

        # Apply per-layer inference with refinement if enabled
        if self.use_refinement and not self.training:
            # Use sampling-based refinement
            (refined_bbox_preds, refined_scores,
             refined_points, refined_centernesses) = self._per_layer_inference(
                cls_scores, bbox_preds, centernesses,
                mlvl_points, mlvl_strides, cfg)

            # Prepare for NMS: need to restructure into expected format
            # Original get_bboxes expects predictions per level, but we have per-level top-k
            # We'll use a simplified approach: directly decode and apply NMS

            result_list = []
            for img_id in range(len(img_metas)):
                cls_score_list = []
                bbox_pred_list = []

                for lvl_idx in range(num_levels):
                    # Use bbox predictions directly (no angle)
                    bbox_pred = refined_bbox_preds[lvl_idx]  # (K, 4)
                    points = refined_points[lvl_idx]

                    # Scale bbox predictions by stride if needed
                    if self.norm_on_bbox:
                        # Distances are already multiplied by stride in forward_single.
                        bbox_pred_scaled = bbox_pred
                    else:
                        bbox_pred_scaled = bbox_pred

                    # DistanceAnglePointCoder expects 5-dim deltas
                    # [l, t, r, b, angle]. CPMVPD predicts bbox-only deltas,
                    # so append a zero-angle channel during decoding.
                    if bbox_pred_scaled.size(-1) == 4:
                        zero_angle = bbox_pred_scaled.new_zeros(
                            (bbox_pred_scaled.size(0), 1))
                        bbox_pred_decode = torch.cat(
                            [bbox_pred_scaled, zero_angle], dim=-1)
                    else:
                        bbox_pred_decode = bbox_pred_scaled

                    decoded_bboxes = self.bbox_coder.decode(
                        points, bbox_pred_decode)
                    decoded_bboxes = torch.nan_to_num(
                        decoded_bboxes, nan=0.0, posinf=1e4, neginf=-1e4)

                    scores = refined_scores[lvl_idx].sigmoid()
                    centernesses = refined_centernesses[lvl_idx]

                    # Combine classification score with centerness
                    scores = scores * centernesses[:, None]
                    scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

                    cls_score_list.append(scores)
                    bbox_pred_list.append(decoded_bboxes)

                # Concatenate all levels
                mlvl_bboxes = torch.cat(bbox_pred_list)
                mlvl_scores = torch.cat(cls_score_list)

                # Apply NMS
                if rescale:
                    scale_factor = mlvl_bboxes.new_tensor(
                        img_metas[img_id]['scale_factor'][:2]).repeat(2)
                    mlvl_bboxes[:, :4] = mlvl_bboxes[:, :4] / scale_factor

                det_bboxes, det_labels = multiclass_nms_rotated(
                    mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                    cfg.max_per_img)

                result_list.append((det_bboxes, det_labels))

            return result_list
        else:
            # Use standard inference (mean only)
            bbox_preds_mean = [bbox_pred[:, :4] for bbox_pred in bbox_preds]

            # Call parent's get_bboxes with mean predictions (no angle)
            # Need to check parent signature - might need to override differently
            raise NotImplementedError("Standard inference without angle needs parent method update")
