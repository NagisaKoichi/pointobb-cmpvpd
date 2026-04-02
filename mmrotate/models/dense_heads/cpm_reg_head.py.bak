# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean

from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from ..builder import ROTATED_HEADS, build_loss
from .rotated_fcos_head import RotatedFCOSHead
import numpy as np
import os
from PIL import Image

INF = 1e8


@ROTATED_HEADS.register_module()
class CPMRegHead(RotatedFCOSHead):
    """CPM Head with regression branch.

    This head extends CPM by adding a regression branch, inspired by VPD framework.
    The regression branch helps to:
    1. Learn robust and accurate bounding box predictions
    2. Bridge classification and regression for better confidence estimation

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        cls_weight (float): Weight for classification loss. Default: 1.0.
        reg_weight (float): Weight for regression loss. Default: 1.0.
        thresh1 (float): Threshold for positive sample assignment. Default: 8.
        alpha (float): Alpha parameter for negative sample assignment. Default: 1.0.
        use_reg_for_cls (bool): Whether to use regression quality to enhance
            classification confidence. Default: False.
    """

    def __init__(self,
                 *args,
                 cls_weight=1.0,
                 reg_weight=1.0,
                 thresh1=8,
                 alpha=1.0,
                 use_reg_for_cls=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.iter = 0
        self.train_duration = 200
        self.visualize = False
        self.test_duration = 32
        self.store_dir = None
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.thresh1 = thresh1
        self.alpha = alpha
        self.use_reg_for_cls = use_reg_for_cls

        train_cfg = kwargs.get('train_cfg', {})
        test_cfg = kwargs.get('test_cfg', {})

        if 'store_dir' in train_cfg:
            self.store_dir = train_cfg['store_dir']
        elif 'store_dir' in test_cfg:
            self.store_dir = test_cfg['store_dir']

        if self.store_dir is not None:
            os.makedirs(self.store_dir + "/visualize/", exist_ok=True)

        if 'cls_weight' in train_cfg:
            self.cls_weight = train_cfg['cls_weight']
        if 'reg_weight' in train_cfg:
            self.reg_weight = train_cfg['reg_weight']
        if 'thresh1' in train_cfg:
            self.thresh1 = train_cfg['thresh1']
        if 'alpha' in train_cfg:
            self.alpha = train_cfg['alpha']
        if 'vis_train_duration' in train_cfg:
            self.train_duration = train_cfg['vis_train_duration']
        if 'visualize' in train_cfg:
            self.visualize = train_cfg['visualize']
        if 'use_reg_for_cls' in train_cfg:
            self.use_reg_for_cls = train_cfg['use_reg_for_cls']

    def get_mask_image(self, max_probs, max_indices, thr, num_width):
        PALETTE = [
            (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
            (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
            (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
            (0, 0, 255), (147, 116, 116), (0, 0, 255), (0, 0, 0),
            (128, 128, 0), (75, 0, 130)
        ]
        mask_image = np.ones((num_width, num_width, 3), dtype=np.uint8) * 255
        for i in range(num_width):
            for j in range(num_width):
                if max_probs[i, j] > thr:
                    mask_image[i, j] = PALETTE[max_indices[i, j]]
        return mask_image

    def _draw_image(self, max_probs, max_indices, thr, img_flip_direction, img_A, num_width):
        mask_image = self.get_mask_image(max_probs, max_indices, thr, num_width)
        img_B = Image.fromarray(mask_image)
        if img_flip_direction is None:
            pass
        elif img_flip_direction == 'horizontal':
            img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
        elif img_flip_direction == 'vertical':
            img_B = img_B.transpose(Image.FLIP_TOP_BOTTOM)
        elif img_flip_direction == 'diagonal':
            img_B = img_B.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
        combined_image = Image.new('RGB', (img_A.width + img_B.width, img_A.height))
        combined_image.paste(img_A, (0, 0))
        combined_image.paste(img_B, (img_A.width, 0))
        return combined_image

    def draw_image(self, img_path, flip, score_probs):
        num_width = score_probs.shape[2]
        img_A = Image.open(img_path).convert("RGB")
        img_A = img_A.resize((num_width, num_width))

        max_probs, max_indices = torch.max(score_probs, dim=0)
        os.makedirs(self.store_dir + "/visualize/" + str(self.iter), exist_ok=True)

        thr_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15]
        for i in range(len(thr_list)):
            thr = thr_list[i]
            conbine_image = self._draw_image(max_probs, max_indices, thr, flip, img_A, num_width)
            output_path = self.store_dir + "/visualize/" + str(self.iter) + "/" + str(thr) + ".jpg"
            conbine_image.save(output_path)

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head with regression branch.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale level.
            angle_preds (list[Tensor]): Box angle for each scale level.
            centernesses (list[Tensor]): Centerness for each scale level.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding boxes
                can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        # Get targets including bbox and angle targets for regression
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, gt_bboxes, gt_labels)

        if self.visualize and self.store_dir is not None and self.iter % self.train_duration == 0:
            self.draw_image(img_metas[0]['filename'], img_metas[0]['flip_direction'],
                          cls_scores[0][0].sigmoid())

        self.iter += 1

        num_imgs = cls_scores[0].size(0)
        # Flatten cls_scores, bbox_preds, angle_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)

        # Repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        neg_inds = (flatten_labels == bg_class_ind).nonzero().reshape(-1)
        num_neg = torch.tensor(
            len(neg_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_neg = max(reduce_mean(num_neg), 1.0)

        avail_inds = (flatten_labels >= 0).nonzero().reshape(-1)
        num_avail = torch.tensor(
            len(avail_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_avail = max(reduce_mean(num_avail), 1.0)

        # Classification loss
        loss_cls = self.loss_cls(
            flatten_cls_scores[avail_inds], flatten_labels[avail_inds], avg_factor=num_avail)

        # Regression loss (only for positive samples)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)

        # Centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            if self.separate_angle:
                bbox_coder = self.h_bbox_coder
            else:
                bbox_coder = self.bbox_coder
            pos_bbox_pred = torch.cat([pos_bbox_preds, pos_angle_preds], dim=-1)
            pos_bbox_target = torch.cat([pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points, pos_bbox_pred)
            pos_decoded_target_preds = bbox_coder.decode(pos_points, pos_bbox_target)

            # Regression loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)

            # Centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)

            if self.separate_angle:
                loss_angle = self.loss_angle(
                    pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            if self.separate_angle:
                loss_angle = pos_angle_preds.sum()

        if self.separate_angle:
            return dict(
                loss_cls=self.cls_weight * loss_cls,
                loss_bbox=self.reg_weight * loss_bbox,
                loss_angle=self.reg_weight * loss_angle,
                loss_centerness=loss_centerness)
        else:
            return dict(
                loss_cls=self.cls_weight * loss_cls,
                loss_bbox=self.reg_weight * loss_bbox,
                loss_centerness=loss_centerness)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 5) for rotated boxes.
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level.
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each level.
                concat_lvl_angle_targets (list[Tensor]): Angle targets of each level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # Expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # Concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # The number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # Get labels and bbox_targets of each image
        labels_list, bbox_targets_list, angle_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # Split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0)
                            for bbox_targets in bbox_targets_list]
        angle_targets_list = [angle_targets.split(num_points, 0)
                             for angle_targets in angle_targets_list]

        # Concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets)

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl, alpha=1, thresh1=8, thresh2_bg=4):
        """Compute regression, classification and angle targets for a single image.

        This method combines CPM label assignment with regression target computation.
        """
        alpha = self.alpha
        thresh1 = self.thresh1
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))

        # Compute bbox targets using standard FCOS approach
        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes_expand = gt_bboxes[None].expand(num_points, num_gts, 5)
        points_expand = points[:, None, :].expand(num_points, num_gts, 2)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes_expand, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points_expand - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # Condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        if self.center_sampling:
            # Condition1: inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # Project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)

        # Condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # CPM label assignment based on distance to GT centers
        center_point_gt = gt_bboxes[:, :2]
        dist_sample_and_gt = torch.cdist(points, center_point_gt)  # [num_sample, num_gt]
        dist_gt_and_gt = torch.cdist(center_point_gt, center_point_gt) + \
                        torch.eye(num_gts).to(dist_sample_and_gt.device) * INF  # [num_gt, num_gt]
        dist_min_gt_and_gt, dist_min_gt_and_gt_index = dist_gt_and_gt.min(dim=1)

        labels = -1 * torch.ones(num_points, dtype=gt_labels.dtype, device=gt_labels.device)

        # Special case: single ground truth
        if num_gts == 1:
            index_pos = (dist_sample_and_gt < 8).nonzero().reshape(-1)
            index_neg = (dist_sample_and_gt > 128).nonzero().reshape(-1)
            labels[index_pos] = gt_labels[0]
            labels[index_neg] = self.num_classes
            # For regression targets, use the standard FCOS assignment
            bbox_targets_single = bbox_targets[:, 0, :]
            angle_targets_single = gt_angle[:, 0, :]
            return labels, bbox_targets_single, angle_targets_single

        # Negative labels: points far from all ground truths
        index_neg = ((alpha * dist_sample_and_gt) > dist_min_gt_and_gt).all(dim=1).nonzero().squeeze(-1)
        if len(index_neg) > 0:
            labels[index_neg] = self.num_classes

        # Positive labels: points close to ground truth centers
        thresh1_tensor = thresh1 * torch.ones_like(dist_min_gt_and_gt)
        dist_min_thresh1_gt = torch.min(dist_min_gt_and_gt / 2, thresh1_tensor)
        index_pos = (dist_sample_and_gt < dist_min_thresh1_gt).nonzero()
        if len(index_pos) > 0:
            labels[index_pos[:, 0]] = gt_labels[index_pos[:, 1]]

        # Additional background labels between same-class instances
        is_nearest_same_class = (gt_labels[dist_min_gt_and_gt_index] == gt_labels)
        valid_middle_point = (center_point_gt[is_nearest_same_class] +
                             center_point_gt[dist_min_gt_and_gt_index][is_nearest_same_class]) / 2
        dist_sample_and_gt_middle = torch.cdist(points, valid_middle_point)
        index_neg_additional = (dist_sample_and_gt_middle < thresh2_bg).any(dim=1).nonzero().squeeze(-1)
        if len(index_neg_additional) > 0:
            labels[index_neg_additional] = self.num_classes

        # For regression targets, use standard FCOS assignment (choose closest valid GT)
        # If there are still more than one objects for a location, choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        bbox_targets_out = bbox_targets[range(num_points), min_area_inds]
        angle_targets_out = gt_angle[range(num_points), min_area_inds]

        # Set regression targets to 0 for background samples
        bbox_targets_out[min_area == INF] = 0
        angle_targets_out[min_area == INF] = 0

        return labels, bbox_targets_out, angle_targets_out
