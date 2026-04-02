# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS
from mmdet.models.dense_heads import FCOSHead

INF = 1e8


@HEADS.register_module()
class CPMHboxHead(FCOSHead):
    """CPM Head for horizontal bounding boxes (Hbox).

    This head uses the CPM label assignment strategy but outputs
    horizontal bounding boxes instead of rotated boxes.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        cls_weight (float): Weight for classification loss. Default: 1.0.
        thresh1 (float): Threshold for positive sample assignment. Default: 8.
        alpha (float): Alpha parameter for negative sample assignment. Default: 1.0.
    """

    def __init__(self,
                 *args,
                 cls_weight=1.0,
                 thresh1=8,
                 alpha=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_weight = cls_weight
        self.thresh1 = thresh1
        self.alpha = alpha

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale level.
            centernesses (list[Tensor]): Centerness for each scale level.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding boxes
                can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        # Use CPM label assignment
        labels = self.get_targets(all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # Flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)

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

        loss_cls = self.loss_cls(
            flatten_cls_scores[avail_inds],
            flatten_labels[avail_inds],
            avg_factor=num_avail)

        return dict(
            loss_cls=self.cls_weight * loss_cls,
            loss_bbox=0. * loss_cls,
            loss_centerness=0. * loss_cls)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification targets for points in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple: Labels of each level.
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

        # Get labels of each image
        labels_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # Split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        concat_lvl_labels = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
        return concat_lvl_labels

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl, thresh2_bg=4):
        """Compute regression, classification targets for a single image.

        Uses CPM label assignment strategy based on center point distances.
        """
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        # Convert horizontal boxes to center points
        # gt_bboxes format: [x1, y1, x2, y2]
        center_point_gt = torch.stack([
            (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2,
            (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
        ], dim=1)

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes)

        # Distance between samples and ground truth centers
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
            return labels

        # Negative labels: points far from all ground truths
        index_neg = ((self.alpha * dist_sample_and_gt) > dist_min_gt_and_gt).all(dim=1).nonzero().squeeze(-1)
        if len(index_neg) > 0:
            labels[index_neg] = self.num_classes

        # Positive labels: points close to ground truth centers
        thresh1_tensor = self.thresh1 * torch.ones_like(dist_min_gt_and_gt)
        dist_min_thresh1_gt = torch.min(dist_min_gt_and_gt / 2, thresh1_tensor)
        index_pos = (dist_sample_and_gt < dist_min_thresh1_gt).nonzero()
        if len(index_pos) > 0:
            labels[index_pos[:, 0]] = gt_labels[index_pos[:, 1]]

        # Additional background labels between same-class instances
        is_nearest_same_class = (gt_labels[dist_min_gt_and_gt_index] == gt_labels)
        valid_middle_point = (center_point_gt[is_nearest_same_class] +
                             center_point_gt[dist_min_gt_and_gt_index][is_nearest_same_class]) / 2
        dist_sample_and_gt = torch.cdist(points, valid_middle_point)
        index_neg_additional = (dist_sample_and_gt < thresh2_bg).any(dim=1).nonzero().squeeze(-1)
        if len(index_neg_additional) > 0:
            labels[index_neg_additional] = self.num_classes

        return labels
