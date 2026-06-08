# Copyright (c) OpenMMLab. All rights reserved.
"""Fused pseudo label head that generates pseudo labels using two methods
and fuses them internally, following the logic of fuse_pseudo_labels.py.

Method 1 (PCA-edge, from PseudoLabelHead): Uses PCA on classification
probability maps to determine orientation, then edge boundary detection
along principal axes.

Method 2 (Probmap-seg, from CPMVPDSegPseudoHead): Builds probability maps
from model predictions, uses GT-guided segmentation, then decodes OBB
from mask using PCA.

The two sets of pseudo labels are fused using weighted averaging of
(cx, cy, w, h) and angular mean for theta.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os

from mmcv.ops import nms_rotated
from mmcv.runner import force_fp32
from mmrotate.core import multiclass_nms_rotated

from ..builder import ROTATED_HEADS
from .cpm_vpd_head import CPMVPDHead

try:
    from tools.analysis_tools.segment_variance_map import (
        build_gt_guided_segmentation_mask,
        _decode_obb_from_probmap,
    )
except Exception:
    from segment_variance_map import (  # type: ignore
        build_gt_guided_segmentation_mask,
        _decode_obb_from_probmap,
    )

INF = 1e8


@ROTATED_HEADS.register_module()
class CPMVPDFusedPseudoHead(CPMVPDHead):
    """Fused pseudo label head combining PCA-edge and probmap-seg methods.

    Generates pseudo labels using both methods and fuses them with weighted
    averaging of (cx, cy, w, h) and angular mean for theta, following
    fuse_pseudo_labels.py logic.

    Args:
        # --- PseudoLabelHead parameters ---
        cls_weight (int): Weight for classification loss. Default: 20.
        thresh3 (float | list): Per-class threshold for edge boundary
            detection. If float, broadcast to all classes. Default: 0.1.
        multiple_factor (float): Factor for center factor computation.
            Default: 1/16.
        pca_length (int): Length of PCA rectangle. Default: 28.
        default_max_length (int): Default max length for edge boundary.
            Default: 128.
        # --- CPMVPDSegPseudoHead parameters ---
        sigma_scale (float): Scale for Gaussian prior sigma. Default: 0.2.
        min_sigma (float): Minimum sigma. Default: 0.7.
        max_sigma (float): Maximum sigma. Default: 20.0.
        seg_score_thr (float): Score threshold for segmentation. Default: 0.05.
        seg_topk (int): Top-k pixels per object. Default: 0.
        bg_std_scale (float | None): Background std scale. Default: 1.0.
        per_gt_thr_ratio (float): Per-GT threshold ratio. Default: 0.5.
        mask_min_pixels (int): Minimum pixels for valid mask. Default: 6.
        enable_final_nms (bool): Enable final NMS. Default: False.
        class_agnostic_nms (bool): Class-agnostic NMS. Default: True.
        class_agnostic_iou_thr (float): IoU threshold for class-agnostic NMS.
            Default: 0.1.
        cls_floor (float): Floor for classification weight. Default: 0.7.
        cls_gamma (float): Gamma for classification weight. Default: 0.5.
        uncert_q_lo (float): Low quantile for uncertainty. Default: 0.01.
        uncert_q_hi (float): High quantile for uncertainty. Default: 0.40.
        uncert_gamma (float): Gamma for uncertainty weight. Default: 0.5.
        alpha_cls (float): Alpha for classification in probmap. Default: 0.66.
        alpha_uncert (float): Alpha for uncertainty in probmap. Default: 0.33.
        prob_smooth_ksize (int): Kernel size for probmap smoothing.
            Default: 3.
        prob_local_contrast (float): Local contrast beta. Default: 0.30.
        # --- Fusion parameters ---
        fuse_w1 (float): Weight for method 1 in fusion. Default: 1.0.
        fuse_w2 (float): Weight for method 2 in fusion. Default: 1.0.
        fuse_score_mode (str): Score fusion mode ('first', 'second', 'avg',
            'min', 'max'). Default: 'first'.
        float_format (str): Float format for export. Default: '{:.1f}'.
    """

    def __init__(self,
                 # --- PseudoLabelHead parameters ---
                 cls_weight=20,
                 thresh3=0.1,
                 multiple_factor=1 / 16,
                 pca_length=28,
                 default_max_length=128,
                 # --- CPMVPDSegPseudoHead parameters ---
                 sigma_scale=0.2,
                 min_sigma=0.7,
                 max_sigma=20.0,
                 seg_score_thr=0.05,
                 seg_topk=0,
                 bg_std_scale=1.0,
                 per_gt_thr_ratio=0.5,
                 mask_min_pixels=6,
                 enable_final_nms=False,
                 class_agnostic_nms=True,
                 class_agnostic_iou_thr=0.1,
                 cls_floor=0.7,
                 cls_gamma=0.5,
                 uncert_q_lo=0.01,
                 uncert_q_hi=0.40,
                 uncert_gamma=0.5,
                 alpha_cls=0.66,
                 alpha_uncert=0.33,
                 prob_smooth_ksize=3,
                 prob_local_contrast=0.30,
                 # --- Fusion parameters ---
                 fuse_w1=1.0,
                 fuse_w2=1.0,
                 fuse_score_mode='first',
                 float_format='{:.1f}',
                 **kwargs):

        super().__init__(**kwargs)

        # ---- PseudoLabelHead params ----
        self.cls_weight = int(cls_weight)
        self.thresh3 = thresh3
        self.multiple_factor = float(multiple_factor)
        self.pca_length = int(pca_length)
        self.default_max_length = int(default_max_length)

        # Override from train_cfg if provided (same pattern as PseudoLabelHead)
        train_cfg = kwargs.get('train_cfg', {}) or {}
        if 'thresh3' in train_cfg:
            self.thresh3 = train_cfg['thresh3']
        if 'cls_weight' in train_cfg:
            self.cls_weight = train_cfg['cls_weight']
        if 'pca_length' in train_cfg:
            self.pca_length = train_cfg['pca_length']
        if 'multiple_factor' in train_cfg:
            self.multiple_factor = train_cfg['multiple_factor']
        # Ensure thresh3 is indexable per class
        if isinstance(self.thresh3, (int, float)):
            self.thresh3 = [float(self.thresh3)] * self.num_classes
        assert len(self.thresh3) == self.num_classes, (
            f'thresh3 length {len(self.thresh3)} != num_classes '
            f'{self.num_classes}')

        # ---- CPMVPDSegPseudoHead params ----
        self.sigma_scale = float(sigma_scale)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.seg_score_thr = float(seg_score_thr)
        self.seg_topk = int(seg_topk)
        self.bg_std_scale = (None if bg_std_scale is None
                             else float(bg_std_scale))
        self.per_gt_thr_ratio = float(per_gt_thr_ratio)
        self.mask_min_pixels = int(mask_min_pixels)
        self.cls_floor = float(cls_floor)
        self.cls_gamma = float(cls_gamma)
        self.uncert_q_lo = float(uncert_q_lo)
        self.uncert_q_hi = float(uncert_q_hi)
        self.uncert_gamma = float(uncert_gamma)
        self.alpha_cls = float(alpha_cls)
        self.alpha_uncert = float(alpha_uncert)
        self.prob_smooth_ksize = int(prob_smooth_ksize)
        self.prob_local_contrast = float(prob_local_contrast)
        self.enable_final_nms = bool(enable_final_nms)
        self.class_agnostic_nms = bool(class_agnostic_nms)
        self.class_agnostic_iou_thr = float(class_agnostic_iou_thr)

        # ---- Fusion params ----
        self.fuse_w1 = float(fuse_w1)
        self.fuse_w2 = float(fuse_w2)
        self.fuse_score_mode = fuse_score_mode
        self.float_format = float_format

        # ---- Directories ----
        self.store_ann_dir = train_cfg.get('store_ann_dir', None)
        if self.store_ann_dir is not None:
            os.makedirs(self.store_ann_dir, exist_ok=True)
        self.store_dir = train_cfg.get('store_dir', None)
        if self.store_dir is not None:
            os.makedirs(os.path.join(self.store_dir, 'visualize'),
                        exist_ok=True)

        self.default_classes = (
            'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field',
            'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    # =====================================================================
    # Method 1 helpers (from PseudoLabelHead)
    # =====================================================================

    def _m1_get_center_factor(self, center_point_gt, gt_labels,
                              cls_scores_first_level):
        """Compute per-GT center factor on the feature map.

        Same logic as PseudoLabelHead.get_center_factor.
        """
        num_gts = center_point_gt.shape[0]
        _, H, W = cls_scores_first_level.shape
        unique_labels = gt_labels.unique()
        center_factors = []
        for label in unique_labels:
            mask = (gt_labels == label)
            gt_ctrs = center_point_gt[mask]
            center_factor_i = self._m1_get_center_factor_cls(gt_ctrs, H, W)
            center_factors.append(center_factor_i)

        dtype = (center_factors[0].dtype if center_factors
                 else torch.float32)
        device = (center_factors[0].device if center_factors
                  else gt_labels.device)
        final_center_factors = torch.zeros((num_gts, H, W),
                                           dtype=dtype, device=device)
        for label, center_factor_i in zip(unique_labels, center_factors):
            if center_factor_i is None:
                continue
            mask = (gt_labels == label)
            final_center_factors[mask] = center_factor_i
        return final_center_factors

    def _m1_get_center_factor_cls(self, gt_ctrs, H, W):
        """Compute center factor for GTs of a single class."""
        num_gts_cls = gt_ctrs.shape[0]
        if num_gts_cls == 0:
            return None
        elif num_gts_cls == 1:
            return torch.ones((1, H, W), dtype=torch.float32,
                              device=gt_ctrs.device)
        else:
            points_rect_x = torch.arange(
                0, W * self.strides[0], self.strides[0],
                device=gt_ctrs.device).float()
            points_rect_y = torch.arange(
                0, H * self.strides[0], self.strides[0],
                device=gt_ctrs.device).float()
            points_rect_xy = torch.stack(
                torch.meshgrid(points_rect_x, points_rect_y), -1
            ).reshape(-1, 2)
            each_gt_factor = torch.cdist(points_rect_xy, gt_ctrs)
            each_gt_factor = each_gt_factor.transpose(0, 1).reshape(
                num_gts_cls, H, W)
            each_gt_factor = each_gt_factor.transpose(1, 2)
            each_gt_factor_exp = torch.exp(
                -each_gt_factor * self.multiple_factor) + 1e-6
            sum_exp = each_gt_factor_exp.sum(dim=0).unsqueeze(0)
            center_factor_cls = each_gt_factor_exp / sum_exp
            return center_factor_cls

    def _m1_get_rectangle_cls_prob(self, cls_score, stride, center_factor,
                                   gt_ctr, pca_length, mode='near'):
        """Extract classification probability in a rectangle around each GT.

        Same logic as PseudoLabelHead.get_rectangle_cls_prob.
        """
        H, W = cls_score.shape[1], cls_score.shape[2]
        gt_ctr_lvl = gt_ctr / stride
        length_lvl = pca_length / stride
        rect_length = 2 * int((length_lvl - 1) / 2) + 1
        padding = (10, 10, 10, 10)
        padded_cls_score = F.pad(cls_score, padding, mode='constant', value=0)
        padded_center_factor = F.pad(center_factor, padding,
                                     mode='constant', value=0)
        gt_ctr_based_rect = torch.zeros(
            gt_ctr_lvl.shape[0], cls_score.shape[0],
            rect_length, rect_length, device=cls_score.device)
        gt_ctr_lvl = gt_ctr_lvl + 10

        if mode == 'near':
            gt_ctr_lvl = gt_ctr_lvl.round().long()
            half = int((length_lvl - 1) / 2)
            x_max = gt_ctr_lvl[:, 0] + half
            x_min = gt_ctr_lvl[:, 0] - half
            y_max = gt_ctr_lvl[:, 1] + half
            y_min = gt_ctr_lvl[:, 1] - half
            for i in range(gt_ctr_lvl.shape[0]):
                try:
                    y0, y1 = int(y_min[i]), int(y_min[i] + length_lvl)
                    x0, x1 = int(x_min[i]), int(x_min[i] + length_lvl)
                    cf_i = padded_center_factor[i, y0:y1, x0:x1]
                    gt_ctr_based_rect[i] = (
                        padded_cls_score[:, y0:y1, x0:x1]
                        * cf_i.unsqueeze(0))
                except Exception:
                    pass
        return gt_ctr_based_rect

    def _m1_get_closest_gt_first_axis(self, gt_labels, eigvec_first,
                                      center_point_gt, angle_threshold=0.866):
        """Get the range of the closest GT along the first PCA axis."""
        num_gts = center_point_gt.shape[0]
        unique_labels = gt_labels.unique()
        first_axis_range = []
        for label in unique_labels:
            mask = (gt_labels == label)
            gt_ctrs = center_point_gt[mask]
            eigvec_first_cls = eigvec_first[mask]
            first_axis_range_i = self._m1_get_closest_gt_first_axis_cls(
                gt_ctrs, eigvec_first_cls, angle_threshold)
            first_axis_range.append(first_axis_range_i)

        dtype = (first_axis_range[0].dtype if first_axis_range
                 else torch.float32)
        device = (first_axis_range[0].device if first_axis_range
                  else gt_labels.device)
        final_first_axis_range = torch.zeros((num_gts,), dtype=dtype,
                                             device=device)
        for label, first_axis_range_i in zip(unique_labels,
                                             first_axis_range):
            if first_axis_range_i is None:
                continue
            mask = (gt_labels == label)
            final_first_axis_range[mask] = first_axis_range_i
        return torch.abs(final_first_axis_range)

    def _m1_get_closest_gt_first_axis_cls(self, gt_ctrs, eigvec_first,
                                          angle_threshold=0.866):
        """Get closest GT first axis range for a single class."""
        num_gts_cls = gt_ctrs.shape[0]
        if num_gts_cls == 0:
            return None
        elif num_gts_cls == 1:
            return 512 * torch.ones((1,), dtype=torch.float32,
                                    device=gt_ctrs.device)
        else:
            first_eigvec_range = torch.zeros(
                (num_gts_cls,), dtype=torch.float32, device=gt_ctrs.device)
            eigvec_first_norm = eigvec_first / torch.norm(
                eigvec_first, dim=1, keepdim=True)
            gt_and_gt_vector = gt_ctrs - gt_ctrs.unsqueeze(1)
            gt_vec_proj = torch.abs(
                (gt_and_gt_vector * eigvec_first_norm.unsqueeze(1)).sum(
                    dim=-1))
            gt_and_gt_norm = torch.norm(gt_and_gt_vector, dim=-1) + 1e-8
            gt_and_gt_norm_cos_angle = gt_vec_proj / gt_and_gt_norm
            mask_valid_angle = gt_and_gt_norm_cos_angle > angle_threshold
            for i in range(num_gts_cls):
                mask_valid_angle_i = mask_valid_angle[i]
                if mask_valid_angle_i.sum() == 0:
                    first_eigvec_range[i] = 512
                    continue
                gt_proj = gt_vec_proj[i, mask_valid_angle_i]
                first_eigvec_range[i] = torch.min(gt_proj)
            return first_eigvec_range

    def _m1_get_edge_boundary_simple(self, gt_labels, eigvec,
                                     center_point_gt, cls_scores_all,
                                     default_max_length, is_secondary=False,
                                     is_nearest_same_class=None,
                                     nearest_gt_point=None,
                                     first_axis_range=None):
        """Get edge boundary along an eigenvector direction.

        Same logic as PseudoLabelHead.get_edge_boundary_simple.
        """
        num_gts = center_point_gt.shape[0]
        center_point_gt_feat = center_point_gt / self.strides[0]
        eigvec_norm = eigvec / torch.norm(eigvec, dim=1, keepdim=True)
        top_bottom = torch.zeros(num_gts, 2, device=center_point_gt.device)
        cls_scores_sigmoid = cls_scores_all[0].sigmoid()

        if first_axis_range is not None:
            first_axis_range = first_axis_range / self.strides[0]

        for i in range(num_gts):
            ctr = center_point_gt_feat[i]
            eigvec_i = eigvec_norm[i]
            is_same_class = is_nearest_same_class[i]
            nearest_gt_point_i = nearest_gt_point[i] / self.strides[0]
            direction = nearest_gt_point_i - ctr
            direction_norm = direction / (torch.norm(direction) + 1e-8)
            distance = torch.abs((direction * eigvec_i).sum())

            if not is_secondary:
                valid_dup = torch.abs(
                    (direction_norm * eigvec_i).sum()) > 0.866
            else:
                valid_dup = torch.abs(
                    (direction_norm * eigvec_i).sum()) > 0.5

            # Positive direction
            for j in range(default_max_length):
                point = (ctr + j * eigvec_i).round().long()
                if (point[0] < 0
                        or point[0] >= cls_scores_all[0].shape[2]
                        or point[1] < 0
                        or point[1] >= cls_scores_all[0].shape[1]):
                    top_bottom[i, 0] = j
                    break
                if (cls_scores_sigmoid[gt_labels[i], point[1], point[0]]
                        < self.thresh3[gt_labels[i]]):
                    top_bottom[i, 0] = j
                    break
                if valid_dup:
                    if is_same_class and j > 0.5 * distance:
                        top_bottom[i, 0] = j
                        break
                if first_axis_range is not None:
                    if j > 0.6 * first_axis_range[i]:
                        top_bottom[i, 0] = j
                        break

            # Negative direction
            for j in range(default_max_length):
                point = (ctr - j * eigvec_i).round().long()
                if (point[0] < 0
                        or point[0] >= cls_scores_all[0].shape[2]
                        or point[1] < 0
                        or point[1] >= cls_scores_all[0].shape[1]):
                    top_bottom[i, 1] = j
                    break
                if (cls_scores_sigmoid[gt_labels[i], point[1], point[0]]
                        < self.thresh3[gt_labels[i]]):
                    top_bottom[i, 1] = j
                    break
                if is_secondary and valid_dup:
                    if is_same_class and j > 0.5 * distance:
                        top_bottom[i, 0] = j - 1
                        break
                if first_axis_range is not None:
                    if j > 0.6 * first_axis_range[i]:
                        top_bottom[i, 0] = j
                        break

        return top_bottom[:, 0], top_bottom[:, 1]

    # =====================================================================
    # Method 1: PCA + edge boundary pseudo label generation
    # =====================================================================

    def _method1_generate_polys(self, cls_scores, gt_bboxes, gt_labels,
                                img_metas):
        """Generate polygon pseudo labels using PCA + edge boundary method.

        Returns:
            list[list[tuple]]: [img_id][gt_idx] = (poly8_tensor, label)
                poly8_tensor is shape (8,) with 4 corner (x,y) pairs.
                Returns None for a GT if generation fails.
        """
        num_imgs = cls_scores[0].shape[0]
        stride = self.strides[0]
        all_results = []

        for img_id in range(num_imgs):
            gt_boxes_this = gt_bboxes[img_id]
            labels_this = gt_labels[img_id]

            if gt_boxes_this is None or gt_boxes_this.numel() == 0:
                all_results.append([])
                continue

            num_gts = gt_boxes_this.shape[0]
            center_point_gt = gt_boxes_this[:, :2]
            cls_score_img = cls_scores[0][img_id]  # [C, H, W]

            # --- Center factor ---
            center_factor = self._m1_get_center_factor(
                center_point_gt, labels_this, cls_score_img)

            # --- Rectangle cls prob ---
            gt_ctr_rect = self._m1_get_rectangle_cls_prob(
                cls_score_img.sigmoid(), stride, center_factor,
                center_point_gt, self.pca_length, mode='near')

            # --- PCA on rectangles ---
            gt_ctr_rect_label = gt_ctr_rect[
                torch.arange(num_gts), labels_this, :, :]
            gt_rect_ctr2edge = gt_ctr_rect_label.shape[-1] // 2
            pr_x = torch.arange(
                -gt_rect_ctr2edge, gt_rect_ctr2edge + 1, 1,
                device=gt_ctr_rect.device)
            pr_y = torch.arange(
                -gt_rect_ctr2edge, gt_rect_ctr2edge + 1, 1,
                device=gt_ctr_rect.device)
            points_rect_xy = torch.stack(
                torch.meshgrid(pr_x, pr_y), -1).reshape(-1, 2)

            gt_ctr_rect_flat = gt_ctr_rect_label.transpose(
                1, 2).contiguous().view(num_gts, -1)
            pts_adapt = (points_rect_xy.unsqueeze(0).repeat(num_gts, 1, 1)
                         * torch.sqrt(gt_ctr_rect_flat).unsqueeze(-1))
            cov = torch.matmul(pts_adapt.transpose(1, 2), pts_adapt) / (
                gt_ctr_rect_flat.shape[-1] ** 2 - 1)

            eigvals, eigvecs = torch.symeig(cov, eigenvectors=True)
            larger_idx = (eigvals[:, 1] > eigvals[:, 0]).int()
            eigvec_first = (
                eigvecs[:, 0, :] * (1 - larger_idx).unsqueeze(1).repeat(1, 2)
                + eigvecs[:, 1, :] * larger_idx.unsqueeze(1).repeat(1, 2))

            mask_eigvec = (eigvec_first[:, 1] > 0).int()
            epsilon = mask_eigvec * 1e-6 + (1 - mask_eigvec) * -1e-6
            angle_targets = -torch.atan(
                eigvec_first[:, 0]
                / (eigvec_first[:, 1] + epsilon)).unsqueeze(-1)

            eigvec_second = torch.stack(
                [-eigvec_first[:, 1], eigvec_first[:, 0]], -1)

            # --- Closest GT first axis range ---
            first_axis_range = self._m1_get_closest_gt_first_axis(
                labels_this, eigvec_first, center_point_gt,
                angle_threshold=0.866)

            # --- Nearest same-class GT info ---
            dist_gt_and_gt = torch.cdist(
                center_point_gt, center_point_gt
            ) + torch.eye(num_gts, device=center_point_gt.device) * INF
            dist_min_gt_idx = dist_gt_and_gt.min(dim=1)[1]
            is_nearest_same_class = (
                labels_this[dist_min_gt_idx] == labels_this)

            # --- Edge boundaries ---
            top, bottom = self._m1_get_edge_boundary_simple(
                labels_this, eigvec_first, center_point_gt,
                [cls_score_img], self.default_max_length,
                is_secondary=False,
                is_nearest_same_class=is_nearest_same_class,
                nearest_gt_point=center_point_gt[dist_min_gt_idx],
                first_axis_range=first_axis_range)

            left, right = self._m1_get_edge_boundary_simple(
                labels_this, eigvec_second, center_point_gt,
                [cls_score_img], self.default_max_length,
                is_secondary=True,
                is_nearest_same_class=is_nearest_same_class,
                nearest_gt_point=center_point_gt[dist_min_gt_idx])

            top = top * stride + 1
            bottom = bottom * stride + 1
            left = left * stride + 1
            right = right * stride + 1

            # --- Build OBB and convert to polygon ---
            w = (left + right).unsqueeze(-1)
            h = (top + bottom).unsqueeze(-1)
            pseudo_obbs = torch.cat(
                [center_point_gt, w, h, angle_targets], -1)

            img_results = []
            for i in range(num_gts):
                poly8 = self._obb_to_poly8_internal(pseudo_obbs[i])
                img_results.append(
                    (poly8.detach(), labels_this[i].long()))

            all_results.append(img_results)

        return all_results

    # =====================================================================
    # Method 2 helpers (from CPMVPDSegPseudoHead)
    # =====================================================================

    @staticmethod
    def _infer_image_shape(img_meta):
        for key in ('img_shape', 'pad_shape', 'ori_shape'):
            shape = img_meta.get(key, None)
            if shape is not None and len(shape) >= 2:
                return float(shape[0]), float(shape[1])
        raise KeyError(
            'img_meta must contain one of img_shape/pad_shape/ori_shape.')

    @staticmethod
    def _extract_center_and_size(gt_bboxes):
        if gt_bboxes is None or gt_bboxes.numel() == 0:
            return None, None, None, None
        boxes = gt_bboxes.float()
        if boxes.dim() == 1:
            boxes = boxes.unsqueeze(0)
        if boxes.size(1) >= 8:
            xs = boxes[:, 0:8:2]
            ys = boxes[:, 1:8:2]
            cx = xs.mean(dim=1)
            cy = ys.mean(dim=1)
            bw = (xs.max(dim=1)[0] - xs.min(dim=1)[0]).clamp(min=1e-3)
            bh = (ys.max(dim=1)[0] - ys.min(dim=1)[0]).clamp(min=1e-3)
        elif boxes.size(1) >= 5:
            cx = boxes[:, 0]
            cy = boxes[:, 1]
            bw = boxes[:, 2].abs().clamp(min=1e-3)
            bh = boxes[:, 3].abs().clamp(min=1e-3)
        elif boxes.size(1) >= 4:
            x1, y1 = boxes[:, 0], boxes[:, 1]
            x2, y2 = boxes[:, 2], boxes[:, 3]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bw = (x2 - x1).abs().clamp(min=1e-3)
            bh = (y2 - y1).abs().clamp(min=1e-3)
        else:
            raise ValueError(f'Unsupported gt bbox shape: {tuple(boxes.shape)}')
        return cx, cy, bw, bh

    @staticmethod
    def _gaussian_priors(h, w, cx, cy, sigma):
        yy, xx = torch.meshgrid(
            torch.arange(h, device=cx.device, dtype=torch.float32),
            torch.arange(w, device=cx.device, dtype=torch.float32),
            indexing='ij')
        dx2 = (xx.unsqueeze(0) - cx.view(-1, 1, 1)) ** 2
        dy2 = (yy.unsqueeze(0) - cy.view(-1, 1, 1)) ** 2
        denom = 2.0 * sigma.view(-1, 1, 1) ** 2 + 1e-6
        return torch.exp(-(dx2 + dy2) / denom)

    def _build_prob_map(self, cls_score_lvl, centerness_lvl, bbox_pred_lvl):
        """Build probability map same as CPMVPDSegPseudoHead._build_prob_map."""
        cls_score_lvl = torch.nan_to_num(
            cls_score_lvl, nan=0.0, posinf=50.0, neginf=-50.0)
        log_sigma = torch.nan_to_num(
            bbox_pred_lvl[2:4], nan=0.0, posinf=1e4, neginf=-1e4)
        std = log_sigma.exp()
        std_geo = torch.sqrt(torch.clamp(std[0] * std[1], min=1e-12))

        max_class_prob = cls_score_lvl.sigmoid().max(dim=0)[0]

        q_lo = torch.quantile(std_geo, self.uncert_q_lo)
        q_hi = torch.quantile(std_geo, self.uncert_q_hi)
        denom = float(max(q_hi - q_lo, 1e-6))
        std_geo_norm = ((std_geo - float(q_lo)) / denom).clamp(0.0, 1.0)
        uncert_weight = torch.pow(
            (1.0 - std_geo_norm).clamp(0.0, 1.0), self.uncert_gamma)

        cls_stretch = (
            (max_class_prob - float(self.cls_floor))
            / max(1.0 - float(self.cls_floor), 1e-6)).clamp(0.0, 1.0)
        cls_weight = torch.pow(
            torch.clamp(cls_stretch, min=1e-6), float(self.cls_gamma))

        probmap = (float(self.alpha_cls) * cls_weight
                   + float(self.alpha_uncert) * uncert_weight)

        k = int(self.prob_smooth_ksize)
        if k % 2 == 0:
            k += 1
        if k > 1:
            pad = k // 2
            probmap = F.avg_pool2d(
                probmap[None, None], kernel_size=k, stride=1,
                padding=pad)[0, 0]

        beta = float(self.prob_local_contrast)
        if beta > 0.0:
            local_avg = F.avg_pool2d(
                probmap[None, None], kernel_size=3, stride=1,
                padding=1)[0, 0]
            probmap = (probmap - beta * local_avg).clamp(min=0.0)

        pmax = torch.clamp(probmap.max(), min=1e-6)
        probmap = (probmap / pmax).clamp(min=1e-6, max=1.0)
        return probmap

    # =====================================================================
    # Method 2: Probmap + segmentation pseudo label generation
    # =====================================================================

    def _method2_generate_polys(self, cls_scores, bbox_preds, centernesses,
                                gt_bboxes, gt_labels, img_metas):
        """Generate polygon pseudo labels using probmap + segmentation method.

        Returns:
            list[list[tuple]]: [img_id][gt_idx] = (poly8_tensor, label, score)
                or None if generation failed for that GT.
        """
        num_levels = len(cls_scores)
        num_imgs = cls_scores[0].shape[0]
        feat_level = 0
        if hasattr(self, 'test_cfg') and self.test_cfg:
            feat_level = int(self.test_cfg.get('feat_level', 0))
        feat_level = max(0, min(feat_level, num_levels - 1))

        all_results = []
        for img_id in range(num_imgs):
            gt_boxes_this = gt_bboxes[img_id]
            labels_this = gt_labels[img_id]

            if gt_boxes_this is None or gt_boxes_this.numel() == 0:
                all_results.append([])
                continue

            stride = self.strides[feat_level]
            bbox_pred_lvl = bbox_preds[feat_level][img_id]
            cls_score_lvl = cls_scores[feat_level][img_id]
            centerness_lvl = centernesses[feat_level][img_id]

            probmap = self._build_prob_map(
                cls_score_lvl, centerness_lvl, bbox_pred_lvl)

            label_map = build_gt_guided_segmentation_mask(
                p_model=probmap,
                gt_bboxes=gt_boxes_this,
                img_meta=img_metas[img_id],
                sigma_scale=self.sigma_scale,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                score_thr=self.seg_score_thr,
                topk=self.seg_topk,
                bg_std_scale=self.bg_std_scale)

            fg_mask = (label_map != -1)
            num_gts = gt_boxes_this.shape[0]

            img_results = []
            for gt_idx in range(num_gts):
                cls_id = labels_this[gt_idx].long()
                per_gt_map = probmap * (label_map == gt_idx).to(probmap.dtype)

                box = _decode_obb_from_probmap(
                    per_gt_map=per_gt_map,
                    fg_mask=fg_mask,
                    stride=float(stride),
                    mask_min_pixels=self.mask_min_pixels)

                if box is None:
                    img_results.append(None)
                    continue

                # Compute score
                mask_for_score = (per_gt_map > 0) & fg_mask
                if mask_for_score.any():
                    score = probmap[mask_for_score].mean()
                else:
                    score = probmap.new_tensor(0.0)
                score = torch.nan_to_num(
                    score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)

                # Convert OBB to polygon using CPMVPDSegPseudoHead convention
                poly8 = self._obb_to_poly8_vpd(box)
                img_results.append((poly8.detach(), cls_id, score))

            all_results.append(img_results)

        return all_results

    # =====================================================================
    # Polygon <-> OBB conversion helpers
    # (from fuse_pseudo_labels.py, ported to PyTorch)
    # =====================================================================

    @staticmethod
    def _sort_rectangle_points(points):
        """Sort 4 points of a rectangle in counter-clockwise order.

        Args:
            points: (4, 2) tensor.

        Returns:
            (4, 2) tensor sorted by polar angle around centroid.
        """
        centroid = points.mean(dim=0)
        angles = torch.atan2(points[:, 1] - centroid[1],
                             points[:, 0] - centroid[0])
        order = torch.argsort(angles)
        return points[order]

    @staticmethod
    def _poly_to_obb_fuse(poly):
        """Convert 8-point polygon to (cx, cy, w, h, theta) OBB.

        Uses the same convention as fuse_pseudo_labels.py::poly_to_obb:
        h >= w, theta aligned with the longer edge.

        Args:
            poly: (8,) or (4,2) tensor.

        Returns:
            (cx, cy, w, h, theta) or None if degenerate.
        """
        pts = poly.reshape(4, 2).float()
        pts = CPMVPDFusedPseudoHead._sort_rectangle_points(pts)

        edge1 = torch.norm(pts[1] - pts[0])
        edge2 = torch.norm(pts[2] - pts[1])

        if edge1 < 1e-6 or edge2 < 1e-6:
            return None

        h = max(edge1, edge2)
        w = min(edge1, edge2)

        if edge1 >= edge2:
            theta = torch.atan2(pts[1, 1] - pts[0, 1],
                                pts[1, 0] - pts[0, 0])
        else:
            theta = torch.atan2(pts[2, 1] - pts[1, 1],
                                pts[2, 0] - pts[1, 0])

        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()
        return cx, cy, w, h, theta

    @staticmethod
    def _obb_to_poly_fuse(cx, cy, w, h, theta):
        """Convert OBB to 8-point polygon.

        Uses the same convention as fuse_pseudo_labels.py::obb_to_poly.
        """
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dx = w / 2.0
        dy = h / 2.0
        corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
        poly_parts = []
        for x, y in corners:
            rx = x * sin_t + y * cos_t + cx
            ry = -x * cos_t + y * sin_t + cy
            poly_parts.extend([rx, ry])
        return torch.stack(poly_parts)

    @staticmethod
    def _obb_to_poly8_internal(obb):
        """Convert OBB [cx, cy, w, h, angle] to poly8 using PseudoLabelHead
        rotation convention (standard rotation matrix).

        This matches PseudoLabelHead.generate_labels.
        """
        cx, cy, w, h, angle = obb[0], obb[1], obb[2], obb[3], obb[4]
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        hw = w / 2.0
        hh = h / 2.0
        corners = obb.new_tensor([
            [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh],
        ])
        rot = obb.new_tensor([[cos_a, -sin_a], [sin_a, cos_a]])
        pts = corners.matmul(rot.t()) + obb.new_tensor([cx, cy])
        return pts.reshape(-1)

    @staticmethod
    def _obb_to_poly8_vpd(obb):
        """Convert OBB [cx, cy, w, h, angle] to poly8 using CPMVPDSegPseudoHead
        convention.

        This matches CPMVPDSegPseudoHead._obb_to_poly8.
        """
        cx, cy, w, h, angle = obb[0], obb[1], obb[2], obb[3], obb[4]
        c = torch.cos(angle)
        s = torch.sin(angle)
        hw = 0.5 * w
        hh = 0.5 * h
        corners = obb.new_tensor([
            [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh],
        ])
        rot = obb.new_tensor([[c, -s], [s, c]])
        pts = corners.matmul(rot.t()) + obb.new_tensor([cx, cy])
        return pts.reshape(-1)

    # =====================================================================
    # Fusion logic (from fuse_pseudo_labels.py)
    # =====================================================================

    @staticmethod
    def _ang_mean(ang1, ang2, w1, w2):
        """Weighted angular mean, handling pi wrap-around.

        Same logic as fuse_pseudo_labels.py::ang_mean.
        """
        a1 = float(ang1)
        a2 = float(ang2)
        if abs(a1 - a2) > np.pi / 2:
            if a1 > a2:
                a2 += np.pi
            else:
                a1 += np.pi
        mean = (a1 * w1 + a2 * w2) / (w1 + w2)
        mean = mean % np.pi
        return mean

    def _fuse_polys(self, poly1, poly2, w1, w2):
        """Fuse two polygon representations.

        Follows fuse_pseudo_labels.py::fuse_pair:
        1. Convert both polys to OBB using consistent _poly_to_obb_fuse
        2. Weighted average of (cx, cy, w, h)
        3. Angular mean for theta (with w1 doubled, as in original)
        4. Convert fused OBB back to polygon using _obb_to_poly_fuse

        Args:
            poly1: (8,) tensor from method 1.
            poly2: (8,) tensor from method 2.
            w1: weight for method 1.
            w2: weight for method 2.

        Returns:
            (8,) fused polygon tensor.
        """
        obb1 = self._poly_to_obb_fuse(poly1)
        obb2 = self._poly_to_obb_fuse(poly2)

        if obb1 is None and obb2 is None:
            # Both failed, return poly1 as fallback
            return poly1
        if obb1 is None:
            return poly2
        if obb2 is None:
            return poly1

        cx1, cy1, bw1, bh1, th1 = obb1
        cx2, cy2, bw2, bh2, th2 = obb2

        total = w1 + w2
        cx = (cx1 * w1 + cx2 * w2) / total
        cy = (cy1 * w1 + cy2 * w2) / total
        bw = (bw1 * w1 + bw2 * w2) / total
        bh = (bh1 * w1 + bh2 * w2) / total

        # Double w1 for theta (same as fuse_pseudo_labels.py::fuse_pair)
        w1_theta = w1 * 2
        theta = self._ang_mean(th1, th2, w1_theta, w2)

        return self._obb_to_poly_fuse(cx, cy, bw, bh, theta)

    def _fuse_score(self, score1, score2):
        """Fuse two scores based on fuse_score_mode."""
        if self.fuse_score_mode == 'avg':
            return (score1 + score2) / 2.0
        elif self.fuse_score_mode == 'max':
            return max(score1, score2)
        elif self.fuse_score_mode == 'min':
            return min(score1, score2)
        elif self.fuse_score_mode == 'second':
            return score2
        else:  # 'first'
            return score1

    # =====================================================================
    # Export helpers
    # =====================================================================

    @staticmethod
    def _undo_flip_rbox(boxes5, img_meta):
        if boxes5.numel() == 0:
            return boxes5
        if not img_meta.get('flip', False):
            return boxes5
        out = boxes5.clone()
        direction = img_meta.get('flip_direction', 'horizontal')
        h, w = img_meta['img_shape'][:2]
        if direction == 'horizontal':
            out[:, 0] = float(w) - out[:, 0]
            out[:, 4] = -out[:, 4]
        elif direction == 'vertical':
            out[:, 1] = float(h) - out[:, 1]
            out[:, 4] = -out[:, 4]
        elif direction == 'diagonal':
            out[:, 0] = float(w) - out[:, 0]
            out[:, 1] = float(h) - out[:, 1]
        return out

    @staticmethod
    def _rescale_rbox_to_ori(boxes5, img_meta):
        if boxes5.numel() == 0:
            return boxes5
        scale = boxes5.new_tensor(img_meta['scale_factor'][:2]).repeat(2)
        out = boxes5.clone()
        out[:, :4] = out[:, :4] / scale
        return out

    def _export_fused_pseudo_txt(self, fused_results, img_metas):
        """Export fused pseudo labels to DOTA-style txt files.

        Args:
            fused_results: list of (det_bboxes, det_labels) per image,
                where det_bboxes is (N, 6) with [cx,cy,w,h,angle,score]
                and det_labels is (N,).
            img_metas: list of image meta dicts.
        """
        classes = getattr(self, 'CLASSES', None)
        if classes is None:
            classes = self.default_classes

        for img_idx, (det_bboxes, det_labels) in enumerate(fused_results):
            meta = img_metas[img_idx]
            filename_raw = os.path.splitext(
                os.path.basename(meta['filename']))[0]
            out_path = os.path.join(self.store_ann_dir,
                                    filename_raw + '.txt')

            if det_bboxes is None or det_bboxes.numel() == 0:
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write('')
                continue

            boxes5 = det_bboxes[:, :5]
            boxes5 = self._undo_flip_rbox(boxes5, meta)
            boxes5 = self._rescale_rbox_to_ori(boxes5, meta)

            lines = []
            for bi in range(boxes5.shape[0]):
                cls_id = int(det_labels[bi].item())
                if cls_id < 0 or cls_id >= len(classes):
                    continue
                # Convert OBB to polygon for export (standard rotation)
                poly8 = self._obb_to_poly8_vpd(boxes5[bi])
                poly8_vals = [float(v.item()) for v in poly8]
                lines.append(
                    f'{poly8_vals[0]:.1f} {poly8_vals[1]:.1f} '
                    f'{poly8_vals[2]:.1f} {poly8_vals[3]:.1f} '
                    f'{poly8_vals[4]:.1f} {poly8_vals[5]:.1f} '
                    f'{poly8_vals[6]:.1f} {poly8_vals[7]:.1f} '
                    f'{classes[cls_id]} 0\n')

            with open(out_path, 'w', encoding='utf-8') as f:
                if lines:
                    f.writelines(lines)
                else:
                    f.write('')

    # =====================================================================
    # Main fused pseudo label generation
    # =====================================================================

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def generate_fused_pseudo_labels(self, cls_scores, bbox_preds,
                                     centernesses, gt_bboxes, gt_labels,
                                     img_metas):
        """Generate pseudo labels using both methods and fuse them.

        Workflow:
        1. Method 1 (PCA-edge) → polygon per GT
        2. Method 2 (Probmap-seg) → polygon per GT
        3. Fuse polygons using _fuse_polys (consistent OBB conversion)
        4. Return fused results in (det_bboxes, det_labels) format

        Returns:
            list[tuple]: Each item is (det_bboxes, det_labels) where
                det_bboxes is (N, 6) with [cx, cy, w, h, angle, score]
                and det_labels is (N,).
        """
        num_imgs = cls_scores[0].shape[0]

        # Generate from both methods
        m1_results = self._method1_generate_polys(
            cls_scores, gt_bboxes, gt_labels, img_metas)
        m2_results = self._method2_generate_polys(
            cls_scores, bbox_preds, centernesses,
            gt_bboxes, gt_labels, img_metas)

        fused_results = []
        for img_id in range(num_imgs):
            gt_boxes_this = gt_bboxes[img_id]
            if gt_boxes_this is None or gt_boxes_this.numel() == 0:
                empty_b = cls_scores[0].new_zeros((0, 6))
                empty_l = gt_labels[img_id].new_zeros((0,), dtype=torch.long)
                fused_results.append((empty_b, empty_l))
                continue

            num_gts = gt_boxes_this.shape[0]
            fused_bboxes = []
            fused_labels = []

            for gt_idx in range(num_gts):
                label = gt_labels[img_id][gt_idx].long()

                # Method 1 result
                m1_item = (m1_results[img_id][gt_idx]
                           if gt_idx < len(m1_results[img_id])
                           else None)
                # Method 2 result
                m2_item = (m2_results[img_id][gt_idx]
                           if gt_idx < len(m2_results[img_id])
                           else None)

                has_m1 = m1_item is not None
                has_m2 = m2_item is not None

                if has_m1 and has_m2:
                    poly1, _ = m1_item
                    poly2, _, score2 = m2_item
                    # Fuse polygons via consistent OBB conversion
                    fused_poly = self._fuse_polys(
                        poly1, poly2, self.fuse_w1, self.fuse_w2)
                    # Convert fused poly back to OBB for output
                    fused_obb = self._poly_to_obb_fuse(fused_poly)
                    if fused_obb is not None:
                        cx, cy, w, h, theta = fused_obb
                        score = self._fuse_score(0.0, float(score2))
                        fused_bboxes.append(
                            fused_poly.new_tensor(
                                [cx, cy, w, h, theta, score]))
                        fused_labels.append(label)
                    else:
                        # Fallback to method 1
                        obb1 = self._poly_to_obb_fuse(poly1)
                        if obb1 is not None:
                            cx, cy, w, h, theta = obb1
                            fused_bboxes.append(
                                poly1.new_tensor(
                                    [cx, cy, w, h, theta, 0.0]))
                            fused_labels.append(label)

                elif has_m1:
                    poly1, _ = m1_item
                    obb1 = self._poly_to_obb_fuse(poly1)
                    if obb1 is not None:
                        cx, cy, w, h, theta = obb1
                        fused_bboxes.append(
                            poly1.new_tensor(
                                [cx, cy, w, h, theta, 0.0]))
                        fused_labels.append(label)

                elif has_m2:
                    poly2, _, score2 = m2_item
                    obb2 = self._poly_to_obb_fuse(poly2)
                    if obb2 is not None:
                        cx, cy, w, h, theta = obb2
                        fused_bboxes.append(
                            poly2.new_tensor(
                                [cx, cy, w, h, theta, float(score2)]))
                        fused_labels.append(label)
                # else: both methods failed, skip this GT

            if len(fused_bboxes) == 0:
                empty_b = cls_scores[0].new_zeros((0, 6))
                empty_l = gt_labels[img_id].new_zeros(
                    (0,), dtype=torch.long)
                fused_results.append((empty_b, empty_l))
            else:
                fused_bboxes_t = torch.stack(fused_bboxes)
                fused_labels_t = torch.stack(fused_labels)
                fused_results.append((fused_bboxes_t, fused_labels_t))

        return fused_results

    # =====================================================================
    # Loss
    # =====================================================================

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self, cls_scores, bbox_preds, centernesses, gt_bboxes,
             gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute loss and generate fused pseudo labels as a side effect.

        The training loss is computed by the parent CPMVPDHead.
        After the loss, fused pseudo labels are generated and exported.
        """
        losses = super().loss(
            cls_scores, bbox_preds, centernesses,
            gt_bboxes, gt_labels, img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)

        if self.store_ann_dir is not None:
            with torch.no_grad():
                fused_results = self.generate_fused_pseudo_labels(
                    cls_scores, bbox_preds, centernesses,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                    img_metas=img_metas)
                self._export_fused_pseudo_txt(fused_results, img_metas)

        return losses

    # =====================================================================
    # Inference (inherit from parent)
    # =====================================================================

    def get_bboxes(self, cls_scores, bbox_preds, centernesses, img_metas,
                   cfg=None, rescale=None):
        """Inference-time bbox decoding. Uses parent CPMVPDHead logic."""
        return super().get_bboxes(
            cls_scores, bbox_preds, centernesses, img_metas,
            cfg=cfg, rescale=rescale)
