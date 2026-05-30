# Copyright (c) OpenMMLab. All rights reserved.
"""CPMVPD pseudo-label head using variance-modulated CPM prob map.

Pipeline:
1) Build class-wise prob map from CPM classification score and centerness.
2) Modulate the map by variance confidence from VPD lstd channels.
3) Decode pseudo boxes with the original PseudoLabelHead flow:
   center-factor -> rectangle extraction -> PCA axis -> boundary walk.
"""

import os
import torch
import torch.nn.functional as F
from mmcv.ops import nms_rotated
from mmcv.runner import force_fp32

from mmrotate.core import multiclass_nms_rotated
from ..builder import ROTATED_HEADS
from .cpm_vpd_head import CPMVPDHead

INF = 1e8


@ROTATED_HEADS.register_module()
class CPMVPDPseudoHead(CPMVPDHead):

    def __init__(self,
                 *args,
                 remap_uncert_q_lo=0.01,
                 remap_uncert_q_hi=0.20,
                 remap_uncert_gamma=1.0,
                 remap_prob_smooth_ksize=3,
                 remap_prob_local_contrast=0.30,
                 enable_final_nms=False,
                 class_agnostic_nms=True,
                 class_agnostic_iou_thr=0.1,
                 use_cpm_directly=True,
                 **kwargs):
        # Backward compatibility: swallow legacy kwargs from older remap decoder.
        legacy_keys = [
            'point_search_radius', 'use_lstd_for_size', 'lstd_size_factor',
            'use_remap_size', 'remap_edge_thr_ratio', 'remap_edge_max_len',
            'remap_size_mix', 'remap_use_gt_guided', 'remap_cls_floor',
            'remap_cls_gamma', 'remap_alpha_cls', 'remap_alpha_uncert',
            'remap_sigma_scale', 'remap_min_sigma', 'remap_max_sigma',
            'remap_score_thr', 'remap_topk', 'remap_bg_std_scale',
            'use_pca_decode', 'pca_window_radius', 'pca_use_adaptive_radius',
            'pca_radius_scale', 'pca_radius_min', 'pca_radius_max',
            'pca_thr_ratio', 'pca_weight_gamma', 'pca_min_pixels',
            'pca_size_factor', 'pca_center_mix', 'pca_size_mix',
            'pca_angle_mix', 'pca_component_select', 'pca_connectivity',
            'pca_use_aniso_gate', 'pca_aniso_thr', 'pca_peak_dist_penalty',
            'pca_enable_instance_gate', 'pca_max_center_offset',
            'pca_min_fill_ratio', 'pca_map_source', 'use_cpm_generated_boxes'
        ]
        for key in legacy_keys:
            kwargs.pop(key, None)

        super().__init__(*args, **kwargs)

        self.remap_uncert_q_lo = float(remap_uncert_q_lo)
        self.remap_uncert_q_hi = float(remap_uncert_q_hi)
        self.remap_uncert_gamma = float(remap_uncert_gamma)
        self.remap_prob_smooth_ksize = int(remap_prob_smooth_ksize)
        self.remap_prob_local_contrast = float(remap_prob_local_contrast)
        self.enable_final_nms = bool(enable_final_nms)
        self.class_agnostic_nms = bool(class_agnostic_nms)
        self.class_agnostic_iou_thr = float(class_agnostic_iou_thr)
        self.use_cpm_directly = bool(use_cpm_directly)

        train_cfg = kwargs.get('train_cfg', {}) or {}
        self.store_ann_dir = train_cfg.get('store_ann_dir', None)
        if self.store_ann_dir is not None:
            os.makedirs(self.store_ann_dir, exist_ok=True)

        self.thresh3 = train_cfg.get('thresh3', [0.1] * self.num_classes)
        if isinstance(self.thresh3, (int, float)):
            self.thresh3 = [float(self.thresh3)] * self.num_classes
        self.pca_length = int(train_cfg.get('pca_length', 28))
        self.multiple_factor = float(train_cfg.get('multiple_factor', 1 / 16))
        assert len(self.thresh3) == self.num_classes

        self.default_classes = (
            'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field',
            'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    @staticmethod
    def _obb_to_poly8(obb):
        cx, cy, w, h, angle = obb
        c = torch.cos(angle)
        s = torch.sin(angle)
        hw = 0.5 * w
        hh = 0.5 * h

        corners = obb.new_tensor([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh],
        ])
        rot = obb.new_tensor([[c, -s], [s, c]])
        pts = corners.matmul(rot.t()) + obb.new_tensor([cx, cy])
        return pts.reshape(-1)

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

    def _export_pseudo_txt(self, bbox_list, img_metas):
        classes = getattr(self, 'CLASSES', None)
        if classes is None:
            classes = self.default_classes

        for img_idx, (det_bboxes, det_labels) in enumerate(bbox_list):
            meta = img_metas[img_idx]
            filename_raw = os.path.splitext(os.path.basename(meta['filename']))[0]
            out_path = os.path.join(self.store_ann_dir, filename_raw + '.txt')

            boxes5 = det_bboxes[:, :5] if det_bboxes.numel() > 0 else det_bboxes.new_zeros((0, 5))
            boxes5 = self._undo_flip_rbox(boxes5, meta)
            boxes5 = self._rescale_rbox_to_ori(boxes5, meta)

            lines = []
            for bi in range(boxes5.shape[0]):
                cls_id = int(det_labels[bi].item())
                if cls_id < 0 or cls_id >= len(classes):
                    continue
                poly8 = self._obb_to_poly8(boxes5[bi])
                poly8 = [float(v.item()) for v in poly8]
                lines.append(
                    f'{poly8[0]:.1f} {poly8[1]:.1f} {poly8[2]:.1f} {poly8[3]:.1f} '
                    f'{poly8[4]:.1f} {poly8[5]:.1f} {poly8[6]:.1f} {poly8[7]:.1f} '
                    f'{classes[cls_id]} 0\n')

            with open(out_path, 'w', encoding='utf-8') as f:
                if lines:
                    f.writelines(lines)
                else:
                    f.write('')

    def _build_probmap(self, cls_score_lvl, bbox_pred_lvl, centerness_lvl):
        cls_score_lvl = torch.nan_to_num(
            cls_score_lvl, nan=0.0, posinf=50.0, neginf=-50.0)
        centerness_lvl = torch.nan_to_num(
            centerness_lvl, nan=0.0, posinf=50.0, neginf=-50.0)
        lstd = torch.nan_to_num(
            bbox_pred_lvl[2:4], nan=0.0, posinf=1e4, neginf=-1e4)
        std = lstd.exp()

        cls_prob = cls_score_lvl.sigmoid()
        centerness_prob = centerness_lvl.sigmoid().squeeze(0)

        std_geo = torch.sqrt(torch.clamp(std[0] * std[1], min=1e-12))
        std_geo = torch.nan_to_num(std_geo, nan=0.0, posinf=1e6, neginf=0.0)
        q_lo = torch.quantile(std_geo, self.remap_uncert_q_lo)
        q_hi = torch.quantile(std_geo, self.remap_uncert_q_hi)
        denom = torch.clamp(q_hi - q_lo, min=1e-6)
        std_geo_norm = ((std_geo - q_lo) / denom).clamp(0.0, 1.0)
        var_conf = torch.pow(
            torch.clamp(1.0 - std_geo_norm, min=1e-6), self.remap_uncert_gamma)

        probmap = cls_prob * var_conf.unsqueeze(0)

        if self.remap_prob_smooth_ksize > 1:
            k = int(self.remap_prob_smooth_ksize)
            if k % 2 == 0:
                k += 1
            pad = k // 2
            probmap = F.avg_pool2d(
                probmap.unsqueeze(0), kernel_size=k, stride=1,
                padding=pad).squeeze(0)

        beta = float(self.remap_prob_local_contrast)
        if beta > 0.0:
            local_avg = F.avg_pool2d(
                probmap.unsqueeze(0), kernel_size=3, stride=1,
                padding=1).squeeze(0)
            probmap = (probmap - beta * local_avg).clamp(min=0.0)

        pmax = torch.clamp(probmap.max(), min=1e-6)
        return (probmap / pmax).clamp(min=1e-6, max=1.0)

    def get_rectangle_cls_prob(self, prob_map, stride, center_factor,
                               gt_ctr, pca_length, mode='near'):
        assert mode in ['near', 'bilinear'], "mode must be either 'near' or 'bilinear'"

        gt_ctr_lvl = gt_ctr / stride
        length_lvl = pca_length / stride
        rect_length = 2 * int((length_lvl - 1) / 2) + 1
        padding = (10, 10, 10, 10)
        padded_prob_map = F.pad(prob_map, padding, mode='constant', value=0)
        padded_center_factor = F.pad(center_factor, padding, mode='constant', value=0)
        gt_ctr_based_rect = torch.zeros(
            gt_ctr_lvl.shape[0], prob_map.shape[0], rect_length, rect_length,
            device=prob_map.device)

        gt_ctr_lvl = gt_ctr_lvl + 10

        if mode == 'near':
            gt_ctr_lvl = gt_ctr_lvl.round().long()
            x_min = gt_ctr_lvl[:, 0] - int((length_lvl - 1) / 2)
            y_min = gt_ctr_lvl[:, 1] - int((length_lvl - 1) / 2)
            for i in range(gt_ctr_lvl.shape[0]):
                center_factor_i = padded_center_factor[
                    i,
                    int(y_min[i]):int(y_min[i] + length_lvl),
                    int(x_min[i]):int(x_min[i] + length_lvl)
                ]
                gt_ctr_based_rect[i] = padded_prob_map[:,
                    int(y_min[i]):int(y_min[i] + length_lvl),
                    int(x_min[i]):int(x_min[i] + length_lvl)
                ] * center_factor_i.unsqueeze(0)
        else:
            x_max = gt_ctr_lvl[:, 0] + (length_lvl - 1) / 2
            x_min = gt_ctr_lvl[:, 0] - (length_lvl - 1) / 2
            y_max = gt_ctr_lvl[:, 1] + (length_lvl - 1) / 2
            y_min = gt_ctr_lvl[:, 1] - (length_lvl - 1) / 2
            for i in range(gt_ctr_lvl.shape[0]):
                x_max_i = int(x_max[i])
                x_min_i = int(x_min[i])
                y_max_i = int(y_max[i])
                y_min_i = int(y_min[i])
                x_max_weight = x_max[i] - x_max_i
                x_min_weight = 1 - x_max_weight
                y_max_weight = y_max[i] - y_max_i
                y_min_weight = 1 - y_max_weight
                gt_ctr_based_rect[i] = (
                    padded_prob_map[:, y_min_i, x_min_i] * x_min_weight * y_min_weight
                    + padded_prob_map[:, y_min_i, x_max_i] * x_max_weight * y_min_weight
                    + padded_prob_map[:, y_max_i, x_min_i] * x_min_weight * y_max_weight
                    + padded_prob_map[:, y_max_i, x_max_i] * x_max_weight * y_max_weight
                )
        return gt_ctr_based_rect

    def get_center_factor(self, center_point_gt, gt_labels, prob_map_lvl0):
        num_gts = center_point_gt.shape[0]
        _, H, W = prob_map_lvl0.shape
        unique_labels = gt_labels.unique()

        center_factors = []
        for label in unique_labels:
            mask = (gt_labels == label)
            gt_ctrs = center_point_gt[mask]
            center_factor_i = self.get_center_factor_cls(gt_ctrs, H, W)
            center_factors.append(center_factor_i)

        final_center_factors = torch.zeros(
            (num_gts, H, W),
            dtype=center_factors[0].dtype,
            device=center_factors[0].device)
        for label, center_factor_i in zip(unique_labels, center_factors):
            if center_factor_i is None:
                continue
            mask = (gt_labels == label)
            final_center_factors[mask] = center_factor_i

        return final_center_factors

    def get_center_factor_cls(self, gt_ctrs, H, W):
        num_gts_cls = gt_ctrs.shape[0]
        if num_gts_cls == 0:
            return None
        if num_gts_cls == 1:
            return torch.ones((1, H, W), dtype=torch.float32, device=gt_ctrs.device)

        points_rect_x = torch.arange(
            0, W * self.strides[0], self.strides[0], device=gt_ctrs.device).float()
        points_rect_y = torch.arange(
            0, H * self.strides[0], self.strides[0], device=gt_ctrs.device).float()
        points_rect_xy = torch.stack(
            torch.meshgrid(points_rect_x, points_rect_y), -1).reshape(-1, 2)
        each_gt_factor = torch.cdist(points_rect_xy, gt_ctrs)
        each_gt_factor = each_gt_factor.transpose(0, 1).reshape(num_gts_cls, H, W)
        each_gt_factor = each_gt_factor.transpose(1, 2)
        each_gt_factor_exp = torch.exp(-each_gt_factor * self.multiple_factor) + 1e-6
        sum_factor = each_gt_factor_exp.sum(dim=0).unsqueeze(0)
        return each_gt_factor_exp / sum_factor

    def get_closest_gt_first_axis(self, gt_labels, eigvec_first,
                                  center_point_gt, angle_threshold):
        num_gts = center_point_gt.shape[0]
        unique_labels = gt_labels.unique()

        first_axis_range = []
        for label in unique_labels:
            mask = (gt_labels == label)
            gt_ctrs = center_point_gt[mask]
            eigvec_first_cls = eigvec_first[mask]
            first_axis_range_i = self.get_closest_gt_first_axis_cls(
                gt_ctrs, eigvec_first_cls, angle_threshold)
            first_axis_range.append(first_axis_range_i)

        final_first_axis_range = torch.zeros(
            (num_gts,), dtype=first_axis_range[0].dtype,
            device=first_axis_range[0].device)
        for label, first_axis_range_i in zip(unique_labels, first_axis_range):
            if first_axis_range_i is None:
                continue
            mask = (gt_labels == label)
            final_first_axis_range[mask] = first_axis_range_i

        return torch.abs(final_first_axis_range)

    def get_closest_gt_first_axis_cls(self, gt_ctrs, eigvec_first,
                                      angle_threshold=0.866):
        num_gts_cls = gt_ctrs.shape[0]
        if num_gts_cls == 0:
            return None
        if num_gts_cls == 1:
            return 512 * torch.ones((1,), dtype=torch.float32, device=gt_ctrs.device)

        first_eigvec_range = torch.zeros(
            (num_gts_cls,), dtype=torch.float32, device=gt_ctrs.device)
        eigvec_first_norm = eigvec_first / torch.norm(eigvec_first, dim=1, keepdim=True)
        gt_and_gt_vector = gt_ctrs - gt_ctrs.unsqueeze(1)
        gt_vec_proj = torch.abs(
            (gt_and_gt_vector * eigvec_first_norm.unsqueeze(1)).sum(dim=-1))
        gt_and_gt_norm_cos_angle = gt_vec_proj / torch.norm(gt_and_gt_vector, dim=-1)
        mask_valid_angle = gt_and_gt_norm_cos_angle > angle_threshold
        for i in range(num_gts_cls):
            mask_valid_angle_i = mask_valid_angle[i]
            if mask_valid_angle_i.sum() == 0:
                first_eigvec_range[i] = 512
                continue
            gt_proj = gt_vec_proj[i, mask_valid_angle_i]
            first_eigvec_range[i] = torch.min(gt_proj)
        return first_eigvec_range

    def get_edge_boundary_simple(self, gt_labels, eigvec, center_point_gt,
                                 prob_map_lvl0, is_secondary=False,
                                 is_nearest_same_class=None, nearest_gt_point=None,
                                 first_axis_range=None, default_max_length=128):
        num_gts = center_point_gt.shape[0]
        center_point_gt = center_point_gt / self.strides[0]

        eigvec_norm = eigvec / torch.norm(eigvec, dim=1, keepdim=True)
        top_bottom = torch.zeros(num_gts, 2, device=center_point_gt.device)
        H, W = prob_map_lvl0.shape[1], prob_map_lvl0.shape[2]
        if first_axis_range is not None:
            first_axis_range = first_axis_range / self.strides[0]

        for i in range(num_gts):
            ctr = center_point_gt[i]
            eigvec_i = eigvec_norm[i]
            is_same_class = is_nearest_same_class[i]
            nearest_gt_point_i = nearest_gt_point[i] / self.strides[0]
            direction = nearest_gt_point_i - ctr
            direction_norm = direction / torch.norm(direction)
            distance = torch.abs((direction * eigvec_i).sum())
            if not is_secondary:
                valid_dup_remove = torch.abs((direction_norm * eigvec_i).sum()) > 0.866
            else:
                valid_dup_remove = torch.abs((direction_norm * eigvec_i).sum()) > 0.5

            for j in range(default_max_length):
                point = (ctr + j * eigvec_i).round().long()
                if point[0] < 0 or point[0] >= W or point[1] < 0 or point[1] >= H:
                    top_bottom[i, 0] = j
                    break
                if prob_map_lvl0[gt_labels[i], point[1], point[0]] < self.thresh3[gt_labels[i]]:
                    top_bottom[i, 0] = j
                    break
                if valid_dup_remove and is_same_class and j > 0.5 * distance:
                    top_bottom[i, 0] = j
                    break
                if first_axis_range is not None and j > 0.6 * first_axis_range[i]:
                    top_bottom[i, 0] = j
                    break

            for j in range(default_max_length):
                point = (ctr - j * eigvec_i).round().long()
                if point[0] < 0 or point[0] >= W or point[1] < 0 or point[1] >= H:
                    top_bottom[i, 1] = j
                    break
                if prob_map_lvl0[gt_labels[i], point[1], point[0]] < self.thresh3[gt_labels[i]]:
                    top_bottom[i, 1] = j
                    break
                if is_secondary and valid_dup_remove and is_same_class and j > 0.5 * distance:
                    top_bottom[i, 0] = j - 1
                    break
                if first_axis_range is not None and j > 0.6 * first_axis_range[i]:
                    top_bottom[i, 0] = j
                    break

        return top_bottom[:, 0], top_bottom[:, 1]

    def _pseudo_boxes_from_probmap(self, gt_bboxes, gt_labels, prob_map_lvl0):
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return None, None

        center_point_gt = gt_bboxes[:, :2]
        stride0 = self.strides[0]

        center_factor = self.get_center_factor(
            center_point_gt, gt_labels, prob_map_lvl0)
        gt_ctr_rect = self.get_rectangle_cls_prob(
            prob_map_lvl0, stride0, center_factor, center_point_gt,
            self.pca_length, mode='near')
        gt_ctr_rect_label = gt_ctr_rect[torch.arange(num_gts), gt_labels, :, :]

        gt_rect_ctr2edge = gt_ctr_rect_label.shape[-1] // 2
        points_rect_x = torch.arange(
            -gt_rect_ctr2edge, gt_rect_ctr2edge + 1, 1, device=gt_ctr_rect.device)
        points_rect_y = torch.arange(
            -gt_rect_ctr2edge, gt_rect_ctr2edge + 1, 1, device=gt_ctr_rect.device)
        points_rect_xy = torch.stack(
            torch.meshgrid(points_rect_x, points_rect_y), -1).reshape(-1, 2)
        gt_ctr_rect_label = gt_ctr_rect_label.transpose(1, 2).contiguous().view(num_gts, -1)
        points_rect_xy_adapt = (
            points_rect_xy.unsqueeze(0).repeat(num_gts, 1, 1)
            * torch.sqrt(gt_ctr_rect_label).unsqueeze(-1))
        points_cov_matrix = (
            torch.matmul(points_rect_xy_adapt.transpose(1, 2), points_rect_xy_adapt)
            / (gt_ctr_rect_label.shape[-1] ** 2 - 1))
        
        # Add a small epsilon to the diagonal to prevent ill-conditioned matrix errors in PCA
        points_cov_matrix = points_cov_matrix + torch.eye(2, device=points_cov_matrix.device) * 1e-6
        eigvals, eigvecs = torch.symeig(points_cov_matrix, eigenvectors=True)

        larger_eigvals_index = (eigvals[:, 1] > eigvals[:, 0]).int()
        eigvec_first = (
            eigvecs[:, 0, :] * (1 - larger_eigvals_index).unsqueeze(1).repeat(1, 2)
            + eigvecs[:, 1, :] * larger_eigvals_index.unsqueeze(1).repeat(1, 2))
        mask_eigvec = (eigvec_first[:, 1] > 0).int()
        epsilon = mask_eigvec * 1e-6 + (1 - mask_eigvec) * -1e-6

        angle_targets = -torch.atan(
            eigvec_first[:, 0] / (eigvec_first[:, 1] + epsilon)).unsqueeze(-1)
        eigvec_second = torch.stack([-eigvec_first[:, 1], eigvec_first[:, 0]], -1)

        dist_gt_and_gt = (
            torch.cdist(center_point_gt, center_point_gt)
            + torch.eye(num_gts, device=center_point_gt.device) * INF)
        _, dist_min_gt_and_gt_index = dist_gt_and_gt.min(dim=1)
        is_nearest_same_class = (gt_labels[dist_min_gt_and_gt_index] == gt_labels)

        first_axis_range = self.get_closest_gt_first_axis(
            gt_labels, eigvec_first, center_point_gt, angle_threshold=0.866)

        top_simple, bottom_simple = self.get_edge_boundary_simple(
            gt_labels, eigvec_first, center_point_gt, prob_map_lvl0,
            is_secondary=False, is_nearest_same_class=is_nearest_same_class,
            nearest_gt_point=center_point_gt[dist_min_gt_and_gt_index],
            first_axis_range=first_axis_range)
        left_simple, right_simple = self.get_edge_boundary_simple(
            gt_labels, eigvec_second, center_point_gt, prob_map_lvl0,
            is_secondary=True, is_nearest_same_class=is_nearest_same_class,
            nearest_gt_point=center_point_gt[dist_min_gt_and_gt_index],
            first_axis_range=None)

        top_simple = top_simple * stride0 + 1
        bottom_simple = bottom_simple * stride0 + 1
        left_simple = left_simple * stride0 + 1
        right_simple = right_simple * stride0 + 1

        pseudo_gt_bboxes = torch.cat([
            center_point_gt,
            (left_simple + right_simple).unsqueeze(-1),
            (top_simple + bottom_simple).unsqueeze(-1),
            angle_targets
        ], -1).detach()

        H, W = prob_map_lvl0.shape[1], prob_map_lvl0.shape[2]
        center_feat = (center_point_gt / stride0).round().long()
        center_feat[:, 0].clamp_(0, W - 1)
        center_feat[:, 1].clamp_(0, H - 1)
        label_idx = gt_labels.long()
        scores = prob_map_lvl0[label_idx, center_feat[:, 1], center_feat[:, 0]]
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)

        return pseudo_gt_bboxes, scores

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_pseudo_bboxes(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          cfg=None,
                          rescale=None):
        cfg = self.test_cfg if cfg is None else cfg
        if cfg is None:
            raise ValueError('Test config is missing for pseudo generation.')

        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        num_imgs = cls_scores[0].shape[0]
        result_list = []

        for img_id in range(num_imgs):
            gt_boxes_this = gt_bboxes[img_id] if len(gt_bboxes[img_id]) > 0 else None
            labels_this = gt_labels[img_id] if len(gt_labels[img_id]) > 0 else None

            if gt_boxes_this is None or gt_boxes_this.numel() == 0:
                empty_bboxes = bbox_preds[0].new_zeros((0, 6))
                empty_labels = gt_labels[img_id].new_zeros((0,), dtype=torch.long)
                result_list.append((empty_bboxes, empty_labels))
                continue

            prob_map_lvl0 = self._build_probmap(
                cls_scores[0][img_id], bbox_preds[0][img_id], centernesses[0][img_id])
            pseudo_bboxes, pseudo_scores = self._pseudo_boxes_from_probmap(
                gt_boxes_this, labels_this, prob_map_lvl0)

            if pseudo_bboxes is None or pseudo_bboxes.numel() == 0:
                empty_bboxes = bbox_preds[0].new_zeros((0, 6))
                empty_labels = gt_labels[img_id].new_zeros((0,), dtype=torch.long)
                result_list.append((empty_bboxes, empty_labels))
                continue

            mlvl_bboxes = pseudo_bboxes
            mlvl_scores = mlvl_bboxes.new_zeros((mlvl_bboxes.shape[0], self.num_classes))
            mlvl_scores[
                torch.arange(mlvl_scores.shape[0], device=mlvl_scores.device),
                labels_this.long()] = pseudo_scores

            if rescale:
                scale_factor = mlvl_bboxes.new_tensor(
                    img_metas[img_id]['scale_factor'][:2]).repeat(2)
                mlvl_bboxes[:, :4] /= scale_factor

            if self.enable_final_nms:
                det_bboxes, det_labels = multiclass_nms_rotated(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.get('score_thr', 0.05),
                    cfg.get('nms', dict(type='nms_rotated', iou_thr=0.1)),
                    cfg.get('max_per_img', 2000))

                if self.class_agnostic_nms and det_bboxes.shape[0] > 0:
                    keep_iou_thr = cfg.get(
                        'class_agnostic_iou_thr', self.class_agnostic_iou_thr)
                    _, keep_inds = nms_rotated(
                        det_bboxes[:, :5], det_bboxes[:, 5], keep_iou_thr)
                    det_bboxes = det_bboxes[keep_inds]
                    det_labels = det_labels[keep_inds]
            else:
                score_thr = cfg.get('score_thr', 0.05)
                max_per_img = cfg.get('max_per_img', 2000)
                max_scores, labels = mlvl_scores.max(dim=1)
                valid = max_scores >= score_thr
                if valid.any():
                    det_labels = labels[valid]
                    det_scores = max_scores[valid]
                    det_boxes5 = mlvl_bboxes[valid]
                    if det_scores.shape[0] > max_per_img:
                        _, order = det_scores.sort(descending=True)
                        order = order[:max_per_img]
                        det_labels = det_labels[order]
                        det_scores = det_scores[order]
                        det_boxes5 = det_boxes5[order]
                    det_bboxes = torch.cat([det_boxes5, det_scores[:, None]], dim=1)
                else:
                    det_bboxes = mlvl_bboxes.new_zeros((0, 6))
                    det_labels = mlvl_bboxes.new_zeros((0,), dtype=torch.long)

            result_list.append((det_bboxes, det_labels))

        return result_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        losses = super().loss(
            cls_scores,
            bbox_preds,
            centernesses,
            gt_bboxes,
            gt_labels,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)

        if self.store_ann_dir is not None:
            with torch.no_grad():
                bbox_list = self.get_pseudo_bboxes(
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                    img_metas=img_metas,
                    cfg=self.test_cfg,
                    rescale=False)
                self._export_pseudo_txt(bbox_list, img_metas)

        return losses
