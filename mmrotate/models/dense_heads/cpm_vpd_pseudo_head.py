# Copyright (c) OpenMMLab. All rights reserved.
"""CPMVPD pseudo-label head based on 8-channel regression outputs.

This head reuses CPMVPD inference outputs:
    [0:4] posterior mean  (dx, dy, log_w, log_h)
    [4:8] posterior lstd  (log_sx, log_sy, log_sw, log_sh)

Pseudo generation pipeline:
1) Build remap score map from cls-score map using mean offsets (dx, dy).
2) Use point-supervised targets (GT centers) as query points.
3) Search local peak around each query point on remap map.
4) Decode one pseudo box from the selected pixel's mean/lstd.
5) Apply multiclass rotated NMS to all pseudo boxes.
"""

import torch
from mmcv.runner import force_fp32
from mmcv.ops import nms_rotated

from mmrotate.core import multiclass_nms_rotated
from ..builder import ROTATED_HEADS
from .cpm_vpd_head import CPMVPDHead


@ROTATED_HEADS.register_module()
class CPMVPDPseudoHead(CPMVPDHead):
    """Point-supervised pseudo-box generator using CPMVPD 8-channel output.

    Args:
        point_search_radius (int): Local search radius (feature pixels)
            around each target point on remap map.
        use_lstd_for_size (bool): Whether to inject lstd into size decoding.
        lstd_size_factor (float): Size correction factor for lstd.
    """

    def __init__(self,
                 *args,
                 point_search_radius=3,
                 use_lstd_for_size=True,
                 lstd_size_factor=0.5,
                 use_remap_size=True,
                 remap_edge_thr_ratio=0.35,
                 remap_edge_max_len=64,
                 remap_size_mix=1.0,
                 enable_final_nms=False,
                 class_agnostic_nms=True,
                 class_agnostic_iou_thr=0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.point_search_radius = int(point_search_radius)
        self.use_lstd_for_size = bool(use_lstd_for_size)
        self.lstd_size_factor = float(lstd_size_factor)
        self.use_remap_size = bool(use_remap_size)
        self.remap_edge_thr_ratio = float(remap_edge_thr_ratio)
        self.remap_edge_max_len = int(remap_edge_max_len)
        self.remap_size_mix = float(remap_size_mix)
        self.enable_final_nms = bool(enable_final_nms)
        self.class_agnostic_nms = bool(class_agnostic_nms)
        self.class_agnostic_iou_thr = float(class_agnostic_iou_thr)

    @staticmethod
    def _build_remap_map(cls_score_lvl, centerness_lvl, bbox_pred_lvl):
        """Build remap map following visualize script logic.

        Remap formula:
            p'(y, x) = log(1 + p(y + mu_y(y, x), x + mu_x(y, x)))
        where p is max-class probability.
        """
        del centerness_lvl  # kept for signature consistency

        mu = torch.nan_to_num(bbox_pred_lvl[0:4], nan=0.0, posinf=1e4, neginf=-1e4)
        cls_score_lvl = torch.nan_to_num(cls_score_lvl, nan=0.0, posinf=50.0, neginf=-50.0)

        max_class_prob = cls_score_lvl.sigmoid().max(dim=0)[0]
        cx_mu = mu[0]
        cy_mu = mu[1]

        h, w = max_class_prob.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, device=max_class_prob.device, dtype=torch.float32),
            torch.arange(w, device=max_class_prob.device, dtype=torch.float32),
            indexing='ij')
        new_x = torch.round(xx + cx_mu).long().clamp(0, w - 1)
        new_y = torch.round(yy + cy_mu).long().clamp(0, h - 1)
        remapped_max_prob = max_class_prob[new_y, new_x]
        remapped_max_prob = torch.log1p(remapped_max_prob)
        return remapped_max_prob

    def _decode_from_stats(self, point, stride, mu, lstd):
        """Decode one rotated box from a selected pixel stats."""
        dx, dy, log_w, log_h = mu
        if self.use_lstd_for_size:
            log_w = log_w + self.lstd_size_factor * lstd[2]
            log_h = log_h + self.lstd_size_factor * lstd[3]

        if self.norm_on_bbox:
            cx = point[0] + dx * stride
            cy = point[1] + dy * stride
            w = torch.exp(log_w) * stride
            h = torch.exp(log_h) * stride
        else:
            cx = point[0] + dx
            cy = point[1] + dy
            w = torch.exp(log_w)
            h = torch.exp(log_h)

        angle = cx.new_zeros(())
        return torch.stack([cx, cy, w, h, angle], dim=0)

    def _pick_peak_from_remap(self, remap_map, target_xy_feat, radius):
        """Pick local peak around a target point on remap map."""
        h, w = remap_map.shape
        tx = int(torch.round(target_xy_feat[0]).item())
        ty = int(torch.round(target_xy_feat[1]).item())

        x0 = max(0, tx - radius)
        x1 = min(w - 1, tx + radius)
        y0 = max(0, ty - radius)
        y1 = min(h - 1, ty + radius)
        if x1 < x0 or y1 < y0:
            return None

        window = remap_map[y0:y1 + 1, x0:x1 + 1]
        if window.numel() == 0:
            return None

        flat_idx = torch.argmax(window).item()
        win_w = window.shape[1]
        dy = flat_idx // win_w
        dx = flat_idx % win_w
        py = y0 + dy
        px = x0 + dx
        score = window[dy, dx]
        return py, px, score

    def _estimate_wh_from_remap_peak(self, remap_map, py, px, stride):
        """Estimate w/h from remap responses around local peak."""
        h, w = remap_map.shape
        peak = float(remap_map[py, px].item())
        thr = max(peak * self.remap_edge_thr_ratio, 1e-6)

        def walk(dx, dy):
            dist = 0
            for step in range(1, self.remap_edge_max_len + 1):
                x = px + dx * step
                y = py + dy * step
                if x < 0 or x >= w or y < 0 or y >= h:
                    break
                if float(remap_map[y, x].item()) < thr:
                    break
                dist = step
            return dist

        left = walk(-1, 0)
        right = walk(1, 0)
        up = walk(0, -1)
        down = walk(0, 1)
        est_w = max(float((left + right + 1) * stride), float(stride))
        est_h = max(float((up + down + 1) * stride), float(stride))
        return remap_map.new_tensor([est_w, est_h])

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
        """Generate pseudo boxes from point-supervised targets and remap peaks.

        Args:
            cls_scores (list[Tensor]): Per-level cls logits, shape [B, C, H, W].
            bbox_preds (list[Tensor]): Per-level 8-ch regression, shape [B, 8, H, W].
            centernesses (list[Tensor]): Per-level centerness logits, shape [B, 1, H, W].
            gt_bboxes (list[Tensor]): Point-supervised GT boxes (center in first 2 dims).
            gt_labels (list[Tensor]): GT labels aligned with gt_bboxes.
            img_metas (list[dict]): Image meta info.
            cfg (ConfigDict | None): Test config for NMS.
            rescale (bool | None): Whether to rescale to original image.

        Returns:
            list[tuple[Tensor, Tensor]]: per-image (det_bboxes, det_labels).
        """
        cfg = self.test_cfg if cfg is None else cfg
        if cfg is None:
            raise ValueError('Test config is missing for pseudo generation.')

        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        num_levels = len(cls_scores)
        num_imgs = cls_scores[0].shape[0]
        result_list = []

        for img_id in range(num_imgs):
            points_this = gt_bboxes[img_id][:, :2] if len(gt_bboxes[img_id]) > 0 else None
            labels_this = gt_labels[img_id] if len(gt_labels[img_id]) > 0 else None

            if points_this is None or points_this.numel() == 0:
                empty_bboxes = bbox_preds[0].new_zeros((0, 6))
                empty_labels = gt_labels[img_id].new_zeros((0,), dtype=torch.long)
                result_list.append((empty_bboxes, empty_labels))
                continue

            per_gt_bboxes = []
            per_gt_scores = []
            per_gt_labels = []

            for gt_idx in range(points_this.shape[0]):
                center_xy = points_this[gt_idx]
                cls_id = labels_this[gt_idx].long()

                best = None
                best_score = None

                for lvl_idx in range(num_levels):
                    stride = self.strides[lvl_idx]
                    bbox_pred_lvl = bbox_preds[lvl_idx][img_id]
                    cls_score_lvl = cls_scores[lvl_idx][img_id]
                    centerness_lvl = centernesses[lvl_idx][img_id]
                    remap_map = self._build_remap_map(
                        cls_score_lvl, centerness_lvl, bbox_pred_lvl)

                    target_xy_feat = center_xy / float(stride)
                    peak = self._pick_peak_from_remap(
                        remap_map, target_xy_feat, self.point_search_radius)
                    if peak is None:
                        continue

                    py, px, score = peak
                    h, w = remap_map.shape
                    pt_idx = py * w + px
                    anchor_point = mlvl_points[lvl_idx][pt_idx]

                    stats = bbox_pred_lvl[:, py, px]
                    mu = torch.nan_to_num(stats[:4], nan=0.0, posinf=1e4, neginf=-1e4)
                    lstd = torch.nan_to_num(stats[4:8], nan=0.0, posinf=1e4, neginf=-1e4)

                    box = self._decode_from_stats(anchor_point, stride, mu, lstd)
                    if self.use_remap_size:
                        wh_remap = self._estimate_wh_from_remap_peak(
                            remap_map, py, px, stride)
                        mix = min(max(self.remap_size_mix, 0.0), 1.0)
                        box[2] = box[2] * (1.0 - mix) + wh_remap[0] * mix
                        box[3] = box[3] * (1.0 - mix) + wh_remap[1] * mix
                    score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)

                    if best_score is None or score > best_score:
                        best_score = score
                        best = (box, cls_id)

                if best is not None:
                    box, cls_id = best
                    per_gt_bboxes.append(box)
                    per_gt_scores.append(best_score)
                    per_gt_labels.append(cls_id)

            if len(per_gt_bboxes) == 0:
                empty_bboxes = bbox_preds[0].new_zeros((0, 6))
                empty_labels = gt_labels[img_id].new_zeros((0,), dtype=torch.long)
                result_list.append((empty_bboxes, empty_labels))
                continue

            mlvl_bboxes = torch.stack(per_gt_bboxes, dim=0)
            mlvl_scores = mlvl_bboxes.new_zeros((mlvl_bboxes.shape[0], self.num_classes))
            label_tensor = torch.stack(per_gt_labels, dim=0)
            score_tensor = torch.stack(per_gt_scores, dim=0)
            mlvl_scores[torch.arange(mlvl_scores.shape[0], device=mlvl_scores.device),
                        label_tensor] = score_tensor

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

                # Optional second-stage class-agnostic suppression to reduce
                # heavy overlaps across different categories.
                if self.class_agnostic_nms and det_bboxes.shape[0] > 0:
                    keep_iou_thr = cfg.get(
                        'class_agnostic_iou_thr', self.class_agnostic_iou_thr)
                    _, keep_inds = nms_rotated(
                        det_bboxes[:, :5],
                        det_bboxes[:, 5],
                        keep_iou_thr)
                    det_bboxes = det_bboxes[keep_inds]
                    det_labels = det_labels[keep_inds]

                    max_per_img = cfg.get('max_per_img', 2000)
                    if det_bboxes.shape[0] > max_per_img:
                        _, order = det_bboxes[:, 5].sort(descending=True)
                        order = order[:max_per_img]
                        det_bboxes = det_bboxes[order]
                        det_labels = det_labels[order]
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
