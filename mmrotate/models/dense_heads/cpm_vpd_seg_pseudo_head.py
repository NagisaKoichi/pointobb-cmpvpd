# Copyright (c) OpenMMLab. All rights reserved.
"""CPMVPD pseudo head that decodes rotated boxes from GT-guided masks.

Design follows `pseudo_label_head.py`/`cpm_vpd_pseudo_head.py` style, but the
box is decoded from segmentation masks built with the same rule as
`generate_vpd_variance_map.py`:

1) Build prob map: sigmoid(exp(log_sigma)).mean * (max_class_prob * centerness)
2) Build GT-guided per-GT component maps P_i(x) = p_model(x) * w_i(x)
3) For each GT i, take mask directly from P_i (NOT by point->mask-id lookup)
4) Estimate dominant direction from mask pixels (PCA)
5) Fit minimum rotated rectangle in dominant/orthogonal coordinates
"""

import torch
import os
from mmcv.ops import nms_rotated
from mmcv.runner import force_fp32

from mmrotate.core import multiclass_nms_rotated
from ..builder import ROTATED_HEADS
from .cpm_vpd_head import CPMVPDHead


@ROTATED_HEADS.register_module()
class CPMVPDSegPseudoHead(CPMVPDHead):

    def __init__(self,
                 *args,
                 sigma_scale=0.5,
                 min_sigma=1.0,
                 max_sigma=20.0,
                 seg_score_thr=0.7,
                 seg_topk=0,
                 bg_std_scale=1.5,
                 per_gt_thr_ratio=0.5,
                 mask_min_pixels=6,
                 enable_final_nms=False,
                 class_agnostic_nms=True,
                 class_agnostic_iou_thr=0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_scale = float(sigma_scale)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.seg_score_thr = float(seg_score_thr)
        self.seg_topk = int(seg_topk)
        self.bg_std_scale = None if bg_std_scale is None else float(bg_std_scale)
        self.per_gt_thr_ratio = float(per_gt_thr_ratio)
        self.mask_min_pixels = int(mask_min_pixels)

        self.enable_final_nms = bool(enable_final_nms)
        self.class_agnostic_nms = bool(class_agnostic_nms)
        self.class_agnostic_iou_thr = float(class_agnostic_iou_thr)

        train_cfg = kwargs.get('train_cfg', {}) or {}
        self.store_ann_dir = train_cfg.get('store_ann_dir', None)
        if self.store_ann_dir is not None:
            os.makedirs(self.store_ann_dir, exist_ok=True)

        self.default_classes = (
            'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field',
            'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    @staticmethod
    def _infer_image_shape(img_meta):
        for key in ('img_shape', 'pad_shape', 'ori_shape'):
            shape = img_meta.get(key, None)
            if shape is not None and len(shape) >= 2:
                return float(shape[0]), float(shape[1])
        raise KeyError('img_meta must contain one of img_shape/pad_shape/ori_shape.')

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
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
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
        dx2 = (xx.unsqueeze(0) - cx.view(-1, 1, 1))**2
        dy2 = (yy.unsqueeze(0) - cy.view(-1, 1, 1))**2
        denom = 2.0 * sigma.view(-1, 1, 1)**2 + 1e-6
        return torch.exp(-(dx2 + dy2) / denom)

    @staticmethod
    def _build_prob_map(cls_score_lvl, centerness_lvl, bbox_pred_lvl):
        cls_score_lvl = torch.nan_to_num(
            cls_score_lvl, nan=0.0, posinf=50.0, neginf=-50.0)
        centerness_lvl = torch.nan_to_num(
            centerness_lvl, nan=0.0, posinf=50.0, neginf=-50.0)
        log_sigma = torch.nan_to_num(
            bbox_pred_lvl[2:4], nan=0.0, posinf=1e4, neginf=-1e4)
        sigma = log_sigma.exp().sum(dim=0)

        max_class_prob = cls_score_lvl.sigmoid().max(dim=0)[0]
        
        fused_map = (1 - sigma.sigmoid() * (1 - max_class_prob)).sqrt()
        from scipy.ndimage import minimum_filter
        fused_np = fused_map.cpu().numpy()
        fused_np = minimum_filter(fused_np, size=3)
        fused_map = torch.from_numpy(fused_np).to(fused_map.device).to(fused_map.dtype)
        
        return fused_map.clamp(min=1e-6, max=1.0)
        

    def _build_per_gt_maps(self, p_model, gt_bboxes, img_meta):
        p_model = torch.nan_to_num(p_model.float(), nan=0.0, posinf=1.0, neginf=0.0)
        h, w = p_model.shape

        if gt_bboxes is None or gt_bboxes.numel() == 0:
            return None, None

        cx, cy, bw, bh = self._extract_center_and_size(gt_bboxes)
        if cx is None:
            return None, None

        img_h, img_w = self._infer_image_shape(img_meta)
        sx = float(w) / max(float(img_w), 1.0)
        sy = float(h) / max(float(img_h), 1.0)

        cx_f = cx * sx
        cy_f = cy * sy
        bw_f = bw * sx
        bh_f = bh * sy

        sigma = self.sigma_scale * torch.sqrt((bw_f * bh_f).clamp(min=1e-6))
        sigma = torch.clamp(sigma, min=self.min_sigma, max=self.max_sigma)

        priors = self._gaussian_priors(h=h, w=w, cx=cx_f, cy=cy_f, sigma=sigma)
        ownership = priors / torch.clamp(priors.sum(dim=0, keepdim=True), min=1e-8)

        per_gt = p_model.unsqueeze(0) * ownership

        if self.seg_score_thr > 0.0:
            per_gt = per_gt * (per_gt >= self.seg_score_thr).to(per_gt.dtype)

        if self.seg_topk > 0:
            n_obj = per_gt.shape[0]
            flat = per_gt.view(n_obj, -1)
            k = min(self.seg_topk, flat.shape[1])
            if k > 0:
                kth = torch.topk(flat, k, dim=1)[0][:, -1].view(n_obj, 1, 1)
                per_gt = per_gt * (per_gt >= kth).to(per_gt.dtype)

        fused = per_gt.sum(dim=0)
        if self.bg_std_scale is not None:
            mu = fused.mean()
            std = fused.std(unbiased=False)
            fg_mask = fused >= (mu + self.bg_std_scale * std)
        else:
            fg_mask = fused > 0

        return per_gt, fg_mask

    def _mask_from_per_gt(self, per_gt_map, fg_mask):
        peak = float(per_gt_map.max().item())
        if peak <= 0:
            return None
        thr = max(self.seg_score_thr, self.per_gt_thr_ratio * peak)
        mask = (per_gt_map >= thr) & fg_mask
        if int(mask.sum().item()) < self.mask_min_pixels:
            # fallback: keep strongest area under fg mask
            mask = (per_gt_map > 0) & fg_mask
        if int(mask.sum().item()) < self.mask_min_pixels:
            return None
        return mask

    def _decode_min_rect_from_mask(self, mask, stride):
        ys, xs = torch.nonzero(mask, as_tuple=True)
        n = ys.numel()
        if n < self.mask_min_pixels:
            return None

        pts = torch.stack([xs.float(), ys.float()], dim=1)
        mean = pts.mean(dim=0)
        centered = pts - mean

        cov = centered.t().matmul(centered) / max(float(n - 1), 1.0)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        major = eigvecs[:, 1]
        if float(eigvals[1].item()) < 1e-8:
            major = pts.new_tensor([1.0, 0.0])
        major = major / torch.clamp(torch.norm(major), min=1e-6)
        minor = torch.stack([-major[1], major[0]])

        proj_u = centered.matmul(major)
        proj_v = centered.matmul(minor)
        umin, umax = proj_u.min(), proj_u.max()
        vmin, vmax = proj_v.min(), proj_v.max()

        width_feat = torch.clamp(umax - umin + 1.0, min=1.0)
        height_feat = torch.clamp(vmax - vmin + 1.0, min=1.0)
        center_feat = mean + 0.5 * ((umin + umax) * major + (vmin + vmax) * minor)

        cx = center_feat[0] * float(stride)
        cy = center_feat[1] * float(stride)
        w = width_feat * float(stride)
        h = height_feat * float(stride)
        angle = torch.atan2(major[1], major[0])
        return torch.stack([cx, cy, w, h, angle], dim=0)

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

        for det_bboxes, det_labels in bbox_list:
            if det_bboxes is None:
                continue

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
        num_levels = len(cls_scores)
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

            per_gt_bboxes = []
            per_gt_scores = []
            per_gt_labels = []

            for gt_idx in range(gt_boxes_this.shape[0]):
                cls_id = labels_this[gt_idx].long()
                best_box = None
                best_score = None

                for lvl_idx in range(num_levels):
                    stride = self.strides[lvl_idx]
                    bbox_pred_lvl = bbox_preds[lvl_idx][img_id]
                    cls_score_lvl = cls_scores[lvl_idx][img_id]
                    centerness_lvl = centernesses[lvl_idx][img_id]

                    probmap = self._build_prob_map(
                        cls_score_lvl=cls_score_lvl,
                        centerness_lvl=centerness_lvl,
                        bbox_pred_lvl=bbox_pred_lvl)
                    per_gt_maps, fg_mask = self._build_per_gt_maps(
                        p_model=probmap,
                        gt_bboxes=gt_boxes_this,
                        img_meta=img_metas[img_id])
                    if per_gt_maps is None:
                        continue

                    # Directly use GT's original component map (index == gt_idx).
                    mask = self._mask_from_per_gt(per_gt_maps[gt_idx], fg_mask)
                    if mask is None:
                        continue

                    box = self._decode_min_rect_from_mask(mask=mask, stride=stride)
                    if box is None:
                        continue

                    score = probmap[mask].mean()
                    score = torch.nan_to_num(
                        score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_box = box

                if best_box is not None:
                    per_gt_bboxes.append(best_box)
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
            mlvl_scores[
                torch.arange(mlvl_scores.shape[0], device=mlvl_scores.device),
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
