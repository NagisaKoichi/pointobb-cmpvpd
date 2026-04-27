# Copyright (c) OpenMMLab. All rights reserved.
"""CPMVPD pseudo-label head based on 4-channel regression outputs.

This head reuses CPMVPD inference outputs:
    [0:2] posterior mean  (dx, dy)
    [2:4] posterior lstd  (log_sx, log_sy)

Pseudo generation pipeline:
1) Build remap score map from cls-score map using mean offsets (dx, dy).
2) Use point-supervised targets (GT centers) as query points.
3) Search local peak around each query point on remap map.
4) Decode one pseudo box from the selected pixel's mean/lstd.
5) Apply multiclass rotated NMS to all pseudo boxes.
"""

import os
import torch
from mmcv.runner import force_fp32
from mmcv.ops import nms_rotated

from mmrotate.core import multiclass_nms_rotated
from ..builder import ROTATED_HEADS
from .cpm_vpd_head import CPMVPDHead


@ROTATED_HEADS.register_module()
class CPMVPDPseudoHead(CPMVPDHead):
    """Point-supervised pseudo-box generator using CPMVPD 4-channel output.

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
                 use_pca_decode=True,
                 pca_window_radius=6,
                 pca_use_adaptive_radius=True,
                 pca_radius_scale=0.30,
                 pca_radius_min=3,
                 pca_radius_max=14,
                 pca_thr_ratio=0.45,
                 pca_weight_gamma=2.0,
                 pca_min_pixels=6,
                 pca_size_factor=1.8,
                 pca_center_mix=0.0,
                 pca_size_mix=0.4,
                 pca_angle_mix=0.7,
                 pca_component_select='peak',
                 pca_connectivity=4,
                 pca_use_aniso_gate=True,
                 pca_aniso_thr=0.35,
                 pca_peak_dist_penalty=0.20,
                 pca_enable_instance_gate=True,
                 pca_max_center_offset=4.0,
                 pca_min_fill_ratio=0.20,
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
        self.use_pca_decode = bool(use_pca_decode)
        self.pca_window_radius = int(pca_window_radius)
        self.pca_use_adaptive_radius = bool(pca_use_adaptive_radius)
        self.pca_radius_scale = float(pca_radius_scale)
        self.pca_radius_min = int(pca_radius_min)
        self.pca_radius_max = int(pca_radius_max)
        self.pca_thr_ratio = float(pca_thr_ratio)
        self.pca_weight_gamma = float(pca_weight_gamma)
        self.pca_min_pixels = int(pca_min_pixels)
        self.pca_size_factor = float(pca_size_factor)
        self.pca_center_mix = float(pca_center_mix)
        self.pca_size_mix = float(pca_size_mix)
        self.pca_angle_mix = float(pca_angle_mix)
        self.pca_component_select = str(pca_component_select)
        self.pca_connectivity = int(pca_connectivity)
        self.pca_use_aniso_gate = bool(pca_use_aniso_gate)
        self.pca_aniso_thr = float(pca_aniso_thr)
        self.pca_peak_dist_penalty = float(pca_peak_dist_penalty)
        self.pca_enable_instance_gate = bool(pca_enable_instance_gate)
        self.pca_max_center_offset = float(pca_max_center_offset)
        self.pca_min_fill_ratio = float(pca_min_fill_ratio)
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
        dx, dy = mu

        if self.norm_on_bbox:
            cx = point[0] + dx * stride
            cy = point[1] + dy * stride
        else:
            cx = point[0] + dx
            cy = point[1] + dy

        # xy-only head has no direct size logits; start from stride-sized box.
        wh_scale = 1.0
        if self.use_lstd_for_size:
            wh_scale = torch.exp(self.lstd_size_factor * lstd.mean())
        w = point.new_tensor(float(stride)) * wh_scale
        h = point.new_tensor(float(stride)) * wh_scale

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

        yy, xx = torch.meshgrid(
            torch.arange(y0, y1 + 1, device=window.device, dtype=torch.float32),
            torch.arange(x0, x1 + 1, device=window.device, dtype=torch.float32),
            indexing='ij')
        dx2 = (xx - float(target_xy_feat[0].item()))**2
        dy2 = (yy - float(target_xy_feat[1].item()))**2
        dist2_norm = (dx2 + dy2) / max(float(radius * radius), 1.0)
        penalized = window - self.pca_peak_dist_penalty * dist2_norm

        flat_idx = torch.argmax(penalized).item()
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

    @staticmethod
    def _blend_angle(base_angle, pca_angle, mix):
        mix = float(min(max(mix, 0.0), 1.0))
        if mix <= 0.0:
            return base_angle
        if mix >= 1.0:
            return pca_angle
        sin_v = (1.0 - mix) * torch.sin(base_angle) + mix * torch.sin(pca_angle)
        cos_v = (1.0 - mix) * torch.cos(base_angle) + mix * torch.cos(pca_angle)
        return torch.atan2(sin_v, cos_v)

    def _resolve_pca_radius(self, stride, base_wh):
        if (not self.pca_use_adaptive_radius) or (base_wh is None):
            return max(int(self.pca_window_radius), 0)
        wf = float(base_wh[0].item()) / max(float(stride), 1e-6)
        hf = float(base_wh[1].item()) / max(float(stride), 1e-6)
        approx = self.pca_radius_scale * max((wf * hf)**0.5, 1.0)
        radius = int(round(approx))
        radius = max(self.pca_radius_min, radius)
        radius = min(self.pca_radius_max, radius)
        return max(radius, 0)

    def _component_offsets(self):
        if self.pca_connectivity == 8:
            return [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _flood_fill_component(self, mask_cpu, seed_y, seed_x):
        h, w = mask_cpu.shape
        if seed_y < 0 or seed_y >= h or seed_x < 0 or seed_x >= w:
            return None
        if not bool(mask_cpu[seed_y, seed_x].item()):
            return None
        visited = torch.zeros((h, w), dtype=torch.bool)
        queue = [(seed_y, seed_x)]
        visited[seed_y, seed_x] = True
        comp = []
        offsets = self._component_offsets()
        while queue:
            y, x = queue.pop()
            comp.append((y, x))
            for dy, dx in offsets:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or (not bool(mask_cpu[ny, nx].item())):
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))
        return comp

    def _all_components(self, mask_cpu):
        h, w = mask_cpu.shape
        visited = torch.zeros((h, w), dtype=torch.bool)
        offsets = self._component_offsets()
        components = []
        points = torch.nonzero(mask_cpu, as_tuple=False)
        for p in points:
            y0 = int(p[0].item())
            x0 = int(p[1].item())
            if visited[y0, x0]:
                continue
            queue = [(y0, x0)]
            visited[y0, x0] = True
            comp = []
            while queue:
                y, x = queue.pop()
                comp.append((y, x))
                for dy, dx in offsets:
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or (not bool(mask_cpu[ny, nx].item())):
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
            components.append(comp)
        return components

    def _select_component_mask(self, mask, seed_y, seed_x, target_xy_window):
        if int(mask.sum().item()) == 0:
            return None
        mask_cpu = mask.detach().to(device='cpu', dtype=torch.bool)
        mode = self.pca_component_select.lower()

        best_comp = None
        if mode == 'peak':
            best_comp = self._flood_fill_component(mask_cpu, seed_y, seed_x)
        else:
            comps = self._all_components(mask_cpu)
            if len(comps) == 0:
                return None
            tx = float(target_xy_window[0].item())
            ty = float(target_xy_window[1].item())
            best_d2 = None
            for comp in comps:
                xs = torch.tensor([p[1] for p in comp], dtype=torch.float32)
                ys = torch.tensor([p[0] for p in comp], dtype=torch.float32)
                cx = float(xs.mean().item())
                cy = float(ys.mean().item())
                d2 = (cx - tx)**2 + (cy - ty)**2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_comp = comp

        if best_comp is None:
            return None

        out_cpu = torch.zeros_like(mask_cpu)
        ys = [p[0] for p in best_comp]
        xs = [p[1] for p in best_comp]
        out_cpu[ys, xs] = True
        return out_cpu.to(device=mask.device)

    def _decode_from_local_weighted_pca(self,
                                        remap_map,
                                        py,
                                        px,
                                        stride,
                                        target_xy_feat,
                                        base_wh=None):
        h, w = remap_map.shape
        radius = self._resolve_pca_radius(stride=stride, base_wh=base_wh)
        x0 = max(0, int(px) - radius)
        x1 = min(w - 1, int(px) + radius)
        y0 = max(0, int(py) - radius)
        y1 = min(h - 1, int(py) + radius)
        if x1 < x0 or y1 < y0:
            return None

        window = remap_map[y0:y1 + 1, x0:x1 + 1]
        if window.numel() == 0:
            return None

        peak = float(remap_map[py, px].item())
        thr = max(peak * self.pca_thr_ratio, 1e-6)
        raw_mask = window >= thr

        seed_y = int(py - y0)
        seed_x = int(px - x0)
        target_xy_window = target_xy_feat.new_tensor([
            float(target_xy_feat[0].item()) - float(x0),
            float(target_xy_feat[1].item()) - float(y0)
        ])
        mask = self._select_component_mask(raw_mask, seed_y, seed_x, target_xy_window)
        if mask is None:
            return None
        if int(mask.sum().item()) < int(self.pca_min_pixels):
            return None

        ys, xs = torch.nonzero(mask, as_tuple=True)
        vals = window[ys, xs]
        weights = torch.pow(torch.clamp(vals, min=1e-6), self.pca_weight_gamma)
        weights = torch.clamp(weights, min=1e-6)

        xs_abs = xs.float() + float(x0)
        ys_abs = ys.float() + float(y0)
        pts = torch.stack([xs_abs, ys_abs], dim=1)

        wsum = torch.clamp(weights.sum(), min=1e-6)
        mean = (pts * weights[:, None]).sum(dim=0) / wsum
        centered = pts - mean

        cov = centered.t().matmul(centered * weights[:, None]) / wsum
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

        width_feat = torch.clamp((umax - umin + 1.0) * self.pca_size_factor, min=1.0)
        height_feat = torch.clamp((vmax - vmin + 1.0) * self.pca_size_factor, min=1.0)

        cx = mean[0] * float(stride)
        cy = mean[1] * float(stride)
        w_box = width_feat * float(stride)
        h_box = height_feat * float(stride)
        angle = torch.atan2(major[1], major[0])

        l1 = torch.clamp(eigvals[1], min=0.0)
        l2 = torch.clamp(eigvals[0], min=0.0)
        aniso = (l1 - l2) / torch.clamp(l1 + l2, min=1e-6)

        xmin = float(xs.min().item())
        xmax = float(xs.max().item())
        ymin = float(ys.min().item())
        ymax = float(ys.max().item())
        box_area = max((xmax - xmin + 1.0) * (ymax - ymin + 1.0), 1.0)
        fill_ratio = float(mask.sum().item()) / box_area
        center_offset = torch.sqrt(
            (mean[0] - float(target_xy_feat[0].item()))**2 +
            (mean[1] - float(target_xy_feat[1].item()))**2)
        instance_ok = True
        if self.pca_enable_instance_gate:
            instance_ok = bool(
                (float(center_offset.item()) <= self.pca_max_center_offset) and
                (fill_ratio >= self.pca_min_fill_ratio))

        return remap_map.new_tensor([cx, cy, w_box, h_box, angle]), aniso, instance_ok

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
            bbox_preds (list[Tensor]): Per-level 4-ch regression, shape [B, 4, H, W].
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
                    mu = torch.nan_to_num(stats[:2], nan=0.0, posinf=1e4, neginf=-1e4)
                    lstd = torch.nan_to_num(stats[2:4], nan=0.0, posinf=1e4, neginf=-1e4)

                    box = self._decode_from_stats(anchor_point, stride, mu, lstd)
                    wh_remap = None
                    if self.use_remap_size or self.use_pca_decode:
                        wh_remap = self._estimate_wh_from_remap_peak(
                            remap_map, py, px, stride)

                    if self.use_remap_size:
                        mix = min(max(self.remap_size_mix, 0.0), 1.0)
                        box[2] = box[2] * (1.0 - mix) + wh_remap[0] * mix
                        box[3] = box[3] * (1.0 - mix) + wh_remap[1] * mix

                    if self.use_pca_decode:
                        pca_out = self._decode_from_local_weighted_pca(
                            remap_map=remap_map,
                            py=py,
                            px=px,
                            stride=stride,
                            target_xy_feat=target_xy_feat,
                            base_wh=wh_remap)
                        if pca_out is not None:
                            pca_box, pca_aniso, instance_ok = pca_out
                            if instance_ok:
                                center_mix = min(max(self.pca_center_mix, 0.0), 1.0)
                                size_mix = min(max(self.pca_size_mix, 0.0), 1.0)
                                angle_mix = min(max(self.pca_angle_mix, 0.0), 1.0)

                                box[0] = box[0] * (1.0 - center_mix) + pca_box[0] * center_mix
                                box[1] = box[1] * (1.0 - center_mix) + pca_box[1] * center_mix
                                box[2] = box[2] * (1.0 - size_mix) + pca_box[2] * size_mix
                                box[3] = box[3] * (1.0 - size_mix) + pca_box[3] * size_mix

                                if (not self.pca_use_aniso_gate) or (float(pca_aniso.item()) >= self.pca_aniso_thr):
                                    box[4] = self._blend_angle(box[4], pca_box[4], angle_mix)
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
