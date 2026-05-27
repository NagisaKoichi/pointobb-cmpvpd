# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from mmcv import Config, DictAction
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import build_dataloader, replace_ImageToTensor

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, get_device, setup_multi_processes
from segment_variance_map import (build_gt_guided_remap_map,
                                  build_gt_guided_segmentation_mask,
                                  _extract_center_and_size,
                                  _infer_image_shape)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference VPD checkpoint and dump variance maps')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='directory to save generated variance maps')
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val', 'test'],
        help='which data split in config to run inference on')
    parser.add_argument('--ann-file', help='optional annotation path override')
    parser.add_argument('--img-prefix', help='optional image prefix override')
    parser.add_argument(
        '--feat-level',
        type=int,
        default=0,
        help='FPN level index used for variance map visualization')
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='max number of images to process, 0 means all')
    parser.add_argument(
        '--seg-score-thr',
        type=float,
        default=0.04,
        help='optional per-object score threshold for GT-guided segmentation')
    parser.add_argument(
        '--seg-topk',
        type=int,
        default=0,
        help='optional top-k pixel keep for each object map, 0 disables')
    parser.add_argument(
        '--sigma-scale',
        type=float,
        default=0.5,
        help='adaptive sigma scale: sigma_i = sigma_scale * sqrt(w_i * h_i)')
    parser.add_argument(
        '--min-sigma',
        type=float,
        default=1.0,
        help='lower bound of adaptive sigma in feature-map coordinates')
    parser.add_argument(
        '--max-sigma',
        type=float,
        default=20.0,
        help='upper bound of adaptive sigma in feature-map coordinates')
    parser.add_argument(
        '--bg-std-scale',
        type=float,
        default=1.5,
        help='background suppression threshold: mean + bg_std_scale * std')
    parser.add_argument(
        '--cls-floor',
        type=float,
        default=0.05,
        help='classification floor before contrast stretching')
    parser.add_argument(
        '--cls-gamma',
        type=float,
        default=0.5,
        help='power applied to stretched classification score')
    parser.add_argument(
        '--uncert-q-lo',
        type=float,
        default=0.01,
        help='lower quantile for uncertainty normalization')
    parser.add_argument(
        '--uncert-q-hi',
        type=float,
        default=0.20,
        help='upper quantile for uncertainty normalization')
    parser.add_argument(
        '--uncert-gamma',
        type=float,
        default=0.7,
        help='power applied to uncertainty confidence weight')
    parser.add_argument(
        '--alpha-cls',
        type=float,
        default=0.5,
        help='fusion exponent for classification branch')
    parser.add_argument(
        '--alpha-uncert',
        type=float,
        default=0.5,
        help='fusion exponent for uncertainty branch')
    parser.add_argument(
        '--prob-smooth-ksize',
        type=int,
        default=3,
        help='avg-pool kernel size for optional probmap smoothing (odd, <=1 disables)')
    parser.add_argument(
        '--prob-local-contrast',
        type=float,
        default=0.30,
        help='local contrast factor beta in p <- max(p - beta*avgpool(p), 0)')
    parser.add_argument(
        '--scatter-class',
        type=int,
        default=-1,
        help='only include this class index in scatter plot, -1 means all')
    parser.add_argument(
        '--scatter-class-name',
        default=None,
        help='only include this class name in scatter plot (overrides --scatter-class)')
    parser.add_argument(
        '--remap-output-mode',
        default='seg',
        help='output processed variance map or GT-instance segmentation mask')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='single gpu id for inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in config, key=value format')
    return parser.parse_args()


def _to_heatmap(var_tensor, flip_direction):
    var_np = var_tensor.detach().cpu().numpy().astype(np.float32)
    var_min = float(var_np.min())
    var_max = float(var_np.max())
    if var_max - var_min < 1e-8:
        norm = np.zeros_like(var_np, dtype=np.float32)
    else:
        norm = (var_np - var_min) / (var_max - var_min)

    heatmap = np.zeros((norm.shape[0], norm.shape[1], 3), dtype=np.uint8)
    heatmap[..., 0] = np.clip(255.0 * norm, 0, 255).astype(np.uint8)
    heatmap[..., 1] = np.clip(
        255.0 * (1.0 - np.abs(2.0 * norm - 1.0)), 0, 255).astype(np.uint8)
    heatmap[..., 2] = np.clip(255.0 * (1.0 - norm), 0, 255).astype(np.uint8)

    img = Image.fromarray(heatmap, mode='RGB')
    if flip_direction == 'horizontal':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_direction == 'vertical':
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif flip_direction == 'diagonal':
        img = img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
    return img


def _label_to_color_mask(label_map, flip_direction):
    labels = label_map.detach().cpu().numpy().astype(np.int64)
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # Background keeps black; each GT instance gets deterministic color.
    max_label = int(labels.max())
    for idx in range(max_label + 1):
        m = labels == idx
        if not np.any(m):
            continue
        r = (37 * (idx + 1) + 17) % 256
        g = (91 * (idx + 1) + 73) % 256
        b = (53 * (idx + 1) + 149) % 256
        out[m, 0] = r
        out[m, 1] = g
        out[m, 2] = b

    img = Image.fromarray(out, mode='RGB')
    if flip_direction == 'horizontal':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_direction == 'vertical':
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif flip_direction == 'diagonal':
        img = img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
    return img

def _grayscale_erode(mask, kernel_size):
    # choose the minimum value in the neighborhood defined by the kernel size
    from scipy.ndimage import minimum_filter
    return minimum_filter(mask, size=kernel_size)


def _save_maps_for_image(img_path,
                         img_meta,
                         flip_direction,
                         bbox_pred_lvl,
                         cls_score_lvl,
                         centerness_lvl,
                         gt_bboxes,
                         out_path,
                         out_mean_path,
                         out_remap_path,
                         seg_score_thr,
                         seg_topk,
                         sigma_scale,
                         min_sigma,
                         max_sigma,
                         bg_std_scale,
                         remap_output_mode,
                         cls_floor,
                         cls_gamma,
                         uncert_q_lo,
                         uncert_q_hi,
                         uncert_gamma,
                         alpha_cls,
                         alpha_uncert,
                         prob_smooth_ksize,
                         prob_local_contrast):
    # Keep compatibility with old 4-channel checkpoints [mu_x, mu_y, log_sx, log_sy].
    num_bbox_channels = int(bbox_pred_lvl.shape[0])
    if num_bbox_channels < 4:
        raise ValueError(
            f'Unsupported bbox_pred channels: {num_bbox_channels}. '
            'Mu visualization requires the 4-channel VPD output '
            '[mu_x, mu_y, log_sx, log_sy].')

    bbox_mu = torch.nan_to_num(
        bbox_pred_lvl[0:2], nan=0.0, posinf=1e4, neginf=-1e4)
    log_sigma = bbox_pred_lvl[2:4]

    lstd = torch.nan_to_num(log_sigma, nan=0.0, posinf=1e4, neginf=-1e4)
    center_lstd = lstd.mean(dim=0)

    center_mu = bbox_mu.mean(dim=0)

    std = lstd.exp()

    cls_score_lvl = torch.nan_to_num(cls_score_lvl, nan=0.0, posinf=50.0, neginf=-50.0)
    centerness_lvl = torch.nan_to_num(centerness_lvl, nan=0.0, posinf=50.0, neginf=-50.0)

    max_class_prob = cls_score_lvl.sigmoid().max(dim=0)[0]
    centerness_prob = centerness_lvl.sigmoid().squeeze(0)
    combined_score = max_class_prob

    # Use geometric uncertainty from two sigma channels, avoiding channel mean.
    std_geo = torch.sqrt(torch.clamp(std[0] * std[1], min=1e-12))
    std_geo = torch.nan_to_num(std_geo, nan=0.0, posinf=1e6, neginf=0.0)
    q_lo = torch.quantile(std_geo, float(uncert_q_lo))
    q_hi = torch.quantile(std_geo, float(uncert_q_hi))
    denom = torch.clamp(q_hi - q_lo, min=1e-6)
    std_geo_norm = ((std_geo - q_lo) / denom).clamp(0.0, 1.0)
    uncert_weight = torch.pow((1.0 - std_geo_norm).clamp(0.0, 1.0), float(uncert_gamma))

    cls_stretch = ((max_class_prob - float(cls_floor)) /
                   max(1.0 - float(cls_floor), 1e-6)).clamp(0.0, 1.0)
    cls_weight = torch.pow(torch.clamp(cls_stretch, min=1e-6), float(cls_gamma))

    # probmap = (
    #     torch.pow(torch.clamp(cls_weight, min=1e-6), float(alpha_cls)) *
    #     torch.pow(torch.clamp(uncert_weight, min=1e-6), float(alpha_uncert)))
    
    # or mean
    probmap = float(alpha_cls) * cls_weight + float(alpha_uncert) * uncert_weight

    if int(prob_smooth_ksize) > 1:
        k = int(prob_smooth_ksize)
        if k % 2 == 0:
            k += 1
        pad = k // 2
        probmap = F.avg_pool2d(
            probmap[None, None], kernel_size=k, stride=1, padding=pad)[0, 0]

    beta = float(prob_local_contrast)
    if beta > 0.0:
        local_avg = F.avg_pool2d(
            probmap[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
        probmap = (probmap - beta * local_avg).clamp(min=0.0)

    pmax = torch.clamp(probmap.max(), min=1e-6)
    probmap = (probmap / pmax).clamp(min=1e-6, max=1.0)

    print(f'std_geo_norm stats - min: {std_geo_norm.min().item():.6f}, max: {std_geo_norm.max().item():.6f}, mean: {std_geo_norm.mean().item():.6f}')
    print(f'probmap stats - min: {probmap.min().item():.6f}, max: {probmap.max().item():.6f}, mean: {probmap.mean().item():.6f}')

    if remap_output_mode == 'seg':
        
        variance = std_geo  # shape [H, W]
        cpm = probmap  # shape [H, W]
        
        # sigmoid-fuse
        # sigma_weight = 0.8
        # fused = sigma_weight * (1 - variance.sigmoid()) + (1 - sigma_weight) * cpm
        
        fused = (1 - (variance.sigmoid() * (1 - cpm))).sqrt()
        # erode
        fused_np = fused.detach().cpu().numpy()
        print(fused_np.shape)
        print(f'Before erosion - fused stats: min: {fused_np.min():.6f}, max: {fused_np.max():.6f}, mean: {fused_np.mean():.6f}')
        fused_np = _grayscale_erode(fused_np, kernel_size=1)
        print(f'After erosion - fused stats: min: {fused_np.min():.6f}, max: {fused_np.max():.6f}, mean: {fused_np.mean():.6f}')
        fused = torch.from_numpy(fused_np).to(fused.device).to(fused.dtype)
        
        remapped_label_map = build_gt_guided_segmentation_mask(
            p_model=fused,
            gt_bboxes=gt_bboxes,
            img_meta=img_meta,
            sigma_scale=sigma_scale,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            # score_thr=seg_score_thr,
            score_thr=0.1,
            topk=seg_topk,
            bg_std_scale=bg_std_scale)
        remapped_img = _label_to_color_mask(remapped_label_map, flip_direction)
        # half-transparent overlay of segmentation mask on original image
        base_img = Image.open(img_path).convert('RGB')
        base_img = base_img.resize(remapped_img.size)
        remapped_img = Image.blend(base_img, remapped_img, alpha=0.8)

    elif remap_output_mode == 'gt_guided':
        # Build remap as distance-from-mapped-position-to-nearest-GT using mu.
        # bbox_mu: [2, H, W] -> mu_x, mu_y offsets in feature-pixel units.
        with torch.no_grad():
            p = probmap
            h, w = p.shape
            device = p.device
            dtype = p.dtype

            remapped_max_prob = build_gt_guided_remap_map(
                p_model=probmap,
                gt_bboxes=gt_bboxes,
                img_meta=img_meta,
                sigma_scale=sigma_scale,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                score_thr=seg_score_thr,
                topk=seg_topk,
                bg_std_scale=bg_std_scale)
            remapped_img = _to_heatmap(remapped_max_prob, flip_direction)
                
    elif remap_output_mode == 'mu_mask':
        mu_scale = 8.0
        mu = bbox_mu  # shape [2, H, W]
        w, h = mu.shape[2], mu.shape[1]
        device = mu.device
        dtype = mu.dtype
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing='ij')
        
        print(f'mu stats - min: {mu.min().item():.6f}, max: {mu.max().item():.6f}, mean: {mu.mean().item():.6f}')

        # mapped position in feature coordinates
        mapped_x = xx + mu[0] * float(mu_scale)
        mapped_y = yy + mu[1] * float(mu_scale)

        cx, cy, bw, bh = _extract_center_and_size(gt_bboxes)
        if cx is None:
            # no GT: produce zero map
            nearest_dist = torch.zeros_like(mapped_x)
        else:
            img_h, img_w = _infer_image_shape(img_meta)
            sx = float(w) / max(float(img_w), 1.0)  # feature-map to image scaling factor
            sy = float(h) / max(float(img_h), 1.0)

            cx_f = (cx.to(device=device, dtype=dtype) * sx).view(-1, 1, 1)
            cy_f = (cy.to(device=device, dtype=dtype) * sy).view(-1, 1, 1)

            # distances: shape [n_gt, H, W]
            dx2 = (mapped_x.unsqueeze(0) - cx_f)**2
            dy2 = (mapped_y.unsqueeze(0) - cy_f)**2
            d2 = dx2 + dy2
            minvals, minidx = d2.min(dim=0)
            nearest_dist = torch.sqrt(minvals)

        # Post-process: within 10 units => keep (set mask=1), else background.
        # Assign pixel to nearest GT index for coloring.
        # Build label map: -1 background, otherwise gt index.
        
        thresh = 5.0
        label_map = torch.full((h, w), -1, dtype=torch.long, device=device)
        if cx is not None:
            keep_mask = nearest_dist <= float(thresh)
            if keep_mask.any():
                nearest_idx = minidx.to(torch.long)
                label_map[keep_mask] = nearest_idx[keep_mask]

        remapped_img = _label_to_color_mask(label_map, flip_direction)
        # Half-transparent overlay of segmentation mask on original image
        base_img = Image.open(img_path).convert('RGB')
        base_img = base_img.resize(remapped_img.size)
        remapped_img = Image.blend(base_img, remapped_img, alpha=0.8)
        
        # or a soft mask based on nearest_dist
        # remapped_img = _to_heatmap(-nearest_dist, flip_direction)  # closer to GT center -> hotter color            
            
    
    elif remap_output_mode == 'mu':
        mu_scale = 8.0
        mu = bbox_mu * mu_scale  # shape [2, H, W]
        w, h = mu.shape[2], mu.shape[1]
        reprojected_x = torch.arange(w, device=mu.device, dtype=mu.dtype) + mu[0]
        reprojected_y = torch.arange(h, device=mu.device, dtype=mu.dtype) + mu[1]
        reprojected_x = reprojected_x.clamp(0, w - 1)
        reprojected_y = reprojected_y.clamp(0, h - 1)
        remapped_img = _to_heatmap(reprojected_x, flip_direction)
        
    elif remap_output_mode == 'mu_density':
        # for each pixel, compute the density of reprojected positions from mu, then visualize the density map.
        mu_scale = 1.0
        mu = bbox_mu * mu_scale  # shape [2, H, W]
        w, h = mu.shape[2], mu.shape[1]
        reprojected_x = torch.arange(w, device=mu.device, dtype=mu.dtype) + mu[0]
        reprojected_y = torch.arange(h, device=mu.device, dtype=mu.dtype) + mu[1]
        reprojected_x = reprojected_x.clamp(0, w - 1)
        reprojected_y = reprojected_y.clamp(0, h - 1)
        # Compute density simply by +1 for every reprojected position.
        density = torch.zeros((h, w), device=mu.device, dtype=mu.dtype)
        for i in range(w):
            for j in range(h):
                x = int(reprojected_x[j, i].item())
                y = int(reprojected_y[j, i].item())
                density[y, x] += 1.0
        remapped_img = _to_heatmap(density, flip_direction)
        
    elif remap_output_mode == 'sigma_fuse':  # cpm + variance
        variance = std_geo  # shape [H, W]
        cpm = probmap  # shape [H, W]
        
        # sigmoid-fuse
        # sigma_weight = 0.8
        # fused = sigma_weight * (1 - variance.sigmoid()) + (1 - sigma_weight) * cpm
        
        fused = (1 - (variance.sigmoid() * (1 - cpm))).sqrt()
        # erode
        fused_np = fused.detach().cpu().numpy()
        print(fused_np.shape)
        print(f'Before erosion - fused stats: min: {fused_np.min():.6f}, max: {fused_np.max():.6f}, mean: {fused_np.mean():.6f}')
        fused_np = _grayscale_erode(fused_np, kernel_size=3)
        print(f'After erosion - fused stats: min: {fused_np.min():.6f}, max: {fused_np.max():.6f}, mean: {fused_np.mean():.6f}')
        fused = torch.from_numpy(fused_np).to(fused.device).to(fused.dtype)
        
        remapped_img = _to_heatmap(fused, flip_direction)
        
    else:
        raise ValueError(f'Unsupported remap_output_mode: {remap_output_mode}')

    center_lstd_img = _to_heatmap(center_lstd, flip_direction)
    center_mu_img = _to_heatmap(center_mu, flip_direction)
    centerness_img = _to_heatmap(centerness_prob, flip_direction)
    max_cls_img = _to_heatmap(max_class_prob, flip_direction)
    combined_img = _to_heatmap(combined_score, flip_direction)

    base_img = Image.open(img_path).convert('RGB')
    base_img = base_img.resize((center_mu_img.width, center_mu_img.height))

    cell_w, cell_h = base_img.width, base_img.height
    merged = Image.new('RGB', (cell_w * 3, cell_h * 2))

    merged.paste(base_img, (0, 0))
    merged.paste(center_mu_img, (cell_w, 0))
    merged.paste(center_lstd_img, (cell_w * 2, 0))

    merged.paste(centerness_img, (0, cell_h))
    merged.paste(max_cls_img, (cell_w, cell_h))
    merged.paste(combined_img, (cell_w * 2, cell_h))

    merged.save(out_path)

    mean_merged = Image.new('RGB', (cell_w * 2, cell_h))
    mean_merged.paste(base_img, (0, 0))
    mean_merged.paste(center_mu_img, (cell_w, 0))
    mean_merged.save(out_mean_path)

    remapped_img.save(out_remap_path)


def _prepare_dataset_cfg(cfg, args):
    split_cfg = copy.deepcopy(cfg.data[args.split])
    split_cfg.test_mode = (args.split != 'train')

    if args.ann_file is not None:
        split_cfg.ann_file = args.ann_file
    if args.img_prefix is not None:
        split_cfg.img_prefix = args.img_prefix

    samples_per_gpu = split_cfg.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        split_cfg.pipeline = replace_ImageToTensor(split_cfg.pipeline)

    return split_cfg, samples_per_gpu


def _unwrap_tensor_list(data, key, device):
    payload = data.get(key, None)
    if payload is None:
        return None

    if isinstance(payload, DataContainer):
        payload = payload.data[0]

    if torch.is_tensor(payload):
        return [payload.to(device)]

    if isinstance(payload, (list, tuple)):
        out = []
        for item in payload:
            if isinstance(item, DataContainer):
                item = item.data[0]
            if torch.is_tensor(item):
                out.append(item.to(device))
            elif isinstance(item, (list, tuple)) and len(item) == 1 and torch.is_tensor(item[0]):
                out.append(item[0].to(device))
            else:
                raise TypeError(
                    f'Unsupported payload element in `{key}`: {type(item)}')
        return out

    raise TypeError(f'Unsupported payload type in `{key}`: {type(payload)}')

def _get_variance_cscore_scatter_data_image(
        bbox_pred_lvl,
        cls_score_lvl,
        gt_bboxes,
        gt_labels,
        img_meta,
        flip_direction):
    # get the (variance, cscore) pairs for all gt centers in an image, for scatter plotting
    device = cls_score_lvl.device
    dtype = cls_score_lvl.dtype
    cx, cy, bw, bh = _extract_center_and_size(gt_bboxes)
    if cx is None:
        return None, None, None
    img_h, img_w = _infer_image_shape(img_meta)
    sx = float(cls_score_lvl.shape[2]) / max(float(img_w), 1.0)
    sy = float(cls_score_lvl.shape[1]) / max(float(img_h), 1.0)
    cx_f = (cx.to(device=device, dtype=dtype) * sx).view(-1)
    cy_f = (cy.to(device=device, dtype=dtype) * sy).view(-1)
        
    # flip gt centers if image flipped
    # if flip_direction == 'horizontal':
    #     cx_f = cls_score_lvl.shape[2] - 1 - cx_f
    # elif flip_direction == 'vertical':
    #     cy_f = cls_score_lvl.shape[1] - 1 - cy_f
    # elif flip_direction == 'diagonal':
    #     cx_f = cls_score_lvl.shape[2] - 1 - cx_f
    #     cy_f = cls_score_lvl.shape[1] - 1 - cy_f
    
    # gt_cls_scores = cls_score_lvl.softmax(dim=0).max(dim=0)[0]
    gt_cls_scores = cls_score_lvl.sigmoid().max(dim=0)[0]
    gt_cls_scores_at_centers = gt_cls_scores[cy_f.long(), cx_f.long()]
    gt_variance_at_centers = bbox_pred_lvl[2:4, cy_f.long(), cx_f.long()].exp().mean(dim=0)
    gt_lstd_at_centers = gt_variance_at_centers.sqrt().log()
    # print(cls_score_lvl.max(), cls_score_lvl.min(), cls_score_lvl.mean())
    # print(cls_score_lvl.sigmoid[:, cy_f.long(), cx_f.long()])
        
    labels_np = None
    if gt_labels is not None:
        labels_np = gt_labels.detach().cpu().numpy().astype(np.int64)

    return (gt_lstd_at_centers.cpu().numpy(),
            gt_cls_scores_at_centers.cpu().numpy(),
            labels_np)

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)

    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck') and cfg.model.neck.get('rfp_backbone'):
        if cfg.model.neck.rfp_backbone.get('pretrained'):
            cfg.model.neck.rfp_backbone.pretrained = None

    dataset_cfg, samples_per_gpu = _prepare_dataset_cfg(cfg, args)
    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
        dist=False,
        shuffle=False)

    cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None or cfg.device == 'npu':
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    device = torch.device('cuda', args.gpu_id) if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    scatter_class_idx = None
    scatter_class_name = None
    if args.scatter_class_name:
        if not hasattr(dataset, 'CLASSES') or dataset.CLASSES is None:
            raise ValueError('Dataset has no CLASSES attribute for name lookup.')
        if args.scatter_class_name not in dataset.CLASSES:
            raise ValueError(
                f'Unknown class name: {args.scatter_class_name}. '
                f'Available: {dataset.CLASSES}')
        scatter_class_name = args.scatter_class_name
        scatter_class_idx = int(list(dataset.CLASSES).index(args.scatter_class_name))
    elif args.scatter_class is not None and int(args.scatter_class) >= 0:
        scatter_class_idx = int(args.scatter_class)
        if hasattr(dataset, 'CLASSES') and dataset.CLASSES is not None:
            if scatter_class_idx < len(dataset.CLASSES):
                scatter_class_name = dataset.CLASSES[scatter_class_idx]

    captured = {}

    def _hook_fn(_module, _inputs, outputs):
        if not isinstance(outputs, (list, tuple)):
            raise RuntimeError('Unexpected bbox_head output type.')
        if len(outputs) == 3:
            cls_scores, bbox_preds, centernesses = outputs
        elif len(outputs) == 4:
            cls_scores, bbox_preds, _, centernesses = outputs
        else:
            raise RuntimeError(f'Unexpected bbox_head output length: {len(outputs)}')
        captured['cls_scores'] = cls_scores
        captured['bbox_preds'] = bbox_preds
        captured['centernesses'] = centernesses

    hook_handle = model.bbox_head.register_forward_hook(_hook_fn)

    mmcv.mkdir_or_exist(args.out_dir)
    progress = mmcv.ProgressBar(len(dataset))

    max_images = args.max_images if args.max_images > 0 else len(dataset)
    processed = 0
    
    # scatter data
    scatter_data = []

    for data in data_loader:
        if processed >= max_images:
            break

        captured.clear()
        with torch.no_grad():
            img = data['img'].data[0].to(device)
            feats = model.extract_feat(img)
            _ = model.bbox_head(feats)

        img_metas = data['img_metas'].data[0]
        batch_size = len(img_metas)
        gt_bboxes_list = _unwrap_tensor_list(data, 'gt_bboxes', device)
        gt_labels_list = _unwrap_tensor_list(data, 'gt_labels', device)

        if ('bbox_preds' not in captured or 'cls_scores' not in captured
                or 'centernesses' not in captured):
            raise RuntimeError('Failed to capture cls_scores/bbox_preds/centernesses from bbox_head forward hook.')

        bbox_preds = captured['bbox_preds']
        cls_scores = captured['cls_scores']
        centernesses = captured['centernesses']
        if args.feat_level < 0 or args.feat_level >= len(bbox_preds):
            raise ValueError(
                f'feat_level={args.feat_level} is out of range [0, {len(bbox_preds) - 1}]')

        for b in range(batch_size):
            if processed >= max_images:
                break

            meta = img_metas[b]
            img_path = meta['filename']
            flip_direction = meta.get('flip_direction')
            bbox_pred_lvl = bbox_preds[args.feat_level][b]
            cls_score_lvl = cls_scores[args.feat_level][b]
            centerness_lvl = centernesses[args.feat_level][b]
            gt_bboxes = None if gt_bboxes_list is None else gt_bboxes_list[b]
            gt_labels = None if gt_labels_list is None else gt_labels_list[b]

            stem = osp.splitext(osp.basename(img_path))[0]
            out_name = f'{processed:06d}_{stem}_lstd.jpg'
            out_path = osp.join(args.out_dir, out_name)
            out_mean_name = f'{processed:06d}_{stem}_mean.jpg'
            out_mean_path = osp.join(args.out_dir, out_mean_name)
            out_remap_name = f'{processed:06d}_{stem}_remap.jpg'
            out_remap_path = osp.join(args.out_dir, out_remap_name)
            _save_maps_for_image(
                img_path,
                meta,
                flip_direction,
                bbox_pred_lvl,
                cls_score_lvl,
                centerness_lvl,
                gt_bboxes,
                out_path,
                out_mean_path,
                out_remap_path,
                args.seg_score_thr,
                args.seg_topk,
                args.sigma_scale,
                args.min_sigma,
                args.max_sigma,
                args.bg_std_scale,
                args.remap_output_mode,
                args.cls_floor,
                args.cls_gamma,
                args.uncert_q_lo,
                args.uncert_q_hi,
                args.uncert_gamma,
                args.alpha_cls,
                args.alpha_uncert,
                args.prob_smooth_ksize,
                args.prob_local_contrast)
            
            # save scatter data
            variance_at_centers, cscore_at_centers, labels_at_centers = _get_variance_cscore_scatter_data_image(
                bbox_pred_lvl,
                cls_score_lvl,
                gt_bboxes,
                gt_labels,
                meta,
                flip_direction)
            if cscore_at_centers is not None and variance_at_centers is not None:
                if scatter_class_idx is not None:
                    if labels_at_centers is None:
                        pass
                    else:
                        keep = labels_at_centers == scatter_class_idx
                        if np.any(keep):
                            for var, cs in zip(variance_at_centers[keep], cscore_at_centers[keep]):
                                scatter_data.append((var, cs))
                else:
                    for var, cs in zip(variance_at_centers, cscore_at_centers):
                        scatter_data.append((var, cs))

            processed += 1
            progress.update()
            
    # plot scatter data
    if scatter_data:
        import matplotlib.pyplot as plt
        scatter_data = np.array(scatter_data)
        cs = scatter_data[:, 1]
        lstd = scatter_data[:, 0]
        plt.figure(figsize=(8, 6))
        plt.scatter(lstd, cs, alpha=0.9, s=2)
        plt.ylabel('Classification Scores')
        plt.xlabel('lstd')
        title = 'Variance vs Classification Score'
        if scatter_class_idx is not None:
            if scatter_class_name:
                title += f' (class: {scatter_class_name})'
            else:
                title += f' (class idx: {scatter_class_idx})'
        plt.title(title)
        plt.grid(True)
        scatter_out_path = osp.join(args.out_dir, '_variance_vs_cscore_scatter.png')
        plt.savefig(scatter_out_path)
        print(f'Scatter plot saved to: {scatter_out_path}')

    hook_handle.remove()
    print(f'Processed {processed} images. Results saved to: {args.out_dir}')


if __name__ == '__main__':
    main()