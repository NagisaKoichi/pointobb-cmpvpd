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
                                  build_gt_guided_segmentation_mask)


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
        '--remap-output-mode',
        default='variance',
        choices=['variance', 'mask'],
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
    # Channels 0:2 are posterior center mean for (x, y).
    mu = torch.nan_to_num(bbox_pred_lvl[0:2], nan=0.0, posinf=1e4, neginf=-1e4)
    center_mu = mu[0:2].mean(dim=0)

    # Channels 2:4 are center log_sigma for (x, y).
    log_sigma = bbox_pred_lvl[2:4]
    lstd = torch.nan_to_num(log_sigma, nan=0.0, posinf=1e4, neginf=-1e4)

    std = lstd.exp()
    center_std = std.mean(dim=0)

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

    if remap_output_mode == 'mask':
        remapped_label_map = build_gt_guided_segmentation_mask(
            p_model=probmap,
            gt_bboxes=gt_bboxes,
            img_meta=img_meta,
            sigma_scale=sigma_scale,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            score_thr=seg_score_thr,
            topk=seg_topk,
            bg_std_scale=bg_std_scale)
        remapped_img = _label_to_color_mask(remapped_label_map, flip_direction)
        # half-transparent overlay of segmentation mask on original image
        base_img = Image.open(img_path).convert('RGB')
        base_img = base_img.resize(remapped_img.size)
        remapped_img = Image.blend(base_img, remapped_img, alpha=0.8)

    else:
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

    center_mu_img = _to_heatmap(center_mu, flip_direction)
    center_std_img = _to_heatmap(center_std, flip_direction)
    centerness_img = _to_heatmap(centerness_prob, flip_direction)
    max_cls_img = _to_heatmap(max_class_prob, flip_direction)
    combined_img = _to_heatmap(combined_score, flip_direction)

    base_img = Image.open(img_path).convert('RGB')
    base_img = base_img.resize((center_std_img.width, center_std_img.height))

    cell_w, cell_h = base_img.width, base_img.height
    merged = Image.new('RGB', (cell_w * 3, cell_h * 2))

    merged.paste(base_img, (0, 0))
    merged.paste(center_std_img, (cell_w, 0))
    merged.paste(center_mu_img, (cell_w * 2, 0))

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

            processed += 1
            progress.update()

    hook_handle.remove()
    print(f'Processed {processed} images. Results saved to: {args.out_dir}')


if __name__ == '__main__':
    main()