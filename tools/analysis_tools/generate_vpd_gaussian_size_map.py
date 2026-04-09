# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os.path as osp

import mmcv
import numpy as np
import torch
from PIL import Image
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import build_dataloader, replace_ImageToTensor

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, get_device, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference VPD checkpoint and dump center-Gaussian maps')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument(
        '--out-dir', required=True, help='directory to save generated maps')
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
        help='FPN level index used for map generation')
    parser.add_argument(
        '--prob-thr',
        type=float,
        default=0.3,
        help='keep pixels with max_class_probability > prob_thr')
    parser.add_argument(
        '--max-sigma',
        type=float,
        default=20.0,
        help='upper bound for generated Gaussian sigma on feature map')
    parser.add_argument(
        '--min-sigma',
        type=float,
        default=0.5,
        help='lower bound for generated Gaussian sigma on feature map')
    parser.add_argument(
        '--window-scale',
        type=float,
        default=3.0,
        help='local accumulation window radius = window_scale * sigma')
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='max number of images to process, 0 means all')
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='single gpu id for inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in config, key=value format')
    return parser.parse_args()


def _to_heatmap(map_tensor, flip_direction):
    map_np = map_tensor.detach().cpu().numpy().astype(np.float32)
    map_min = float(map_np.min())
    map_max = float(map_np.max())
    if map_max - map_min < 1e-8:
        norm = np.zeros_like(map_np, dtype=np.float32)
    else:
        norm = (map_np - map_min) / (map_max - map_min)

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


def _accumulate_size_gaussians(max_class_prob,
                               center_mu,
                               center_log_sigma,
                               prob_thr,
                               min_sigma,
                               max_sigma,
                               window_scale):
    # max_class_prob: [H, W]
    # center_mu: [2, H, W], corresponds to (dx_mu, dy_mu)
    # center_log_sigma: [2, H, W], corresponds to (log_sx, log_sy)
    h, w = max_class_prob.shape
    out_map = torch.zeros((h, w), dtype=max_class_prob.dtype, device=max_class_prob.device)

    keep_mask = max_class_prob > prob_thr
    keep_idx = torch.nonzero(keep_mask, as_tuple=False)
    if keep_idx.numel() == 0:
        return out_map

    std = torch.exp(torch.clamp(center_log_sigma, min=-100.0, max=100.0))

    for i in range(keep_idx.shape[0]):
        y = int(keep_idx[i, 0].item())
        x = int(keep_idx[i, 1].item())

        dx = center_mu[0, y, x].item()
        dy = center_mu[1, y, x].item()
        cx = int(np.clip(np.round(x + dx), 0, w - 1))
        cy = int(np.clip(np.round(y + dy), 0, h - 1))

        sigma_x = float(np.clip(std[0, y, x].item(), min_sigma, max_sigma))
        sigma_y = float(np.clip(std[1, y, x].item(), min_sigma, max_sigma))

        rx = max(1, int(np.ceil(window_scale * sigma_x)))
        ry = max(1, int(np.ceil(window_scale * sigma_y)))

        x0 = max(0, cx - rx)
        x1 = min(w - 1, cx + rx)
        y0 = max(0, cy - ry)
        y1 = min(h - 1, cy + ry)

        xx = torch.arange(x0, x1 + 1, device=out_map.device, dtype=out_map.dtype) - float(cx)
        yy = torch.arange(y0, y1 + 1, device=out_map.device, dtype=out_map.dtype) - float(cy)

        gx = torch.exp(-0.5 * (xx / sigma_x)**2)
        gy = torch.exp(-0.5 * (yy / sigma_y)**2)
        kernel = gy[:, None] * gx[None, :]

        # Normalize each local kernel so each pixel contributes by its class confidence.
        kernel = kernel / (kernel.sum() + 1e-6)
        weight = max_class_prob[y, x]
        out_map[y0:y1 + 1, x0:x1 + 1] += weight * kernel

    return out_map


def _save_map_for_image(img_path,
                        flip_direction,
                        bbox_pred_lvl,
                        cls_score_lvl,
                        out_path,
                        prob_thr,
                        min_sigma,
                        max_sigma,
                        window_scale):
    # Channels 0:2 are posterior center mean (dx, dy).
    mu = torch.nan_to_num(bbox_pred_lvl[0:2], nan=0.0, posinf=1e4, neginf=-1e4)
    # Channels 2:4 are center log_sigma (log_sx, log_sy).
    log_sigma = torch.nan_to_num(
        bbox_pred_lvl[2:4], nan=0.0, posinf=10.0, neginf=-10.0)

    cls_score_lvl = torch.nan_to_num(
        cls_score_lvl, nan=0.0, posinf=50.0, neginf=-50.0)
    max_class_prob = cls_score_lvl.sigmoid().max(dim=0)[0]

    gaussian_map = _accumulate_size_gaussians(
        max_class_prob=max_class_prob,
        center_mu=mu,
        center_log_sigma=log_sigma,
        prob_thr=prob_thr,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        window_scale=window_scale)

    map_img = _to_heatmap(gaussian_map, flip_direction)

    base_img = Image.open(img_path).convert('RGB')
    base_img = base_img.resize((map_img.width, map_img.height))

    merged = Image.new('RGB', (base_img.width * 2, base_img.height))
    merged.paste(base_img, (0, 0))
    merged.paste(map_img, (base_img.width, 0))
    merged.save(out_path)


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

    device = torch.device(
        'cuda', args.gpu_id) if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    captured = {}

    def _hook_fn(_module, _inputs, outputs):
        if not isinstance(outputs, (list, tuple)):
            raise RuntimeError('Unexpected bbox_head output type.')
        if len(outputs) == 3:
            cls_scores, bbox_preds, _ = outputs
        elif len(outputs) == 4:
            cls_scores, bbox_preds, _, _ = outputs
        else:
            raise RuntimeError(
                f'Unexpected bbox_head output length: {len(outputs)}')
        captured['cls_scores'] = cls_scores
        captured['bbox_preds'] = bbox_preds

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

        if 'bbox_preds' not in captured or 'cls_scores' not in captured:
            raise RuntimeError(
                'Failed to capture cls_scores/bbox_preds from bbox_head hook.')

        bbox_preds = captured['bbox_preds']
        cls_scores = captured['cls_scores']
        if args.feat_level < 0 or args.feat_level >= len(bbox_preds):
            raise ValueError(
                f'feat_level={args.feat_level} is out of range '
                f'[0, {len(bbox_preds) - 1}]')

        for b in range(batch_size):
            if processed >= max_images:
                break

            meta = img_metas[b]
            img_path = meta['filename']
            flip_direction = meta.get('flip_direction')
            bbox_pred_lvl = bbox_preds[args.feat_level][b]
            cls_score_lvl = cls_scores[args.feat_level][b]

            stem = osp.splitext(osp.basename(img_path))[0]
            out_name = f'{processed:06d}_{stem}_center_gauss.jpg'
            out_path = osp.join(args.out_dir, out_name)
            _save_map_for_image(
                img_path=img_path,
                flip_direction=flip_direction,
                bbox_pred_lvl=bbox_pred_lvl,
                cls_score_lvl=cls_score_lvl,
                out_path=out_path,
                prob_thr=args.prob_thr,
                min_sigma=args.min_sigma,
                max_sigma=args.max_sigma,
                window_scale=args.window_scale)

            processed += 1
            progress.update()

    hook_handle.remove()
    print(f'Processed {processed} images. Results saved to: {args.out_dir}')


if __name__ == '__main__':
    main()