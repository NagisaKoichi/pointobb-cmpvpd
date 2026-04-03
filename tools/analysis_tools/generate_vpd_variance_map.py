# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
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


def _save_maps_for_image(img_path,
                         flip_direction,
                         bbox_pred_lvl,
                         cls_score_lvl,
                         centerness_lvl,
                         out_path,
                         out_mean_path):
    # Channels 0:4 are posterior mean for (x, y, w, h).
    mu = torch.nan_to_num(bbox_pred_lvl[0:4], nan=0.0, posinf=1e4, neginf=-1e4)
    center_mu = mu[0:2].mean(dim=0)
    scale_mu = mu[2:4].mean(dim=0)

    # Channels 4:8 are log_sigma for (x, y, w, h).
    log_sigma = bbox_pred_lvl[4:8]
    lstd = torch.nan_to_num(log_sigma, nan=0.0, posinf=1e4, neginf=-1e4)

    center_lstd = lstd[0:2].mean(dim=0)
    scale_lstd = lstd[2:4].mean(dim=0)

    cls_score_lvl = torch.nan_to_num(cls_score_lvl, nan=0.0, posinf=50.0, neginf=-50.0)
    centerness_lvl = torch.nan_to_num(centerness_lvl, nan=0.0, posinf=50.0, neginf=-50.0)

    max_class_prob = cls_score_lvl.sigmoid().max(dim=0)[0]
    centerness_prob = centerness_lvl.sigmoid().squeeze(0)
    combined_score = max_class_prob * centerness_prob

    center_mu_img = _to_heatmap(center_mu, flip_direction)
    scale_mu_img = _to_heatmap(scale_mu, flip_direction)
    center_img = _to_heatmap(center_lstd, flip_direction)
    scale_img = _to_heatmap(scale_lstd, flip_direction)
    centerness_img = _to_heatmap(centerness_prob, flip_direction)
    max_cls_img = _to_heatmap(max_class_prob, flip_direction)
    combined_img = _to_heatmap(combined_score, flip_direction)

    base_img = Image.open(img_path).convert('RGB')
    base_img = base_img.resize((center_img.width, center_img.height))

    cell_w, cell_h = base_img.width, base_img.height
    merged = Image.new('RGB', (cell_w * 3, cell_h * 2))

    merged.paste(base_img, (0, 0))
    merged.paste(center_img, (cell_w, 0))
    merged.paste(scale_img, (cell_w * 2, 0))

    merged.paste(centerness_img, (0, cell_h))
    merged.paste(max_cls_img, (cell_w, cell_h))
    merged.paste(combined_img, (cell_w * 2, cell_h))

    merged.save(out_path)

    mean_merged = Image.new('RGB', (cell_w * 3, cell_h))
    mean_merged.paste(base_img, (0, 0))
    mean_merged.paste(center_mu_img, (cell_w, 0))
    mean_merged.paste(scale_mu_img, (cell_w * 2, 0))
    mean_merged.save(out_mean_path)


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

            stem = osp.splitext(osp.basename(img_path))[0]
            out_name = f'{processed:06d}_{stem}_lstd.jpg'
            out_path = osp.join(args.out_dir, out_name)
            out_mean_name = f'{processed:06d}_{stem}_mean.jpg'
            out_mean_path = osp.join(args.out_dir, out_mean_name)
            _save_maps_for_image(
                img_path,
                flip_direction,
                bbox_pred_lvl,
                cls_score_lvl,
                centerness_lvl,
                out_path,
                out_mean_path)

            processed += 1
            progress.update()

    hook_handle.remove()
    print(f'Processed {processed} images. Results saved to: {args.out_dir}')


if __name__ == '__main__':
    main()