#!/usr/bin/env python3
"""Generate non-rotated pseudo labels from a CPMVPD checkpoint.

This script runs inference with a CPMVPD model and writes one DOTA-style txt
file per image in `out_dir`:
    x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult

Boxes are exported as axis-aligned polygons (no rotation), i.e., the model's
(cx, cy, w, h) is converted to rectangle corners while ignoring angle.
"""

import argparse
import copy
import os
import os.path as osp
from pathlib import Path

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import build_dataloader, replace_ImageToTensor

from mmrotate.core import rbbox2result
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, get_device, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate pseudo labels using CPMVPDHead inference')
    parser.add_argument('config', help='model config path')
    parser.add_argument('checkpoint', help='checkpoint path')
    parser.add_argument('--out-dir', required=True, help='pseudo label output dir')
    parser.add_argument(
        '--split',
        default='val',
        choices=['train', 'val', 'test'],
        help='data split to traverse')
    parser.add_argument('--ann-file', default=None, help='override ann_file for split')
    parser.add_argument('--img-prefix', default=None, help='override img_prefix for split')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.05,
        help='filter predicted boxes by score')
    parser.add_argument(
        '--edge-thr',
        type=float,
        default=0.03,
        help='class-probability threshold used to estimate axis-aligned '
        'box edges from the first-level cls heatmap')
    parser.add_argument(
        '--edge-max-len',
        type=int,
        default=128,
        help='max probing steps per direction on cls heatmap when estimating '
        'box width/height')
    parser.add_argument(
        '--disable-heatmap-size',
        action='store_true',
        help='disable heatmap-based width/height estimation and use model '
        'predicted w/h directly')
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='max number of images to process, 0 means all')
    parser.add_argument('--gpu-id', type=int, default=0, help='single gpu id')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options in key=value format')
    return parser.parse_args()


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


def _unwrap_img_metas(data):
    img_metas = data['img_metas']
    if isinstance(img_metas, DataContainer):
        img_metas = img_metas.data[0]

    # test pipeline may wrap one extra list level
    if isinstance(img_metas, list) and len(img_metas) == 1 and isinstance(img_metas[0], list):
        img_metas = img_metas[0]
    return img_metas


def _unwrap_imgs_for_forward(data, device):
    """Build `img` argument for detector.forward_test.

    Returns:
        list[Tensor]: usually length 1 (single augmentation), each tensor shape
        is [batch, C, H, W].
    """
    def _extract_tensors(obj):
        if isinstance(obj, DataContainer):
            return _extract_tensors(obj.data[0])
        if torch.is_tensor(obj):
            return [obj]
        if isinstance(obj, (list, tuple)):
            out = []
            for item in obj:
                out.extend(_extract_tensors(item))
            return out
        raise TypeError(f'Unsupported img payload type: {type(obj)}')

    imgs = _extract_tensors(data['img'])
    if not imgs:
        raise RuntimeError('No image tensor found in batch `data["img"]`.')
    return [img.to(device) for img in imgs]


def _to_axis_aligned_poly(cx, cy, w, h):
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = y1
    x3 = x2
    y3 = cy + h / 2.0
    x4 = x1
    y4 = y3
    return x1, y1, x2, y2, x3, y3, x4, y4


def _estimate_wh_from_heatmap(cx, cy, cls_id, cls_prob_lvl0, stride,
                              edge_thr, max_len):
    """Estimate axis-aligned width/height by probing class heatmap edges."""
    _, h, w = cls_prob_lvl0.shape
    x0 = int(round(cx / stride))
    y0 = int(round(cy / stride))

    if x0 < 0 or x0 >= w or y0 < 0 or y0 >= h:
        return None

    cls_map = cls_prob_lvl0[cls_id]

    def walk(dx, dy):
        dist = 0
        for step in range(1, max_len + 1):
            x = x0 + dx * step
            y = y0 + dy * step
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            if float(cls_map[y, x]) < edge_thr:
                break
            dist = step
        return dist

    left = walk(-1, 0)
    right = walk(1, 0)
    up = walk(0, -1)
    down = walk(0, 1)

    est_w = (left + right + 1) * stride
    est_h = (up + down + 1) * stride
    est_w = max(float(est_w), float(stride))
    est_h = max(float(est_h), float(stride))
    return est_w, est_h


def _write_one_txt(txt_path, det_result, class_names, score_thr,
                   cls_prob_lvl0, stride, edge_thr, edge_max_len,
                   disable_heatmap_size):
    lines = []
    for cls_id, cls_dets in enumerate(det_result):
        if cls_dets is None or len(cls_dets) == 0:
            continue
        cls_name = class_names[cls_id]
        for det in cls_dets:
            # det layout from rbbox2result: [cx, cy, w, h, angle, score]
            if len(det) < 6:
                continue
            cx, cy, w, h, _angle, score = det[:6]
            if float(score) < score_thr:
                continue

            if not disable_heatmap_size:
                est = _estimate_wh_from_heatmap(
                    float(cx), float(cy), cls_id, cls_prob_lvl0,
                    stride=float(stride), edge_thr=float(edge_thr),
                    max_len=int(edge_max_len))
                if est is not None:
                    w, h = est

            x1, y1, x2, y2, x3, y3, x4, y4 = _to_axis_aligned_poly(
                float(cx), float(cy), float(w), float(h))
            lines.append(
                f'{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} '
                f'{x3:.1f} {y3:.1f} {x4:.1f} {y4:.1f} {cls_name} 0\n')

    with open(txt_path, 'w', encoding='utf-8') as f:
        if lines:
            f.writelines(lines)
        else:
            # Keep an empty file so downstream annfile counting is stable.
            f.write('')


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
    cfg.model.train_cfg = None
    if cfg.model.get('test_cfg') is None:
        cfg.model.test_cfg = dict()
    cfg.model.test_cfg['use_remap_score'] = True
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None or cfg.device == 'npu':
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        class_names = checkpoint['meta']['CLASSES']
    else:
        class_names = dataset.CLASSES
    model.CLASSES = class_names

    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_id)
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    max_images = args.max_images if args.max_images > 0 else total
    progress = mmcv.ProgressBar(min(total, max_images))

    processed = 0
    with torch.no_grad():
        for data in data_loader:
            if processed >= max_images:
                break

            imgs = _unwrap_imgs_for_forward(data, device)
            img_metas = _unwrap_img_metas(data)

            # Run bbox head directly so we can reuse cls heatmap for size estimation.
            feats = model.extract_feat(imgs[0])
            outs = model.bbox_head(feats)
            bbox_list = model.bbox_head.get_bboxes(*outs, img_metas, rescale=True)
            results = [
                rbbox2result(det_bboxes, det_labels, model.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]

            cls_scores = outs[0]  # list[level], tensor shape [B, C, H, W]
            stride0 = model.bbox_head.strides[0]

            for b, (meta, det_result) in enumerate(zip(img_metas, results)):
                if processed >= max_images:
                    break

                cls_prob_lvl0 = cls_scores[0][b].sigmoid().detach()

                stem = osp.splitext(osp.basename(meta['filename']))[0]
                txt_path = out_dir / f'{stem}.txt'
                _write_one_txt(
                    txt_path,
                    det_result=det_result,
                    class_names=class_names,
                    score_thr=args.score_thr,
                    cls_prob_lvl0=cls_prob_lvl0,
                    stride=stride0,
                    edge_thr=args.edge_thr,
                    edge_max_len=args.edge_max_len,
                    disable_heatmap_size=args.disable_heatmap_size)

                processed += 1
                progress.update()

    print(f'Processed {processed} images. Pseudo labels saved to: {out_dir}')


if __name__ == '__main__':
    main()
