#!/usr/bin/env python3
"""Visualize pseudo labels on original images.

Supported label line formats (space-separated):
1) x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
2) x1 y1 x2 y2 x3 y3 x4 y4 class_name
3) x1 y1 x2 y2 x3 y3 x4 y4 score class_name
4) x1 y1 x2 y2 x3 y3 x4 y4 class_name score
"""

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Draw pseudo labels on images and save visualization results.')
    parser.add_argument('image_dir', help='Directory containing original images.')
    parser.add_argument('label_dir', help='Directory containing pseudo label txt files.')
    parser.add_argument('out_dir', help='Directory to save visualization images.')
    parser.add_argument(
        '--config',
        default=None,
        help='If set, use dataset traversal order from config (same as '
        'generate_vpd_variance_map.py) to pick examples.')
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val', 'test'],
        help='Data split used with --config. Ignored when --config is unset.')
    parser.add_argument('--ann-file', help='Optional annotation path override (with --config).')
    parser.add_argument('--img-prefix', help='Optional image prefix override (with --config).')
    parser.add_argument(
        '--image-exts',
        nargs='+',
        default=['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'],
        help='Image extensions to match (case-insensitive).')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='Skip boxes with score lower than this threshold. '
        'If a label line has no score, it will be kept.')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Line thickness for polygons.')
    parser.add_argument(
        '--font-scale',
        type=float,
        default=0.5,
        help='Font scale for text.')
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='Maximum number of images to process. 0 means all.')
    parser.add_argument(
        '--skip-empty',
        action='store_true',
        help='Skip saving images that have no valid parsed boxes.')
    return parser.parse_args()


def is_float_token(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_label_line(line: str) -> Optional[Tuple[np.ndarray, str, Optional[float]]]:
    """Parse one pseudo label line to polygon, class name and optional score."""
    parts = line.strip().split()
    if len(parts) < 9:
        return None

    # First 8 values should be polygon coordinates.
    try:
        coords = [float(v) for v in parts[:8]]
    except ValueError:
        return None
    pts = np.array(coords, dtype=np.float32).reshape(4, 2)

    remain = parts[8:]
    class_name: Optional[str] = None
    score: Optional[float] = None

    non_float_tokens = [tok for tok in remain if not is_float_token(tok)]
    float_tokens = [float(tok) for tok in remain if is_float_token(tok)]

    if non_float_tokens:
        class_name = non_float_tokens[0]
    else:
        class_name = 'unknown'

    # Use the first float in trailing fields as score if possible.
    if float_tokens:
        candidate = float_tokens[0]
        if 0.0 <= candidate <= 1.0:
            score = candidate

    return pts, class_name, score


def random_color_from_name(name: str) -> Tuple[int, int, int]:
    """Generate deterministic BGR color from class name."""
    h = abs(hash(name))
    b = 64 + (h % 192)
    g = 64 + ((h >> 8) % 192)
    r = 64 + ((h >> 16) % 192)
    return int(b), int(g), int(r)


def build_image_index(image_dir: Path, image_exts: List[str]) -> Dict[str, Path]:
    exts = {e.lower() for e in image_exts}
    index: Dict[str, Path] = {}
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        index[path.stem] = path
    return index


def _prepare_dataset_cfg(cfg, args):
    split_cfg = copy.deepcopy(cfg.data[args.split])
    split_cfg.test_mode = (args.split != 'train')

    if args.ann_file is not None:
        split_cfg.ann_file = args.ann_file
    if args.img_prefix is not None:
        split_cfg.img_prefix = args.img_prefix

    samples_per_gpu = split_cfg.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        from mmdet.datasets import replace_ImageToTensor
        split_cfg.pipeline = replace_ImageToTensor(split_cfg.pipeline)

    return split_cfg, samples_per_gpu


def select_label_paths(args, label_dir: Path) -> List[Path]:
    if args.config is None:
        label_paths = sorted(label_dir.glob('*.txt'))
        if args.max_images > 0:
            label_paths = label_paths[:args.max_images]
        return label_paths

    from mmcv import Config
    from mmdet.datasets import build_dataloader

    from mmrotate.datasets import build_dataset
    from mmrotate.utils import compat_cfg, setup_multi_processes

    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)

    dataset_cfg, samples_per_gpu = _prepare_dataset_cfg(cfg, args)
    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
        dist=False,
        shuffle=False)

    max_images = args.max_images if args.max_images > 0 else len(dataset)
    selected: List[Path] = []
    for data in data_loader:
        if len(selected) >= max_images:
            break
        img_metas = data['img_metas'].data[0]
        for meta in img_metas:
            if len(selected) >= max_images:
                break
            stem = Path(meta['filename']).stem
            selected.append(label_dir / f'{stem}.txt')
    return selected


def draw_boxes(
        img: Image.Image,
        labels: List[Tuple[np.ndarray, str, Optional[float]]],
        thickness: int,
        font_scale: float) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)

    # PIL default font does not support scalable TTF size without an external file.
    # We keep font_scale argument for API compatibility and map it to y-offset.
    text_offset = max(10, int(18 * font_scale))

    for pts, class_name, score in labels:
        poly = np.round(pts).astype(np.int32).reshape(-1, 2)
        poly_points = [tuple(map(int, p.tolist())) for p in poly]
        color = random_color_from_name(class_name)

        draw.line(poly_points + [poly_points[0]], fill=color, width=thickness)

        text = class_name if score is None else f'{class_name}:{score:.2f}'
        anchor = poly_points[0]
        text_pos = (anchor[0], max(0, anchor[1] - text_offset))
        draw.text(text_pos, text, fill=color)
    return out


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        raise NotADirectoryError(f'Image dir not found: {image_dir}')
    if not label_dir.is_dir():
        raise NotADirectoryError(f'Label dir not found: {label_dir}')

    image_index = build_image_index(image_dir, args.image_exts)
    label_paths = select_label_paths(args, label_dir)

    total = len(label_paths)
    matched = 0
    saved = 0
    skipped_score = 0
    parse_failed = 0

    for i, label_path in enumerate(label_paths, 1):
        if not label_path.is_file():
            print(f'[{i}/{total}] missing label file: {label_path.name}')
            continue

        stem = label_path.stem
        image_path = image_index.get(stem)
        if image_path is None:
            print(f'[{i}/{total}] missing image for label: {label_path.name}')
            continue
        matched += 1

        try:
            img = Image.open(image_path).convert('RGB')
        except (OSError, ValueError):
            print(f'[{i}/{total}] failed to read image: {image_path}')
            continue

        parsed: List[Tuple[np.ndarray, str, Optional[float]]] = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                item = parse_label_line(line)
                if item is None:
                    parse_failed += 1
                    continue
                pts, class_name, score = item
                if score is not None and score < args.score_thr:
                    skipped_score += 1
                    continue
                parsed.append((pts, class_name, score))

        if args.skip_empty and not parsed:
            print(f'[{i}/{total}] skip empty: {image_path.name}')
            continue

        vis = draw_boxes(img, parsed, args.thickness, args.font_scale)
        out_path = out_dir / image_path.name
        vis.save(out_path)
        saved += 1
        print(f'[{i}/{total}] saved: {out_path}')

    print('\nDone')
    print(f'Label files total: {total}')
    print(f'Matched images: {matched}')
    print(f'Saved images: {saved}')
    print(f'Skipped by score: {skipped_score}')
    print(f'Parse failed lines: {parse_failed}')


if __name__ == '__main__':
    main()