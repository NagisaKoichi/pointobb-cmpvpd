# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmrotate.utils import dump_vpd_heatmaps_from_case_dir, find_vpd_case_dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate heatmaps from saved VPD intermediate tensors.')
    parser.add_argument(
        'input_dir',
        type=str,
        help='One case directory or a root directory that contains many cases.')
    parser.add_argument(
        '--cmap',
        type=str,
        default='turbo',
        help='Matplotlib colormap for heatmaps.')
    parser.add_argument(
        '--no-variance-channels',
        action='store_true',
        help='Only save variance mean heatmap, do not save each variance channel.')
    return parser.parse_args()


def main():
    args = parse_args()
    case_dirs = find_vpd_case_dirs(args.input_dir)

    if not case_dirs:
        print(f'No valid VPD case directory found under: {args.input_dir}')
        return

    success = 0
    for case_dir in case_dirs:
        ok = dump_vpd_heatmaps_from_case_dir(
            case_dir=case_dir,
            cmap=args.cmap,
            save_variance_channels=not args.no_variance_channels)
        if ok:
            success += 1

    print(f'Processed {len(case_dirs)} case(s), success: {success}.')


if __name__ == '__main__':
    main()