# Copyright (c) OpenMMLab. All rights reserved.
import os

import torch


def _to_2d_map(tensor, reduce_mode='max'):
    """Convert input tensor to a 2D map.

    Args:
        tensor (Tensor): Input tensor in shape (H, W), (1, H, W), or (C, H, W).
        reduce_mode (str): Channel reduce mode for (C, H, W). Options are
            'max' and 'mean'.

    Returns:
        Tensor: 2D tensor in shape (H, W).
    """
    if tensor.dim() == 2:
        return tensor
    if tensor.dim() == 3:
        if tensor.size(0) == 1:
            return tensor[0]
        if reduce_mode == 'mean':
            return tensor.mean(dim=0)
        return tensor.max(dim=0)[0]
    raise ValueError(f'Unsupported tensor shape: {tuple(tensor.shape)}')


def _normalize_map(array):
    """Normalize map to [0, 1] with robust percentile clipping."""
    flat = array.reshape(-1)
    if flat.numel() == 0:
        return array

    # Robust clipping avoids outliers dominating color contrast.
    q_low = torch.quantile(flat, 0.01)
    q_high = torch.quantile(flat, 0.99)

    if (q_high - q_low).abs().item() < 1e-12:
        return torch.zeros_like(array)

    array = array.clamp(min=q_low.item(), max=q_high.item())
    array = (array - q_low) / (q_high - q_low)
    return array


def _save_heatmap(map_tensor, out_file, cmap='turbo'):
    """Save one 2D tensor as heatmap image."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    map_tensor = map_tensor.detach().float().cpu()
    map_tensor = _normalize_map(map_tensor)
    plt.imsave(out_file, map_tensor.numpy(), cmap=cmap)
    return True


def dump_vpd_heatmaps(case_dir,
                      cls_scores_per_level,
                      centerness_per_level,
                      variance_per_level,
                      cmap='turbo',
                      save_variance_channels=True):
    """Dump heatmaps for one VPD case directory.

    Args:
        case_dir (str): Case directory path.
        cls_scores_per_level (list[Tensor]): Class score tensors (C, H, W).
        centerness_per_level (list[Tensor]): Centerness tensors.
        variance_per_level (list[Tensor]): Variance tensors (C, H, W).
        cmap (str): Matplotlib colormap.
        save_variance_channels (bool): Whether to save each variance channel.

    Returns:
        bool: True if heatmap backend is available and files are written.
    """
    heatmap_dir = os.path.join(case_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)

    # Try once so online inference does not fail when matplotlib is absent.
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        return False

    for lvl_idx, centerness in enumerate(centerness_per_level):
        centerness_map = _to_2d_map(centerness, reduce_mode='max')
        _save_heatmap(
            centerness_map,
            os.path.join(heatmap_dir, f'centerness_lvl{lvl_idx}.png'),
            cmap=cmap)

    for lvl_idx, cls_score in enumerate(cls_scores_per_level):
        cls_max_map = _to_2d_map(cls_score, reduce_mode='max')
        _save_heatmap(
            cls_max_map,
            os.path.join(heatmap_dir, f'cls_score_max_lvl{lvl_idx}.png'),
            cmap=cmap)

    for lvl_idx, variance in enumerate(variance_per_level):
        var_mean_map = _to_2d_map(variance, reduce_mode='mean')
        _save_heatmap(
            var_mean_map,
            os.path.join(heatmap_dir, f'variance_mean_lvl{lvl_idx}.png'),
            cmap=cmap)

        if save_variance_channels and variance.dim() == 3:
            for ch in range(variance.size(0)):
                _save_heatmap(
                    variance[ch],
                    os.path.join(
                        heatmap_dir, f'variance_c{ch}_lvl{lvl_idx}.png'),
                    cmap=cmap)

    return True


def dump_vpd_heatmaps_from_case_dir(case_dir,
                                    cmap='turbo',
                                    save_variance_channels=True):
    """Load saved tensors from case directory and generate heatmaps."""
    cls_path = os.path.join(case_dir, 'cls_scores.pt')
    cent_path = os.path.join(case_dir, 'centerness.pt')
    var_path = os.path.join(case_dir, 'variance.pt')

    if not (os.path.isfile(cls_path) and os.path.isfile(cent_path)
            and os.path.isfile(var_path)):
        return False

    cls_scores_per_level = torch.load(cls_path, map_location='cpu')
    centerness_per_level = torch.load(cent_path, map_location='cpu')
    variance_per_level = torch.load(var_path, map_location='cpu')

    return dump_vpd_heatmaps(
        case_dir=case_dir,
        cls_scores_per_level=cls_scores_per_level,
        centerness_per_level=centerness_per_level,
        variance_per_level=variance_per_level,
        cmap=cmap,
        save_variance_channels=save_variance_channels)


def find_vpd_case_dirs(input_dir):
    """Find VPD case dirs under input directory.

    Returns:
        list[str]: Sorted list of directories that contain required pt files.
    """
    required = {'cls_scores.pt', 'centerness.pt', 'variance.pt'}

    if os.path.isdir(input_dir):
        files = set(os.listdir(input_dir))
        if required.issubset(files):
            return [input_dir]

    case_dirs = []
    for entry in sorted(os.listdir(input_dir)):
        case_dir = os.path.join(input_dir, entry)
        if not os.path.isdir(case_dir):
            continue
        files = set(os.listdir(case_dir))
        if required.issubset(files):
            case_dirs.append(case_dir)
    return case_dirs