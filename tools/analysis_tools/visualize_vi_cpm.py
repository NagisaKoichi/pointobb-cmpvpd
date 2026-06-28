# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmrotate.datasets import build_dataset
from mmdet.datasets import build_dataloader
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, get_device

# 注意：不再手动 import cpm_vi_head，依赖环境安装

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize VI Head outputs (Mu, Kappa, Centerness)')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument(
        '--out-dir', required=True, help='directory to save visualization results')
    parser.add_argument(
        '--split', default='train', choices=['train', 'val', 'test'],
        help='which data split in config to run inference on')
    parser.add_argument(
        '--feat-level', type=int, default=0,
        help='FPN level index used for visualization (0=P3/stride8)')
    parser.add_argument(
        '--max-images', type=int, default=0,
        help='max number of images to process, 0 means all')
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='single gpu id for inference')
    args = parser.parse_args()
    return args

def _to_heatmap(data, flip_direction, cmap='jet', vmin=None, vmax=None, is_kappa=False):
    """Convert 2D tensor to heatmap image."""
    if is_kappa:
        data = np.clip(data, 0, 20) # Clamp kappa for visualization
        
    data_np = data.detach().cpu().numpy().astype(np.float32)
    
    if vmin is None: vmin = data_np.min()
    if vmax is None: vmax = data_np.max()
    
    # Normalize
    norm = (data_np - vmin) / (vmax - vmin + 1e-6)
    norm = np.clip(norm, 0, 1)
    
    # Apply colormap
    cmap_fn = plt.get_cmap(cmap)
    heatmap = cmap_fn(norm)
    heatmap_uint8 = (heatmap[:, :, :3] * 255).astype(np.uint8)
    
    img = Image.fromarray(heatmap_uint8, mode='RGB')
    
    # Handle flip
    if flip_direction == 'horizontal':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_direction == 'vertical':
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif flip_direction == 'diagonal':
        img = img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
        
    return img

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    
    # Set distributed settings (single GPU here)
    cfg.model.pretrained = None
    cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    
    # Build Dataset
    dataset_cfg = cfg.data[args.split]
    dataset_cfg.test_mode = (args.split != 'train')
    dataset = build_dataset(dataset_cfg)
    
    # Build DataLoader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
        dist=False,
        shuffle=False
    )
    
    # Build Model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model = model.to(cfg.device)
    model.eval()
    
    # Hook to capture outputs
    captured = {}
    def _hook_fn(module, inputs, outputs):
        # CPMVIHead forward returns: 
        # (cls_scores, bbox_preds, angle_preds, centernesses, vi_preds)
        if not isinstance(outputs, (list, tuple)):
            raise RuntimeError('Unexpected bbox_head output type.')
            
        print(f"Output length: {len(outputs)}")
        
        # 根据输出长度自动解析
        # 兼容不同版本：标准FCOS返回3个，CPM返回4个，CPMVI返回5个
        if len(outputs) == 5:
            cls_scores, bbox_preds, angle_preds, centernesses, vi_preds = outputs
            captured['vi_preds'] = vi_preds
        elif len(outputs) == 4:
            cls_scores, bbox_preds, centernesses, _ = outputs
            captured['vi_preds'] = None # No VI output
        elif len(outputs) == 3:
            cls_scores, bbox_preds, centernesses = outputs
            captured['vi_preds'] = None
            
        captured['cls_scores'] = cls_scores
        captured['centernesses'] = centernesses
        captured['bbox_preds'] = bbox_preds # Sometimes needed for reference
        
        # Debug print
        if captured['vi_preds'] is not None:
            print(f"VI pred shape: {captured['vi_preds'][0].shape}")
            
    hook_handle = model.bbox_head.register_forward_hook(_hook_fn)
    
    mmcv.mkdir_or_exist(args.out_dir)
    
    progress = mmcv.ProgressBar(len(dataset))
    max_images = args.max_images if args.max_images > 0 else len(dataset)
    processed = 0
    
    for data in data_loader:
        if processed >= max_images:
            break
        
        captured.clear()
        
        # Forward
        with torch.no_grad():
            img = data['img'].data[0].to(cfg.device)
            # 只需要提取特征并通过 head，不需要完整的 inference_detector
            feats = model.extract_feat(img)
            # 触发 hook
            _ = model.bbox_head(feats)
            
        img_metas = data['img_metas'].data[0]
        
        # Check capture
        if 'vi_preds' not in captured or captured['vi_preds'] is None:
            print("Warning: vi_preds not captured. Check head type.")
            continue
            
        # Process single image (Batch size = 1 usually)
        meta = img_metas[0]
        img_path = meta['filename']
        flip_direction = meta.get('flip_direction')
        
        # Extract data for the chosen level
        lvl = args.feat_level
        vi_pred_lvl = captured['vi_preds'][lvl][0] # (2, H, W)
        cls_score_lvl = captured['cls_scores'][lvl][0] # (C, H, W)
        centerness_lvl = captured['centernesses'][lvl][0] # (1, H, W)
        
        # 1. VI Mu & Kappa
        # Channel 0: mu_logit, Channel 1: log_kappa (Assuming CPMVIHead definition)
        mu_logit = vi_pred_lvl[0]
        log_kappa = vi_pred_lvl[1]
        
        mu_prob = torch.sigmoid(mu_logit)
        kappa_val = torch.exp(log_kappa)
        
        # 2. CPM Centerness
        cpm_prob = torch.sigmoid(centerness_lvl[0])
        
        # 3. Classification Score (Max prob)
        max_cls_prob = torch.sigmoid(cls_score_lvl).max(dim=0)[0]
        
        # Resize to original image shape for better visualization
        # infer shape from meta
        img_h, img_w = meta['img_shape'][:2]
        
        # Upsample function
        def upsample(t, h, w):
            return F.interpolate(t.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze()
            
        mu_up = upsample(mu_prob, img_h, img_w)
        kappa_up = upsample(kappa_val, img_h, img_w)
        cpm_up = upsample(cpm_prob, img_h, img_w)
        cls_up = upsample(max_cls_prob, img_h, img_w)
        
        # Generate heatmaps
        img_ori = Image.open(img_path).convert('RGB').resize((img_w, img_h))
        
        vis_mu = _to_heatmap(mu_up, flip_direction, cmap='jet', vmin=0, vmax=1)
        vis_kappa = _to_heatmap(kappa_up, flip_direction, cmap='hot', is_kappa=True)
        vis_cpm = _to_heatmap(cpm_up, flip_direction, cmap='jet', vmin=0, vmax=1)
        vis_cls = _to_heatmap(cls_up, flip_direction, cmap='jet', vmin=0, vmax=1)
        
        # Concat: Row 1: Original | Mu | Kappa
        # Row 2: CPM | Cls Score | (Empty or Combined)
        row1_w = img_w * 3
        row2_w = img_w * 3
        total_h = img_h * 2
        
        merged = Image.new('RGB', (row1_w, total_h))
        merged.paste(img_ori, (0, 0))
        merged.paste(vis_mu, (img_w, 0))
        merged.paste(vis_kappa, (img_w * 2, 0))
        
        merged.paste(vis_cpm, (0, img_h))
        merged.paste(vis_cls, (img_w, img_h))
        # Paste a blank or combined map at last slot if needed
        
        # Save
        stem = osp.splitext(osp.basename(img_path))[0]
        out_name = f'{processed:06d}_{stem}_vi_vis.jpg'
        save_path = osp.join(args.out_dir, out_name)
        merged.save(save_path)
        
        processed += 1
        progress.update()
        
    hook_handle.remove()
    print(f'\nProcessed {processed} images. Results saved to: {args.out_dir}')

if __name__ == '__main__':
    main()
