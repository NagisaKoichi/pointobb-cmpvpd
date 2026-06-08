_base_ = ['./train_cpm_vpd_point_dotav10.py']

# Fused pseudo-label generation combining PCA-edge (Method 1, from
# PseudoLabelHead) and GT-guided segmentation (Method 2, from
# CPMVPDSegPseudoHead) approaches, with internal fusion following
# fuse_pseudo_labels.py logic.

model = dict(
    bbox_head=dict(
        type='CPMVPDFusedPseudoHead',
        # ---- Method 1 (PseudoLabelHead) direct parameters ----
        # cls_weight / thresh3 / pca_length / multiple_factor are
        # read from train_cfg below (same pattern as PseudoLabelHead)
        default_max_length=128,
        # ---- Method 2 (CPMVPDSegPseudoHead) parameters ----
        sigma_scale=0.2,
        min_sigma=0.7,
        max_sigma=20.0,
        seg_score_thr=0.05,
        seg_topk=0,
        bg_std_scale=1.0,
        per_gt_thr_ratio=0.5,
        mask_min_pixels=6,
        enable_final_nms=False,
        class_agnostic_nms=True,
        class_agnostic_iou_thr=0.1,
        cls_floor=0.7,
        cls_gamma=0.5,
        uncert_q_lo=0.01,
        uncert_q_hi=0.40,
        uncert_gamma=0.5,
        alpha_cls=0.66,
        alpha_uncert=0.33,
        prob_smooth_ksize=3,
        prob_local_contrast=0.30,
        # ---- Fusion parameters ----
        fuse_w1=1.0,
        fuse_w2=1.0,
        fuse_score_mode='first',
        float_format='{:.1f}',
    ),
    # Parameters read from train_cfg inside CPMVPDFusedPseudoHead.__init__
    # (same mechanism as PseudoLabelHead)
    train_cfg=dict(
        store_dir='fused_pseudo_label',
        cls_weight=1.0,
        thresh3=[0.03, 0.04, 0.1, 0.01, 0.10, 0.06, 0.08, 0.02,
                 0.01, 0.03, 0.005, 0.02, 0.05, 0.1, 0.015],
        pca_length=20,
        multiple_factor=1 / 4,
        store_ann_dir='your_fused_pseudo_label_dir',  # 修改为实际输出路径
    ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        nms=dict(iou_thr=0.1),
        max_per_img=5000,
    ))
