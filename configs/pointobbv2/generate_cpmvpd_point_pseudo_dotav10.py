_base_ = ['./train_cpm_vpd_point_dotav10.py']

# Use the 8-channel CPMVPD branch for point-supervised pseudo generation.
model = dict(
    bbox_head=dict(
        type='CPMVPDPseudoHead',
        point_search_radius=3,
        use_lstd_for_size=False,
        lstd_size_factor=0.0,
        use_remap_size=True,
        remap_edge_thr_ratio=0.35,
        remap_edge_max_len=64,
        remap_size_mix=1.0,
        enable_final_nms=False,
        class_agnostic_nms=False,
        class_agnostic_iou_thr=0.1),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        class_agnostic_iou_thr=0.1,
        max_per_img=5000,
        use_remap_score=False))
