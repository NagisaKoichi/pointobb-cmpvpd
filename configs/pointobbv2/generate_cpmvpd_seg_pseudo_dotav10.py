_base_ = ['./train_cpm_vpd_point_dotav10.py']

# Stage-2 pseudo-label generation by GT-guided segmentation masks.
# It keeps CPMVPD outputs (mu/log_sigma) and writes DOTA txt annotations.
model = dict(
    bbox_head=dict(type='CPMVPDSegPseudoHead'),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        nms=dict(iou_thr=0.1),
        max_per_img=5000))
