_base_ = [
    './train_cpm_vpd_point_dotav10.py'
]

# Stable recipe for 4-GPU training (recommended start point).
# Keep these paths aligned with your local dataset/output directories.
data_root = 'E:/Programs/PointOBB-v2-master/DOTAv10_split_ss/'
store_dir = 'E:/Programs/PointOBB-v2-master/exps/cpm_vpd_point_dotav10_stable4gpu/'

# Global batch = 4 GPUs x 2 imgs/gpu = 8
# Use a conservative LR for VPD stability before scaling up.
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/'),
    val=dict(
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/'),
    test=dict(
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/',
        samples_per_gpu=2))

model = dict(
    train_cfg=dict(
        store_dir=store_dir,
        cls_weight=8,
        thresh1=8,
        alpha=1,
        use_point_supervised=True,
        js_weight=1.0,
        log_std_min=-7.0,
        log_std_max=5.0),
    test_cfg=dict(
        store_dir=store_dir,
        num_samples=6,
        use_refinement=True,
        log_std_min=-7.0,
        log_std_max=5.0,
        nms_pre=2000,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

# Stronger warmup helps prevent early-iteration divergence in VPD heads.
optimizer = dict(lr=0.02)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=8)

lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    step=[6])

evaluation = dict(interval=2, metric='mAP')
find_unused_parameters = True
