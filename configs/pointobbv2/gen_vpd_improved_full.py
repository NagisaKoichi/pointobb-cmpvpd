_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
data_root = '/media/ps/passport2/zlk/datasets/DOTAv10/split_ss_dota/'
store_dir = 'vpd_improved'
angle_version = 'le90'
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'], version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(pipeline=train_pipeline,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        version=angle_version, classes=classes),
    val=dict(ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        version=angle_version, classes=classes),
    test=dict(ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/',
        version=angle_version, classes=classes, samples_per_gpu=4))
model = dict(
    type='RotatedFCOS',
    backbone=dict(type='ResNet', depth=50, num_stages=4, out_indices=(0,1,2,3),
        frozen_stages=1, zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(type='FPN', in_channels=[256,512,1024,2048], out_channels=256,
        start_level=0, add_extra_convs='on_output', num_outs=6,
        relu_before_extra_convs=True),
    bbox_head=dict(type='VPDPseudoLabelHead', num_classes=len(classes),
        in_channels=256, stacked_convs=4, feat_channels=256,
        regress_ranges=((-1,32),(32,64),(64,128),(128,256),(256,512),(512,1e8)),
        strides=[4,8,16,32,64,128], center_sampling=True,
        center_sample_radius=1.5, norm_on_bbox=True, centerness_on_reg=True,
        separate_angle=False, scale_angle=True,
        bbox_coder=dict(type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(store_dir=store_dir, cls_weight=1.0,
        thresh3=[0.03,0.04,0.1,0.01,0.10,0.06,0.08,0.02,0.01,0.03,0.005,0.02,0.05,0.1,0.015],
        pca_length=28,
        store_ann_dir='/media/ps/passport2/zlk/results/frozen_vpd_v3/pseudo_labels_pca_07b_ep3',
        multiple_factor=1/4,
        sigma_power=1.0,
        sigma_spike_thresh=2.0,
        sigma_pca=True,
        mu_refine_radius=0,
        remap_use=True,
        remap_use_gt_guided=True,
        remap_sigma_scale=0.5,
        remap_min_sigma=1.0,
        remap_max_sigma=20.0,
        remap_score_thr=0.04,
        remap_topk=0,
        remap_bg_std_scale=1.5,
        remap_cls_floor=0.05,
        remap_cls_gamma=0.5,
        remap_uncert_q_lo=0.01,
        remap_uncert_q_hi=0.20,
        remap_uncert_gamma=0.7,
        remap_alpha_cls=0.5,
        remap_alpha_uncert=0.5,
        remap_prob_smooth_ksize=3,
        remap_prob_local_contrast=0.30),
    test_cfg=dict(store_dir=store_dir, nms_pre=2000, min_bbox_size=0,
        score_thr=0.05, nms=dict(iou_thr=0.1), max_per_img=2000))
find_unused_parameters = True
runner = dict(type='EpochBasedRunner', max_epochs=1)
lr_config = dict(_delete_=True, policy='step', warmup='linear',
    warmup_iters=500, warmup_ratio=1.0/3, step=[4])
evaluation = dict(interval=3, metric='mAP')
optimizer = dict(lr=0.0)
load_from = 'work_dirs/frozen_vpd_v2/epoch_3.pth'
