_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

data_root = '/media/passport2/zlk/datasets/DOTAv10_split_ss/'

store_dir = '/media/passport2/zlk/PointOBB-v2/work_dirs/frozen_vpd_v3/'

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
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        version=angle_version,
        classes=classes),
    val=dict(
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        version=angle_version,
        classes=classes),
    test=dict(
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/',
        version=angle_version,
        classes=classes,
        samples_per_gpu=4))

# model settings - CPM with Point-Supervised VPD (frozen backbone)
model = dict(
    type='RotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=6,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CPMVPDHead',
        num_classes=len(classes),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        regress_ranges=((-1, 32), (32, 64), (64, 128), (128, 256), (256, 512),
                                 (512, 1e8)),
        strides=[4, 8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        visualize=True,
        store_dir=store_dir,
        cls_weight=20,
        thresh1=8,
        alpha=1,
        use_point_supervised=True,
        js_weight=1.0,
        freeze_base=True,
        # Stronger sigma training:
        dense_radius=14,          # was 8 → covers pca_length=28/stride=4=7 stride units
        sigma_min_target=0.5,     # was 1.0 → finer resolution near center
        sigma_max_target=6.0,     # was 4.0 → allow higher sigma far from center
        lambda_sigma_dense=2.0,   # was 1.0 → emphasize sigma loss
        ambiguity_weight=2.0,     # keep same
        ),
    test_cfg=dict(
        store_dir=store_dir,
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

# Load baseline 6-epoch checkpoint
load_from = 'work_dirs/cpm_dotav10/epoch_6.pth'

find_unused_parameters = True

# 3 epochs for stronger sigma (was 1)
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=3)
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    step=[2])
evaluation = dict(interval=999, metric='mAP')  # skip eval, no meaningful result

# Only optimize sigma parameters, freeze everything else
optimizer = dict(
    _delete_=True,
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone.conv1': dict(lr_mult=0, decay_mult=0),
            'backbone.bn1': dict(lr_mult=0, decay_mult=0),
            'backbone.layer1': dict(lr_mult=0, decay_mult=0),
            'backbone.layer2': dict(lr_mult=0, decay_mult=0),
            'backbone.layer3': dict(lr_mult=0, decay_mult=0),
            'backbone.layer4': dict(lr_mult=0, decay_mult=0),
            'neck': dict(lr_mult=0, decay_mult=0),
            'bbox_head.cls_convs': dict(lr_mult=0, decay_mult=0),
            'bbox_head.conv_cls': dict(lr_mult=0, decay_mult=0),
            'bbox_head.reg_convs': dict(lr_mult=0, decay_mult=0),
            'bbox_head.conv_centerness': dict(lr_mult=0, decay_mult=0),
            'bbox_head.conv_angle': dict(lr_mult=0, decay_mult=0),
            'bbox_head.scales': dict(lr_mult=0, decay_mult=0),
            'bbox_head.scale_angle': dict(lr_mult=0, decay_mult=0),
        }))
