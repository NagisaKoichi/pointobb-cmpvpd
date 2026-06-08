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

data_root = '/media/ps/passport2/zlk/datasets/DOTAv10_split_ss/'

store_dir = '/media/ps/passport2/zlk/PointOBB-v2/exps/exp1/cpm_vpd_point_dotav10/'

pseudo_dir = '/media/ps/passport2/zlk/results/0601_xy_vpdstyle_clsw1_l5e-2_lkl0p5_onlylr/vpd_cpm_dotav10/pseudo_labels_cpm/'

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
        # ann_file=data_root + 'trainval/annfiles/',
        ann_file=pseudo_dir,
        img_prefix=data_root + 'trainval/images/',
        version=angle_version,
        classes=classes),
    val=dict(
        # ann_file=data_root + 'trainval/annfiles/',
        ann_file=pseudo_dir,
        img_prefix=data_root + 'trainval/images/',
        version=angle_version,
        classes=classes),
    test=dict(
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/',
        version=angle_version,
        classes=classes,
        samples_per_gpu=1))