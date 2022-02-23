# data settings
_base_ = [
    '../../_base_/data.py'
]

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

img_norm_cfg = dict(
    mean=(103.53, 116.28, 123.675), std=(1.0, 1.0, 1.0), to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='InstaBoostMPA',
        action_candidate=('normal', 'horizontal', 'skip'),
        action_prob=(1, 0, 0),
        scale=(0.8, 1.2),
        dx=0.5,
        dy=0.5,
        theta=(-1, 1),
        color_prob=0.5,
        hflag=False,
        aug_ratio=0.5,
        resize_scale=(608,608),
        max_instance_num=9),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='Resize',
          img_scale=[(352, 352), (352, 384), (384, 352), (384, 384), (384, 416),
                   (416, 384), (416, 416), (416, 448), (448, 416), (448, 448),
                   (448, 480), (480, 448), (480, 480), (480, 512), (512, 480),
                   (512, 512), (512, 544), (544, 512), (544, 544), (544, 576),
                   (576, 544), (576, 576), (576, 608), (608, 576), (608, 608)],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 480),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
                type=dataset_type,
                ann_file='data/coco/annotations/instances_train2017.json',
                img_prefix='data/coco/train2017/',
                pipeline=train_pipeline,
                classes=[])),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline)
)
