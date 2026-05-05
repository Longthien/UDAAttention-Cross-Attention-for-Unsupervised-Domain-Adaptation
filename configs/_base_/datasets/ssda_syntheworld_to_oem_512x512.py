
dataset_type = 'LoveDADataset'
data_root = '/home/Hung_Data/HungData/mmseg_data/Datasets/LoveDA/loveDA/loveDA_rural/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)  # follow base line of DAFormer
loveda_scale = (512, 512)
source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=loveda_scale),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
target_unlabel_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=loveda_scale),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
target_label_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=loveda_scale),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg',]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=loveda_scale,
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train_target_unlabel_dataset=dict(
    type='OpenEarthMapDataset',
    data_root='data/OpenEarthMap/OpenEarthMap_wo_xBD',
    img_dir='',
    ann_dir='',
    split='/home/ubuntu/workdirs/UDAAttention-Cross-Attention-for-Unsupervised-Domain-Adaptation/configs/split/train.txt',
    pipeline=target_unlabel_train_pipeline)

train_target_label_dataset=dict(
    type='OpenEarthMapDataset',
    data_root='data/OpenEarthMap/OpenEarthMap_wo_xBD',
    img_dir='',
    ann_dir='',
    split='/home/ubuntu/workdirs/UDAAttention-Cross-Attention-for-Unsupervised-Domain-Adaptation/configs/split/train_5percent.txt',
    pipeline=target_label_train_pipeline)

train_source_dataset=dict(
    type='SyntheWorldDataset',
    data_root='data/SyntheWorld',
    img_dir='images',
    ann_dir='ss_mask',
    pipeline=source_train_pipeline),
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='SSDADataset',
        source=train_source_dataset,
        target_unlabeled=train_target_unlabel_dataset,
        target_labeled=train_target_label_dataset,
    ),
    val=dict(
        type='OpenEarthMapDataset',
        data_root='data/OpenEarthMap/OpenEarthMap_wo_xBD',
        img_dir='',
        ann_dir='',
        split='/home/ubuntu/workdirs/UDAAttention-Cross-Attention-for-Unsupervised-Domain-Adaptation/configs/split/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type='OpenEarthMapDataset',
        data_root='data/OpenEarthMap/OpenEarthMap_wo_xBD',
        img_dir='',
        ann_dir='',
        split='/home/ubuntu/workdirs/UDAAttention-Cross-Attention-for-Unsupervised-Domain-Adaptation/configs/split/test.txt',
        pipeline=test_pipeline),
)