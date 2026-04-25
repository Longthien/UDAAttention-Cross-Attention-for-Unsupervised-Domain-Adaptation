# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture with MIT-B3
    '../_base_/models/daformer_sepaspp_mitb3.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]

seed = 0

model = dict(
    neck=dict(
        type='DeformableCrossDomainAttNeck',
        rescale=0.2,
        key_query_num_convs=2,
        out_cat_and_conv=True,
        # Optional: explicitly define in_channels for maing stage sizes
        in_channels=[64, 128, 320, 512],
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU')),
)

uda = dict(
    type='UDANeck_DACS',
    alpha=0.999,
    # imnet_feature_dist_lambda=0.005,
    # imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    # imnet_feature_dist_scale_min_ratio=0.75,
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))

optimizer_config = None
optimizer = dict(
    lr=2e-04,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=5.0),
            neck=dict(lr_mult=5.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')

name = 'udaneck_daformer_gta2cs_uda_warm_fdthings_rcs_croppl_a999_mitb3_deformable'
exp = 'UDANeck-DAFormer-MitB3-Deformable'
name_dataset = 'gta2cityscapes'
name_architecture = 'daformer_sepaspp_mitb3'
name_encoder = 'mitb3'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_2e-04_pmTrue_poly10warm_1x2_40k'
