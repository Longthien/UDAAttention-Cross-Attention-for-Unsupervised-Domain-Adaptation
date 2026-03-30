# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/ours_model/segformer_efficientViM.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA
uda = dict(
    type='PartiallyFroozeEmaDacs',
    # Increased Alpha
    alpha=0.999,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,)
# Optimizer Hyperparameters
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
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
# Meta Information for Result Analysis
name = 'lr_2e_04-freeze_patchembed-stage1_block0_1-efficient_segformer_gta2cs_uda_warm_a999'
exp = 'EfficientViM-FroozeEMA-DACS'
name_dataset = 'gta2cityscapes'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
