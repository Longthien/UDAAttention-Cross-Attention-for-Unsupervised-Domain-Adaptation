# Obtained from https://github.com/lhoyer/DAFormer.git

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/udaneck_dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0

model = dict(
    neck=dict(
        type='LinearCrossDomainAttNeck',
        rescale=0.5,
        out_cat_and_conv=False,
        hybrid_route=False,
        n_points=12,
        sr_ratios=(8, 4, 2, 1),
        in_channels=[64, 128, 320, 512],
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='GELU')),
)
# Modifications to Basic UDA
uda = dict(
    type='TADA_DACS',
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    nce_loss_weight=0.1,
    nce_loss_temp = 0.07,
    nce_active_classes = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15], # majority + medium classes
    # nce_active_classes = [0, 1, 2, 3, 8, 9, 10, 13], # only majority classes
)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            neck=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
# Meta Information for Result Analysis
name = 'tada_nce0.1_majorclass_baseline_mitb5_40k'
exp = 'tada'
name_dataset = 'gta2cityscapes'
name_architecture = 'daformer_sepaspp_mitb5_crossattn'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_rcs0.01_cpl_crossattn'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
