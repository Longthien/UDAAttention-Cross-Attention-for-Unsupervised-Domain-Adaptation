# Obtained from https://github.com/lhoyer/DAFormer.git

# Baseline UDA
uda = dict(
    type='UDANeck_DACS',
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=2000,
    print_grad_magnitude=True,
)
use_ddp_wrapper = True
