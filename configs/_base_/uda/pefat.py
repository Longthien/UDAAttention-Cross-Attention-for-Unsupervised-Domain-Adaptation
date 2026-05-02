uda = dict(
    type='PEFAT',
    alpha=0.999,
    warmup_iters=15000,
    pefat_k_steps=1,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    blur=True,
)
use_ddp_wrapper = True
