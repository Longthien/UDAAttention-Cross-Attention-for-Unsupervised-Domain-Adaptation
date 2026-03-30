# obtained from https://github.com/mlvlab/EfficientViM

# model settings
find_unused_parameters = True
_base_ = ['daformer_conv1_mitb5.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/EfficientViM_M4_e450.pth',
    backbone=dict(
        type='EfficientViM_M4',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/EfficientViM_M4_e450.pth',
        ),
        norm_cfg=norm_cfg,
        norm_eval=False,
        contract_dilation=True),
    neck=dict(
        type='CrossDomainAttNeck',
        in_channels=[224, 320, 512],
        isa_channels=256,
        key_query_num_convs=1,
        out_cat_and_conv=True,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),),   
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[224, 320, 512],
        in_index=[0, 1, 2],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg)
        ),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))