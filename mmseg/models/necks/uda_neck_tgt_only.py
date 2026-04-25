import random

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from timm.models.layers import DropPath

from ..builder import NECKS
from mmcv.cnn import ConvModule
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from mmcv.utils import print_log


class SelfAttentionBlock(_SelfAttentionBlock):
    """Self-Attention Module.

    Args:
        in_channels (int): Input channels of key/query feature.
        channels (int): Output channels of key/query transform.
        conv_cfg (dict | None): Config of conv layers.
        norm_cfg (dict | None): Config of norm layers.
        act_cfg (dict | None): Config of activation layers.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 conv_cfg,
                 norm_cfg,
                 act_cfg,
                 key_query_num_convs=2):
        super(SelfAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=key_query_num_convs,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=True,
            with_out=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.output_project = self.build_project(
            in_channels,
            in_channels,
            num_convs=1,
            use_conv_module=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, query, key):
        context = super(SelfAttentionBlock, self).forward(query, key)
        return self.output_project(context)


class TgtOnlyUdaAttentionBlock(BaseModule):
    def __init__(self,
                 channels,
                 isa_channels,
                 key_query_num_convs=2,
                 rescale=0.5,
                 out_cat_and_conv=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 drop_path=0.):
        super(TgtOnlyUdaAttentionBlock, self).__init__()

        self.decoder = SelfAttentionBlock(
            channels,
            isa_channels,
            key_query_num_convs=key_query_num_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mix_att = SelfAttentionBlock(
            channels,
            isa_channels,
            key_query_num_convs=key_query_num_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.rescale = rescale
        self.out_cat_and_conv = out_cat_and_conv
        if out_cat_and_conv:
            self.out_conv = ConvModule(
                channels * 2,
                channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        num_images, C, H, W = x.shape
        if num_images == 1:
            x_src = x
            x_tgt = x.clone()
        else:
            if num_images % 2 != 0:
                raise ValueError(
                    'TgtOnlyUdaAttentionBlock expects an even number of '
                    f'concatenated source/target features (2B), but got '
                    f'{num_images}.')
            half = num_images // 2
            x_src = x[:half]
            x_tgt = x[half:]

        dec_out = self.decoder(x_tgt, x_tgt) * self.rescale
        cross_out = self.mix_att(x_src, dec_out)

        if self.out_cat_and_conv:
            out = self.out_conv(torch.cat([cross_out, x_src], dim=1))
        else:
            out = x_src + cross_out

        out = self.drop_path(out)
        if self.training and random.random() < 0.005:
            print_log(
                f"Decode: {torch.mean(dec_out)}, {torch.std(dec_out)}, "
                f"Cross: {torch.mean(cross_out)}, {torch.std(cross_out)}",
                'mmseg')
            print_log(
                f"Out mean & std: {torch.mean(out)}, {torch.std(out)}, "
                f"Feat shape: {out.shape}",
                'mmseg')

        return out


@NECKS.register_module()
class TgtOnlyCrossDomainAttNeck(BaseModule):
    def __init__(
            self,
            in_channels,
            rescale=0.5,
            key_query_num_convs=2,
            out_cat_and_conv=False,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            **kwargs):
        super(TgtOnlyCrossDomainAttNeck, self).__init__()

        self.in_channels = in_channels
        self.uda_blocks = nn.ModuleList([
            TgtOnlyUdaAttentionBlock(
                channels=in_channels[i],
                isa_channels=in_channels[i] * 2,
                key_query_num_convs=key_query_num_convs,
                out_cat_and_conv=out_cat_and_conv,
                rescale=rescale,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop_path=0.1)
            for i in range(len(in_channels))
        ])

    def forward(self, inputs):
        """
        inputs: list of feature maps from backbone
        each tensor: [2B, C_i, H_i, W_i], with source first and target second.
        """
        return [block(inputs[i]) for i, block in enumerate(self.uda_blocks)]
