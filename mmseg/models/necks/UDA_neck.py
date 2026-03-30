import torch.nn as nn
import torch
import torch.nn.functional as F
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
        """Forward function."""
        context = super(SelfAttentionBlock, self).forward(query, key)
        return self.output_project(context)


class UdaAttentionBlock(BaseModule):
    def __init__(self,
                 channels,
                 isa_channels,
                 key_query_num_convs=2,
                 rescale=0.5,
                 out_cat_and_conv=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 drop_path=0.,):
        super(UdaAttentionBlock, self).__init__()
        

        self.encoder = SelfAttentionBlock(
            channels,
            isa_channels,
            key_query_num_convs=key_query_num_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
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

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        B, C, H, W = x.shape
        B = int(B /2)
        if B < 1:
            B= 1  # No source images, only target
            # return x  # No source images, return target as is
            x_src = x  # [B, C, H, W]
            x_tgt = x.clone()  # [B, C, H, W]
        else:
            # Split source and target
            x_src = x[:-B]  # [B, C, H, W]
            x_tgt = x[-B:]  # [B, C, H, W]
            
        # Flatten for attention
        # x_src_flat = x_src.flatten(2).transpose(1, 2)  # [B, HW, C]
        # x_tgt_flat = x_tgt.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # # Normalize
        # x_src_norm = self.norm1(x_src_flat)
        # x_tgt_norm = self.norm1(x_tgt_flat)
        
        # Encoder-Decoder Attention
        enc_out = self.encoder(x_src, x_src)  # [B, HW, C]
        dec_out = self.decoder(x_tgt, x_tgt)*self.rescale  # [B, HW, C]
        
        
        # # Normalize
        # enc_out = self.norm1(enc_out)
        # dec_out = self.norm1(dec_out)
        # Cross Attention
        cross_out = self.mix_att(enc_out, dec_out)  # [B, HW
        
        # cross_out = self.drop_path(cross_out)
        # cross_out = cross_out * self.rescale
        if self.out_cat_and_conv:
            out = self.out_conv(torch.cat([cross_out, enc_out], dim=1))
        else:
            out = enc_out + cross_out
        
        import random
        if random.random() < 0.005:
            print_log(f"Encode: {torch.mean(enc_out)}, {torch.std(enc_out)}, Decode: {torch.mean(dec_out)}, {torch.std(dec_out)},\
                Cross: {torch.mean(cross_out)}, {torch.std(cross_out)}", 'mmseg')
            
            print_log(f"Out mean & std: {torch.mean(out)}, {torch.std(out)}\
                , Feat shape: {out.shape}", 'mmseg')
        # out = self.norm2(out)
        # out = out.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        return out

@NECKS.register_module()
class CrossDomainAttNeck(BaseModule):
    def __init__(self, 
                 in_channels, 
                 rescale=0.5,
                 key_query_num_convs=2,
                out_cat_and_conv=False,
                conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 **kwargs):
        super(CrossDomainAttNeck, self).__init__()

        self.in_channels = in_channels
        self.uda_blocks = nn.ModuleList([
            UdaAttentionBlock(
                channels=in_channels[i],
                isa_channels=in_channels[i]*2,
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
        each tensor: [B+1, C_i, H_i, W_i]
        """
        outputs = []
        for i, block in enumerate(self.uda_blocks):
            outputs.append(block(inputs[i]))
        # B, C, H, W = inputs[-1].shape
        # B = int(B /2)
        # for i, feat in enumerate(inputs):
        #     # print(feat.shape)
        #     # print('index', i)
        #     # print(len(self.in_channels))
        #     if i == 0:
        #         outputs.append(self.uda_blocks(feat))
        #     elif B < 1:
        #         outputs.append(feat)
        #     else:
        #         outputs.append(feat[:-B])
        #     # print(outputs[-1].shape)

        return outputs