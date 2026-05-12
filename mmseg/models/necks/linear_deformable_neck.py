import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from timm.models.layers import DropPath
from ..builder import NECKS

from mmcv.cnn import ConvModule, build_norm_layer
from ..utils import SelfSRAttentionBlock, LinearDeformableCrossAttnBlock, SelfSRAttention, FeedForward

from mmcv.utils import print_log

class UdaAttentionBlock(BaseModule):
    def __init__(self, channels, rescale=0.5,
                 out_cat_and_conv=False, conv_cfg=None, norm_cfg=None, act_cfg=None, drop_path=0.,
                 use_deformable=False, detach_target=False, n_points=4, sr_ratio=1):
        super(UdaAttentionBlock, self).__init__()
        
        self.rescale = nn.Parameter(torch.full((1, channels, 1, 1), float(rescale)))
        self.out_cat_and_conv = out_cat_and_conv
        self.detach_target = detach_target

        # Number of heads for attention mechanisms
        num_heads = max(1, channels // 32)

        # Self-Attention for Source and Target
        self.encoder = SelfSRAttention(dim=channels, num_heads=num_heads, 
                                                sr_ratio=sr_ratio, 
                                                qkv_bias=True)
        self.decoder = SelfSRAttentionBlock(dim=channels, num_heads=num_heads, 
                                                sr_ratio=sr_ratio, drop_path=drop_path,
                                                qkv_bias=True)
        self.norm_e = build_norm_layer(norm_cfg, channels)[1] if norm_cfg else nn.Identity()
        self.norm_d = build_norm_layer(norm_cfg, channels)[1] if norm_cfg else nn.Identity()
        # Cross-Attention: Hybrid Routing
        if use_deformable:
            # Use Deformable Cross-Attention for High-Res Stages
            self.mix_att = LinearDeformableCrossAttnBlock(
                in_channels=channels,
                n_heads=num_heads, # Dynamic head scaling based on channel depth
                n_points=n_points
            )
        else:
            # Use Standard SelfAttentionBlock for Low-Res Stages
            self.mix_att = SelfSRAttention(dim=channels, num_heads=num_heads, 
                                                sr_ratio=sr_ratio, 
                                                qkv_bias=True)
        self.mix_ffn = FeedForward(dim=channels, mlp_ratio=4., drop=drop_path, drop_path=drop_path)
        self.ffn_norm = build_norm_layer(norm_cfg, channels)[1] if norm_cfg else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if out_cat_and_conv:
            self.out_conv = ConvModule(channels * 2, channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.out_norm = build_norm_layer(norm_cfg, channels)[1] if norm_cfg else nn.Identity()

    def forward(self, x, ema_tgt_feat=None):
        x_src, x_tgt = self.parse_inputs(x, ema_tgt_feat)

        # 1. Intra-domain processing
        enc_out = x_src + self.drop_path(self.encoder(x_src, x_src))

        rescale = self.rescale.clamp(0.1, 2.0)
        dec_out = self.decoder(x_tgt, x_tgt) * rescale

        # 2. Cross-domain attention (Deformable or Standard based on initialization)
        # enc_out = self._norm_4d(enc_out, self.norm_e)
        # dec_out = self._norm_4d(dec_out, self.norm_d)
        enc_out = self.norm_e(enc_out)
        dec_out = self.norm_d(dec_out)
        cross_out = self.mix_att(enc_out, dec_out)

        # 4. Feature merging
        if self.out_cat_and_conv:
            out = self.out_conv(torch.cat([self.drop_path(cross_out), enc_out], dim=1))
        else:
            out = enc_out + self.out_norm(cross_out) 
        
        out = self.mix_ffn(self.ffn_norm(out))

        # if self.training and random.random() < 0.005:
        #     print_log(f"Encode(abs): {torch.abs(enc_out).mean():.4f}, Decode(abs): {torch.abs(dec_out).mean():.4f}, Cross(abs): {torch.abs(cross_out).mean():.4f}, Out(abs): {torch.abs(out).mean():.4f}", 'mmseg')
        #     print_log(f"Feat shape: {out.shape}", 'mmseg')

        return out, dec_out
    def _norm_4d(self, x, norm):
        B, C, H, W = x.shape
        return norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)
    def parse_inputs(self, x, ema_tgt_feat=None):
        num_images, C, H, W = x.shape
        if ema_tgt_feat is not None:
            x_src, x_tgt = x, ema_tgt_feat
        elif num_images == 1:
            x_src, x_tgt = x, x.clone() 
        else:
            if num_images % 2 != 0:
                raise ValueError(f'UdaAttentionBlock expects an even number (2B), but got {num_images}.')
            half = num_images // 2
            x_src, x_tgt = x[:half], x[half:]
        x_tgt = x_tgt.detach().clone() if self.detach_target else x_tgt.clone()
        
        return x_src, x_tgt

@NECKS.register_module()
class LinearCrossDomainAttNeck(BaseModule):
    def __init__(self, in_channels, rescale=0.5, out_cat_and_conv=False, hybrid_route=True,
                 conv_cfg=None, norm_cfg=None, act_cfg=None, drop_path=0.1, n_points=4, sr_ratios=(8, 4, 2, 1), **kwargs):
        super(LinearCrossDomainAttNeck, self).__init__()

        self.in_channels = in_channels
        self.uda_blocks = nn.ModuleList()

        for i in range(len(in_channels)):
            # Hybrid Architecture Logic: 
            # i=0 (Stage 1) and i=1 (Stage 2) trigger Deformable and Stop-Gradient
            use_deformable = (i < 2) and hybrid_route
            detach_target = True
            sr_ratio = sr_ratios[i] if i < len(sr_ratios) else 1

            self.uda_blocks.append(
                UdaAttentionBlock(
                    channels=in_channels[i],
                    out_cat_and_conv=out_cat_and_conv,
                    rescale=rescale,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    drop_path=drop_path,
                    use_deformable=use_deformable,
                    detach_target=detach_target,
                    n_points=n_points,
                    sr_ratio=sr_ratio
                )
            )
        
    def forward(self, inputs):
        """
        inputs: list of feature maps from backbone [Stage1, Stage2, Stage3, Stage4]
        each tensor: [2B, C_i, H_i, W_i]
        """
        outputs = []
        self.target_features = []
        has_ema_feat = hasattr(self, 'ema_target_features') and self.ema_target_features is not None

        for i, block in enumerate(self.uda_blocks):
            ema_tgt_feat = self.ema_target_features[i] if has_ema_feat else None
            out, dec_out = block(inputs[i], ema_tgt_feat=ema_tgt_feat)
            outputs.append(out)
            self.target_features.append(dec_out)

        if has_ema_feat:
            # Clear to prevent memory leaks across iterations
            self.ema_target_features = None

        return outputs