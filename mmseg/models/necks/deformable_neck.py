import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from timm.models.layers import DropPath
from ..builder import NECKS

from mmcv.cnn import ConvModule, build_norm_layer
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from mmcv.utils import print_log

class SelfAttentionBlock(_SelfAttentionBlock):
    # [Your exact baseline SelfAttentionBlock goes here - unchanged]
    def __init__(self, in_channels, channels, conv_cfg, norm_cfg, act_cfg, key_query_num_convs=2):
        super(SelfAttentionBlock, self).__init__(
            key_in_channels=in_channels, query_in_channels=in_channels, channels=channels,
            out_channels=in_channels, share_key_query=False, query_downsample=None,
            key_downsample=None, key_query_num_convs=key_query_num_convs, key_query_norm=True,
            value_out_num_convs=1, value_out_norm=False, matmul_norm=True, with_out=False,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.output_project = self.build_project(
            in_channels, in_channels, num_convs=1, use_conv_module=True,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, query, key):
        context = super(SelfAttentionBlock, self).forward(query, key)
        return self.output_project(context)


class DeformableCrossAttnBlock(nn.Module):
    """
    Native PyTorch Deformable Cross-Attention.
    Computes offsets and weights from Source Query, and samples from Target Key/Value.
    """
    def __init__(self, in_channels, n_heads=4, n_points=4, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_head = in_channels // n_heads

        # Predicts both (x,y) offsets and attention weights directly from Query
        # Output channels = 3 * heads * points (2 for offset, 1 for weight)
        self.offset_weight_net = nn.Conv2d(in_channels, n_heads * n_points * 3, kernel_size=3, padding=1)
        
        # Initialize offsets to 0 so the model starts by looking at the exact spatial counterpart
        nn.init.constant_(self.offset_weight_net.weight, 0.)
        nn.init.constant_(self.offset_weight_net.bias, 0.)

        # Output projection back to in_channels
        self.proj = ConvModule(
            in_channels, in_channels, kernel_size=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, query, key):
        # query: Source features [B, C, H, W]
        # key: Target features [B, C, H, W]
        B, C, H, W = query.shape

        # 1. Predict offsets and weights
        out = self.offset_weight_net(query) 
        out = out.view(B, self.n_heads, self.n_points, 3, H, W)

        offsets = out[:, :, :, :2, :, :] # [B, heads, points, 2, H, W]
        weights = out[:, :, :, 2, :, :]  # [B, heads, points, H, W]
        weights = F.softmax(weights, dim=2) # Softmax over the K points

        # 2. Create normalized reference grid [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=query.device),
            torch.linspace(-1, 1, W, device=query.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, 1, H, W, 2)

        # 3. Compute sampling locations
        # Scale offsets by 0.1 to prevent chaotic jumps in early iterations
        offsets = offsets.permute(0, 1, 2, 4, 5, 3)  # [B, heads, pts, H, W, 2]
        sampling_locs = grid + offsets * 0.1  # [B, heads, pts, H, W, 2]

        # 4. Bilinear Sampling from Target per point
        sampled_list = []
        key_reshaped = key.view(B, self.n_heads, self.d_head, H, W).reshape(B * self.n_heads, self.d_head, H, W)
        for k in range(self.n_points):
            locs_k = sampling_locs[:, :, k, :, :, :]  # [B, heads, H, W, 2]
            locs_k = locs_k.reshape(B * self.n_heads, H, W, 2)
            s = F.grid_sample(key_reshaped, locs_k, mode='bilinear', align_corners=True)
            sampled_list.append(s)  # [B*heads, d_head, H, W]
        sampled = torch.stack(sampled_list, dim=1)  # [B*heads, pts, d_head, H, W]
        sampled = sampled.view(B, self.n_heads, self.n_points, self.d_head, H, W)
        sampled_feat = sampled.permute(0, 1, 3, 4, 5, 2)  # [B, heads, d_head, H, W, points]

        # 5. Apply attention weights and sum
        weights = weights.permute(0, 1, 3, 4, 2).unsqueeze(2)  # [B, heads, 1, H, W, points]
        out_feat = (sampled_feat * weights).sum(dim=-1)  # [B, heads, d_head, H, W]
        out_feat = out_feat.reshape(B, C, H, W)

        return self.proj(out_feat)


class UdaAttentionBlock(BaseModule):
    def __init__(self, channels, isa_channels, key_query_num_convs=2, rescale=0.5,
                 out_cat_and_conv=False, conv_cfg=None, norm_cfg=None, act_cfg=None, drop_path=0.,
                 use_deformable=False, detach_target=False, n_points=4):
        super(UdaAttentionBlock, self).__init__()
        
        self.rescale = nn.Parameter(torch.full((1, channels, 1, 1), float(rescale)))
        self.out_cat_and_conv = out_cat_and_conv
        self.detach_target = detach_target

        # Self-Attention for Source and Target
        self.encoder = SelfAttentionBlock(channels, isa_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, key_query_num_convs=key_query_num_convs)
        self.decoder = SelfAttentionBlock(channels, isa_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, key_query_num_convs=key_query_num_convs)
        
        # Cross-Attention: Hybrid Routing
        if use_deformable:
            # Use Deformable Cross-Attention for High-Res Stages
            self.mix_att = DeformableCrossAttnBlock(
                in_channels=channels,
                n_heads=max(1, channels // 32), # Dynamic head scaling based on channel depth
                n_points=n_points,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
            )
        else:
            # Use Standard SelfAttentionBlock for Low-Res Stages
            self.mix_att = SelfAttentionBlock(channels, isa_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, key_query_num_convs=key_query_num_convs)

        if out_cat_and_conv:
            self.out_conv = ConvModule(channels * 2, channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.out_norm = build_norm_layer(norm_cfg, channels)[1] if norm_cfg else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        num_images, C, H, W = x.shape
        if num_images == 1:
            # For inference/debug with single image, treat as both source and target
            x_src = x
            x_tgt = x.clone()
        else:
            if num_images % 2 != 0:
                raise ValueError(f'UdaAttentionBlock expects an even number (2B), but got {num_images}.')
            half = num_images // 2
            x_src = x[:half]
            x_tgt = x[half:]

        # 1. Intra-domain processing
        enc_out = self.encoder(x_src, x_src)
        tgt_input = x_tgt.detach() if self.detach_target else x_tgt
        dec_out = self.decoder(tgt_input, tgt_input) * self.rescale

        # 2. Cross-domain attention (Deformable or Standard based on initialization)
        cross_out = self.mix_att(enc_out, dec_out)

        # 4. Feature merging
        if self.out_cat_and_conv:
            out = self.out_conv(torch.cat([self.drop_path(cross_out), enc_out], dim=1))
        else:
            out = enc_out + self.drop_path(cross_out)
            out = self.out_norm(out)

        if self.training and random.random() < 0.005:
            print_log(f"Encode(abs): {torch.abs(enc_out).mean():.4f}, Decode(abs): {torch.abs(dec_out).mean():.4f}, Cross(abs): {torch.abs(cross_out).mean():.4f}, Out(abs): {torch.abs(out).mean():.4f}", 'mmseg')
            print_log(f"Feat shape: {out.shape}", 'mmseg')

        return out, dec_out

@NECKS.register_module()
class DeformableCrossDomainAttNeck(BaseModule):
    def __init__(self, in_channels, isa_channels=None, rescale=0.5, key_query_num_convs=2, out_cat_and_conv=False,
                 conv_cfg=None, norm_cfg=None, act_cfg=None, n_points=4, **kwargs):
        super(DeformableCrossDomainAttNeck, self).__init__()

        self.in_channels = in_channels
        self.isa_channels = isa_channels or [c * 2 for c in in_channels]
        self.uda_blocks = nn.ModuleList()

        for i in range(len(in_channels)):
            # Hybrid Architecture Logic: 
            # i=0 (Stage 1) and i=1 (Stage 2) trigger Deformable and Stop-Gradient
            use_deformable = (i < 2) 
            detach_target = (i < 2)

            self.uda_blocks.append(
                UdaAttentionBlock(
                    channels=in_channels[i],
                    isa_channels=self.isa_channels[i],
                    key_query_num_convs=key_query_num_convs,
                    out_cat_and_conv=out_cat_and_conv,
                    rescale=rescale,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    drop_path=0.1,
                    use_deformable=use_deformable,
                    detach_target=detach_target,
                    n_points=n_points
                )
            )
        
    def forward(self, inputs):
        """
        inputs: list of feature maps from backbone [Stage1, Stage2, Stage3, Stage4]
        each tensor: [2B, C_i, H_i, W_i]
        """
        outputs = []
        self.target_features = []
        for i, block in enumerate(self.uda_blocks):
            out, dec_out = block(inputs[i])
            outputs.append(out)
            self.target_features.append(dec_out)

        return outputs