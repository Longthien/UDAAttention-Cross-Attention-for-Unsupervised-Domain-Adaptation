

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.backbones.mix_transformer import Mlp

from timm.models.layers import DropPath

class SelfSRAttention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads=8, 
                 qkv_bias=False, 
                 attn_drop=0., 
                 proj_drop=0., 
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        # Scaling factor: 1 / sqrt(d_k)
        self.scale = head_dim ** -0.5

        # Separate Query projection (for query_feats)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Unified Key/Value projection (for key_feats)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, query_feats, key_feats):
        """
        Args:
            query_feats: Tensor of shape [B, C, H_q, W_q]
            key_feats: Tensor of shape [B, C, H_k, W_k]
        """
        B, C, H_q, W_q = query_feats.shape
        
        N_q = H_q * W_q
        
        # ---------------------------------------------------------
        # 1. Process Query (Q)
        # Flatten: [B, C, H_q, W_q] -> [B, C, N_q] -> [B, N_q, C]
        # ---------------------------------------------------------
        q_flat = query_feats.flatten(2).transpose(1, 2)
        q = self.q(q_flat).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # ---------------------------------------------------------
        # 2. Process Key/Value (K, V) with Spatial Reduction
        # ---------------------------------------------------------
        if self.sr_ratio > 1:
            # Operate directly on the 2D feature map [B, C, H_k, W_k]
            x_kv = self.sr(key_feats)
            
            # Flatten reduced map: [B, C, H_k/R, W_k/R] -> [B, C, N_kv] -> [B, N_kv, C]
            x_kv_flat = x_kv.flatten(2).transpose(1, 2)
            
            # Apply LayerNorm to the transposed shape where channels are the last dimension
            x_kv_flat = self.norm(x_kv_flat)
        else:
            # Flatten directly if no spatial reduction
            x_kv_flat = key_feats.flatten(2).transpose(1, 2)
            
        N_kv = x_kv_flat.shape[1]
        
        # Project and split into K and V
        kv = self.kv(x_kv_flat).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        # ---------------------------------------------------------
        # 3. Scaled Dot-Product Attention
        # ---------------------------------------------------------
        # q: [B, num_heads, N_q, head_dim]
        # k.transpose: [B, num_heads, head_dim, N_kv]
        # attn: [B, num_heads, N_q, N_kv]
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # ---------------------------------------------------------
        # 4. Output Projection and Reshaping
        # ---------------------------------------------------------
        # attn @ v: [B, num_heads, N_q, head_dim]
        # Transpose/Reshape -> [B, N_q, C]
        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, N_q, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out
class SelfSRAttentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfSRAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, query, key):
        B, C, H, W = query.shape
        
        q_flat = query.flatten(2).transpose(1, 2)
        k_flat = key.flatten(2).transpose(1, 2)
        
        q_norm = self.norm1(q_flat).transpose(1, 2).reshape(B, C, H, W)
        k_norm = self.norm1(k_flat).transpose(1, 2).reshape(B, C, H, W)
        
        attn_out = self.attn(q_norm, k_norm)
        
        x = q_flat + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class LinearDeformableCrossAttn(nn.Module):
    """
    Native PyTorch Deformable Cross-Attention using Linear Projections.
    Computes offsets and weights from flattened Source Query tokens, 
    and samples from Target Key/Value feature maps.
    """
    def __init__(self, in_channels, n_heads=4, n_points=4):
        super().__init__()
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_head = in_channels // n_heads
        self.offset_weight_net = nn.Linear(in_channels, n_heads * n_points * 3)
        
        nn.init.constant_(self.offset_weight_net.weight, 0.)
        nn.init.constant_(self.offset_weight_net.bias, 0.)

        # Final Output projection
        self.proj = nn.Linear(in_channels, in_channels)
        self.out_norm = nn.LayerNorm(in_channels)

    def forward(self, query, key):
        # query: Source features [B, C, H, W]
        # key: Target features [B, C, H, W]
        B, C, H, W = query.shape
        N = H * W

        # Flatten query: [B, C, H, W] -> [B, C, N] -> [B, N, C]
        query_flat = query.flatten(2).transpose(1, 2)

        # Apply Linear Layer: [B, N, C] -> [B, N, heads * points * 3]
        out = self.offset_weight_net(query_flat) 
        
        # Reshape to easily extract offsets and weights
        # Shape: [B, H, W, heads, points, 3]
        out = out.view(B, H, W, self.n_heads, self.n_points, 3)

        offsets = out[..., :2] # [B, H, W, heads, points, 2]
        weights = out[..., 2]  # [B, H, W, heads, points]
        
        # Softmax over the K points
        weights = F.softmax(weights, dim=-1) # [B, H, W, heads, points]

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=query.device),
            torch.linspace(-1, 1, W, device=query.device),
            indexing='ij'
        )
        # Reshape grid for broadcasting: [1, H, W, 1, 1, 2]
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, H, W, 1, 1, 2)

        sampling_locs = grid + offsets * 0.1 # [B, H, W, heads, points, 2]

        sampling_locs = sampling_locs.permute(0, 3, 4, 1, 2, 5).contiguous()

        sampled_list = []
        key_reshaped = key.view(B, self.n_heads, self.d_head, H, W).reshape(B * self.n_heads, self.d_head, H, W)
        
        for k in range(self.n_points):
            locs_k = sampling_locs[:, :, k, :, :, :]  # [B, heads, H, W, 2]
            locs_k = locs_k.reshape(B * self.n_heads, H, W, 2)
            s = F.grid_sample(key_reshaped, locs_k, mode='bilinear', align_corners=True)
            sampled_list.append(s)  # [B*heads, d_head, H, W]
            
        sampled = torch.stack(sampled_list, dim=1)  # [B*heads, pts, d_head, H, W]
        sampled = sampled.view(B, self.n_heads, self.n_points, self.d_head, H, W)

        weights = weights.permute(0, 3, 4, 1, 2).unsqueeze(3)
        out_feat = (sampled * weights).sum(dim=2)  # [B, heads, d_head, H, W]
        out_feat = out_feat.reshape(B, C, H, W)

        out_feat_flat = out_feat.flatten(2).transpose(1, 2) # [B, N, C]
        out_proj = self.proj(out_feat_flat) # [B, N, C]

        return out_proj
        
class LinearDeformableCrossAttnBlock(nn.Module):
    def __init__(self, in_channels, n_heads=4, n_points=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(in_channels)
        self.attn = LinearDeformableCrossAttn(
            in_channels,
            n_heads=n_heads,
            n_points=n_points)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_channels)
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, query, key):
        B, C, H, W = query.shape
        
        q_flat = query.flatten(2).transpose(1, 2)
        k_flat = key.flatten(2).transpose(1, 2)
        
        q_norm = self.norm1(q_flat).transpose(1, 2).reshape(B, C, H, W)
        k_norm = self.norm1(k_flat).transpose(1, 2).reshape(B, C, H, W)
        
        attn_out = self.attn(q_norm, k_norm)
        
        x = q_flat + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

