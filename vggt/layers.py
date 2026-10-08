"""Transformer building blocks used by VGGT.

A from-scratch re-implementation of the layers described in the VGGT paper
(Sec. 3.3 / Appendix): pre-norm ViT blocks with LayerScale, QK-norm and 2D
rotary position embeddings (RoPE). Attribute names follow the official
release so that its checkpoints load with ``load_state_dict(strict=True)``.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class PositionGetter:
    """Returns the (y, x) grid coordinate of every patch, cached per grid size."""

    def __init__(self):
        self.cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size, height, width, device):
        if (height, width) not in self.cache:
            ys = torch.arange(height, device=device)
            xs = torch.arange(width, device=device)
            self.cache[height, width] = torch.cartesian_prod(ys, xs)
        pos = self.cache[height, width]
        return pos.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D RoPE: half of each head's channels is rotated by the y index, half by x."""

    def __init__(self, frequency: float = 100.0):
        super().__init__()
        self.base_frequency = frequency
        self.cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _cos_sin(self, dim, seq_len, device, dtype):
        key = (dim, seq_len, device, dtype)
        if key not in self.cache:
            inv_freq = 1.0 / (self.base_frequency ** (torch.arange(0, dim, 2, device=device).float() / dim))
            angles = torch.outer(torch.arange(seq_len, device=device, dtype=inv_freq.dtype), inv_freq).to(dtype)
            angles = torch.cat([angles, angles], dim=-1)
            self.cache[key] = (angles.cos(), angles.sin())
        return self.cache[key]

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _rope_1d(self, x, positions, cos, sin):
        cos = F.embedding(positions, cos)[:, None]
        sin = F.embedding(positions, sin)[:, None]
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x, positions):
        # x: (B, heads, N, head_dim); positions: (B, N, 2) integer (y, x)
        dim = x.size(-1) // 2
        cos, sin = self._cos_sin(dim, int(positions.max()) + 1, x.device, x.dtype)
        x_y, x_x = x.chunk(2, dim=-1)
        x_y = self._rope_1d(x_y, positions[..., 0], cos, sin)
        x_x = self._rope_1d(x_x, positions[..., 1], cos, sin)
        return torch.cat([x_y, x_x], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_bias=True, qk_norm=False, rope=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = rope

    def forward(self, x, pos=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    """Pre-norm transformer block: x + LS(Attn(LN(x))), x + LS(MLP(LN(x)))."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, proj_bias=True, ffn_bias=True,
                 init_values=None, qk_norm=False, rope=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, proj_bias, qk_norm, rope)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()

    def forward(self, x, pos=None):
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    """Plain conv patchifier: (B, 3, H, W) -> (B, H/p * W/p, D)."""

    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)
