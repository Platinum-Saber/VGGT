"""Update transformer and correlation pyramid of the tracker (after CoTracker / VGGSfM)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import bilinear_sampler
from .modules import Mlp, AttnBlock, CrossAttnBlock, ResidualBlock  # noqa: F401 (ResidualBlock kept for API parity)


class EfficientUpdateFormer(nn.Module):
    """Transformer that predicts coordinate and feature updates for all tracks.

    Input (B, N, T, D): N tracks over T frames. Time blocks attend along T for each track; space attention
    goes through ``num_virtual_tracks`` learned virtual tracks (tracks -> virtual -> self-attention -> tracks),
    which keeps the cost linear in N.
    """

    def __init__(self, space_depth=6, time_depth=6, input_dim=320, hidden_size=384, num_heads=8, output_dim=130,
                 mlp_ratio=4.0, add_space_attn=True, num_virtual_tracks=64):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.flow_head = nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks

        # (sic) "virual_tracks": the official parameter name, needed to load the checkpoint.
        self.virual_tracks = nn.Parameter(torch.randn(1, num_virtual_tracks, 1, hidden_size)) if add_space_attn else None

        self.time_blocks = nn.ModuleList([
            AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
            for _ in range(time_depth)
        ])
        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList([
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                for _ in range(space_depth)
            ])
            self.space_point2virtual_blocks = nn.ModuleList([
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(space_depth)
            ])
            self.space_virtual2point_blocks = nn.ModuleList([
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(space_depth)
            ])
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            torch.nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None):
        tokens = self.input_transform(self.input_norm(input_tensor))
        init_tokens = tokens
        B, _, T, _ = tokens.shape
        if self.add_space_attn:
            tokens = torch.cat([tokens, self.virual_tracks.repeat(B, 1, T, 1)], dim=1)
        _, N, _, _ = tokens.shape

        j = 0
        for i in range(len(self.time_blocks)):
            time_tokens = self.time_blocks[i](tokens.contiguous().view(B * N, T, -1))  # attend over time
            tokens = time_tokens.view(B, N, T, -1)
            if self.add_space_attn and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0):
                space = tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)  # attend over tracks
                point_tokens = space[:, : N - self.num_virtual_tracks]
                virtual_tokens = space[:, N - self.num_virtual_tracks:]
                virtual_tokens = self.space_virtual2point_blocks[j](virtual_tokens, point_tokens, mask=mask)
                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](point_tokens, virtual_tokens, mask=mask)
                space = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space.view(B, T, N, -1).permute(0, 2, 1, 3)
                j += 1

        if self.add_space_attn:
            tokens = tokens[:, : N - self.num_virtual_tracks]
        tokens = tokens + init_tokens
        return self.flow_head(self.output_norm(tokens)), None


class CorrBlock:
    """Multi-scale correlation between track features and feature maps.

    Builds an average-pooled pyramid of the feature maps; for each level, correlates the track features with
    every location and samples a (2r+1) x (2r+1) window around the current track position.
    """

    def __init__(self, fmaps, num_levels=4, radius=4, multiple_track_feats=False, padding_mode="zeros"):
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.num_levels = num_levels
        self.radius = radius
        self.padding_mode = padding_mode
        self.multiple_track_feats = multiple_track_feats

        self.fmaps_pyramid = [fmaps]
        current = fmaps
        for _ in range(num_levels - 1):
            B, S, C, H, W = current.shape
            current = F.avg_pool2d(current.reshape(B * S, C, H, W), kernel_size=2, stride=2)
            _, _, h, w = current.shape
            current = current.reshape(B, S, C, h, w)
            self.fmaps_pyramid.append(current)

        r = self.radius
        dx = torch.linspace(-r, r, 2 * r + 1, device=fmaps.device, dtype=fmaps.dtype)
        dy = torch.linspace(-r, r, 2 * r + 1, device=fmaps.device, dtype=fmaps.dtype)
        self.delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=-1)  # (2r+1, 2r+1, 2)

    def corr_sample(self, targets, coords):
        """targets (B, S, N, C) track features; coords (B, S, N, 2) full-resolution positions.
        Returns (B, S, N, num_levels * (2r+1)^2)."""
        B, S, N, C = targets.shape
        if self.multiple_track_feats:
            targets_split = torch.split(targets, C // self.num_levels, dim=-1)

        out = []
        for i, fmaps in enumerate(self.fmaps_pyramid):
            B, S, C, H, W = fmaps.shape
            fmap1 = targets_split[i] if self.multiple_track_feats else targets
            corrs = compute_corr_level(fmap1, fmaps.view(B, S, C, H * W), C).view(B, S, N, H, W)
            centroid = coords.reshape(B * S * N, 1, 1, 2) / (2 ** i)
            delta = self.delta.to(coords.device).to(coords.dtype)
            grid = centroid + delta.view(1, 2 * self.radius + 1, 2 * self.radius + 1, 2)
            sampled = bilinear_sampler(corrs.reshape(B * S * N, 1, H, W), grid, padding_mode=self.padding_mode)
            out.append(sampled.view(B, S, N, -1))
        return torch.cat(out, dim=-1).contiguous()


def compute_corr_level(fmap1, fmap2s, C):
    """(B, S, N, C) x (B, S, C, HW) -> (B, S, N, HW), scaled by 1/sqrt(C)."""
    corrs = torch.matmul(fmap1, fmap2s)
    corrs = corrs.view(fmap1.shape[0], fmap1.shape[1], fmap1.shape[2], -1)
    return corrs / math.sqrt(C)
