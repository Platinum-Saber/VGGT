"""Positional embeddings and bilinear sampling helpers for the tracker."""

from typing import Tuple, Union

import torch
import torch.nn.functional as F


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Union[int, Tuple[int, int]], return_grid=False) -> torch.Tensor:
    """Sin-cos embedding of an (h, w) integer grid; returns (1, embed_dim, h, w)."""
    gh, gw = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)
    grid = torch.meshgrid(torch.arange(gw, dtype=torch.float), torch.arange(gh, dtype=torch.float), indexing="xy")
    grid = torch.stack(grid, dim=0).reshape([2, 1, gh, gw])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = pos_embed.reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
    return (pos_embed, grid) if return_grid else pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=2)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """[sin(pos * w_k), cos(pos * w_k)], w_k = 10000^(-k / (embed_dim/2)); returns (1, M, embed_dim)."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)[None].float()


def get_2d_embedding(xy: torch.Tensor, C: int, cat_coords: bool = True) -> torch.Tensor:
    """Interleaved sin/cos embedding of (B, N, 2) coordinates, C channels per axis."""
    B, N, D = xy.shape
    assert D == 2
    x, y = xy[:, :, 0:1], xy[:, :, 1:2]
    div_term = (torch.arange(0, C, 2, device=xy.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C / 2))
    pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    pe = torch.cat([pe_x, pe_y], dim=2)
    if cat_coords:
        pe = torch.cat([xy, pe], dim=2)
    return pe


def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    """grid_sample with pixel coordinates: with align_corners, x in [0, W-1] maps to the pixel centres.

    input (B, C, H, W) with coords (B, Ho, Wo, 2) as (x, y), or (B, C, T, H, W) with coords as (t, x, y).
    The coordinates are detached, as in the official code.
    """
    coords = coords.detach().clone().to(input.device).to(input.dtype)
    sizes = input.shape[2:]
    assert len(sizes) in [2, 3]
    if len(sizes) == 3:
        coords = coords[..., [1, 2, 0]]
    if align_corners:
        scale = torch.tensor([2 / max(s - 1, 1) for s in reversed(sizes)], device=coords.device, dtype=coords.dtype)
    else:
        scale = torch.tensor([2 / s for s in reversed(sizes)], device=coords.device, dtype=coords.dtype)
    coords.mul_(scale)
    coords.sub_(1)
    return F.grid_sample(input, coords, align_corners=align_corners, padding_mode=padding_mode)


def sample_features4d(input, coords):
    """Sample (B, C, H, W) features at (B, R, 2) pixel coordinates -> (B, R, C)."""
    B = input.shape[0]
    feats = bilinear_sampler(input, coords.unsqueeze(2))  # (B, C, R, 1)
    return feats.permute(0, 2, 1, 3).view(B, -1, feats.shape[1] * feats.shape[3])
