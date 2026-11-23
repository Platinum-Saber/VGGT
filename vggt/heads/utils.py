"""Sin-cos positional embedding of a normalised UV grid (used by the DPT head)."""

import torch


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """(H, W, 2) grid -> (H, W, embed_dim): half of the channels encode x, half encode y."""
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    flat = pos_grid.reshape(-1, grid_dim)
    emb_x = make_sincos_pos_embed(embed_dim // 2, flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, flat[:, 1], omega_0=omega_0)
    return torch.cat([emb_x, emb_y], dim=-1).view(H, W, embed_dim)


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """[sin(pos * w_k), cos(pos * w_k)] with w_k = omega_0^(-k / (embed_dim/2)), computed in float64."""
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0 ** omega
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def create_uv_grid(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None,
                   device: torch.device = None) -> torch.Tensor:
    """Pixel-centre UV coordinates normalised by the image diagonal; returns (height, width, 2)."""
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x, span_y = aspect_ratio / diag, 1.0 / diag
    x = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, steps=width, dtype=dtype,
                       device=device)
    y = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, steps=height, dtype=dtype,
                       device=device)
    uu, vv = torch.meshgrid(x, y, indexing="xy")
    return torch.stack((uu, vv), dim=-1)
