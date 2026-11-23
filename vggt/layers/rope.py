"""2D rotary position embedding (RoPE) for patch tokens.

Half of each attention head's channels is rotated according to the patch row (y), the other half according to
the column (x). With frequencies theta_j = base^(-2j/d), a token at position m has channel pairs (j, j + d/2)
rotated by m * theta_j, so the dot product of a rotated query and key depends only on their relative offset.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionGetter:
    """Returns the (y, x) grid index of every patch, cached per grid size."""

    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            ys = torch.arange(height, device=device)
            xs = torch.arange(width, device=device)
            self.position_cache[height, width] = torch.cartesian_prod(ys, xs)
        pos = self.position_cache[height, width]
        return pos.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype):
        key = (dim, seq_len, device, dtype)
        if key not in self.frequency_cache:
            inv_freq = 1.0 / (self.base_frequency ** (torch.arange(0, dim, 2, device=device).float() / dim))
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.outer(positions, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.frequency_cache[key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return tokens * cos + self._rotate_features(tokens) * sin

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """tokens (B, heads, N, head_dim); positions (B, N, 2) integer (y, x)."""
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"
        dim = tokens.size(-1) // 2
        cos_comp, sin_comp = self._compute_frequency_components(dim, int(positions.max()) + 1, tokens.device,
                                                                tokens.dtype)
        vertical, horizontal = tokens.chunk(2, dim=-1)
        vertical = self._apply_1d_rope(vertical, positions[..., 0], cos_comp, sin_comp)
        horizontal = self._apply_1d_rope(horizontal, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical, horizontal), dim=-1)
