"""Image to patch tokens: (B, C, H, W) -> (B, H/p * W/p, D) with a stride-p convolution."""

from typing import Callable, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()
        image_hw = make_2tuple(img_size)
        patch_hw = make_2tuple(patch_size)
        grid = (image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1])
        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = grid
        self.num_patches = grid[0] * grid[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        ph, pw = self.patch_size
        assert H % ph == 0, f"Input image height {H} is not a multiple of patch height {ph}"
        assert W % pw == 0, f"Input image width {W} is not a multiple of patch width: {pw}"
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        x = self.norm(x.flatten(2).transpose(1, 2))
        if not self.flatten_embedding:
            x = x.reshape(-1, h, w, self.embed_dim)
        return x

    def flops(self) -> float:
        ho, wo = self.patches_resolution
        flops = ho * wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += ho * wo * self.embed_dim
        return flops
