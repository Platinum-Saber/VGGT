"""DINOv2-with-registers ViT used as VGGT's patchifier (paper Sec. 3.3).

VGGT initialises its patch embedding from DINOv2 ViT-L/14 (with 4 register
tokens) and only keeps the normalised patch tokens. Parameter names match
the official ``DinoVisionTransformer`` so checkpoints load directly.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Block, PatchEmbed

VIT_CONFIGS = {
    "dinov2_vits14_reg": dict(embed_dim=384, depth=12, num_heads=6),
    "dinov2_vitb14_reg": dict(embed_dim=768, depth=12, num_heads=12),
    "dinov2_vitl14_reg": dict(embed_dim=1024, depth=24, num_heads=16),
    "dinov2_vitg2_reg": dict(embed_dim=1536, depth=40, num_heads=24),
}


class DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
                 num_register_tokens=4, init_values=1.0, interpolate_antialias=True):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, init_values=init_values, norm_layer=norm_layer) for _ in range(depth)]
        )
        self.norm = norm_layer(embed_dim)
        # Unused at inference (masked-image-modelling token), kept for checkpoint compatibility.
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=False)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.register_tokens, std=1e-6)
        self.apply(_init_linear)

    def interpolate_pos_encoding(self, x, H, W):
        N = self.pos_embed.shape[1] - 1
        gh, gw = H // self.patch_size, W // self.patch_size
        if x.shape[1] - 1 == N and H == W:
            return self.pos_embed
        pos = self.pos_embed.float()
        cls_pos, patch_pos = pos[:, :1], pos[:, 1:]
        M = int(math.sqrt(N))
        patch_pos = F.interpolate(
            patch_pos.reshape(1, M, M, -1).permute(0, 3, 1, 2), size=(gh, gw), mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
        return torch.cat([cls_pos, patch_pos], dim=1).to(x.dtype)

    def forward(self, images):
        _, _, H, W = images.shape
        x = self.patch_embed(images)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.interpolate_pos_encoding(x, H, W)
        x = torch.cat([x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1 + self.num_register_tokens:]  # normalised patch tokens


def _init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_dino(name, img_size, patch_size, num_register_tokens=4):
    return DinoVisionTransformer(img_size=img_size, patch_size=patch_size, num_register_tokens=num_register_tokens,
                                 **VIT_CONFIGS[name])
