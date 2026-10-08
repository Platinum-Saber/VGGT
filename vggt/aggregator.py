"""Alternating-Attention aggregator (paper Sec. 3.3, Fig. 2).

Every frame is patchified into tokens, and gets one camera token plus
``num_register_tokens`` register tokens. The first frame uses a different
(learned) camera/register token than the remaining frames, which tells the
network which frame defines the world coordinate system. The tokens are then
processed by L pairs of (frame-wise self-attention, global self-attention)
blocks. For every pair the frame and global outputs are concatenated
(-> 2C channels) and handed to the prediction heads.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .dino import build_dino
from .layers import Block, PatchEmbed, PositionGetter, RotaryPositionEmbedding2D

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        patch_embed="dinov2_vitl14_reg",
        aa_order: Sequence[str] = ("frame", "global"),
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        cached_layer_indices: Optional[Sequence[int]] = (4, 11, 17, 23),
        grad_checkpoint=True,
    ):
        super().__init__()
        if patch_embed == "conv":
            self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        else:
            self.patch_embed = build_dino(patch_embed, img_size, patch_size, num_register_tokens)

        self.rope = RotaryPositionEmbedding2D(rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        def make_blocks():
            return nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, init_values=init_values, qk_norm=qk_norm, rope=self.rope)
                for _ in range(depth)
            ])

        # Only the attention types that appear in aa_order get parameters, so
        # ablations such as global-only attention have a matching param budget.
        self.frame_blocks = make_blocks() if "frame" in aa_order else None
        self.global_blocks = make_blocks() if "global" in aa_order else None

        self.depth = depth
        self.aa_order = list(aa_order)
        self.patch_size = patch_size
        self.grad_checkpoint = grad_checkpoint
        # None = keep the output of every layer.
        self.cached_layer_indices = None if cached_layer_indices is None else set(cached_layer_indices) | {depth - 1}

        # Index 0: token for the first (reference) frame, index 1: for all others.
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim) * 1e-6)
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim) * 1e-6)
        self.patch_start_idx = 1 + num_register_tokens

        self.register_buffer("_resnet_mean", torch.tensor(_RESNET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_resnet_std", torch.tensor(_RESNET_STD).view(1, 1, 3, 1, 1), persistent=False)

    def _run(self, block, tokens, pos):
        if self.training and self.grad_checkpoint:
            return checkpoint(block, tokens, pos, use_reentrant=False)
        return block(tokens, pos=pos)

    def forward(self, images: torch.Tensor):
        """images: (B, S, 3, H, W) in [0, 1]. Returns (list of (B, S, P, 2C) or None per layer, patch_start_idx)."""
        B, S, _, H, W = images.shape
        images = (images - self._resnet_mean) / self._resnet_std
        patch_tokens = self.patch_embed(images.reshape(B * S, 3, H, W))
        C = patch_tokens.shape[-1]

        camera_token = _first_vs_rest(self.camera_token, B, S)
        register_token = _first_vs_rest(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        P = tokens.shape[1]

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, images.device) + 1
            # Special tokens get position 0, i.e. no rotation.
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        outputs: List[Optional[torch.Tensor]] = []
        for i in range(self.depth):
            inter = []
            for attn_type in self.aa_order:
                if attn_type == "frame":  # attention within each frame: (B*S, P, C)
                    tokens = self._run(self.frame_blocks[i], tokens.reshape(B * S, P, C),
                                       None if pos is None else pos.reshape(B * S, P, 2))
                elif attn_type == "global":  # attention across all frames: (B, S*P, C)
                    tokens = self._run(self.global_blocks[i], tokens.reshape(B, S * P, C),
                                       None if pos is None else pos.reshape(B, S * P, 2))
                else:
                    raise ValueError(attn_type)
                inter.append(tokens.reshape(B, S, P, C))
            if self.cached_layer_indices is None or i in self.cached_layer_indices:
                # With a single attention type, duplicate it so heads always see 2C channels.
                outputs.append(torch.cat(inter if len(inter) == 2 else inter * 2, dim=-1))
            else:
                outputs.append(None)
        return outputs, self.patch_start_idx


def _first_vs_rest(token, B, S):
    """(1, 2, X, C) -> (B*S, X, C): slot 0 for frame 0, slot 1 for frames 1..S-1."""
    first = token[:, :1].expand(B, 1, *token.shape[2:])
    rest = token[:, 1:].expand(B, S - 1, *token.shape[2:])
    return torch.cat([first, rest], dim=1).reshape(B * S, *token.shape[2:])
