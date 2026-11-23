"""Alternating-Attention aggregator (VGGT paper, Sec. 3.3).

Each frame is patchified, gets one camera token and ``num_register_tokens`` register tokens (the first frame uses a
separate learned pair, which defines the world frame), and the tokens then pass through ``depth`` pairs of
frame-wise and global self-attention blocks. For the cached layers, the frame and global outputs are concatenated
to 2C channels for the prediction heads.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    Args:
        img_size, patch_size, embed_dim: image size, patch size and token width.
        depth: number of frame blocks (and of global blocks).
        num_heads, mlp_ratio, qkv_bias, proj_bias, ffn_bias, qk_norm, init_values: block settings.
        num_register_tokens: register tokens per frame.
        block_fn: transformer block class.
        patch_embed: "conv" or a DINOv2 name such as "dinov2_vitl14_reg".
        aa_order: order of the attention types inside one AA step, e.g. ["frame", "global"].
        aa_block_size: blocks of one type run before switching to the next type.
        rope_freq: RoPE base frequency (<= 0 disables RoPE).
        cached_layer_indices: layers whose outputs are returned (the last layer is always added).
    Call ``model.train()`` to enable gradient checkpointing.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        cached_layer_indices: Tuple[int, ...] = (4, 11, 17, 23),
    ):
        super().__init__()
        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        def make_blocks():
            return nn.ModuleList([
                block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                         proj_bias=proj_bias, ffn_bias=ffn_bias, init_values=init_values, qk_norm=qk_norm,
                         rope=self.rope)
                for _ in range(depth)
            ])

        self.frame_blocks = make_blocks()
        self.global_blocks = make_blocks()

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.cached_layer_indices.add(depth - 1)
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")
        self.aa_block_num = self.depth // self.aa_block_size

        # Slot 0: token for the first (reference) frame; slot 1: shared by all other frames.
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.patch_start_idx = 1 + num_register_tokens
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False

    def __build_patch_embed__(self, patch_embed, img_size, patch_size, num_register_tokens, interpolate_antialias=True,
                              interpolate_offset=0.0, block_chunks=0, init_values=1.0, embed_dim=1024):
        """A plain conv patchifier for "conv", otherwise a DINOv2 ViT with registers."""
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
            return
        vit_models = {
            "dinov2_vitl14_reg": vit_large,
            "dinov2_vitb14_reg": vit_base,
            "dinov2_vits14_reg": vit_small,
            "dinov2_vitg2_reg": vit_giant2,
        }
        self.patch_embed = vit_models[patch_embed](
            img_size=img_size, patch_size=patch_size, num_register_tokens=num_register_tokens,
            interpolate_antialias=interpolate_antialias, interpolate_offset=interpolate_offset,
            block_chunks=block_chunks, init_values=init_values,
        )
        if hasattr(self.patch_embed, "mask_token"):
            self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[Optional[torch.Tensor]], int]:
        """
        images: (B, S, 3, H, W) in [0, 1].
        Returns the per-layer outputs (B, S, P, 2C) for cached layers and None elsewhere (so layer indices stay
        stable), and the index where patch tokens start.
        """
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
        if self.patch_start_idx > 0:
            # Special tokens get position 0 (no rotation); patches are shifted to start at 1.
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape
        frame_idx = global_idx = 0
        output_list = []
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos)
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                layer_idx = len(output_list)
                if layer_idx in self.cached_layer_indices:
                    output_list.append(torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1))
                else:
                    output_list.append(None)

        del frame_intermediates, global_intermediates
        return output_list, self.patch_start_idx

    def _run_block(self, block, tokens, pos):
        if self.training:
            return checkpoint(block, tokens, pos, use_reentrant=self.use_reentrant)
        return block(tokens, pos=pos)

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """Attention inside each frame: sequences of shape (B*S, P, C)."""
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)
        intermediates = []
        for _ in range(self.aa_block_size):
            tokens = self._run_block(self.frame_blocks[frame_idx], tokens, pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """Attention across all frames: one sequence of shape (B, S*P, C) per scene."""
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)
        intermediates = []
        for _ in range(self.aa_block_size):
            tokens = self._run_block(self.global_blocks[global_idx], tokens, pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """(1, 2, X, C) -> (B*S, X, C): slot 0 for frame 0, slot 1 for frames 1..S-1."""
    first = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    combined = torch.cat([first, others], dim=1)
    return combined.view(B * S, *combined.shape[2:])
