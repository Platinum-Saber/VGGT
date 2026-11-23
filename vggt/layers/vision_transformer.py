"""DINOv2 vision transformer (with register tokens), used by VGGT as its patchifier.

Same module names as DINOv2 / the official VGGT so the released checkpoint loads with strict=True.
"""

import logging
import math
from functools import partial
from typing import Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint

from . import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        qk_norm=False,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.use_reentrant = False

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None

        dpr = [drop_path_rate] * depth if drop_path_uniform else [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer in ("swiglufused", "swiglu"):
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            def ffn_layer(*args, **kwargs):
                return nn.Identity()
        else:
            raise NotImplementedError

        blocks = [
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias,
                     ffn_bias=ffn_bias, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                     ffn_layer=ffn_layer, init_values=init_values, qk_norm=qk_norm)
            for i in range(depth)
        ]
        if block_chunks > 0:
            # FSDP-style chunking; each chunk is padded with Identity so block indices stay global.
            self.chunked_blocks = True
            size = depth // block_chunks
            self.blocks = nn.ModuleList(
                [BlockChunk([nn.Identity()] * i + blocks[i: i + size]) for i in range(0, depth, size)])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        """Resample the learned (square) patch position embedding to the current patch grid.
        As in DINOv2, ``w`` is the image's first spatial size (height) and ``h`` the second (width)."""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos, patch_pos = pos_embed[:, 0], pos_embed[:, 1:]
        dim = x.shape[-1]
        w0, h0 = w // self.patch_size, h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        if self.interpolate_offset:
            kwargs = {"scale_factor": (float(w0 + self.interpolate_offset) / M, float(h0 + self.interpolate_offset) / M)}
        else:
            kwargs = {"size": (w0, h0)}
        patch_pos = nn.functional.interpolate(patch_pos.reshape(1, M, M, dim).permute(0, 3, 1, 2), mode="bicubic",
                                              antialias=self.interpolate_antialias, **kwargs)
        assert (w0, h0) == patch_pos.shape[-2:]
        patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos.unsqueeze(0), patch_pos), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            # Registers go after cls and do not get a position embedding.
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def _run_blocks(self, x):
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=self.use_reentrant) if self.training else blk(x)
        return x

    def _split_output(self, x, masks):
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1: self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward_features_list(self, x_list, masks_list):
        xs = [self.prepare_tokens_with_masks(x, m) for x, m in zip(x_list, masks_list)]
        xs = self._run_blocks(xs)  # needs xFormers nested tensors, see NestedTensorBlock
        return [self._split_output(x, m) for x, m in zip(xs, masks_list)]

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self._run_blocks(self.prepare_tokens_with_masks(x, masks))
        return self._split_output(x, masks)

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        total = len(self.blocks)
        take = range(total - n, total) if isinstance(n, int) else n
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take:
                output.append(x)
        assert len(output) == len(take), f"only {len(output)} / {len(take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total = [], 0, len(self.blocks[-1])
        take = range(total - n, total) if isinstance(n, int) else n
        for chunk in self.blocks:
            for blk in chunk[i:]:
                x = blk(x)
                if i in take:
                    output.append(x)
                i += 1
        assert len(output) == len(take), f"only {len(output)} / {len(take)} blocks found"
        return output

    def get_intermediate_layers(self, x: torch.Tensor, n: Union[int, Sequence] = 1, reshape: bool = False,
                                return_class_token: bool = False, norm=True) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        outputs = self._get_intermediate_layers_chunked(x, n) if self.chunked_blocks else \
            self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(o) for o in outputs]
        class_tokens = [o[:, 0] for o in outputs]
        outputs = [o[:, 1 + self.num_register_tokens:] for o in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [o.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                       for o in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=True, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret if is_training else self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _vit(embed_dim, depth, num_heads, patch_size, num_register_tokens, **kwargs):
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                 mlp_ratio=4, block_fn=partial(Block, attn_class=MemEffAttention),
                                 num_register_tokens=num_register_tokens, **kwargs)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    return _vit(384, 12, 6, patch_size, num_register_tokens, **kwargs)


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    return _vit(768, 12, 12, patch_size, num_register_tokens, **kwargs)


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    return _vit(1024, 24, 16, patch_size, num_register_tokens, **kwargs)


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """Close to ViT-giant: embed dim 1536, 24 heads (64 channels per head)."""
    return _vit(1536, 40, 24, patch_size, num_register_tokens, **kwargs)
