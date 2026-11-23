"""Pre-norm transformer block: x + LS(Attn(LN(x))), then x + LS(MLP(LN(x)))."""

from typing import Callable

import torch
from torch import Tensor, nn

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop,
                               proj_drop=drop, qk_norm=qk_norm, fused_attn=fused_attn, rope=rope)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop,
                             bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None) -> Tensor:
        def attn_residual(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # Large drop rates: compute the residual only for the kept samples (saves compute).
            x = drop_add_residual_stochastic_depth(x, residual_func=attn_residual, pos=pos,
                                                   sample_drop_ratio=self.sample_drop_ratio)
            x = drop_add_residual_stochastic_depth(x, residual_func=ffn_residual,
                                                   sample_drop_ratio=self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            # The official code reuses drop_path1 for both branches; kept for identical behaviour.
            x = x + self.drop_path1(attn_residual(x, pos=pos))
            x = x + self.drop_path1(ffn_residual(x))
        else:
            x = x + attn_residual(x, pos=pos)
            x = x + ffn_residual(x)
        return x


def drop_add_residual_stochastic_depth(x: Tensor, residual_func: Callable, sample_drop_ratio: float = 0.0,
                                       pos=None) -> Tensor:
    b = x.shape[0]
    keep = max(int(b * (1 - sample_drop_ratio)), 1)
    idx = torch.randperm(b, device=x.device)[:keep]
    residual = residual_func(x[idx], pos=pos[idx]) if pos is not None else residual_func(x[idx])
    out = torch.index_add(x.flatten(1), 0, idx, residual.flatten(1).to(dtype=x.dtype), alpha=b / keep)
    return out.view_as(x)


class NestedTensorBlock(Block):
    """Block used by DINOv2. The official version also accepts lists of tensors, which needs xFormers;
    for a single tensor (the only case VGGT uses) it is exactly ``Block``."""

    def forward(self, x_or_x_list, pos=None):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list, pos=pos)
        raise AssertionError("Nested tensors (lists of tensors) require xFormers, which is not supported here")
