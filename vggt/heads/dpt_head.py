"""DPT dense prediction head (Ranftl et al., "Vision Transformers for Dense Prediction", 2021).

Patch tokens from four aggregator layers are reshaped to feature maps, projected, resized to four scales
(x4, x2, x1, x0.5), fused coarse-to-fine with RefineNet-style blocks and upsampled to the image resolution
(divided by ``down_ratio``). With ``feature_only`` the head returns these features (used by the track head);
otherwise a last convolution and ``activate_head`` give the prediction and its confidence.
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn

from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


class DPTHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1, head_features_2 = features, 32
        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1,
                                                  padding=1)
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, aggregated_tokens_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int,
                frames_chunk_size: int = 8) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        images: (B, S, 3, H, W). Frames are processed in chunks of ``frames_chunk_size`` to bound memory.
        Returns features (B, S, C, H/r, W/r) if feature_only, else (prediction (B, S, H, W, d), confidence (B, S, H, W)).
        """
        S = images.shape[1]
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)
        assert frames_chunk_size > 0

        preds, confs = [], []
        for start in range(0, S, frames_chunk_size):
            end = min(start + frames_chunk_size, S)
            out = self._forward_impl(aggregated_tokens_list, images, patch_start_idx, start, end)
            if self.feature_only:
                preds.append(out)
            else:
                preds.append(out[0])
                confs.append(out[1])
        if self.feature_only:
            return torch.cat(preds, dim=1)
        return torch.cat(preds, dim=1), torch.cat(confs, dim=1)

    def _forward_impl(self, aggregated_tokens_list, images, patch_start_idx, frames_start_idx=None,
                      frames_end_idx=None):
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            x = self.norm(x.reshape(B * S, -1, x.shape[-1]))
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))  # tokens -> feature map
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            out.append(self.resize_layers[dpt_idx](x))

        out = self.scratch_forward(out)
        out = custom_interpolate(
            out, (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear", align_corners=True)
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)
        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)
        return preds.view(B, S, *preds.shape[1:]), conf.view(B, S, *conf.shape[1:])

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Add 0.1 x the sin-cos embedding of the UV grid of x's resolution."""
        grid = create_uv_grid(x.shape[-1], x.shape[-2], aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        emb = position_grid_to_embed(grid, x.shape[1]) * ratio
        return x + emb.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse the 4-level pyramid coarse-to-fine."""
        layer_1, layer_2, layer_3, layer_4 = features
        l1 = self.scratch.layer1_rn(layer_1)
        l2 = self.scratch.layer2_rn(layer_2)
        l3 = self.scratch.layer3_rn(layer_3)
        l4 = self.scratch.layer4_rn(layer_4)
        out = self.scratch.refinenet4(l4, size=l3.shape[2:])
        out = self.scratch.refinenet3(out, l3, size=l2.shape[2:])
        out = self.scratch.refinenet2(out, l2, size=l1.shape[2:])
        out = self.scratch.refinenet1(out, l1)
        return self.scratch.output_conv1(out)


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(features, nn.ReLU(inplace=True), deconv=False, bn=False, expand=False,
                              align_corners=True, size=size, has_residual=has_residual, groups=groups)


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    """3x3 convs bringing each pyramid level to the common feature width (x1, x2, x4, x8 if expand)."""
    scratch = nn.Module()
    mult = [1, 2, 4, 8] if expand else [1, 1, 1, 1]
    for i in range(min(len(in_shape), 4)):
        setattr(scratch, f"layer{i + 1}_rn", nn.Conv2d(in_shape[i], out_shape * mult[i], kernel_size=3, stride=1,
                                                       padding=1, bias=False, groups=groups))
    return scratch


class ResidualConvUnit(nn.Module):
    """act -> conv -> act -> conv, plus a skip connection.

    The activation is an in-place ReLU, so after the first call ``x`` itself holds relu(x) and the skip adds
    relu(x), not x. The released weights depend on this, so it is kept.
    """

    def __init__(self, features, activation, bn, groups=1):
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """out = Conv1x1(Upsample(RCU2(x + RCU1(skip))))."""

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None,
                 has_residual=True, groups=1):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=groups)
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=groups)
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=groups)
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)

        if size is None and self.size is None:
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(output)


def custom_interpolate(x: torch.Tensor, size: Tuple[int, int] = None, scale_factor: float = None,
                       mode: str = "bilinear", align_corners: bool = True) -> torch.Tensor:
    """F.interpolate, split along the batch when the output would exceed the INT_MAX element limit."""
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    n = size[0] * size[1] * x.shape[0] * x.shape[1]
    if n > INT_MAX:
        chunks = torch.chunk(x, chunks=(n // INT_MAX) + 1, dim=0)
        x = torch.cat([nn.functional.interpolate(c, size=size, mode=mode, align_corners=align_corners)
                       for c in chunks], dim=0)
        return x.contiguous()
    return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
