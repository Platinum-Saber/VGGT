"""DPT dense prediction head (paper Sec. 3.3; Ranftl et al. 2021).

Patch tokens from four aggregator layers are reshaped to feature maps,
projected, resized to 4 scales (x4, x2, x1, x0.5), fused coarse-to-fine by
RefineNet-style blocks and upsampled to full resolution. The last channel is
the aleatoric-uncertainty confidence (1 + exp(.)), the others the prediction.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_log_transform(y):
    return torch.sign(y) * torch.expm1(torch.abs(y))


def activate_head(out, activation, conf_activation="expp1"):
    fmap = out.permute(0, 2, 3, 1)
    xyz, conf = fmap[..., :-1], fmap[..., -1]
    if activation == "exp":
        pred = torch.exp(xyz)
    elif activation == "inv_log":
        pred = inverse_log_transform(xyz)
    elif activation == "linear":
        pred = xyz
    else:
        raise ValueError(activation)
    if conf_activation == "expp1":
        conf = 1 + conf.exp()
    else:
        raise ValueError(conf_activation)
    return pred, conf


def make_sincos_pos_embed(embed_dim, pos, omega_0=100):
    omega = torch.arange(embed_dim // 2, dtype=torch.double, device=pos.device) / (embed_dim / 2.0)
    omega = 1.0 / omega_0 ** omega
    out = torch.outer(pos.reshape(-1).double(), omega)
    return torch.cat([out.sin(), out.cos()], 1).float()


def position_grid_to_embed(grid, embed_dim, omega_0=100):
    H, W, _ = grid.shape
    flat = grid.reshape(-1, 2)
    emb = torch.cat([make_sincos_pos_embed(embed_dim // 2, flat[:, 0], omega_0),
                     make_sincos_pos_embed(embed_dim // 2, flat[:, 1], omega_0)], -1)
    return emb.view(H, W, embed_dim)


def create_uv_grid(width, height, aspect_ratio, dtype=None, device=None):
    """UV coordinates in [-span, span], normalised by the image diagonal. Returns (height, width, 2)."""
    diag = (aspect_ratio ** 2 + 1.0) ** 0.5
    sx, sy = aspect_ratio / diag, 1.0 / diag
    xs = torch.linspace(-sx * (width - 1) / width, sx * (width - 1) / width, width, dtype=dtype, device=device)
    ys = torch.linspace(-sy * (height - 1) / height, sy * (height - 1) / height, height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack([uu, vv], -1)


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(F.relu(x)))) + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, has_residual=True):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1)

    def forward(self, x, skip=None, size=None):
        if self.has_residual:
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        if size is None:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        else:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class DPTHead(nn.Module):
    def __init__(self, dim_in, patch_size=14, output_dim=4, activation="inv_log", conf_activation="expp1",
                 features=256, out_channels: Sequence[int] = (256, 512, 1024, 1024),
                 intermediate_layer_idx: Sequence[int] = (4, 11, 17, 23), pos_embed=True):
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.intermediate_layer_idx = list(intermediate_layer_idx)

        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(dim_in, oc, 1) for oc in out_channels])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
        ])
        self.scratch = nn.Module()
        for i, oc in enumerate(out_channels):
            setattr(self.scratch, f"layer{i + 1}_rn", nn.Conv2d(oc, features, 3, padding=1, bias=False))
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet4 = FeatureFusionBlock(features, has_residual=False)
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, output_dim, 1)
        )

    def _add_pos_embed(self, x, W, H, ratio=0.1):
        grid = create_uv_grid(x.shape[-1], x.shape[-2], aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        emb = position_grid_to_embed(grid, x.shape[1]) * ratio
        return x + emb.permute(2, 0, 1)[None]

    def forward(self, aggregated_tokens_list, images, patch_start_idx, frames_chunk_size=8):
        S = images.shape[1]
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward(aggregated_tokens_list, images, patch_start_idx, 0, S)
        chunks = [self._forward(aggregated_tokens_list, images, patch_start_idx, s, min(s + frames_chunk_size, S))
                  for s in range(0, S, frames_chunk_size)]
        return torch.cat([c[0] for c in chunks], 1), torch.cat([c[1] for c in chunks], 1)

    def _forward(self, aggregated_tokens_list, images, patch_start_idx, s0, s1):
        B, _, _, H, W = images.shape
        S = s1 - s0
        ph, pw = H // self.patch_size, W // self.patch_size
        feats = []
        for i, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx][:, s0:s1, patch_start_idx:]
            x = self.norm(x.reshape(B * S, -1, x.shape[-1]))
            x = x.permute(0, 2, 1).reshape(B * S, -1, ph, pw)
            x = self.projects[i](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            feats.append(self.resize_layers[i](x))

        l1, l2, l3, l4 = [getattr(self.scratch, f"layer{i + 1}_rn")(f) for i, f in enumerate(feats)]
        out = self.scratch.refinenet4(l4, size=l3.shape[2:])
        out = self.scratch.refinenet3(out, l3, size=l2.shape[2:])
        out = self.scratch.refinenet2(out, l2, size=l1.shape[2:])
        out = self.scratch.refinenet1(out, l1)
        out = self.scratch.output_conv1(out)
        out = F.interpolate(out, size=(ph * self.patch_size, pw * self.patch_size), mode="bilinear",
                            align_corners=True)
        if self.pos_embed:
            out = self._add_pos_embed(out, W, H)
        out = self.scratch.output_conv2(out)
        pred, conf = activate_head(out, self.activation, self.conf_activation)
        return pred.reshape(B, S, *pred.shape[1:]), conf.reshape(B, S, *conf.shape[1:])
