"""VGGT: Visual Geometry Grounded Transformer (Wang et al., CVPR 2025).

f((I_i)_{i=1..N}) = (g_i, D_i, P_i, T_i): for every input frame predicts the
camera (pose encoding g), a depth map D, a point map P in the first camera's
frame, and (in the paper) tracking features T. This implementation covers the
camera, depth and point heads; the tracking head is not re-implemented.

``VGGT()`` with default arguments is the released VGGT-1B configuration.
``VGGT.small(...)`` builds a scaled-down variant that is trainable on a CPU.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .aggregator import Aggregator
from .heads.camera_head import CameraHead
from .heads.dpt_head import DPTHead


class VGGT(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_embed="dinov2_vitl14_reg",
        aa_order: Sequence[str] = ("frame", "global"),
        dpt_layers: Sequence[int] = (4, 11, 17, 23),
        dpt_features=256,
        dpt_out_channels: Sequence[int] = (256, 512, 1024, 1024),
        camera_trunk_depth=4,
        camera_num_heads=16,
        enable_camera=True,
        enable_depth=True,
        enable_point=True,
        grad_checkpoint=True,
    ):
        super().__init__()
        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            patch_embed=patch_embed, aa_order=aa_order, cached_layer_indices=dpt_layers,
            grad_checkpoint=grad_checkpoint,
        )
        dim = 2 * embed_dim
        dpt_kw = dict(patch_size=patch_size, features=dpt_features, out_channels=dpt_out_channels,
                      intermediate_layer_idx=dpt_layers)
        self.camera_head = CameraHead(dim, trunk_depth=camera_trunk_depth, num_heads=camera_num_heads) \
            if enable_camera else None
        self.point_head = DPTHead(dim, output_dim=4, activation="inv_log", **dpt_kw) if enable_point else None
        self.depth_head = DPTHead(dim, output_dim=2, activation="exp", **dpt_kw) if enable_depth else None

    @classmethod
    def small(cls, img_size=112, embed_dim=192, depth=6, num_heads=6, **kw):
        """A ~10M-parameter VGGT: conv patchifier, `depth` AA layers, light DPT heads."""
        last = depth - 1
        cfg = dict(
            img_size=img_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, patch_embed="conv",
            dpt_layers=(last // 4, last // 2, (3 * last) // 4, last), dpt_features=64,
            dpt_out_channels=(48, 96, 192, 192), camera_trunk_depth=2, camera_num_heads=num_heads,
            grad_checkpoint=False,
        )
        cfg.update(kw)
        return cls(**cfg)

    def forward(self, images: torch.Tensor, num_camera_iterations: int = 4):
        """images: (S, 3, H, W) or (B, S, 3, H, W) in [0, 1]; H, W multiples of the patch size."""
        if images.dim() == 4:
            images = images[None]
        tokens, patch_start_idx = self.aggregator(images)
        preds = {}
        if self.camera_head is not None:
            pose_enc_list = self.camera_head(tokens, num_iterations=num_camera_iterations)
            preds["pose_enc"] = pose_enc_list[-1]
            preds["pose_enc_list"] = pose_enc_list
        if self.depth_head is not None:
            preds["depth"], preds["depth_conf"] = self.depth_head(tokens, images, patch_start_idx)
        if self.point_head is not None:
            preds["world_points"], preds["world_points_conf"] = self.point_head(tokens, images, patch_start_idx)
        return preds

    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu", **kw):
        """Load an official VGGT-1B ``model.pt`` (tracking-head weights are ignored)."""
        model = cls(**kw)
        state = torch.load(path, map_location=map_location)
        state = {k: v for k, v in state.items() if not k.startswith("track_head.")}
        model.load_state_dict(state, strict=True)
        return model
