"""VGGT: Visual Geometry Grounded Transformer (Wang et al., CVPR 2025).

f((I_i)_{i=1..S}) = (g_i, D_i, P_i, T_i): from one or more images of a scene, predicts per frame the camera
(pose encoding g), a depth map D with confidence, a point map P in the first camera's frame with confidence, and
point tracks T for given query points. Default arguments give the released VGGT-1B configuration.
"""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log",
                                  conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp",
                                  conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        images: (S, 3, H, W) or (B, S, 3, H, W) in [0, 1].
        query_points: optional (N, 2) or (B, N, 2) pixel coordinates in the first frame to track.

        Returns a dict with pose_enc (B, S, 9), pose_enc_list, depth (B, S, H, W, 1), depth_conf (B, S, H, W),
        world_points (B, S, H, W, 3), world_points_conf (B, S, H, W); with query points also track (B, S, N, 2),
        vis (B, S, N) and conf (B, S, N); in eval mode also the input images.
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        # The heads always run in fp32, also when the caller uses mixed precision.
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(aggregated_tokens_list, images=images,
                                                    patch_start_idx=patch_start_idx)
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(aggregated_tokens_list, images=images,
                                                    patch_start_idx=patch_start_idx)
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(aggregated_tokens_list, images=images,
                                                    patch_start_idx=patch_start_idx, query_points=query_points)
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images
        return predictions
