"""Track head (VGGT paper, Sec. 3.3, "Tracking").

A DPT head in feature-only mode turns the aggregator tokens into dense tracking features at half resolution; a
CoTracker/VGGSfM-style predictor then follows the query points of the first frame through all frames with
iterative correlation-based refinement.
"""

import torch.nn as nn

from .dpt_head import DPTHead
from .track_modules.base_track_predictor import BaseTrackerPredictor


class TrackHead(nn.Module):
    def __init__(self, dim_in, patch_size=14, features=128, iters=4, predict_conf=True, stride=2, corr_levels=7,
                 corr_radius=4, hidden_size=384):
        """
        dim_in: token width (2C); features: channels of the tracking feature maps; iters: refinement steps;
        stride: feature-map stride relative to the image; corr_levels / corr_radius: correlation pyramid levels and
        search radius; hidden_size: width of the update transformer.
        """
        super().__init__()
        self.patch_size = patch_size
        self.feature_extractor = DPTHead(dim_in=dim_in, patch_size=patch_size, features=features, feature_only=True,
                                         down_ratio=2, pos_embed=False)
        self.tracker = BaseTrackerPredictor(latent_dim=features, predict_conf=predict_conf, stride=stride,
                                            corr_levels=corr_levels, corr_radius=corr_radius, hidden_size=hidden_size)
        self.iters = iters

    def forward(self, aggregated_tokens_list, images, patch_start_idx, query_points=None, iters=None):
        """
        images: (B, S, 3, H, W); query_points: (B, N, 2) pixel coordinates in frame 0.
        Returns (list of track predictions (B, S, N, 2) per iteration, visibility (B, S, N), confidence (B, S, N)).
        """
        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)  # (B, S, C, H/2, W/2)
        if iters is None:
            iters = self.iters
        return self.tracker(query_points=query_points, fmaps=feature_maps, iters=iters)
