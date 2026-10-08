"""Camera head (paper Sec. 3.3, "Prediction heads").

Takes the camera token of every frame from the last aggregator layer and
refines a 9-d pose encoding over ``num_iterations`` steps. Each step embeds
the current estimate, uses it to modulate (adaLN) the camera tokens, runs a
small self-attention trunk across frames, and predicts a residual update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import Block, Mlp


class CameraHead(nn.Module):
    def __init__(self, dim_in=2048, trunk_depth=4, num_heads=16, mlp_ratio=4, init_values=0.01):
        super().__init__()
        self.target_dim = 9
        self.trunk = nn.Sequential(
            *[Block(dim_in, num_heads, mlp_ratio, init_values=init_values) for _ in range(trunk_depth)]
        )
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(dim_in, dim_in // 2, self.target_dim)

    def forward(self, aggregated_tokens_list, num_iterations=4):
        tokens = self.token_norm(aggregated_tokens_list[-1][:, :, 0])  # (B, S, 2C) camera tokens
        B, S, _ = tokens.shape
        pred, preds = None, []
        for _ in range(num_iterations):
            # No backprop through previous iterations.
            inp = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1) if pred is None else pred.detach())
            shift, scale, gate = self.poseLN_modulation(inp).chunk(3, dim=-1)
            x = gate * (self.adaln_norm(tokens) * (1 + scale) + shift) + tokens
            delta = self.pose_branch(self.trunk_norm(self.trunk(x)))
            pred = delta if pred is None else pred + delta
            # Translation and quaternion are linear; FoV goes through ReLU to stay positive.
            preds.append(torch.cat([pred[..., :7], F.relu(pred[..., 7:])], -1))
        return preds
