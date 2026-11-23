"""Camera head (VGGT paper, Sec. 3.3, "Prediction heads").

Takes the camera token of every frame from the last aggregator layer and refines a 9-d pose encoding over
``num_iterations`` steps. Each step embeds the current estimate, uses it to modulate the camera tokens with
adaptive LayerNorm (adaLN, as in DiT), runs a small self-attention trunk across frames and predicts an update.
"""

import torch
import torch.nn as nn

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose


class CameraHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # keeps the field of view positive
    ):
        super().__init__()
        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        self.trunk = nn.Sequential(*[
            Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
            for _ in range(trunk_depth)
        ])
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """Returns the activated pose encodings (B, S, 9) of every refinement iteration."""
        pose_tokens = self.token_norm(aggregated_tokens_list[-1][:, :, 0])
        return self.trunk_fn(pose_tokens, num_iterations)

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []
        for _ in range(num_iterations):
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # No gradient through the conditioning on the previous estimate.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            shift, scale, gate = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            x = gate * modulate(self.adaln_norm(pose_tokens), shift, scale) + pose_tokens
            x = self.trunk(x)
            delta = self.pose_branch(self.trunk_norm(x))
            pred_pose_enc = delta if pred_pose_enc is None else pred_pose_enc + delta

            pred_pose_enc_list.append(activate_pose(pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act,
                                                    fl_act=self.fl_act))
        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift
