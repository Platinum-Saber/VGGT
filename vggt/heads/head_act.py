"""Output activations of the camera and dense heads."""

import torch
import torch.nn.functional as F


def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    """Apply separate activations to the translation (3), quaternion (4) and field-of-view (2) parts."""
    T = base_pose_act(pred_pose_enc[..., :3], trans_act)
    quat = base_pose_act(pred_pose_enc[..., 3:7], quat_act)
    fl = base_pose_act(pred_pose_enc[..., 7:], fl_act)
    return torch.cat([T, quat, fl], dim=-1)


def base_pose_act(pose_enc, act_type="linear"):
    if act_type == "linear":
        return pose_enc
    if act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    if act_type == "exp":
        return torch.exp(pose_enc)
    if act_type == "relu":
        return F.relu(pose_enc)
    raise ValueError(f"Unknown act_type: {act_type}")


def activate_head(out, activation="norm_exp", conf_activation="expp1"):
    """Split a (B, C, H, W) head output into the prediction (first C-1 channels) and confidence (last channel)."""
    fmap = out.permute(0, 2, 3, 1)
    xyz, conf = fmap[:, :, :, :-1], fmap[:, :, :, -1]

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pts3d = (xyz / d) * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":
        xy, z = xyz.split([2, 1], dim=-1)
        z = inverse_log_transform(z)
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif activation == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")
    return pts3d, conf_out


def inverse_log_transform(y):
    """sign(y) * (exp(|y|) - 1), the inverse of sign(x) * log(1 + |x|)."""
    return torch.sign(y) * torch.expm1(torch.abs(y))
