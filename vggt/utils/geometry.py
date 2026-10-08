"""Geometry helpers: SE(3) inversion and depth unprojection (all torch, batched)."""

import torch


def inverse_se3(T: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of (..., 3, 4) or (..., 4, 4) rigid transforms; returns (..., 4, 4)."""
    R, t = T[..., :3, :3], T[..., :3, 3:]
    Rt = R.transpose(-1, -2)
    out = torch.zeros(T.shape[:-2] + (4, 4), dtype=T.dtype, device=T.device)
    out[..., :3, :3] = Rt
    out[..., :3, 3:] = -Rt @ t
    out[..., 3, 3] = 1
    return out


def to_homogeneous(T: torch.Tensor) -> torch.Tensor:
    if T.shape[-2] == 4:
        return T
    bottom = torch.zeros(T.shape[:-2] + (1, 4), dtype=T.dtype, device=T.device)
    bottom[..., 0, 3] = 1
    return torch.cat([T, bottom], -2)


def unproject_depth_to_world(depth, extrinsics, intrinsics):
    """depth (..., H, W), extrinsics (..., 3, 4) cam-from-world, intrinsics (..., 3, 3) -> (..., H, W, 3)."""
    H, W = depth.shape[-2:]
    v, u = torch.meshgrid(torch.arange(H, dtype=depth.dtype, device=depth.device),
                          torch.arange(W, dtype=depth.dtype, device=depth.device), indexing="ij")
    fx, fy = intrinsics[..., 0, 0, None, None], intrinsics[..., 1, 1, None, None]
    cx, cy = intrinsics[..., 0, 2, None, None], intrinsics[..., 1, 2, None, None]
    cam = torch.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], -1)
    c2w = inverse_se3(extrinsics)
    R, t = c2w[..., None, None, :3, :3], c2w[..., None, None, :3, 3]
    return (R @ cam[..., None])[..., 0] + t
