"""Camera geometry: depth unprojection, SE(3) inversion and projection (same API as the official vggt.utils.geometry).

Lens distortion (``distortion_params`` / ``extra_params``) lives in the official ``vggt.dependency`` package, which
is not part of this re-implementation; passing distortion parameters raises NotImplementedError.
"""

import numpy as np
import torch


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsics_cam: np.ndarray,
                                     intrinsics_cam: np.ndarray) -> np.ndarray:
    """(S, H, W[, 1]) depth, (S, 3, 4) camera-from-world extrinsics, (S, 3, 3) intrinsics -> (S, H, W, 3) world points."""
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()
    points = [depth_to_world_coords_points(depth_map[i].squeeze(-1), extrinsics_cam[i], intrinsics_cam[i])[0]
              for i in range(depth_map.shape[0])]
    return np.stack(points, axis=0)


def depth_to_world_coords_points(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray, eps=1e-8):
    """Returns (world points (H, W, 3), camera points (H, W, 3), valid mask depth > eps)."""
    if depth_map is None:
        return None, None, None
    point_mask = depth_map > eps
    cam_points = depth_to_cam_coords_points(depth_map, intrinsic)
    cam_to_world = closed_form_inverse_se3(extrinsic[None])[0]
    R, t = cam_to_world[:3, :3], cam_to_world[:3, 3]
    world_points = np.dot(cam_points, R.T) + t
    return world_points, cam_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """X_c = D * K^-1 [u, v, 1]^T for every pixel (pinhole, zero skew)."""
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - cu) * depth_map / fu
    y = (v - cv) * depth_map / fv
    return np.stack((x, y, depth_map), axis=-1).astype(np.float32)


def closed_form_inverse_se3(se3, R=None, T=None):
    """Batched inverse of rigid transforms (N, 4, 4) or (N, 3, 4): [R | t]^-1 = [R^T | -R^T t]. NumPy or torch."""
    is_numpy = isinstance(se3, np.ndarray)
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")
    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]
    if is_numpy:
        Rt = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(Rt, T)
        out = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        Rt = R.transpose(1, 2)
        top_right = -torch.bmm(Rt, T)
        out = torch.eye(4, 4)[None].repeat(len(R), 1, 1).to(R.dtype).to(R.device)
    out[:, :3, :3] = Rt
    out[:, :3, 3:] = top_right
    return out


def project_world_points_to_camera_points_batch(world_points, cam_extrinsics):
    """world_points (B, S, H, W, 3), extrinsics (B, S, 3, 4) -> camera points (B, S, H, W, 3)."""
    world_h = torch.cat([world_points, torch.ones_like(world_points[..., :1])], dim=-1)
    return torch.matmul(cam_extrinsics.unsqueeze(2).unsqueeze(3), world_h.unsqueeze(-1)).squeeze(-1)


def project_world_points_to_cam(world_points, cam_extrinsics, cam_intrinsics=None, distortion_params=None, default=0,
                                only_points_cam=False):
    """world_points (N, 3), extrinsics (B, 3, 4), intrinsics (B, 3, 3) -> (pixels (B, N, 2), camera points (B, 3, N))."""
    device = world_points.device
    with torch.autocast(device_type=device.type, enabled=False):
        B = cam_extrinsics.shape[0]
        world_h = torch.cat([world_points, torch.ones_like(world_points[..., 0:1])], dim=1)
        world_h = world_h.unsqueeze(0).expand(B, -1, -1)
        cam_points = torch.bmm(cam_extrinsics, world_h.transpose(-1, -2))
        if only_points_cam:
            return None, cam_points
        return img_from_cam(cam_intrinsics, cam_points, distortion_params, default=default), cam_points


def img_from_cam(cam_intrinsics, cam_points, distortion_params=None, default=0.0):
    """Camera points (B, 3, N) -> pixel coordinates (B, N, 2); invalid divisions become ``default``."""
    if distortion_params is not None:
        raise NotImplementedError("lens distortion is part of the official vggt.dependency package")
    cam_points = cam_points / cam_points[:, 2:3, :]
    xy = cam_points[:, :2, :]
    homo = torch.cat((xy, torch.ones_like(xy[:, :1, :])), dim=1)
    pixels = torch.bmm(cam_intrinsics, homo)[:, :2, :]
    pixels = torch.nan_to_num(pixels, nan=default)
    return pixels.transpose(1, 2)


def cam_from_img(pred_tracks, intrinsics, extra_params=None):
    """Pixel tracks (B, N, 2) -> normalised camera coordinates using K."""
    if extra_params is not None:
        raise NotImplementedError("lens distortion is part of the official vggt.dependency package")
    principal_point = intrinsics[:, [0, 1], [2, 2]].unsqueeze(-2)
    focal_length = intrinsics[:, [0, 1], [0, 1]].unsqueeze(-2)
    return (pred_tracks - principal_point) / focal_length
