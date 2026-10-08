"""Camera parameterisation g = [q, t, f] (paper Sec. 3.1).

Encoded as a 9-vector: translation t (3), rotation quaternion q (4, xyzw)
and field of view (fov_h, fov_w). Extrinsics are OpenCV camera-from-world
[R|t]; the principal point is assumed at the image centre.
"""

import torch

from .rotation import mat_to_quat, quat_to_mat


def extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw):
    R, T = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    return torch.cat([T, mat_to_quat(R), fov_h[..., None], fov_w[..., None]], -1).float()


def pose_encoding_to_extri_intri(pose_enc, image_size_hw=None, build_intrinsics=True):
    T, quat, fov_h, fov_w = pose_enc[..., :3], pose_enc[..., 3:7], pose_enc[..., 7], pose_enc[..., 8]
    extrinsics = torch.cat([quat_to_mat(quat), T[..., None]], -1)
    intrinsics = None
    if build_intrinsics:
        H, W = image_size_hw
        intrinsics = torch.zeros(pose_enc.shape[:-1] + (3, 3), device=pose_enc.device, dtype=pose_enc.dtype)
        intrinsics[..., 0, 0] = (W / 2.0) / torch.tan(fov_w / 2.0)
        intrinsics[..., 1, 1] = (H / 2.0) / torch.tan(fov_h / 2.0)
        intrinsics[..., 0, 2] = W / 2
        intrinsics[..., 1, 2] = H / 2
        intrinsics[..., 2, 2] = 1.0
    return extrinsics, intrinsics
