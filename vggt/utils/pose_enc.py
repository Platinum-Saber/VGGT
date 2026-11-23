"""Camera parameterisation g = [t, q, f] (VGGT paper, Sec. 3.1).

"absT_quaR_FoV": translation t (3), rotation quaternion q (4, scalar-last xyzw) and field of view (fov_h, fov_w).
Extrinsics are OpenCV camera-from-world [R | t]; the principal point is the image centre.
"""

import torch

from .rotation import quat_to_mat, mat_to_quat


def extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw=None, pose_encoding_type="absT_quaR_FoV"):
    """extrinsics (B, S, 3, 4), intrinsics (B, S, 3, 3), image_size_hw (H, W) -> (B, S, 9)."""
    if pose_encoding_type != "absT_quaR_FoV":
        raise NotImplementedError
    R = extrinsics[:, :, :3, :3]
    T = extrinsics[:, :, :3, 3]
    quat = mat_to_quat(R)
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    return torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()


def pose_encoding_to_extri_intri(pose_encoding, image_size_hw=None, pose_encoding_type="absT_quaR_FoV",
                                 build_intrinsics=True):
    """(B, S, 9) -> extrinsics (B, S, 3, 4) and intrinsics (B, S, 3, 3) (or None)."""
    if pose_encoding_type != "absT_quaR_FoV":
        raise NotImplementedError
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h, fov_w = pose_encoding[..., 7], pose_encoding[..., 8]
    extrinsics = torch.cat([quat_to_mat(quat), T[..., None]], dim=-1)
    intrinsics = None
    if build_intrinsics:
        H, W = image_size_hw
        fy = (H / 2.0) / torch.tan(fov_h / 2.0)
        fx = (W / 2.0) / torch.tan(fov_w / 2.0)
        intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = W / 2
        intrinsics[..., 1, 2] = H / 2
        intrinsics[..., 2, 2] = 1.0
    return extrinsics, intrinsics
