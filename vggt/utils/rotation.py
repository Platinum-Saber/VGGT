"""Quaternion <-> rotation matrix conversion. Quaternions are scalar-last (x, y, z, w)."""

import torch
import torch.nn.functional as F


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    i, j, k, r = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    m = torch.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], -1)
    return m.reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Numerically stable conversion (picks the best-conditioned of four candidates)."""
    batch = R.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = R.reshape(batch + (9,)).unbind(-1)
    q_abs = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                         1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1).clamp(min=0).sqrt()
    cand = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1),
    ], -2)
    cand = cand / (2.0 * q_abs[..., None].clamp(min=0.1))
    out = cand[F.one_hot(q_abs.argmax(-1), 4) > 0.5].reshape(batch + (4,))
    out = out[..., [1, 2, 3, 0]]  # rijk -> ijkr
    return torch.where(out[..., 3:4] < 0, -out, out)
