"""Evaluation metrics used in the VGGT paper.

* Camera pose (Sec. 4.1): for every pair of frames, the relative rotation
  error (RRA) and relative translation-direction error (RTA) in degrees;
  AUC@30 is the area under the accuracy curve of max(RRA, RTA) for
  thresholds 1..30 degrees.
* Depth (Sec. 4.3): AbsRel and delta<1.25, after a per-sequence median scale
  alignment (VGGT predicts depth up to one global scale per sequence).
* Point maps (Sec. 4.3): predictions are aligned to GT with a similarity
  (Umeyama) transform; we report the mean Euclidean point error.
"""

import numpy as np
import torch

from vggt.utils.geometry import inverse_se3, to_homogeneous


def _rotation_angle_deg(R):
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def relative_pose_errors(pred_extr, gt_extr):
    """pred/gt (S, 3, 4) camera-from-world. Returns RRA, RTA (degrees) over all ordered pairs i != j."""
    P, G = to_homogeneous(pred_extr.double()), to_homogeneous(gt_extr.double())
    S = P.shape[0]
    i, j = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    keep = i != j
    i, j = i[keep], j[keep]
    rel_p = P[j] @ inverse_se3(P[i])  # cam_j from cam_i
    rel_g = G[j] @ inverse_se3(G[i])
    r_err = _rotation_angle_deg(rel_p[:, :3, :3].transpose(-1, -2) @ rel_g[:, :3, :3])
    tp, tg = rel_p[:, :3, 3], rel_g[:, :3, 3]
    cos = (tp * tg).sum(-1) / (tp.norm(dim=-1) * tg.norm(dim=-1)).clamp(min=1e-12)
    t_err = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    return r_err.float(), t_err.float()


def auc_at(r_err, t_err, max_threshold=30):
    err = torch.maximum(r_err, t_err).cpu().numpy()
    bins = np.arange(max_threshold + 1)
    hist, _ = np.histogram(err, bins=bins)
    acc = np.cumsum(hist) / len(err)
    return float(acc.mean())


def depth_metrics(pred, gt, mask):
    """pred/gt (S, H, W), mask (S, H, W). One median scale per sequence."""
    p, g = pred[mask], gt[mask]
    p = p * (g.median() / p.median().clamp(min=1e-8))
    absrel = ((p - g).abs() / g).mean()
    delta = torch.maximum(p / g, g / p)
    return float(absrel), float((delta < 1.25).float().mean())


def umeyama_sim3(src, dst):
    """Least-squares s, R, t with dst ~ s R src + t. src/dst (N, 3)."""
    src, dst = src.double(), dst.double()
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / src.shape[0]
    U, D, Vt = torch.linalg.svd(cov)
    E = torch.eye(3, dtype=src.dtype)
    if torch.det(U) * torch.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (D * E.diagonal()).sum() / xs.pow(2).sum(-1).mean()
    t = mu_d - s * R @ mu_s
    return s, R, t


def point_error(pred_pts, gt_pts, mask):
    """Mean L2 error after Sim(3) alignment. pred/gt (S, H, W, 3)."""
    p, g = pred_pts[mask], gt_pts[mask]
    s, R, t = umeyama_sim3(p, g)
    aligned = s * p.double() @ R.T + t
    return float((aligned - g.double()).norm(dim=-1).mean())
