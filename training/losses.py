"""Training objective (paper Sec. 3.4): L = L_camera + L_depth + L_pmap.

* L_camera: L1 between predicted and GT pose encodings, summed over the
  camera head's refinement iterations with weights gamma^(n-1-i). (The paper
  states a Huber loss; the official code uses L1 as more stable.)
* L_depth / L_pmap: aleatoric-uncertainty loss
      sum ||Sigma * (D_hat - D)|| + ||Sigma * (grad D_hat - grad D)|| - alpha * log Sigma
  plus the plain regression term, as in the official training code.
"""

import torch

from vggt.utils.pose_enc import extri_intri_to_pose_encoding


def camera_loss(pose_enc_list, gt_extrinsics, gt_intrinsics, image_hw, gamma=0.6, w_trans=1.0, w_rot=1.0,
                w_fov=0.5):
    gt = extri_intri_to_pose_encoding(gt_extrinsics, gt_intrinsics, image_hw)
    n = len(pose_enc_list)
    lt = lr = lf = 0.0
    for i, pred in enumerate(pose_enc_list):
        w = gamma ** (n - i - 1)
        lt = lt + w * (pred[..., :3] - gt[..., :3]).abs().clamp(max=100).mean()
        lr = lr + w * (pred[..., 3:7] - gt[..., 3:7]).abs().mean()
        lf = lf + w * (pred[..., 7:] - gt[..., 7:]).abs().mean()
    lt, lr, lf = lt / n, lr / n, lf / n
    return w_trans * lt + w_rot * lr + w_fov * lf, {"loss_T": lt, "loss_R": lr, "loss_FL": lf}


def _gradient_loss(pred, gt, mask, conf, alpha):
    """L1 on spatial gradients of the residual, confidence-weighted. pred/gt (N,H,W,C), mask/conf (N,H,W)."""
    m = mask[..., None].float()
    diff = (pred - gt) * m
    gx = (diff[:, :, 1:] - diff[:, :, :-1]).abs() * (m[:, :, 1:] * m[:, :, :-1])
    gy = (diff[:, 1:] - diff[:, :-1]).abs() * (m[:, 1:] * m[:, :-1])
    gx, gy = gx.clamp(max=100), gy.clamp(max=100)
    cx, cy = conf[:, :, 1:, None], conf[:, 1:, :, None]
    gx = gx * cx - alpha * torch.log(cx) * (m[:, :, 1:] * m[:, :, :-1])
    gy = gy * cy - alpha * torch.log(cy) * (m[:, 1:] * m[:, :-1])
    denom = m.sum() * pred.shape[-1]
    return (gx.sum() + gy.sum()) / denom.clamp(min=1)


def _quantile_filter(x, q=0.98, min_elements=1000, hard_max=100):
    if x.numel() <= min_elements:
        return x
    x = x.clamp(max=hard_max)
    keep = x < torch.quantile(x.detach().float()[: 2 ** 24], q)
    return x[keep] if keep.sum() > min_elements else x


def regression_loss(pred, gt, mask, conf, alpha=0.2, grad_scales=4, valid_range=0.98):
    """pred/gt (B,S,H,W,C), mask/conf (B,S,H,W). Returns conf-weighted + gradient + plain regression losses."""
    err = torch.norm(gt[mask] - pred[mask], dim=-1)
    l_conf = _quantile_filter(err * conf[mask] - alpha * torch.log(conf[mask]), valid_range).mean()
    l_reg = _quantile_filter(err, valid_range).mean()
    B, S, H, W, C = pred.shape
    p, g = pred.reshape(B * S, H, W, C), gt.reshape(B * S, H, W, C)
    m, c = mask.reshape(B * S, H, W), conf.reshape(B * S, H, W)
    l_grad = sum(
        _gradient_loss(p[:, ::2 ** s, ::2 ** s], g[:, ::2 ** s, ::2 ** s], m[:, ::2 ** s, ::2 ** s],
                       c[:, ::2 ** s, ::2 ** s], alpha)
        for s in range(grad_scales)
    ) / grad_scales
    return l_conf, l_grad, l_reg


def vggt_loss(preds, batch, w_camera=5.0, w_depth=1.0, w_point=1.0):
    logs = {}
    total = 0.0
    hw = batch["images"].shape[-2:]
    mask = batch["point_masks"]
    if "pose_enc_list" in preds:
        lc, d = camera_loss(preds["pose_enc_list"], batch["extrinsics"], batch["intrinsics"], hw)
        total = total + w_camera * lc
        logs.update(d, loss_camera=lc)
    if "depth" in preds:
        a, b, c = regression_loss(preds["depth"], batch["depths"][..., None], mask, preds["depth_conf"])
        total = total + w_depth * (a + b + c)
        logs.update(loss_conf_depth=a, loss_grad_depth=b, loss_reg_depth=c)
    if "world_points" in preds:
        a, b, c = regression_loss(preds["world_points"], batch["world_points"], mask, preds["world_points_conf"])
        total = total + w_point * (a + b + c)
        logs.update(loss_conf_point=a, loss_grad_point=b, loss_reg_point=c)
    logs["objective"] = total
    return total, logs
