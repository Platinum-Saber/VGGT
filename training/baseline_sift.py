"""Classical two-view baseline: SIFT + ratio test + 5-point RANSAC (OpenCV), with GT intrinsics.

For every ordered pair of the first S views of each test scene, estimates the
relative pose from the essential matrix and reports the same pairwise
RRA / RTA / AUC as ``training.evaluate``. Failed pairs count as 180 deg error.

    python -m training.baseline_sift --frames 4
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch

from training.metrics import auc_at
from training.synthetic import MultiViewDataset
from vggt.utils.geometry import inverse_se3, to_homogeneous


def rel_pose(img_i, img_j, K_i, K_j, sift, matcher):
    gi = cv2.cvtColor(img_i, cv2.COLOR_RGB2GRAY)
    gj = cv2.cvtColor(img_j, cv2.COLOR_RGB2GRAY)
    ki, di = sift.detectAndCompute(gi, None)
    kj, dj = sift.detectAndCompute(gj, None)
    if di is None or dj is None or len(ki) < 8 or len(kj) < 8:
        return None
    good = [m for m, n in (p for p in matcher.knnMatch(di, dj, k=2) if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < 8:
        return None
    pi = np.float64([ki[m.queryIdx].pt for m in good])
    pj = np.float64([kj[m.trainIdx].pt for m in good])
    # Normalise with each camera's intrinsics, then estimate E with identity K.
    ni = cv2.undistortPoints(pi[:, None], K_i, None)[:, 0]
    nj = cv2.undistortPoints(pj[:, None], K_j, None)[:, 0]
    E, inl = cv2.findEssentialMat(ni, nj, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1e-3)
    if E is None or E.shape != (3, 3):
        return None
    _, R, t, _ = cv2.recoverPose(E, ni, nj, np.eye(3), mask=inl)
    return R, t[:, 0]


def angle(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.pt")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    ds = MultiViewDataset(torch.load(args.data))
    sift, matcher = cv2.SIFT_create(), cv2.BFMatcher()
    r_errs, t_errs, fails = [], [], 0
    for idx in range(ds.n):
        s = ds.get(idx, torch.arange(args.frames))
        imgs = (s["images"].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        G = to_homogeneous(s["extrinsics"].double())
        Ks = s["intrinsics"].double().numpy()
        for i in range(args.frames):
            for j in range(args.frames):
                if i == j:
                    continue
                gt = (G[j] @ inverse_se3(G[i])).numpy()  # cam_j from cam_i
                est = rel_pose(imgs[i], imgs[j], Ks[i], Ks[j], sift, matcher)
                if est is None:
                    fails += 1
                    r_errs.append(180.0)
                    t_errs.append(180.0)
                    continue
                R, t = est
                cos = (np.trace(R.T @ gt[:3, :3]) - 1) / 2
                r_errs.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                t_errs.append(angle(t, gt[:3, 3]))
    r, t = torch.tensor(r_errs), torch.tensor(t_errs)
    res = {"AUC@30": auc_at(r, t, 30), "AUC@15": auc_at(r, t, 15), "AUC@5": auc_at(r, t, 5),
           "RRA@15": float((r < 15).float().mean()), "RTA@15": float((t < 15).float().mean()),
           "median_rot_err_deg": float(r.median()), "median_trans_err_deg": float(t.median()),
           "failed_pairs": fails / len(r_errs)}
    print(json.dumps(res, indent=1))
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    results[f"sift_5pt_ransac_gtK/S={args.frames}"] = res
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
