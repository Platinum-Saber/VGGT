"""Evaluate trained checkpoints on held-out procedural scenes.

    python -m training.evaluate runs/aa runs/global --frames 4 --out results/eval.json
"""

import argparse
import json
import os

import numpy as np
import torch

from training.metrics import auc_at, depth_metrics, point_error, relative_pose_errors
from training.synthetic import MultiViewDataset, generate_dataset
from training.train import build_model
from vggt.utils.geometry import unproject_depth_to_world
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


@torch.no_grad()
def evaluate(model, ds, S):
    model.eval()
    r_all, t_all, absrel, d125, pe_head, pe_dc, fov_err = [], [], [], [], [], [], []
    for idx in range(ds.n):
        s = ds.get(idx, torch.arange(S))
        img = s["images"]
        H, W = img.shape[-2:]
        p = model(img[None])
        extr, intr = pose_encoding_to_extri_intri(p["pose_enc"], (H, W))
        extr, intr = extr[0], intr[0]
        r, t = relative_pose_errors(extr, s["extrinsics"])
        r_all.append(r)
        t_all.append(t)
        fov_err.append((intr[:, 0, 0] / s["intrinsics"][:, 0, 0] - 1).abs().mean())
        mask = s["point_masks"]
        a, d = depth_metrics(p["depth"][0, ..., 0], s["depths"], mask)
        absrel.append(a)
        d125.append(d)
        pe_head.append(point_error(p["world_points"][0], s["world_points"], mask))
        pts_dc = unproject_depth_to_world(p["depth"][0, ..., 0], extr, intr)  # depth + camera
        pe_dc.append(point_error(pts_dc, s["world_points"], mask))
    r, t = torch.cat(r_all), torch.cat(t_all)
    return {
        "AUC@30": auc_at(r, t, 30), "AUC@15": auc_at(r, t, 15), "AUC@5": auc_at(r, t, 5),
        "RRA@15": float((r < 15).float().mean()), "RTA@15": float((t < 15).float().mean()),
        "median_rot_err_deg": float(r.median()), "median_trans_err_deg": float(t.median()),
        "focal_rel_err": float(torch.stack(fov_err).mean()),
        "depth_AbsRel": float(np.mean(absrel)), "depth_delta<1.25": float(np.mean(d125)),
        "point_err_pointhead": float(np.mean(pe_head)), "point_err_depth+cam": float(np.mean(pe_dc)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default="data/test.pt")
    ap.add_argument("--num-scenes", type=int, default=200)
    ap.add_argument("--frames", type=int, nargs="+", default=[4])
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    if os.path.exists(args.data):
        data = torch.load(args.data)
    else:
        os.makedirs(os.path.dirname(args.data) or ".", exist_ok=True)
        data = generate_dataset(args.data, args.num_scenes, seed=999_999)  # disjoint seed from training
    ds = MultiViewDataset(data)
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for run in args.runs:
        ck = torch.load(os.path.join(run, "model.pt"))
        model = build_model(ck["cfg"])
        model.load_state_dict(ck["model"])
        for S in args.frames:
            res = evaluate(model, ds, S)
            res.update(step=ck["step"], aa_order=ck["cfg"]["aa_order"])
            results[f"{os.path.basename(run.rstrip('/'))}/S={S}"] = res
            print(run, S, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}),
                  flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
