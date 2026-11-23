"""Zero-shot evaluation of the released VGGT-1B against exact ground truth on rendered scenes.

Renders (or loads) test scenes with exact depth, points and cameras (``evaluation.synthetic``), runs VGGT on the
first S views of every scene and reports the paper's metrics: pairwise pose AUC / RRA / RTA, depth AbsRel and
delta<1.25 after median scaling, and point-map error after Sim(3) alignment for the point head and for depth +
predicted cameras. Optionally evaluates the classical SIFT + 5-point RANSAC baseline on the same pairs.

    python -m evaluation.evaluate_pretrained --num-scenes 50 --frames 2 4 8 --sift
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from evaluation.baseline_sift import evaluate_sift
from evaluation.geometry import unproject_depth_to_world
from evaluation.metrics import auc_at, depth_metrics, point_error, relative_pose_errors
from evaluation.synthetic import MultiViewDataset, generate_dataset
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def load_vggt(checkpoint, device):
    """Released VGGT-1B: a local model.pt, or the Hugging Face checkpoint."""
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download("facebook/VGGT-1B", "model.pt")
    model = VGGT()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, ds, S, device, dtype):
    r_all, t_all = [], []
    absrel, d125, pe_head, pe_dc, focal_err, secs = [], [], [], [], [], []
    for idx in range(ds.n):
        s = ds.get(idx, torch.arange(S))
        images = s["images"].to(device)
        H, W = images.shape[-2:]
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
            pred = model(images)
        if device == "cuda":
            torch.cuda.synchronize()
        secs.append(time.time() - t0)

        extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], (H, W))
        extr, intr = extr[0].float().cpu(), intr[0].float().cpu()
        depth = pred["depth"][0, ..., 0].float().cpu()
        points = pred["world_points"][0].float().cpu()

        r, t = relative_pose_errors(extr, s["extrinsics"])
        r_all.append(r)
        t_all.append(t)
        focal_err.append(float((intr[:, 0, 0] / s["intrinsics"][:, 0, 0] - 1).abs().mean()))
        mask = s["point_masks"]
        a, d = depth_metrics(depth, s["depths"], mask)
        absrel.append(a)
        d125.append(d)
        pe_head.append(point_error(points, s["world_points"], mask))
        pe_dc.append(point_error(unproject_depth_to_world(depth, extr, intr), s["world_points"], mask))

    r, t = torch.cat(r_all), torch.cat(t_all)
    return {
        "AUC@30": auc_at(r, t, 30), "AUC@15": auc_at(r, t, 15), "AUC@5": auc_at(r, t, 5),
        "RRA@15": float((r < 15).float().mean()), "RTA@15": float((t < 15).float().mean()),
        "median_rot_err_deg": float(r.median()), "median_trans_err_deg": float(t.median()),
        "focal_rel_err": float(np.mean(focal_err)),
        "depth_AbsRel": float(np.mean(absrel)), "depth_delta<1.25": float(np.mean(d125)),
        "point_err_pointhead": float(np.mean(pe_head)), "point_err_depth+cam": float(np.mean(pe_dc)),
        "sec_per_scene": float(np.mean(secs[1:] if len(secs) > 1 else secs)),  # first call includes warm-up
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="local VGGT-1B model.pt (default: download from Hugging Face)")
    ap.add_argument("--data", default="data/test_518.pt")
    ap.add_argument("--num-scenes", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--frames", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--sift", action="store_true", help="also evaluate SIFT + 5-point RANSAC on the same pairs")
    ap.add_argument("--out", default="results/eval_pretrained.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    num_views = max(args.frames)
    if os.path.exists(args.data):
        data = torch.load(args.data)
    else:
        os.makedirs(os.path.dirname(args.data) or ".", exist_ok=True)
        t = time.time()
        data = generate_dataset(args.data, args.num_scenes, img_size=args.img_size, num_views=num_views, seed=999_999)
        print(f"rendered {args.num_scenes} scenes x {num_views} views at {args.img_size}px in {time.time() - t:.0f}s",
              flush=True)
    ds = MultiViewDataset(data)
    assert ds.V >= num_views, f"{args.data} has {ds.V} views per scene, need {num_views}"

    dtype = torch.bfloat16 if args.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = load_vggt(args.checkpoint, args.device)

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for S in args.frames:
        res = evaluate(model, ds, S, args.device, dtype)
        results[f"vggt1b/S={S}"] = res
        print("VGGT-1B", S, json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
        if args.sift:
            res = evaluate_sift(ds, S)
            results[f"sift_5pt_ransac_gtK/S={S}"] = res
            print("SIFT   ", S, json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
