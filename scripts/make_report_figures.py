"""Figures for report/vggt_report.tex (depth predictions on held-out scenes, training curves).

Run from the repository root. Each figure is produced only when its inputs are given / exist, e.g.

    # 3k-step CPU runs (local training outputs)
    python scripts/make_report_figures.py --cpu

    # 15k-step Kaggle GPU runs (logs are in results/kaggle; checkpoint and test set from the Kaggle output)
    python scripts/make_report_figures.py --gpu-curves \
        --gpu-ckpt path/to/runs/aa_long/model.pt --test-data path/to/data/test.pt
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.synthetic import MultiViewDataset  # noqa: E402
from training.train import build_model  # noqa: E402

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
SCENES = [3, 17]
RUNS = [("aa", "Alternating-Attention", "#1f77b4"), ("global", "Global-only", "#d62728")]


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = build_model(ck["cfg"])
    m.load_state_dict(ck["model"])
    return m.eval()


def qualitative_depth(model, ds, out, label):
    """Input views, GT depth and median-scaled predicted depth for SCENES (4 views each)."""
    fig, axes = plt.subplots(len(SCENES) * 3, 4, figsize=(6.6, 5.2 * len(SCENES) / 1.35))
    for si, idx in enumerate(SCENES):
        s = ds.get(idx, torch.arange(4))
        with torch.no_grad():
            pd = model(s["images"][None])["depth"][0, ..., 0]
        for v in range(4):
            gt, p = s["depths"][v], pd[v]
            p = p * gt.median() / p.median()
            vmin, vmax = gt.min().item(), gt.max().item()
            rows = [(s["images"][v].permute(1, 2, 0).numpy(), None, "input"),
                    (gt.numpy(), (vmin, vmax), "GT depth"),
                    (p.numpy(), (vmin, vmax), label)]
            for r, (im, rng, lab) in enumerate(rows):
                ax = axes[si * 3 + r, v]
                if rng is None:
                    ax.imshow(im)
                else:
                    ax.imshow(im, cmap="turbo_r", vmin=rng[0], vmax=rng[1])
                ax.set_xticks([]); ax.set_yticks([])
                if v == 0:
                    ax.set_ylabel(lab, fontsize=7)
                if r == 0:
                    ax.set_title(f"scene {idx}, view {v}", fontsize=7)
    plt.tight_layout(pad=0.2, h_pad=0.5)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("wrote", out)


def training_curves(logs, out, k):
    """Camera / depth / point regression losses (moving average over k log entries, log scale)."""
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.9))
    for name, lab, c in RUNS:
        rec = [json.loads(l) for l in open(logs[name])]
        st = np.array([r["step"] for r in rec])
        for ax, key, title in zip(axes, ["loss_camera", "loss_reg_depth", "loss_reg_point"],
                                  ["camera loss", "depth regression", "point regression"]):
            y = np.array([r[key] for r in rec])
            ys = np.convolve(y, np.ones(k) / k, mode="valid")
            ax.plot(st[k - 1:], ys, color=c, label=lab, lw=1.2)
            ax.set_title(title, fontsize=8); ax.set_xlabel("step", fontsize=7); ax.set_yscale("log")
            ax.tick_params(labelsize=6); ax.grid(alpha=0.3, which="both", lw=0.4)
    axes[0].legend(fontsize=6)
    plt.tight_layout(pad=0.3)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="report/figures", help="output directory")
    ap.add_argument("--test-data", default="data/test.pt", help="held-out synthetic scenes (test.pt)")
    ap.add_argument("--cpu", action="store_true",
                    help="3k-step CPU figures: training_curves.pdf from results/train_log_*.jsonl and "
                         "qualitative_depth.pdf from --cpu-ckpt")
    ap.add_argument("--cpu-ckpt", default="runs/aa/model.pt", help="3k-step CPU AA checkpoint")
    ap.add_argument("--gpu-curves", action="store_true",
                    help="training_curves_gpu.pdf from <--gpu-log-dir>/train_log_{aa,global}_long.jsonl")
    ap.add_argument("--gpu-log-dir", default="results/kaggle", help="directory with the 15k-step GPU logs")
    ap.add_argument("--gpu-ckpt", default=None,
                    help="15k-step GPU AA checkpoint (runs/aa_long/model.pt); writes qualitative_depth_gpu.pdf")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.cpu:
        training_curves({n: f"results/train_log_{n}.jsonl" for n, _, _ in RUNS},
                        os.path.join(a.out, "training_curves.pdf"), k=5)
    if a.gpu_curves:
        # Both logs have ~300 entries (GPU: every 50 steps, CPU: every 10), so the same 5-entry window is used.
        training_curves({n: os.path.join(a.gpu_log_dir, f"train_log_{n}_long.jsonl") for n, _, _ in RUNS},
                        os.path.join(a.out, "training_curves_gpu.pdf"), k=5)

    ckpts = []
    if a.cpu:
        ckpts.append((a.cpu_ckpt, "qualitative_depth.pdf", "pred. depth (AA)"))
    if a.gpu_ckpt:
        ckpts.append((a.gpu_ckpt, "qualitative_depth_gpu.pdf", "pred. depth (AA)"))
    if ckpts:
        ds = MultiViewDataset(torch.load(a.test_data))
        for path, name, label in ckpts:
            qualitative_depth(load_model(path), ds, os.path.join(a.out, name), label)


if __name__ == "__main__":
    main()
