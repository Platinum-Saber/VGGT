"""Figures for report/vggt_report.tex (synthetic views, depth predictions, training curves)."""
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

OUT = "report/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif"})

# 1) Input views + GT / predicted depth on held-out scenes (CPU 3k-step models).
ds = MultiViewDataset(torch.load("data/test.pt"))
models = {}
for name in ["aa", "global"]:
    ck = torch.load(f"runs/{name}/model.pt", map_location="cpu")
    m = build_model(ck["cfg"]); m.load_state_dict(ck["model"]); m.eval()
    models[name] = m
scenes = [3, 17]
fig, axes = plt.subplots(len(scenes) * 3, 4, figsize=(6.6, 5.2 * len(scenes) / 1.35))
for si, idx in enumerate(scenes):
    s = ds.get(idx, torch.arange(4))
    with torch.no_grad():
        pd = models["aa"](s["images"][None])["depth"][0, ..., 0]
    for v in range(4):
        gt, p = s["depths"][v], pd[v]
        p = p * gt.median() / p.median()
        vmin, vmax = gt.min().item(), gt.max().item()
        rows = [(s["images"][v].permute(1, 2, 0).numpy(), None, "input"),
                (gt.numpy(), (vmin, vmax), "GT depth"),
                (p.numpy(), (vmin, vmax), "pred. depth (AA)")]
        for r, (im, rng, lab) in enumerate(rows):
            ax = axes[si * 3 + r, v]
            ax.imshow(im, cmap=None if rng is None else "turbo_r", vmin=None if rng is None else rng[0],
                      vmax=None if rng is None else rng[1])
            ax.set_xticks([]); ax.set_yticks([])
            if v == 0:
                ax.set_ylabel(lab, fontsize=7)
            if r == 0:
                ax.set_title(f"scene {idx}, view {v}", fontsize=7)
plt.tight_layout(pad=0.2)
plt.savefig(f"{OUT}/qualitative_depth.pdf", bbox_inches="tight")
plt.close()

# 2) Training curves (CPU runs, 3k steps).
def load(path):
    return [json.loads(l) for l in open(path)]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.9))
for name, lab, c in [("aa", "Alternating-Attention", "#1f77b4"), ("global", "Global-only", "#d62728")]:
    rec = load(f"results/train_log_{name}.jsonl")
    st = np.array([r["step"] for r in rec])
    for ax, key, title in zip(axes, ["loss_camera", "loss_reg_depth", "loss_reg_point"],
                              ["camera loss", "depth regression", "point regression"]):
        y = np.array([r[key] for r in rec])
        k = 5
        ys = np.convolve(y, np.ones(k) / k, mode="valid")
        ax.plot(st[k - 1:], ys, color=c, label=lab, lw=1.2)
        ax.set_title(title, fontsize=8); ax.set_xlabel("step", fontsize=7); ax.set_yscale("log")
        ax.tick_params(labelsize=6); ax.grid(alpha=0.3, which="both", lw=0.4)
axes[0].legend(fontsize=6)
plt.tight_layout(pad=0.3)
plt.savefig(f"{OUT}/training_curves.pdf", bbox_inches="tight")
plt.close()
print("ok")
