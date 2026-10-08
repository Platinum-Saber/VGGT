# VGGT: Visual Geometry Grounded Transformer — from-scratch re-implementation

- Paper: Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025 — https://arxiv.org/pdf/2503.11651
- Official code: https://github.com/facebookresearch/vggt

This repository re-implements VGGT (camera, depth and point-map heads) from the paper, verifies it is
**numerically identical** to the official implementation, and reproduces the paper's core claims at small
scale by training on CPU on procedurally rendered multi-view scenes.

## What is implemented

| Paper component (Sec. 3) | File |
|---|---|
| DINOv2-ViT-L/14 + registers patchifier (with pos-embed interpolation) | `vggt/dino.py` |
| Camera token + 4 register tokens per frame; separate learned tokens for the first frame | `vggt/aggregator.py` |
| **Alternating-Attention**: L=24 × (frame-wise self-attn, global self-attn), 2D RoPE, QK-norm, LayerScale | `vggt/aggregator.py`, `vggt/layers.py` |
| Camera head: adaLN-modulated 4-block trunk, 4 iterative refinement steps, `[t, q, fov]` encoding | `vggt/heads/camera_head.py`, `vggt/utils/pose_enc.py` |
| DPT dense heads for depth (`exp`) and point maps (`inv_log`), aleatoric confidence `1 + exp(·)` | `vggt/heads/dpt_head.py` |
| Losses: camera L1 over refinement iterations; confidence-weighted depth/point losses with gradient term and `−α log Σ` | `training/losses.py` |
| Metrics: pairwise RRA / RTA, AUC@30, depth AbsRel/δ<1.25, Sim(3)-aligned point error | `training/metrics.py` |

Not re-implemented: the tracking head (CoTracker-style), and BA post-processing.

## Verification against the official implementation

`tests/test_equivalence.py` builds the official model and ours, loads the **same randomised weights**
(`load_state_dict(strict=True)` — parameter names are identical), and compares outputs on non-square
input (exercising DINOv2 pos-embed interpolation, RoPE, AA, camera trunk and DPT):

```
pose_enc / depth / depth_conf / world_points / world_points_conf   max |diff| = 0.0  (bit-exact)
```

- The default `VGGT()` is the VGGT-1B configuration: 1.19 B parameters (1.26 B in the official model, the
  difference being the tracking head). An official-format 1.26 B checkpoint loads strictly with
  `VGGT.from_checkpoint(path)` (track-head keys dropped) and runs end-to-end through `scripts/demo_pretrained.py`.
- One subtle detail needed for exactness: the official DPT `ResidualConvUnit` applies an *in-place* ReLU to
  its input, so its skip connection carries `relu(x)`, not `x`. The released weights were trained with this.

```bash
git clone https://github.com/facebookresearch/vggt /path/to/official
VGGT_OFFICIAL=/path/to/official python -m pytest tests -q     # 2 passed
```

## Small-scale reproduction (CPU)

The pretrained weights (Hugging Face) were not reachable from the sandbox used here, and it had no GPU (4 CPU cores).
VGGT-1B was trained on 17 datasets with 64 A100s for 9 days, so the reproduction is necessarily small:

- **Data** (`training/synthetic.py`): ray-cast indoor rooms with 3–6 spheres/boxes, solid (3D-consistent) textures,
  2–6 views on an arc around the scene (30–100° spread), random FoV 50–75°, 112×112 px. Exact depth, points and cameras.
  Normalised like the official training code (world = first camera, mean point distance = 1).
  3000 training scenes × 6 views; 200 held-out test scenes (different seed).
- **Model**: `VGGT.small()` — same architecture, conv patchifier (no DINO pre-training), C=192, 6 heads,
  6 AA layers (6 frame + 6 global blocks), light DPT heads: **10.66 M params**.
- **Ablation** (paper Table 5): *global-attention only* with 12 global blocks — **same 10.66 M params**.
- **Training**: AdamW (lr 4e-4, wd 0.05), warm-up + cosine, batch 8, 2–6 random frames per batch, grad-clip 1,
  **3000 steps** (~2.5 h per model on 2 CPU threads). Paper's loss weights (camera ×5, depth ×1, points ×1).

### Results (200 held-out scenes, 4 frames)

| Method | AUC@30 ↑ | RRA@15 ↑ | RTA@15 ↑ | med. rot err ↓ | med. transl-dir err ↓ | Depth AbsRel ↓ | δ<1.25 ↑ | Point err (point head) ↓ | Point err (depth + cam) ↓ |
|---|---|---|---|---|---|---|---|---|---|
| SIFT + 5-pt RANSAC (GT intrinsics) | **0.206** | **0.583** | **0.226** | **10.4°** | 69.6° | – | – | – | – |
| VGGT-small, global attention only | 0.003 | 0.228 | 0.002 | 25.3° | 91.7° | **0.218** | **0.857** | 0.250 | 0.251 |
| VGGT-small, **Alternating-Attention** | 0.042 | 0.377 | 0.038 | 19.4° | **67.8°** | 0.233 | 0.848 | 0.254 | **0.235** |

Point errors are in normalised scene units (mean point distance to origin = 1) after Sim(3) alignment.
Full numbers for 2/4/6 frames are in `results/eval.json`, training logs in `results/train_log_*.jsonl`.

### What reproduces, and what does not

1. **Alternating-Attention > global-only attention at equal parameter count** (paper Table 5) — reproduced for
   camera pose: RRA@15 0.38 vs 0.23, translation-direction error 68° vs 92° (92° ≈ chance), AUC@30 0.042 vs 0.003,
   consistently for 2, 4 and 6 frames. Depth is slightly better for global-only (AbsRel 0.218 vs 0.233), consistent with depth being largely a single-view cue.
2. **Point maps from depth + camera beat the dedicated point head** (paper Table 3, "Ours (Depth + Cam)" vs
   "Ours (Point)") — reproduced for the AA model at every frame count (0.235 vs 0.254 at 4 frames; 0.205 vs 0.227 at 2;
   0.241 vs 0.259 at 6). For the global-only model, whose cameras are worse, the two are tied.
3. **Feed-forward camera pose beating classical SfM** — *not* reproduced at this scale. After 3k CPU steps the
   small model has learned depth/geometry well (δ<1.25 ≈ 0.85) but camera translation is still close to the
   mean pose (rotations are partially learned). The same metrics on training scenes are equally poor,
   i.e. this is **under-training, not over-fitting**; the camera loss was still decreasing at the end.
   The paper's model uses DINOv2 pre-training, 160k steps, batch up to 24 frames × 64 GPUs.

## Running

```bash
pip install torch numpy pillow opencv-python-headless pytest   # (+ huggingface_hub for pretrained weights)

python -m training.train --out runs/aa --aa-order frame global --depth 6 --steps 3000           # renders data/train.pt on first run
python -m training.train --out runs/global --aa-order global --depth 12 --steps 3000
python -m training.evaluate runs/aa runs/global --frames 2 4 6                                  # -> results/eval.json
python -m training.baseline_sift --frames 4

# Pretrained VGGT-1B (needs huggingface.co access; a T4 GPU is enough)
python scripts/demo_pretrained.py --images path/to/*.jpg --out out/                             # depth PNGs, points.ply, cameras.json
```

`--device cuda` and `--init <ckpt>` let you train on a GPU and continue training from a checkpoint.

### Kaggle / Colab (free GPU)

`notebooks/kaggle_pretrained_vggt.ipynb` (GPU + Internet enabled):
1. downloads `facebook/VGGT-1B`, runs the official model and this implementation on the official example images
   and prints the output differences;
2. times inference vs. number of frames;
3. trains the AA and global-only small models for 30k steps (10× the CPU run) and evaluates them.

## Layout

```
vggt/            model: layers, DINOv2 patchifier, aggregator, camera & DPT heads, pose/geometry utils
training/        synthetic data, losses, metrics, train / evaluate / SIFT baseline
tests/           equivalence test vs. official code, VGGT-1B parameter count
scripts/         pretrained-weights demo
notebooks/       Kaggle notebook
results/         evaluation JSON, training logs and configs
```
