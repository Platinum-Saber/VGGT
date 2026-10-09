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

### With the released VGGT-1B weights (Kaggle, Tesla T4)

`notebooks/kaggle_pretrained_vggt.ipynb` loads `facebook/VGGT-1B` into both the official model and this
implementation and runs them (fp32) on the 8 first images of the official `examples/kitchen` scene:

```
pose_enc           max|diff|=0.00e+00  mean|diff|=0.00e+00  max|ref|=1.36e+00
depth              max|diff|=0.00e+00  mean|diff|=0.00e+00  max|ref|=3.57e+00
depth_conf         max|diff|=0.00e+00  mean|diff|=0.00e+00  max|ref|=3.03e+01
world_points       max|diff|=0.00e+00  mean|diff|=0.00e+00  max|ref|=2.86e+00
world_points_conf  max|diff|=0.00e+00  mean|diff|=0.00e+00  max|ref|=3.16e+01
```

i.e. this implementation reproduces the released model's camera, depth and point-map predictions exactly.

Inference with this implementation (T4, fp16 autocast, 518 px wide, aggregator + camera/depth/point heads;
peak memory includes the ~4.7 GiB of fp32 weights; the 1-frame run includes GPU warm-up):

| Frames | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Time (s) | 0.76 | 0.59 | 1.05 | 2.21 | 5.56 | 15.68 |
| Peak GPU memory (GiB) | 7.2 | 7.3 | 7.9 | 8.9 | 9.2 | 9.8 |

Exporting the full 25-image kitchen scene with `scripts/demo_pretrained.py` produced 25 depth maps and a
2.27 M-point cloud (depth + predicted cameras, top-50% confidence).

## Small-scale reproduction (trained from scratch)

The pretrained weights (Hugging Face) were not reachable from the sandbox used for training, and it had no GPU (4 CPU cores).
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

### Longer training on a GPU (Kaggle T4, 15k steps)

Same data, model, losses and test set; batch 16 instead of 8 and 15,000 steps instead of 3,000
(10× more training samples), warm-up 1000 steps (`notebooks/kaggle_pretrained_vggt.ipynb`, Part 2).
The SIFT baseline numbers are identical to the CPU run, confirming the same test set.

| Method (4 frames) | AUC@30 ↑ | AUC@15 ↑ | AUC@5 ↑ | RRA@15 ↑ | RTA@15 ↑ | med. rot err ↓ | med. transl-dir err ↓ | failed pairs | Depth AbsRel ↓ | δ<1.25 ↑ | Point err (point head) ↓ | Point err (depth + cam) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SIFT + 5-pt RANSAC (GT intrinsics) | 0.206 | **0.135** | **0.050** | 0.583 | 0.226 | 10.4° | 69.6° | 21.6 % | – | – | – | – |
| VGGT-small, global attention only | 0.003 | 0.000 | 0.000 | 0.231 | 0.002 | 25.3° | 92.4° | 0 % | **0.087** | **0.937** | 0.110 | 0.212 |
| VGGT-small, **Alternating-Attention** | **0.270** | 0.082 | 0.004 | **0.865** | **0.269** | **7.0°** | **23.6°** | 0 % | **0.087** | **0.937** | 0.099 | **0.096** |

AA model at 2 / 4 / 6 frames: AUC@30 0.249 / 0.270 / 0.272, RRA@15 0.860 / 0.865 / 0.872 (more frames help slightly).
Full numbers: `results/eval_long_kaggle.json`.

### What reproduces, and what does not

1. **Feed-forward camera pose beats classical two-view geometry on AUC@30** (paper Tables 1–2, at toy scale):
   after 15k steps the AA model reaches AUC@30 0.270 vs 0.206 for SIFT + 5-point RANSAC, even though the baseline
   is given the GT intrinsics. It has far lower median errors (rotation 7.0° vs 10.4°, translation direction 23.6° vs 69.6°)
   and never fails, while SIFT fails on 22 % of pairs (wide baselines, low resolution, repetitive textures).
   SIFT is still better at tight thresholds (AUC@5 0.050 vs 0.004): when it succeeds it is more precise, matching the
   paper's observation that feed-forward predictions benefit from refinement (e.g. BA) for high accuracy.
   This was *not* visible after 3k CPU steps (AUC@30 0.042): the camera head was under-trained, not over-fitting.
2. **Alternating-Attention ≫ global-only attention at equal parameter count** (paper Table 5) — reproduced, and the
   gap grows with training: global-only stays at chance for translation direction (≈ 92°) and ~25° rotation error at
   both 3k and 15k steps, while AA improves to 23.6° / 7.0°. Depth quality is the same for both (AbsRel 0.087),
   consistent with depth being largely a single-view cue; the difference is in cross-view reasoning.
3. **Point maps from depth + camera beat the dedicated point head** (paper Table 3) — reproduced for the AA model at
   every frame count and both training lengths (15k: 0.086 vs 0.089 at 2 frames, 0.096 vs 0.099 at 4, 0.098 vs 0.101
   at 6; 3k: 0.235 vs 0.254 at 4 frames). The margin is small. For global-only, whose cameras are poor, depth + camera
   is much worse (0.212 vs 0.110), as expected: the combination is only as good as the predicted cameras.

Caveats: one seed per configuration, a synthetic domain, a 10.7 M-parameter model without DINOv2 pre-training,
and a two-view classical baseline rather than full SfM (COLMAP / VGGSfM). The paper's model uses
DINOv2 initialisation, 160k steps and up to 24 frames × 64 A100s.

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
