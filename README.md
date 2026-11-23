# VGGT: Visual Geometry Grounded Transformer (re-implementation and claim check)

- Paper: Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025, https://arxiv.org/pdf/2503.11651
- Official code: https://github.com/facebookresearch/vggt

This repository re-implements the original VGGT package **without modifications**: the same folder layout, class
and function names, defaults and parameter names as `facebookresearch/vggt`, including the tracking head. Meta's
released `facebook/VGGT-1B` weights load unchanged (`strict=True`). VGGT-1B is **not retrained** (that took
64 A100 GPUs for 9 days on 17 datasets); instead we check that the re-implementation is exact and that the
paper's claims hold for the released model, on real example images and against exact ground truth.

## Layout (mirrors the official package)

```
vggt/models/vggt.py            VGGT(img_size, patch_size, embed_dim, enable_camera/point/depth/track)
vggt/models/aggregator.py      Alternating-Attention aggregator (frame + global blocks, camera/register tokens)
vggt/layers/                   attention (QK-norm), block (LayerScale, drop path), rope (2D RoPE), mlp, patch_embed,
                               swiglu_ffn, drop_path, layer_scale, vision_transformer (DINOv2 ViT with registers)
vggt/heads/camera_head.py      iterative camera refinement with adaLN, pose activations in head_act.py
vggt/heads/dpt_head.py         DPT dense head (depth, point map, and feature-only mode for tracking), utils.py
vggt/heads/track_head.py       tracking: DPT features + track_modules/ (correlation pyramid, update transformer)
vggt/utils/                    pose_enc, rotation, geometry, load_fn, helper, visual_track
demo_pretrained.py             run VGGT-1B on images: depth, point cloud, cameras, tracks, 3D viewer
scripts/export_viewer.py       self-contained interactive 3D viewer (one HTML file)
evaluation/                    rendered test scenes with exact GT, paper metrics, SIFT baseline, evaluate_pretrained.py
tests/test_equivalence.py      equivalence with the official code
notebooks/kaggle_vggt.ipynb    everything above on a Kaggle GPU
report/                        LaTeX report and code walkthrough
```

Not included: the official `vggt/dependency/` folder (VGGSfM tracker, bundle adjustment and COLMAP export used by the
official `demo_colmap.py`). It is post-processing outside the network; the geometry helpers raise
`NotImplementedError` if lens-distortion parameters (which live there) are passed.

```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

model = VGGT()
model.load_state_dict(torch.load("model.pt"))           # facebook/VGGT-1B
images = load_and_preprocess_images(["a.png", "b.png"])
predictions = model(images, query_points=torch.tensor([[100.0, 200.0]]))
```

## Verified so far

**Same outputs as the official code.** `tests/test_equivalence.py` (run against an official checkout):

- random weights, reduced config with every component (DINOv2-S patchifier, RoPE, Alternating-Attention, camera trunk,
  DPT and track heads), non-square input: pose, depth, points, confidences, tracks, visibility and track confidence
  all match;
- the full VGGT-1B configurations have identical parameter names and shapes; 1,256,537,516 parameters.

With the released checkpoint (CPU, fp32, 2 kitchen images at 350x518, 3 query points) every output is
**bit-identical** to the official implementation (max |diff| = 0 for pose_enc, depth, depth_conf, world_points,
world_points_conf, track, vis, conf). The notebook repeats this on a GPU with 8 images.

```bash
git clone https://github.com/facebookresearch/vggt /path/to/official
VGGT_OFFICIAL=/path/to/official python -m pytest tests -q     # 3 passed
```

**Speed** (earlier Kaggle T4 run of the same network, fp16, 518 px wide; peak memory includes 4.7 GiB of weights):

| Frames | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Time (s) | 0.76 | 0.59 | 1.05 | 2.21 | 5.56 | 15.68 |
| Peak GPU memory (GiB) | 7.2 | 7.3 | 7.9 | 8.9 | 9.2 | 9.8 |

**Real images.** On the official 25-image kitchen example the model produces 25 depth maps, a 2.27 M-point cloud
and cameras on a ring around the object (`results/kaggle/kitchen_cameras.json`, figures in the report).

To be filled in by the Kaggle rerun (`notebooks/kaggle_vggt.ipynb`): GPU equivalence incl. tracks, timing, the
example scenes in the 3D viewer, and the ground-truth evaluation below.

## Checking the paper's claims against exact ground truth

Real benchmarks (CO3D, RealEstate10K, DTU, ETH3D) are too large for a Kaggle session, so `evaluation/` renders test
scenes with exact depth, points and cameras (ray-cast rooms with solid-textured objects, 518 x 518, 8 views each) and
evaluates the released VGGT-1B zero-shot with the paper's metrics:

| Claim (paper) | Measured by |
|---|---|
| Feed-forward camera prediction is competitive with or better than classical methods (Tables 1-2) | pairwise pose AUC@30/15/5, RRA@15, RTA@15 vs. SIFT + 5-point RANSAC given the GT intrinsics |
| Accurate depth (Table 4) | AbsRel and delta<1.25 after per-sequence median scaling |
| Point maps from depth + predicted cameras beat the point head (Table 3) | point error after Sim(3) alignment for both |
| More views help | all metrics for 2, 4 and 8 frames |
| Fast (Table 9) | seconds per scene |

```bash
python -m evaluation.evaluate_pretrained --num-scenes 50 --frames 2 4 8 --sift   # -> results/eval_pretrained.json
```

## 3D viewer

`demo_pretrained.py` writes `viewer.html` next to the other outputs: a single file (three.js from a CDN) with the
point cloud (depth + camera, or the point head), the predicted camera frustums with their images, 3D tracks, a
confidence slider (keep the top x % of points), point size, and "look through this camera" by clicking an image.
It opens in any browser and is displayed inline in the Kaggle notebook.

```bash
python demo_pretrained.py --images path/to/images --out out/          # out/viewer.html, points.ply, cameras.json, tracks/
python scripts/export_viewer.py out/predictions.npz out/viewer.html   # rebuild the viewer from --save-npz output
```

## Kaggle

`notebooks/kaggle_vggt.ipynb` (GPU T4, Internet on): clones this repository (branch `claude/clever-curie-rl6bco`
until merged) and the official one, downloads `facebook/VGGT-1B`, then runs (1) the equivalence tests and the
released-weights comparison incl. tracks, (2) timing for 1 to 32 frames, (3) the demo with the 3D viewer on the
kitchen, room and LLFF examples, (4) the ground-truth evaluation with the SIFT baseline. Results are written to
`/kaggle/working/results` and zipped to `vggt_results.zip`.

## Report

`report/vggt_report.pdf` (source `report/vggt_report.tex`): method, verification, results and the claim-by-claim
check. `report/code_walkthrough.pdf` (source `report/code_walkthrough.tex`): code-review companion that splits the
architecture into phases, each with its source file, input/output tensors and shapes, implementation and
mathematics. Rebuild with `cd report && pdflatex vggt_report.tex && pdflatex vggt_report.tex` (or `tectonic`).

The license of the original code and weights (VGGT License, Meta) applies to the released weights used here.
