# VGGT: Visual Geometry Grounded Transformer (re-implementation and claim check)

- Paper: Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025, https://arxiv.org/pdf/2503.11651
- Official code: https://github.com/facebookresearch/vggt
- Interactive walkthrough and results: https://platinum-saber.github.io/VGGT/ (source in `docs/`)

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
results/                       JSON results of the Kaggle run
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

## Results (Kaggle, Tesla T4)

All numbers come from one committed run of `notebooks/kaggle_vggt.ipynb` (23 min on a T4); the raw values are in
`results/equivalence_vggt1b.json`, `results/timing.json` and `results/eval_pretrained.json`.

**Same outputs as the official code.** `tests/test_equivalence.py`: 3 passed (random weights with every component incl.
the track head; identical parameter names and shapes for VGGT-1B, 1,256,537,516 parameters). With the released checkpoint
every output is **bit-identical** to the official implementation, max |diff| = 0 for pose_enc, depth, depth_conf,
world_points, world_points_conf, track, vis and conf (fp32; 8 kitchen images on the T4, 2 images on CPU).

```bash
git clone https://github.com/facebookresearch/vggt /path/to/official
VGGT_OFFICIAL=/path/to/official python -m pytest tests -q     # 3 passed
```

**Speed** (fp16, 518 px wide, warm run; peak memory includes 4.7 GiB of weights):

| Frames | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Time (s) | 0.38 | 0.68 | 1.34 | 2.85 | 6.90 | 19.03 |
| Peak GPU memory (GiB) | 6.8 | 7.0 | 7.5 | 8.7 | 9.0 | 9.6 |

**Real images.** The demo ran on the official kitchen (25 images, 2.27 M points), room (8, 0.81 M), LLFF fern
(20, 2.03 M) and LLFF flower (25, 2.54 M) examples, each with depth maps, cameras, track images and a 3D viewer.

**Accuracy against exact ground truth** (50 rendered scenes, 518 x 518; SIFT + 5-point RANSAC is given the true
intrinsics, VGGT-1B is not):

| Method | Views | AUC@30 | AUC@5 | RRA@15 | median rot. err. | median transl. err. | failed pairs |
|---|---|---|---|---|---|---|---|
| SIFT + 5-pt RANSAC | 4 | 0.473 | 0.239 | 0.693 | 2.87 deg | 12.2 deg | 8.3 % |
| **VGGT-1B** | 2 | 0.773 | 0.314 | 1.000 | 0.50 deg | 4.1 deg | 0 |
| **VGGT-1B** | 4 | **0.850** | 0.374 | **1.000** | **0.48 deg** | 3.8 deg | 0 |
| **VGGT-1B** | 8 | 0.849 | **0.413** | **1.000** | 0.51 deg | **3.4 deg** | 0 |

| Views | Depth AbsRel | delta<1.25 | Point head error | Depth + camera error | s / scene |
|---|---|---|---|---|---|
| 2 | 0.044 | 0.968 | **0.0202** | 0.0205 | 0.97 |
| 4 | 0.036 | 0.976 | **0.0164** | 0.0175 | 1.95 |
| 8 | 0.032 | 0.981 | **0.0142** | 0.0156 | 4.37 |

SIFT with 2 / 8 views: AUC@30 0.439 / 0.467, failed pairs 12 % / 8.6 %.

## The paper's claims

| Claim (paper) | Result here | Status |
|---|---|---|
| Released model and architecture as described | bit-identical outputs, all heads incl. tracking | verified |
| Feed-forward, about a second (Table 9) | 0.4-2.9 s for 1-8 frames on a T4 (paper: H100) | consistent |
| Camera pose better than classical methods (Tables 1-2) | AUC@30 0.85 vs 0.47, median rotation 0.48 vs 2.87 deg, no failures | verified |
| Accurate depth (Table 4) | AbsRel 0.032-0.044 zero-shot | verified |
| More views help | depth, points, focal length and AUC@5 improve from 2 to 8 views; AUC@30 saturates at 4 | verified |
| Depth + camera beats the point head (Table 3) | point head 2-10 % better on these scenes | not reproduced here |
| Point tracking | tracks follow scene points on the examples; no ground truth to score | qualitative only |

The ground-truth scenes are synthetic (ray-cast rooms with textured spheres and boxes), not the paper's real benchmarks,
so they test geometric accuracy under exact ground truth on an unseen domain. To rerun:

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
