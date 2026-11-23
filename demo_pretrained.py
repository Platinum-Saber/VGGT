"""Run the released VGGT-1B on a set of images and export the results, including an interactive 3D viewer.

    python demo_pretrained.py --images path/to/images --out out/

Writes to ``out/``: per-frame depth PNGs, ``points.ply`` (unprojected depth with predicted cameras, filtered by
depth confidence), ``cameras.json``, ``tracks_grid.png`` (tracks of a grid of query points from frame 0) and
``viewer.html`` (open in a browser: point cloud, cameras, 3D tracks). Preprocessing and inference follow the
official README: ``load_and_preprocess_images`` and bf16/fp16 autocast on GPU.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from scripts.export_viewer import write_viewer


def list_images(spec):
    if len(spec) == 1 and os.path.isdir(spec[0]):
        spec = [os.path.join(spec[0], f) for f in os.listdir(spec[0])]
    paths = []
    for s in spec:
        paths += glob.glob(s) if any(ch in s for ch in "*?[") else [s]
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted(p for p in paths if p.lower().endswith(exts))


def load_model(checkpoint, low_memory=False):
    """VGGT-1B from a local model.pt or from Hugging Face. low_memory memory-maps the file instead of copying it."""
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download("facebook/VGGT-1B", "model.pt")
    state = torch.load(checkpoint, map_location="cpu", mmap=low_memory, weights_only=True)
    if low_memory:
        model = _build_on_meta()
        model.load_state_dict(state, strict=True, assign=True)
    else:
        model = VGGT()
        model.load_state_dict(state, strict=True)
    return model


def _build_on_meta():
    """Construct VGGT with parameters on the meta device (no memory), to be filled with load_state_dict(assign=True)."""
    orig = torch.nn.Module.register_parameter

    def register(module, name, param):
        orig(module, name, param)
        if param is not None:
            module._parameters[name] = torch.nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)

    torch.nn.Module.register_parameter = register
    try:
        return VGGT()
    finally:
        torch.nn.Module.register_parameter = orig


def save_ply(path, pts, cols):
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                 % len(pts)).encode())
        rec = np.empty(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = pts, cols
        f.write(rec.tobytes())


def lift_tracks(tracks, vis, depth, extrinsics, intrinsics, conf=None, thr=0.5):
    """2D tracks (S, N, 2) -> 3D (N, S, 3) by reading each frame's predicted depth.

    NaN where the tracker says the point is not visible (or not confident). For a static scene a well-tracked
    point lands on the same 3D location in every frame, so the 3D tracks show multi-view consistency.
    """
    S, N, _ = tracks.shape
    H, W = depth.shape[1:3]
    out = np.full((N, S, 3), np.nan, dtype=np.float32)
    for s in range(S):
        x, y = tracks[s, :, 0], tracks[s, :, 1]
        good = vis[s] > thr if conf is None else (vis[s] > thr) & (conf[s] > thr)
        ok = good & (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
        xi, yi = np.round(x[ok]).astype(int), np.round(y[ok]).astype(int)
        d = depth[s, yi, xi]
        K, E = intrinsics[s], np.vstack([extrinsics[s], [0, 0, 0, 1]])
        cam = np.stack([(x[ok] - K[0, 2]) / K[0, 0] * d, (y[ok] - K[1, 2]) / K[1, 1] * d, d], 1)
        c2w = np.linalg.inv(E)
        out[ok, s] = cam @ c2w[:3, :3].T + c2w[:3, 3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="a folder, a glob, or image files")
    ap.add_argument("--out", default="out")
    ap.add_argument("--checkpoint", default=None, help="local model.pt; default downloads facebook/VGGT-1B")
    ap.add_argument("--mode", default="crop", choices=["crop", "pad"])
    ap.add_argument("--max-frames", type=int, default=0, help="use only the first N images (0 = all)")
    ap.add_argument("--conf-percentile", type=float, default=50, help="PLY keeps points above this depth-conf percentile")
    ap.add_argument("--track-grid", type=int, default=12, help="track a G x G grid of frame-0 points (0 = no tracks)")
    ap.add_argument("--max-points", type=int, default=250_000, help="points embedded in viewer.html")
    ap.add_argument("--save-npz", action="store_true", help="also save all predictions to predictions.npz")
    ap.add_argument("--low-memory", action="store_true", help="memory-map the checkpoint (for machines with little RAM)")
    args = ap.parse_args()

    paths = list_images(args.images)
    if args.max_frames:
        paths = paths[: args.max_frames]
    assert paths, "no images found"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = load_model(args.checkpoint, args.low_memory).to(device).eval()
    images = load_and_preprocess_images(paths, mode=args.mode).to(device)
    S, _, H, W = images.shape

    query = None
    if args.track_grid > 0:
        g = args.track_grid
        ys, xs = torch.meshgrid(torch.linspace(0.08, 0.92, g) * (H - 1), torch.linspace(0.08, 0.92, g) * (W - 1),
                                indexing="ij")
        query = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1).to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
            pred = model(images, query_points=query)

    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], (H, W))
    extr, intr = extr[0].cpu().numpy(), intr[0].cpu().numpy()
    depth = pred["depth"][0].float().cpu().numpy()  # (S, H, W, 1)
    depth_conf = pred["depth_conf"][0].float().cpu().numpy()
    world_points = pred["world_points"][0].float().cpu().numpy()
    world_points_conf = pred["world_points_conf"][0].float().cpu().numpy()
    points = unproject_depth_map_to_point_map(depth, extr, intr)  # depth + predicted cameras
    rgb = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    os.makedirs(args.out, exist_ok=True)
    for i, d in enumerate(depth[..., 0]):
        v = (d - d.min()) / (d.max() - d.min() + 1e-8)
        Image.fromarray((255 * (1 - v)).astype(np.uint8)).save(os.path.join(args.out, f"depth_{i:03d}.png"))
    keep = depth_conf > np.percentile(depth_conf, args.conf_percentile)
    save_ply(os.path.join(args.out, "points.ply"), points[keep].astype(np.float32), rgb[keep])
    json.dump({"images": [os.path.basename(p) for p in paths], "extrinsics_cam_from_world": extr.tolist(),
               "intrinsics": intr.tolist(), "image_hw": [H, W]},
              open(os.path.join(args.out, "cameras.json"), "w"), indent=1)

    tracks3d = None
    if query is not None:
        from vggt.utils.visual_track import visualize_tracks_on_images
        tracks = pred["track"][0].float().cpu()
        vis = pred["vis"][0].float().cpu()
        visualize_tracks_on_images(images.cpu(), tracks, vis > 0.5, out_dir=os.path.join(args.out, "tracks"))
        tconf = pred["conf"][0].float().cpu().numpy()
        tracks3d = lift_tracks(tracks.numpy(), vis.numpy(), depth[..., 0], extr, intr, conf=tconf)

    folder = os.path.dirname(os.path.abspath(paths[0]))
    scene = os.path.basename(os.path.dirname(folder)) if os.path.basename(folder) == "images" else os.path.basename(folder)
    write_viewer(os.path.join(args.out, "viewer.html"), points, rgb, depth_conf, extr, intr, images=rgb,
                 tracks3d=tracks3d, points_alt=world_points, conf_alt=world_points_conf,
                 title=f"VGGT-1B: {scene} ({S} images)", max_points=args.max_points)
    if args.save_npz:
        np.savez_compressed(os.path.join(args.out, "predictions.npz"), images=rgb, extrinsics=extr, intrinsics=intr,
                            depth=depth.astype(np.float16), depth_conf=depth_conf.astype(np.float16),
                            world_points=world_points.astype(np.float16),
                            world_points_conf=world_points_conf.astype(np.float16),
                            points_depth_cam=points.astype(np.float16),
                            **({"tracks3d": tracks3d} if tracks3d is not None else {}))
    print(f"wrote {int(keep.sum())} points, {S} depth maps, cameras and viewer.html to {args.out}")


if __name__ == "__main__":
    main()
