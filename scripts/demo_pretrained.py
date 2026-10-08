"""Run the released VGGT-1B weights through this implementation.

Needs internet access to huggingface.co (e.g. Google Colab / Kaggle with a T4 GPU):

    pip install torch torchvision huggingface_hub pillow
    git clone <this repo> && cd VGGT
    python scripts/demo_pretrained.py --images path/to/imgs/*.jpg --out out/

Writes per-frame depth PNGs, a coloured point cloud (points.ply, from depth +
predicted cameras) and cameras.json. Preprocessing follows the official
``load_and_preprocess_images`` ("crop" mode): width resized to 518, height
rounded to a multiple of 14 and centre-cropped to at most 518.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_to_world  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


def load_images(paths, target=518):
    out = []
    for p in paths:
        img = Image.open(p)
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        w, h = img.size
        new_h = round(h * (target / w) / 14) * 14
        img = img.resize((target, new_h), Image.Resampling.BICUBIC)
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255
        if new_h > target:
            top = (new_h - target) // 2
            x = x[:, top:top + target]
        out.append(x)
    assert len({tuple(x.shape) for x in out}) == 1, "all images must share an aspect ratio"
    return torch.stack(out)


def save_ply(path, pts, cols):
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                 % len(pts)).encode())
        rec = np.empty(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = pts, cols
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--checkpoint", default=None, help="local model.pt; default downloads facebook/VGGT-1B")
    ap.add_argument("--conf-percentile", type=float, default=50)
    args = ap.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download("facebook/VGGT-1B", "model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT.from_checkpoint(ckpt).to(device).eval()

    images = load_images(sorted(args.images)).to(device)
    H, W = images.shape[-2:]
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.autocast(device, dtype=dtype, enabled=device == "cuda"):
        tokens, ps = model.aggregator(images[None])
    with torch.no_grad():  # heads in fp32, as in the official model
        tokens = [t.float() if t is not None else None for t in tokens]
        pose_enc = model.camera_head(tokens)[-1]
        depth, depth_conf = model.depth_head(tokens, images[None], ps)
    extr, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))
    pts = unproject_depth_to_world(depth[0, ..., 0], extr[0], intr[0])

    os.makedirs(args.out, exist_ok=True)
    d = depth[0, ..., 0].cpu().numpy()
    for i, di in enumerate(d):
        v = (di - di.min()) / (di.max() - di.min() + 1e-8)
        Image.fromarray((255 * (1 - v)).astype(np.uint8)).save(os.path.join(args.out, f"depth_{i:03d}.png"))
    conf = depth_conf[0].cpu().numpy()
    keep = conf > np.percentile(conf, args.conf_percentile)
    cols = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    save_ply(os.path.join(args.out, "points.ply"), pts.cpu().numpy()[keep].astype(np.float32), cols[keep])
    json.dump({"extrinsics_cam_from_world": extr[0].cpu().tolist(), "intrinsics": intr[0].cpu().tolist(),
               "image_hw": [H, W]}, open(os.path.join(args.out, "cameras.json"), "w"), indent=1)
    print(f"wrote {keep.sum()} points and {len(d)} depth maps to {args.out}")


if __name__ == "__main__":
    main()
