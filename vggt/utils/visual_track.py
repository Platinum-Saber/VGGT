"""Draw predicted point tracks on the input frames (colour = position of the track in its first visible frame)."""

import os

import numpy as np
import torch


def color_from_xy(x, y, W, H, cmap_name="hsv"):
    import matplotlib
    c = (x / max(W - 1, 1) + y / max(H - 1, 1)) / 2.0
    r, g, b, _ = matplotlib.colormaps[cmap_name](c)
    return (r, g, b)


def get_track_colors_by_position(tracks_b, vis_mask_b=None, image_width=None, image_height=None, cmap_name="hsv"):
    """tracks_b (S, N, 2) -> (N, 3) uint8 colours."""
    S, N, _ = tracks_b.shape
    colors = np.zeros((N, 3), dtype=np.uint8)
    if vis_mask_b is None:
        vis_mask_b = torch.ones(S, N, dtype=torch.bool, device=tracks_b.device)
    for i in range(N):
        visible = torch.where(vis_mask_b[:, i])[0]
        if len(visible) == 0:
            continue
        x, y = tracks_b[int(visible[0].item()), i].tolist()
        r, g, b = color_from_xy(x, y, W=image_width, H=image_height, cmap_name=cmap_name)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def visualize_tracks_on_images(images, tracks, track_vis_mask=None, out_dir="track_visuals_concat_by_xy",
                               image_format="CHW", normalize_mode="[0,1]", cmap_name="hsv", frames_per_row=4,
                               save_grid=True):
    """Save one PNG per frame with the visible track points drawn, plus a grid of all frames.

    images (S, 3, H, W) or (S, H, W, 3); tracks (S, N, 2); track_vis_mask (S, N) bool. A leading batch dim of 1 is
    accepted.
    """
    import cv2

    if len(tracks.shape) == 4:
        tracks, images = tracks.squeeze(0), images.squeeze(0)
        if track_vis_mask is not None:
            track_vis_mask = track_vis_mask.squeeze(0)

    os.makedirs(out_dir, exist_ok=True)
    S = images.shape[0]
    N = tracks.shape[1]
    images, tracks = images.cpu().clone(), tracks.cpu().clone()
    if track_vis_mask is not None:
        track_vis_mask = track_vis_mask.cpu().clone()
    H, W = (images.shape[2], images.shape[3]) if image_format == "CHW" else (images.shape[1], images.shape[2])
    colors = get_track_colors_by_position(tracks, vis_mask_b=track_vis_mask, image_width=W, image_height=H,
                                          cmap_name=cmap_name)

    frames = []
    for s in range(S):
        img = images[s].permute(1, 2, 0) if image_format == "CHW" else images[s]
        img = img.numpy().astype(np.float32)
        img = np.clip(img, 0, 1) * 255.0 if normalize_mode == "[0,1]" else np.clip((img + 1.0) * 127.5, 0, 255.0)
        bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        valid = torch.where(track_vis_mask[s])[0] if track_vis_mask is not None else range(N)
        pts = tracks[s].numpy()
        for i in valid:
            r, g, b = colors[i]
            cv2.circle(bgr, (int(round(pts[i][0])), int(round(pts[i][1]))), radius=3,
                       color=(int(b), int(g), int(r)), thickness=-1)
        cv2.imwrite(os.path.join(out_dir, f"frame_{s:04d}.png"), bgr)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if save_grid:
        rows = []
        for start in range(0, S, frames_per_row):
            row = frames[start: start + frames_per_row]
            row += [np.zeros((H, W, 3), dtype=np.uint8)] * (frames_per_row - len(row))
            rows.append(np.concatenate(row, axis=1))
        out_path = os.path.join(out_dir, "tracks_grid.png")
        cv2.imwrite(out_path, cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved color-by-XY track visualization grid -> {out_path}")
    print(f"[INFO] Saved {S} individual frames to {out_dir}/frame_*.png")
