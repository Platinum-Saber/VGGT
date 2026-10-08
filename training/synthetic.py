"""Procedural multi-view dataset with exact geometry ground truth.

The pretrained VGGT was trained on 17 large real/synthetic 3D datasets that
are not available in this environment, so for a small-scale reproduction we
ray-cast random indoor scenes on the fly:

* a closed, axis-aligned room (6 inward-facing walls) with 3-6 random spheres
  and boxes in the middle;
* every surface has a solid-texture (a function of the 3D point: per-object
  colour, sinusoid bands, checkerboard) with fixed Lambertian shading, so
  appearance is multi-view consistent and textured enough for matching;
* S cameras on an arc around the scene centre, looking at a jittered target,
  with random field of view.

Each sample gives images, depth, world points, OpenCV camera-from-world
extrinsics and pinhole intrinsics, normalised like the official training code:
world frame = first camera, scale such that the mean distance of all valid
points to the origin is 1.
"""

import math

import numpy as np
import torch

from vggt.utils.geometry import inverse_se3, to_homogeneous, unproject_depth_to_world


def _look_at(cam_pos, target, up=np.array([0.0, 1.0, 0.0])):
    """OpenCV convention (x right, y down, z forward) with world +y pointing up."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_c2w = np.stack([x, y, z], 1)
    R = R_c2w.T
    t = -R @ cam_pos
    return np.concatenate([R, t[:, None]], 1)


class SceneSampler:
    def __init__(self, img_size=112, num_views=6, seed=None):
        self.H = self.W = img_size
        self.num_views = num_views
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- scene
    def _random_scene(self):
        r = self.rng
        room = np.array([r.uniform(3, 5), r.uniform(2, 3), r.uniform(3, 5)])  # half extents
        objects = []
        for _ in range(r.integers(3, 7)):
            c = np.array([r.uniform(-1, 1), r.uniform(-room[1] + 0.3, 0.6), r.uniform(-1, 1)])
            size = r.uniform(0.2, 0.45)
            kind = "sphere" if r.random() < 0.5 else "box"
            if kind == "box":
                he = size * r.uniform(0.5, 1.0, 3)
                c[1] = -room[1] + he[1] if r.random() < 0.5 else c[1]  # some boxes rest on the floor
                objects.append(dict(kind="box", lo=c - he, hi=c + he))
            else:
                objects.append(dict(kind="sphere", c=c, r=size))
        n_surf = 6 + len(objects)
        tex = dict(
            color=r.uniform(0.15, 1.0, (n_surf, 3)),
            color2=r.uniform(0.0, 1.0, (n_surf, 3)),
            freq=r.normal(0, 1, (n_surf, 3, 3)) * r.uniform(2, 8, (n_surf, 1, 1)),
            phase=r.uniform(0, 2 * np.pi, (n_surf, 3)),
            checker=r.uniform(1.0, 4.0, n_surf),
        )
        light = np.array([r.uniform(-1, 1), 1.0, r.uniform(-1, 1)])
        return dict(room=room, objects=objects, tex=tex, light=light / np.linalg.norm(light))

    def _random_cameras(self):
        r = self.rng
        target = np.array([r.uniform(-0.3, 0.3), r.uniform(-0.6, 0.0), r.uniform(-0.3, 0.3)])
        az0 = r.uniform(0, 2 * np.pi)
        spread = r.uniform(np.deg2rad(30), np.deg2rad(100))
        extr, fovs = [], []
        for _ in range(self.num_views):
            az = az0 + r.uniform(-spread / 2, spread / 2)
            radius = r.uniform(1.9, 2.6)
            height = r.uniform(-0.6, 0.9)
            pos = np.array([radius * math.cos(az), height, radius * math.sin(az)])
            extr.append(_look_at(pos, target + r.normal(0, 0.15, 3)))
            fovs.append(np.deg2rad(r.uniform(50, 75)))
        return np.stack(extr), np.array(fovs)

    # --------------------------------------------------------------- render
    def _render(self, scene, extrinsic, fov):
        H, W = self.H, self.W
        f = (W / 2) / math.tan(fov / 2)
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        v, u = np.mgrid[0:H, 0:W].astype(np.float64)
        d_cam = np.stack([(u - W / 2) / f, (v - H / 2) / f, np.ones_like(u)], -1)  # z = 1 -> t is depth
        R, t = extrinsic[:, :3], extrinsic[:, 3]
        origin = -R.T @ t
        d = d_cam @ R  # world-space directions (R^T d_cam)

        best_t = np.full((H, W), np.inf)
        surf = np.zeros((H, W), dtype=np.int64)
        normal = np.zeros((H, W, 3))
        room = scene["room"]
        # Room walls: hit the far side of the box from the inside.
        with np.errstate(divide="ignore", invalid="ignore"):
            for axis in range(3):
                for side, sign in ((0, -1), (1, 1)):
                    tt = (sign * room[axis] - origin[axis]) / d[..., axis]
                    tt = np.where(tt > 1e-4, tt, np.inf)
                    upd = tt < best_t
                    best_t = np.where(upd, tt, best_t)
                    surf = np.where(upd, axis * 2 + side, surf)
                    n = np.zeros(3)
                    n[axis] = -sign
                    normal = np.where(upd[..., None], n, normal)
            for k, ob in enumerate(scene["objects"]):
                sid = 6 + k
                if ob["kind"] == "sphere":
                    oc = origin - ob["c"]
                    a = (d * d).sum(-1)
                    b = 2 * (d @ oc)
                    c = oc @ oc - ob["r"] ** 2
                    disc = b * b - 4 * a * c
                    tt = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
                    tt = np.where((disc > 0) & (tt > 1e-4), tt, np.inf)
                    upd = tt < best_t
                    p = origin + d * tt[..., None]
                    n = (p - ob["c"]) / ob["r"]
                else:
                    t0 = (ob["lo"] - origin) / d
                    t1 = (ob["hi"] - origin) / d
                    tmin, tmax = np.minimum(t0, t1), np.maximum(t0, t1)
                    tn, tf = tmin.max(-1), tmax.min(-1)
                    tt = np.where((tn <= tf) & (tn > 1e-4), tn, np.inf)
                    upd = tt < best_t
                    ax = tmin.argmax(-1)
                    n = np.zeros((H, W, 3))
                    np.put_along_axis(n, ax[..., None], -np.sign(np.take_along_axis(d, ax[..., None], -1)), -1)
                best_t = np.where(upd, tt, best_t)
                surf = np.where(upd, sid, surf)
                normal = np.where(upd[..., None], n, normal)

        depth = best_t
        valid = np.isfinite(depth)
        depth = np.where(valid, depth, 0.0)
        p = origin + d * depth[..., None]

        tex = scene["tex"]
        col, col2 = tex["color"][surf], tex["color2"][surf]
        bands = np.sin(np.einsum("hwc,hwkc->hwk", p, tex["freq"][surf]) + tex["phase"][surf])  # (H,W,3)
        pattern = 0.5 + 0.5 * bands.mean(-1, keepdims=True)
        chk = (np.floor(p * tex["checker"][surf][..., None]).sum(-1) % 2)[..., None]
        albedo = col * (0.55 + 0.45 * pattern) * (0.75 + 0.25 * chk) + 0.25 * col2 * bands[..., :1].clip(0, 1)
        shade = 0.45 + 0.55 * np.clip((normal * scene["light"]).sum(-1, keepdims=True), 0, 1)
        img = np.clip(albedo * shade, 0, 1)
        return img, depth, valid, K

    def sample(self):
        scene = self._random_scene()
        extr, fovs = self._random_cameras()
        imgs, depths, masks, Ks = zip(*(self._render(scene, e, f) for e, f in zip(extr, fovs)))
        return dict(
            images=torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float(),  # (V,3,H,W)
            depths=torch.from_numpy(np.stack(depths)).float(),
            point_masks=torch.from_numpy(np.stack(masks)),
            extrinsics=torch.from_numpy(extr).float(),
            intrinsics=torch.from_numpy(np.stack(Ks)).float(),
        )


def normalize_views(sample):
    """Re-express a (subset of) views in the first camera's frame and normalise scale. Adds world_points."""
    extr = to_homogeneous(sample["extrinsics"])
    extr = extr @ inverse_se3(extr[:1])  # camera-from-world, world = cam 0
    extr = extr[:, :3]
    depths = sample["depths"]
    pts = unproject_depth_to_world(depths, extr, sample["intrinsics"])
    mask = sample["point_masks"]
    scale = pts[mask].norm(dim=-1).mean().clamp(min=1e-6)
    extr = extr.clone()
    extr[:, :, 3] /= scale
    out = dict(sample)
    out.update(extrinsics=extr, depths=depths / scale, world_points=pts / scale)
    return out


def generate_dataset(path, num_scenes, img_size=112, num_views=6, seed=0):
    """Render and store scenes as compact tensors (uint8 images, fp16 depths)."""
    sampler = SceneSampler(img_size, num_views, seed=seed)
    samples = [sampler.sample() for _ in range(num_scenes)]
    data = dict(
        images=torch.stack([(s["images"] * 255).round().to(torch.uint8) for s in samples]),
        depths=torch.stack([s["depths"].half() for s in samples]),
        point_masks=torch.stack([s["point_masks"] for s in samples]),
        extrinsics=torch.stack([s["extrinsics"] for s in samples]),
        intrinsics=torch.stack([s["intrinsics"] for s in samples]),
    )
    torch.save(data, path)
    return data


class MultiViewDataset:
    """Serves batches of `S` random views from stored scenes, normalised per sample."""

    def __init__(self, data, seed=0):
        self.data = data
        self.n, self.V = data["images"].shape[:2]
        self.gen = torch.Generator().manual_seed(seed)

    def get(self, idx, view_ids):
        d = self.data
        s = dict(
            images=d["images"][idx, view_ids].float() / 255,
            depths=d["depths"][idx, view_ids].float(),
            point_masks=d["point_masks"][idx, view_ids],
            extrinsics=d["extrinsics"][idx, view_ids],
            intrinsics=d["intrinsics"][idx, view_ids],
        )
        return normalize_views(s)

    def batch(self, batch_size, S):
        items = []
        for _ in range(batch_size):
            idx = int(torch.randint(self.n, (1,), generator=self.gen))
            views = torch.randperm(self.V, generator=self.gen)[:S]
            items.append(self.get(idx, views))
        return {k: torch.stack([it[k] for it in items]) for k in items[0]}
