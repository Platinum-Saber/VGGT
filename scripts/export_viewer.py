"""Write a self-contained interactive 3D viewer (one HTML file) for VGGT predictions.

The page shows the coloured point cloud, the predicted cameras as frustums with their images, optional 3D point
tracks, and controls for point size, confidence threshold and camera views. Data is embedded (quantised, base64);
three.js is loaded from a CDN, so the file opens in any browser with internet access and inside notebooks.

Used by ``demo_pretrained.py``; can also be run on a saved ``predictions.npz``:

    python scripts/export_viewer.py out/predictions.npz out/viewer.html
"""

import argparse
import base64
import io
import json

import numpy as np
from PIL import Image


def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _thumb(img, height=96):
    im = Image.fromarray(img)
    im = im.resize((max(1, round(im.width * height / im.height)), height), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _pack_points(points, colors, conf, max_points, rng):
    """Subsample to max_points, quantise xyz to int16 in the bounding box, conf to its percentile rank (uint8)."""
    pts = points.reshape(-1, 3).astype(np.float32)
    col = colors.reshape(-1, 3).astype(np.uint8)
    cf = conf.reshape(-1).astype(np.float32)
    ok = np.isfinite(pts).all(1) & np.isfinite(cf)
    pts, col, cf = pts[ok], col[ok], cf[ok]
    # Rank on all points so the slider means "keep the top x % by confidence".
    rank = np.empty(len(cf), dtype=np.float32)
    rank[np.argsort(cf, kind="stable")] = np.linspace(0, 1, len(cf), dtype=np.float32)
    if len(pts) > max_points:
        keep = rng.choice(len(pts), max_points, replace=False)
        pts, col, rank = pts[keep], col[keep], rank[keep]
    lo = np.percentile(pts, 0.5, axis=0)
    hi = np.percentile(pts, 99.5, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo, hi = lo - 0.5 * span, hi + 0.5 * span  # leave room for outliers before clipping
    q = np.clip((pts - lo) / (hi - lo) * 65535 - 32768, -32768, 32767).astype(np.int16)
    return {"n": int(len(pts)), "lo": lo.tolist(), "hi": hi.tolist(), "xyz": _b64(q), "rgb": _b64(col),
            "conf": _b64(np.round(rank * 255).astype(np.uint8))}


def write_viewer(path, points, colors, conf, extrinsics, intrinsics, images=None, tracks3d=None, points_alt=None,
                 conf_alt=None, title="VGGT reconstruction", max_points=250_000, seed=0):
    """
    points (S, H, W, 3) world points (OpenCV convention, first camera = world); colors (S, H, W, 3) uint8;
    conf (S, H, W); extrinsics (S, 3, 4) camera-from-world; intrinsics (S, 3, 3); images (S, H, W, 3) uint8 for the
    camera thumbnails; tracks3d (N, S, 3) with NaN where invisible; points_alt / conf_alt: a second point set
    (e.g. the point head) that the viewer can switch to.
    """
    rng = np.random.default_rng(seed)
    S, H, W = points.shape[:3]
    sets = {"depth + camera": _pack_points(points, colors, conf, max_points, rng)}
    if points_alt is not None:
        sets["point head"] = _pack_points(points_alt, colors, conf_alt if conf_alt is not None else conf, max_points, rng)
    cams = []
    for i in range(S):
        E = np.eye(4)
        E[:3] = extrinsics[i]
        c2w = np.linalg.inv(E)
        cams.append({"c2w": c2w[:3].round(6).tolist(), "fx": float(intrinsics[i][0, 0]), "fy": float(intrinsics[i][1, 1]),
                     "cx": float(intrinsics[i][0, 2]), "cy": float(intrinsics[i][1, 2]),
                     "thumb": _thumb(images[i]) if images is not None else None})
    # Robust framing box (2-98 % of the points), so a few outliers do not shrink the initial view.
    flat = points.reshape(-1, 3)[np.isfinite(points.reshape(-1, 3)).all(1)]
    lo, hi = np.percentile(flat, 2, axis=0), np.percentile(flat, 98, axis=0)
    data = {"title": title, "H": H, "W": W, "sets": sets, "cams": cams,
            "fit": {"center": ((lo + hi) / 2).tolist(), "size": float(np.linalg.norm(hi - lo))},
            "tracks": None if tracks3d is None else np.nan_to_num(tracks3d, nan=1e9).round(5).tolist()}
    html = _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
/* full-bleed WebGL canvas; a floating control panel top-left and a strip of camera images at the bottom */
:root{--bg:#eef1f4;--panel:#ffffffe6;--text:#18202b;--muted:#566273;--line:#cfd6df;--accent:#1f6fb0;--cam:#1f6fb0;--cam0:#c8352b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e1217;--panel:#161b22e6;--text:#e6eaef;--muted:#98a3b1;--line:#2a323d;--accent:#5aa9e6;--cam:#5aa9e6;--cam0:#ff6b5e;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0e1217;--panel:#161b22e6;--text:#e6eaef;--muted:#98a3b1;--line:#2a323d;--accent:#5aa9e6;--cam:#5aa9e6;--cam0:#ff6b5e;color-scheme:dark}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--text);font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}
#c{position:fixed;inset:0;width:100%;height:100%;display:block}
#ui{position:fixed;top:calc(12px + env(safe-area-inset-top, 0px));left:calc(12px + env(safe-area-inset-left, 0px));width:250px;
max-width:calc(100vw - 24px);background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:12px 14px;box-sizing:border-box;backdrop-filter:blur(6px)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#ui h1{font-size:14px;margin:0 0 2px}#ui .sub{color:var(--muted);margin-bottom:10px}
label{display:flex;justify-content:space-between;align-items:center;margin:8px 0 2px;color:var(--muted)}
input[type=range]{width:100%;accent-color:var(--accent)}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}.row label{margin:0;gap:4px;color:var(--text)}
select,button{font:inherit;color:var(--text);background:transparent;border:1px solid var(--line);border-radius:6px;padding:3px 8px;cursor:pointer}
button:hover{border-color:var(--accent)}
#thumbs{position:fixed;left:calc(12px + env(safe-area-inset-left, 0px));right:calc(12px + env(safe-area-inset-right, 0px));
bottom:calc(12px + env(safe-area-inset-bottom, 0px));display:flex;gap:6px;overflow-x:auto;padding-bottom:4px}
@media (max-width:520px){#ui{width:auto;right:calc(12px + env(safe-area-inset-right, 0px))}#hint{display:none}#thumbs img{height:48px}}
#thumbs img{height:64px;border-radius:6px;border:2px solid transparent;cursor:pointer;flex:none}
#thumbs img.on{border-color:var(--accent)}
#hint{color:var(--muted);margin-top:10px;font-size:12px}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<canvas id="c"></canvas>
<div id="ui"><h1 id="title"></h1><div class="sub" id="stats"></div>
<label for="set">Point set <select id="set"></select></label>
<label for="conf">Keep top confidence <span id="confv"></span></label><input id="conf" type="range" min="0" max="100" value="50">
<label for="size">Point size <span id="sizev"></span></label><input id="size" type="range" min="1" max="60" value="12">
<div class="row"><label><input id="showcams" type="checkbox" checked>cameras</label>
<label id="trkl" title="tracks lifted to 3D with each frame's depth; a consistent track collapses to a point"><input id="showtracks" type="checkbox" checked>3D tracks</label></div>
<div class="row"><button id="reset">Reset view</button></div>
<div id="hint">Drag to orbit, right-drag to pan, scroll to zoom. Click an image to look through that camera.</div></div>
<div id="thumbs"></div>
<script type="module">
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
const D = __DATA__;
const $ = id => document.getElementById(id);
$("title").textContent = D.title;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const renderer = new THREE.WebGLRenderer({canvas: $("c"), antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene(); scene.background = new THREE.Color(css("--bg") || "#0e1217");
const camera = new THREE.PerspectiveCamera(55, 1, 0.001, 1000);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
// OpenCV (x right, y down, z forward) -> three.js (y up, z backward)
const world = new THREE.Group(); world.scale.set(1, -1, -1); scene.add(world);

function decode(b64, Type){const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Type(u.buffer);}
const sets = {};
for (const [name, P] of Object.entries(D.sets)) {
  const q = decode(P.xyz, Int16Array), rgb = decode(P.rgb, Uint8Array), cf = decode(P.conf, Uint8Array);
  const pos = new Float32Array(P.n * 3), col = new Float32Array(P.n * 3), conf = new Float32Array(P.n);
  for (let i = 0; i < P.n; i++) { for (let k = 0; k < 3; k++) { pos[3*i+k] = P.lo[k] + (q[3*i+k] + 32768) / 65535 * (P.hi[k] - P.lo[k]); col[3*i+k] = rgb[3*i+k] / 255; } conf[i] = cf[i] / 255; }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setAttribute("conf", new THREE.BufferAttribute(conf, 1));
  const m = new THREE.ShaderMaterial({vertexColors: true, transparent: false,
    uniforms: {size: {value: 0.012}, thr: {value: 0.5}, scale: {value: 1}},
    vertexShader: `attribute float conf; uniform float size, thr, scale; varying vec3 vC; varying float vK;
      void main(){ vC = color; vK = conf >= thr ? 1.0 : 0.0; vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = vK > 0.5 ? clamp(size * scale / -mv.z, 1.0, 14.0) : 0.0; gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `varying vec3 vC; varying float vK; void main(){ if (vK < 0.5) discard;
      vec2 d = gl_PointCoord - 0.5; if (dot(d, d) > 0.25) discard; gl_FragColor = vec4(vC, 1.0); }`});
  const pts = new THREE.Points(g, m); pts.visible = false; world.add(pts); sets[name] = pts;
  const o = document.createElement("option"); o.textContent = name; $("set").appendChild(o);
}
let active = Object.keys(sets)[0]; sets[active].visible = true;

// scene scale from the camera spread and point cloud
const sz = D.fit.size || 1;
const camGroup = new THREE.Group(); world.add(camGroup);
const camPose = [];
D.cams.forEach((c, i) => {
  const M = new THREE.Matrix4().set(...c.c2w[0], ...c.c2w[1], ...c.c2w[2], 0, 0, 0, 1);
  const d = sz * 0.04, w = D.W / 2 / c.fx * d, h = D.H / 2 / c.fy * d;
  const corners = [[-w,-h,d],[w,-h,d],[w,h,d],[-w,h,d]].map(p => new THREE.Vector3(...p).applyMatrix4(M));
  const o = new THREE.Vector3(0,0,0).applyMatrix4(M); const v = [];
  for (const p of corners) v.push(o, p);
  for (let k = 0; k < 4; k++) v.push(corners[k], corners[(k+1)%4]);
  const mat = new THREE.LineBasicMaterial({color: new THREE.Color(css(i === 0 ? "--cam0" : "--cam") || "#1f6fb0")});
  mat.userData.token = i === 0 ? "--cam0" : "--cam";
  camGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(v), mat));
  camPose.push({M, o});
});
const trackGroup = new THREE.Group(); world.add(trackGroup);
if (D.tracks) {
  for (const tr of D.tracks) {
    const pts = tr.filter(p => Math.abs(p[0]) < 1e8).map(p => new THREE.Vector3(...p));
    if (pts.length < 2) continue;
    trackGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color: 0xff7f0e})));
  }
} else { $("trkl").style.display = "none"; }

function fit(){ const c = new THREE.Vector3(D.fit.center[0], -D.fit.center[1], -D.fit.center[2]);
  controls.target.copy(c); camera.position.set(c.x, c.y + sz * 0.15, c.z + sz * 0.9); camera.up.set(0, 1, 0); controls.update(); }
function lookThrough(i){
  const {M} = camPose[i]; const o = new THREE.Vector3(0,0,0).applyMatrix4(M), f = new THREE.Vector3(0,0,sz*0.3).applyMatrix4(M), up = new THREE.Vector3(0,-1,0).transformDirection(M);
  const flip = v => new THREE.Vector3(v.x, -v.y, -v.z);
  camera.position.copy(flip(o)); controls.target.copy(flip(f)); camera.up.copy(flip(up)).normalize();
  camera.fov = 2 * Math.atan(D.H / 2 / D.cams[i].fy) * 180 / Math.PI; camera.updateProjectionMatrix(); controls.update();
  document.querySelectorAll("#thumbs img").forEach((im, k) => im.classList.toggle("on", k === i));
}
D.cams.forEach((c, i) => { if (!c.thumb) return; const im = new Image(); im.src = c.thumb; im.title = `frame ${i}`; im.onclick = () => lookThrough(i); $("thumbs").appendChild(im); });

const upd = () => {
  const keep = +$("conf").value, size = +$("size").value / 2500 * sz;
  $("confv").textContent = keep + " %"; $("sizev").textContent = $("size").value;
  for (const p of Object.values(sets)) { p.material.uniforms.thr.value = 1 - keep / 100; p.material.uniforms.size.value = size; }
  camGroup.visible = $("showcams").checked; trackGroup.visible = $("showtracks").checked;
  const n = Math.round(D.sets[active].n * keep / 100);
  $("stats").textContent = `${D.cams.length} cameras, ${n.toLocaleString()} points shown`;
};
["conf","size","showcams","showtracks"].forEach(id => $(id).addEventListener("input", upd));
$("set").addEventListener("change", e => { sets[active].visible = false; active = e.target.value; sets[active].visible = true; upd(); });
$("reset").onclick = () => { camera.fov = 55; camera.updateProjectionMatrix(); fit(); document.querySelectorAll("#thumbs img").forEach(im => im.classList.remove("on")); };

function resize(){ const w = innerWidth, h = innerHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  for (const p of Object.values(sets)) p.material.uniforms.scale.value = h / (2 * Math.tan(camera.fov * Math.PI / 360)); }
addEventListener("resize", resize); resize(); fit(); upd();
// follow the viewer's theme (system setting or an explicit data-theme on the root element)
const retheme = () => { scene.background = new THREE.Color(css("--bg") || "#0e1217");
  camGroup.children.forEach(l => l.material.color = new THREE.Color(css(l.material.userData.token) || "#1f6fb0")); };
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
new MutationObserver(retheme).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
renderer.setAnimationLoop(() => { controls.update(); for (const p of Object.values(sets)) p.material.uniforms.scale.value = innerHeight / (2 * Math.tan(camera.fov * Math.PI / 360)); renderer.render(scene, camera); });
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="predictions.npz written by demo_pretrained.py --save-npz")
    ap.add_argument("out", help="output HTML path")
    ap.add_argument("--max-points", type=int, default=250_000)
    args = ap.parse_args()
    d = np.load(args.npz)
    images = (d["images"] * 255).astype(np.uint8) if d["images"].dtype != np.uint8 else d["images"]
    write_viewer(args.out, d["points_depth_cam"], images, d["depth_conf"], d["extrinsics"], d["intrinsics"], images=images,
                 tracks3d=d["tracks3d"] if "tracks3d" in d else None,
                 points_alt=d["world_points"] if "world_points" in d else None,
                 conf_alt=d["world_points_conf"] if "world_points_conf" in d else None, max_points=args.max_points)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
