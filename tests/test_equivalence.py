"""Numerical equivalence against the official implementation.

Builds the official ``vggt`` model and ours with identical weights
(``load_state_dict(strict=True)`` from one to the other) and checks that
camera, depth and point outputs match. Requires a checkout of
https://github.com/facebookresearch/vggt, given by $VGGT_OFFICIAL.

    VGGT_OFFICIAL=/path/to/vggt python -m pytest tests/test_equivalence.py -q
"""

import importlib
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.environ.get("VGGT_OFFICIAL")


def _import_official():
    """Import the official package (also named ``vggt``) without clobbering ours."""
    saved = {k: v for k, v in sys.modules.items() if k == "vggt" or k.startswith("vggt.")}
    for k in saved:
        del sys.modules[k]
    # The official package has no __init__.py (namespace package), so ours would shadow it:
    # hide our repo root from sys.path while importing.
    old_path = list(sys.path)
    sys.path[:] = [OFFICIAL] + [p for p in sys.path if os.path.abspath(p or ".") != REPO_ROOT]
    importlib.invalidate_caches()
    try:
        mod = importlib.import_module("vggt.models.vggt")
        official = {k: v for k, v in sys.modules.items() if k == "vggt" or k.startswith("vggt.")}
    finally:
        sys.path[:] = old_path
        for k in [k for k in sys.modules if k == "vggt" or k.startswith("vggt.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return mod.VGGT, official


def _small_official(VGGTOfficial, official_mods, embed_dim, depth, heads, layers):
    """Instantiate the official model with a reduced config by patching its constructors' defaults."""
    agg_mod = official_mods["vggt.models.aggregator"]
    dpt_mod = official_mods["vggt.heads.dpt_head"]
    cam_mod = official_mods["vggt.heads.camera_head"]
    agg_init, dpt_init, cam_init = agg_mod.Aggregator.__init__, dpt_mod.DPTHead.__init__, cam_mod.CameraHead.__init__

    def agg(self, **kw):
        kw.update(depth=depth, num_heads=heads, patch_embed="dinov2_vits14_reg", cached_layer_indices=layers)
        agg_init(self, **kw)

    def dpt(self, dim_in, **kw):
        kw.update(intermediate_layer_idx=list(layers), features=32, out_channels=[16, 32, 64, 64])
        dpt_init(self, dim_in, **kw)

    def cam(self, dim_in, **kw):
        kw.update(trunk_depth=2, num_heads=heads)
        cam_init(self, dim_in, **kw)

    agg_mod.Aggregator.__init__, dpt_mod.DPTHead.__init__, cam_mod.CameraHead.__init__ = agg, dpt, cam
    try:
        return VGGTOfficial(img_size=518, patch_size=14, embed_dim=embed_dim, enable_track=False)
    finally:
        agg_mod.Aggregator.__init__, dpt_mod.DPTHead.__init__, cam_mod.CameraHead.__init__ = agg_init, dpt_init, cam_init


@pytest.mark.skipif(not OFFICIAL, reason="set VGGT_OFFICIAL to an official vggt checkout")
def test_matches_official():
    from vggt import VGGT

    torch.manual_seed(0)
    VGGTOfficial, mods = _import_official()
    # Small config, but with the real DINOv2-S patchifier, RoPE, QK-norm, AA, camera trunk and DPT.
    layers = (0, 1, 2, 3)
    ref = _small_official(VGGTOfficial, mods, embed_dim=384, depth=4, heads=6, layers=layers).eval()
    ours = VGGT(embed_dim=384, depth=4, num_heads=6, patch_embed="dinov2_vits14_reg", dpt_layers=layers,
                dpt_features=32, dpt_out_channels=(16, 32, 64, 64), camera_trunk_depth=2, camera_num_heads=6).eval()

    # Randomise every parameter (incl. LayerScale / tokens) so nothing matches by accident.
    state = {k: torch.randn_like(v) * 0.05 if v.is_floating_point() else v for k, v in ref.state_dict().items()}
    ref.load_state_dict(state)
    ours.load_state_dict(state, strict=True)

    imgs = torch.rand(1, 3, 3, 182, 252)  # non-square: exercises pos-embed interpolation
    with torch.no_grad():
        a, b = ref(imgs), ours(imgs)
    for key in ["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"]:
        torch.testing.assert_close(b[key], a[key], rtol=1e-4, atol=1e-5, msg=key)


def test_full_config_param_count():
    """Default config = VGGT-1B. Without the tracking head it should be ~1.18B params."""
    from vggt import VGGT

    with torch.device("meta"):
        model = VGGT()
    n = sum(p.numel() for p in model.parameters())
    assert 1.1e9 < n < 1.3e9, n
