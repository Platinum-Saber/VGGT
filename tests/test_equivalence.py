"""Numerical equivalence against the official implementation.

Builds the official ``vggt`` model and ours with identical (random) weights and checks that camera, depth, point
and track outputs match, and that the full VGGT-1B configurations have identical parameter names and shapes.
Requires a checkout of https://github.com/facebookresearch/vggt, given by $VGGT_OFFICIAL.

    VGGT_OFFICIAL=/path/to/vggt python -m pytest tests -q
"""

import importlib
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.environ.get("VGGT_OFFICIAL")
needs_official = pytest.mark.skipif(not OFFICIAL, reason="set VGGT_OFFICIAL to an official vggt checkout")


def _import_vggt(root, hide=None, extra=()):
    """Import ``vggt.models.vggt`` (and the ``extra`` modules) from ``root``; return (VGGT class, its vggt.* modules).

    Both packages are called ``vggt``, so the modules of one are removed from sys.modules before importing the
    other, and restored afterwards.
    """
    saved = {k: v for k, v in sys.modules.items() if k == "vggt" or k.startswith("vggt.")}
    for k in saved:
        del sys.modules[k]
    old_path = list(sys.path)
    sys.path[:] = [root] + [p for p in sys.path if os.path.abspath(p or ".") != os.path.abspath(hide or "")]
    importlib.invalidate_caches()
    try:
        mod = importlib.import_module("vggt.models.vggt")
        for name in extra:
            importlib.import_module(name)
        mods ={k: v for k, v in sys.modules.items() if k == "vggt" or k.startswith("vggt.")}
    finally:
        sys.path[:] = old_path
        for k in [k for k in sys.modules if k == "vggt" or k.startswith("vggt.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return mod.VGGT, mods


def _small(VGGTcls, mods, embed_dim=384, depth=4, heads=6, layers=(0, 1, 2, 3)):
    """Instantiate a reduced config by patching the constructors' defaults (same patch for both implementations).

    Keeps the real DINOv2-S patchifier, RoPE, QK-norm, Alternating-Attention, camera trunk, DPT and track heads.
    """
    agg_mod, dpt_mod, cam_mod = mods["vggt.models.aggregator"], mods["vggt.heads.dpt_head"], mods["vggt.heads.camera_head"]
    inits = agg_mod.Aggregator.__init__, dpt_mod.DPTHead.__init__, cam_mod.CameraHead.__init__

    def agg(self, **kw):
        kw.update(depth=depth, num_heads=heads, patch_embed="dinov2_vits14_reg", cached_layer_indices=layers)
        inits[0](self, **kw)

    def dpt(self, dim_in, **kw):
        kw.update(intermediate_layer_idx=list(layers), out_channels=[16, 32, 64, 64])
        kw.setdefault("features", 32)
        inits[1](self, dim_in, **kw)

    def cam(self, dim_in, **kw):
        kw.update(trunk_depth=2, num_heads=heads)
        inits[2](self, dim_in, **kw)

    agg_mod.Aggregator.__init__, dpt_mod.DPTHead.__init__, cam_mod.CameraHead.__init__ = agg, dpt, cam
    try:
        return VGGTcls(img_size=518, patch_size=14, embed_dim=embed_dim)
    finally:
        agg_mod.Aggregator.__init__, dpt_mod.DPTHead.__init__, cam_mod.CameraHead.__init__ = inits


@needs_official
def test_matches_official():
    torch.manual_seed(0)
    RefCls, ref_mods = _import_vggt(OFFICIAL, hide=REPO_ROOT)
    OursCls, our_mods = _import_vggt(REPO_ROOT)
    ref = _small(RefCls, ref_mods).eval()
    ours = _small(OursCls, our_mods).eval()

    # Randomise every parameter (incl. LayerScale and tokens) so nothing matches by accident.
    state = {k: torch.randn_like(v) * 0.05 if v.is_floating_point() else v for k, v in ref.state_dict().items()}
    ref.load_state_dict(state)
    ours.load_state_dict(state, strict=True)

    imgs = torch.rand(1, 3, 3, 182, 252)  # non-square: exercises position-embedding interpolation
    query = torch.tensor([[[30.0, 40.0], [120.0, 90.0], [200.0, 150.0]]])
    with torch.no_grad():
        a, b = ref(imgs, query_points=query), ours(imgs, query_points=query)
    for key in ["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf", "track", "vis", "conf"]:
        torch.testing.assert_close(b[key], a[key], rtol=1e-4, atol=1e-5, msg=key)


class _empty_weights:
    """Create parameters on the meta device without allocating memory (both models call .item() on CPU tensors
    during construction, so ``with torch.device("meta")`` cannot be used)."""

    def __enter__(self):
        self._orig = torch.nn.Module.register_parameter

        def register(module, name, param):
            self._orig(module, name, param)
            if param is not None:
                module._parameters[name] = torch.nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)

        torch.nn.Module.register_parameter = register

    def __exit__(self, *exc):
        torch.nn.Module.register_parameter = self._orig


@needs_official
def test_full_config_same_parameters():
    """VGGT-1B: identical parameter names and shapes as the official model, so the checkpoint loads strictly."""
    RefCls, _ = _import_vggt(OFFICIAL, hide=REPO_ROOT)
    OursCls, _ = _import_vggt(REPO_ROOT)
    with _empty_weights():
        ref, ours = RefCls(), OursCls()
    a = {k: tuple(v.shape) for k, v in ref.state_dict().items()}
    b = {k: tuple(v.shape) for k, v in ours.state_dict().items()}
    assert a == b


def test_full_config_param_count():
    """Default config = VGGT-1B including the track head (~1.26B parameters)."""
    from vggt.models.vggt import VGGT

    with _empty_weights():
        model = VGGT()
    n = sum(p.numel() for p in model.parameters())
    assert 1.2e9 < n < 1.3e9, n
