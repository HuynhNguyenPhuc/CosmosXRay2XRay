from __future__ import annotations

import pytest
import torch

try:
    import torchio as tio
    import diffdrr
    HAS_DIFFDRR = True
except ImportError:
    HAS_DIFFDRR = False


@pytest.mark.skipif(not HAS_DIFFDRR, reason="diffdrr or torchio not installed")
def test_diffdrr_imports_and_constants():
    from renderers.diffdrr.data import create_ct_transforms
    from renderers.diffdrr.renderer import FIXED_LINE_INTEGRAL_MAX

    assert FIXED_LINE_INTEGRAL_MAX == 1.0
    transforms = create_ct_transforms(vol_shape=128)
    assert transforms is not None


@pytest.mark.skipif(not HAS_DIFFDRR, reason="diffdrr or torchio not installed")
def test_diffdrr_renderer_instantiation():
    from renderers.diffdrr.renderer import create_diffdrr_renderer

    renderer = create_diffdrr_renderer(img_shape=128, device="cpu")
    assert renderer.image_width == 128
    assert renderer.image_height == 128

