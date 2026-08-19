from __future__ import annotations

import inspect
import re

import pytest
import torch

from transfer2_5.control_signal import generate_control_signal


def test_control_signal_generation():
    # Test control signal generation on a dummy image tensor
    dummy_image = torch.zeros(1, 1, 256, 256, dtype=torch.float32)
    dummy_image[:, :, 50:150, 50:150] = 1.0

    edge_signal = generate_control_signal(dummy_image, control_type="edge_map")
    assert edge_signal is not None
    assert isinstance(edge_signal, torch.Tensor)


def test_denoise_passes_control_context_scale_as_a_list():
    """Regression guard for a real bug confirmed on A100 hardware (2026-08-19):
    MinimalV4LVGControlVaceDiT.forward() unconditionally does
    `control_context_scale = control_context_scale[0]` on the single-control-branch path
    (num_control_branches == 1, our config) before any isinstance/type check — despite its
    own type hint saying `float | torch.Tensor`. A bare float crashes every forward call with
    `TypeError: 'float' object is not subscriptable`; the fix is passing a length-1 list.

    A real forward-pass test would catch this more directly, but constructing
    CosmosXRay2XRayTransferMultiview triggers real checkpoint downloads (__init__ calls
    _setup_network() directly), and MinimalV4LVGControlVaceDiT's atten_backend="minimal_a2a"
    needs Ampere+ GPU hardware (confirmed unavailable on Turing — see
    docs/cosmos-predict3/PLAN.md's sibling findings), neither of which CI can rely on. This
    source-level check is the fast, dependency-free alternative: it fails loudly if the fix
    is ever reverted, rather than only failing silently at real training time.
    """
    source = inspect.getsource(__import__("transfer2_5.module", fromlist=["denoise"]).__dict__["CosmosXRay2XRayTransferMultiview"].denoise)

    match = re.search(r'"control_context_scale":\s*(.+?),?\n', source)
    assert match is not None, "denoise() no longer builds a 'control_context_scale' kwarg — update this test."

    value_expr = match.group(1).strip()
    assert value_expr.startswith("["), (
        f"control_context_scale must be passed as a list (got {value_expr!r}) — "
        "MinimalV4LVGControlVaceDiT.forward() subscripts it unconditionally."
    )
