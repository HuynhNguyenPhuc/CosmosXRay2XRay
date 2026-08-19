from __future__ import annotations

import pytest
from shared.constants import (
    COSMOS_2B_CAMERA_PRETRAINED_UUID,
    COSMOS_2B_PRETRAINED_UUID,
    NUM_LATENT_FRAMES,
)
from predict2_5.module import _CHECKPOINT_HF_FILES, _resolve_checkpoint_uuid
from predict2_5.text_encoder import CR1TextEncoder


def test_predict25_configuration():
    assert COSMOS_2B_PRETRAINED_UUID == "d20b7120-df3e-4911-919d-db6e08bad31c"
    assert NUM_LATENT_FRAMES == 24


def test_checkpoint_uuid_resolves_camera_conditioned_for_plucker():
    """Regression guard: camera_cond="plucker" must load the real pretrained
    camera-conditioned checkpoint (has cam_encoder weights, see
    COSMOS_2B_CAMERA_PRETRAINED_UUID's docstring / docs/CONDITIONING.md §2.1) by default, not
    the base checkpoint — which would leave cam_encoder randomly initialized on every run
    without ever raising, since non_strict_load_model treats missing keys as a soft warning.
    """
    assert _resolve_checkpoint_uuid(None, "plucker") == COSMOS_2B_CAMERA_PRETRAINED_UUID
    assert _resolve_checkpoint_uuid(None, "text") == COSMOS_2B_PRETRAINED_UUID
    # An explicit UUID always overrides the camera_cond-based default, for either arm.
    assert _resolve_checkpoint_uuid("some-other-uuid", "plucker") == "some-other-uuid"
    assert _resolve_checkpoint_uuid("some-other-uuid", "text") == "some-other-uuid"
    # Both resolvable UUIDs must have a real HF filename registered, or _setup_network raises.
    assert COSMOS_2B_PRETRAINED_UUID in _CHECKPOINT_HF_FILES
    assert COSMOS_2B_CAMERA_PRETRAINED_UUID in _CHECKPOINT_HF_FILES


def test_prompt_to_filename():
    prompt = "A 360-degree rotational view, of a chest CT scan."
    filename = CR1TextEncoder.prompt_to_filename(prompt)
    assert "," not in filename
    assert " " not in filename
    assert filename.startswith("A_360-degree")

