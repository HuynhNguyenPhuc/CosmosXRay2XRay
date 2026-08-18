from __future__ import annotations

import pytest
import torch
import numpy as np

from shared.constants import (
    NUM_FRAMES,
    VOL_SIZE,
    XRAY_CAMERAS,
    XRAY_EXTRINSICS,
    XRAY_VIEW_MAPPING,
    COSMOS_2B_PRETRAINED_UUID,
)


def test_shared_constants():
    assert NUM_FRAMES == 93
    assert VOL_SIZE == 256
    assert len(XRAY_CAMERAS) == 7
    assert "xray_ap" in XRAY_CAMERAS
    assert "xray_pa" in XRAY_CAMERAS
    assert "xray_cranial" in XRAY_CAMERAS
    assert len(XRAY_EXTRINSICS) == 7
    assert XRAY_VIEW_MAPPING["xray_ap"] == 0


def test_camera_extrinsics_values():
    assert XRAY_EXTRINSICS["xray_ap"]["azimuth"] == 0.0
    assert XRAY_EXTRINSICS["xray_pa"]["azimuth"] == 180.0
    assert XRAY_EXTRINSICS["xray_cranial"]["elevation"] == 30.0
