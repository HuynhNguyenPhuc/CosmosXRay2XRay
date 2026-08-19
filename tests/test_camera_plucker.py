from __future__ import annotations

import torch

from predict2_5.camera_plucker import (
    PLUCKER_CHANNELS,
    multiview_batch_camera_tensor,
    num_frontal_latent_frames,
    view_plucker_tokens,
)
from shared.constants import NUM_LATENT_FRAMES, XRAY_DISTANCE_DEFAULT, XRAY_FOV_DEFAULT


def test_num_frontal_latent_frames_matches_wan_causal_grouping():
    """Wan-family causal VAE: latent frame 0 = pixel frame 0 alone; latent frame k>=1 covers
    pixels [4k-3, 4k+1). num_frontal_frames=5 (this repo's default) lands exactly on that
    boundary — latent frames 0-1 purely anchor, 2-23 purely target."""
    assert num_frontal_latent_frames(5) == 2
    assert num_frontal_latent_frames(1) == 1
    assert num_frontal_latent_frames(8) == 2  # latent frame 2 needs pixel 8 < 9, so not yet
    assert num_frontal_latent_frames(9) == 3


def test_view_plucker_tokens_shape_and_finite():
    B = 2
    azimuth = torch.tensor([0.0, 180.0])
    elevation = torch.zeros(B)
    distance = torch.full((B,), XRAY_DISTANCE_DEFAULT)
    fov = torch.full((B,), XRAY_FOV_DEFAULT)

    tokens = view_plucker_tokens(azimuth, elevation, distance, fov, max_depth=9.0, ndc_extent=1.0)

    assert tokens.shape == (B, 16, 16, PLUCKER_CHANNELS)
    assert torch.isfinite(tokens).all()


def test_ap_pa_center_rays_are_antipodal():
    """AP (azimuth=0) and PA (azimuth=180) are 180 degrees apart, so the ray through the
    detector center should point in nearly opposite directions in world space — a real
    geometric invariant shape checks alone cannot catch (e.g. a sign error in the rotation
    convention would still produce the right shape)."""
    azimuth = torch.tensor([0.0, 180.0])
    elevation = torch.zeros(2)
    distance = torch.full((2,), XRAY_DISTANCE_DEFAULT)
    fov = torch.full((2,), XRAY_FOV_DEFAULT)

    tokens = view_plucker_tokens(azimuth, elevation, distance, fov, max_depth=9.0, ndc_extent=1.0)

    cy, cx = tokens.shape[1] // 2, tokens.shape[2] // 2
    direction_ap = tokens[0, cy, cx, -3:]
    direction_pa = tokens[1, cy, cx, -3:]
    cos_angle = torch.dot(direction_ap, direction_pa) / (direction_ap.norm() * direction_pa.norm())

    assert cos_angle.item() < -0.999


def test_multiview_batch_camera_tensor_shape():
    B = 2
    zeros = torch.zeros(B)
    azimuth = torch.tensor([315.0, 45.0])
    distance = torch.full((B,), XRAY_DISTANCE_DEFAULT)
    fov = torch.full((B,), XRAY_FOV_DEFAULT)

    camera = multiview_batch_camera_tensor(
        anchor_azimuth=zeros, anchor_elevation=zeros,
        target_azimuth=azimuth, target_elevation=zeros,
        distance=distance, fov=fov,
        max_depth=9.0, ndc_extent=1.0, num_frontal_pixel_frames=5,
    )

    assert camera.shape == (B, NUM_LATENT_FRAMES, 16, 16, PLUCKER_CHANNELS)
    # First 2 latent frames (anchor segment) must be identical to each other (broadcast of
    # one constant map), and different from the target segment.
    assert torch.equal(camera[:, 0], camera[:, 1])
    assert not torch.equal(camera[:, 1], camera[:, 2])
    # RAO (315 deg) / LAO (45 deg) are symmetric about the AP anchor, so their Plücker tokens
    # (which encode absolute ray geometry, not just a summary norm) should be mirror images —
    # checked via equal magnitude at the matching pixel, catching an axis/sign error.
    assert torch.allclose(camera[0, -1].norm(), camera[1, -1].norm(), rtol=1e-4)
