"""Plücker-ray camera conditioning for `CameraMiniTrainDITwithConditionalMask`.

`predict2_5`'s baseline conditions on camera geometry by formatting it into the text
prompt (`XRAY_PROMPT_TEMPLATE`), which a BPE tokenizer turns into token sequences with no
metric or cyclic structure. NVIDIA ships a purpose-built alternative for exactly this kind
of dense, spatial conditioning: `cosmos_predict2._src.predict2.camera` — a DiT variant
(`CameraMiniTrainDITwithConditionalMask`) that adds a Plücker ray embedding directly to
every spatial token, in every transformer block:

    cam_emb = self.cam_encoder(camera)          # per-token Linear(1536 -> model_dim)
    self.self_attn(normalized_x + cam_emb, ...)  # added to spatial tokens pre-attention

This is a materially better match for novel-view X-ray synthesis than the alternative this
repo considered and rejected — `ActionConditionedMinimalV1LVGDiT`, the mechanism NVIDIA's
own Bridge/AgiBot robot-arm post-training recipe uses. That mechanism injects a single
*global* low-DOF vector via AdaLN — appropriate for "what state is the robot arm in" (a
genuinely global quantity), not for camera pose in NVS, where every output PIXEL
corresponds to a specific 3D ray. Plücker rays are inherently per-pixel; `cam_dim=1536`
is exactly `6 * 16 * 16` (Plücker's 6 coordinates, flattened over a 16x16 pixel patch) —
patch-token camera geometry, not a global summary.

Two things are reused verbatim from elsewhere rather than re-derived, because a silent
mismatch would corrupt the training signal without ever raising:

* **Camera geometry** comes from `renderers.diffdrr.renderer.look_at_view_matrices` — the
  exact same look-at math the DRR renderer uses to render the views that are actually fed
  to the model, so Plücker rays cannot drift from rendered geometry.
* **Plücker computation** comes from `cosmos_predict2._src.predict2.camera.utils`'s
  `convert_camera_to_plucker_rays` / `Camera.get_plucker_rays` — NVIDIA's own
  implementation, already patch-token-shaped for direct use by `CameraMiniTrainDIT`.

One thing IS re-derived here, carefully: DiffDRR is a cone-beam X-ray renderer, not a
pinhole camera, so it has no `(fx, fy, cx, cy)` to hand over directly — see
`_intrinsics_from_fov`'s docstring for how that's reconciled with a standard pinhole `K`.
"""

from __future__ import annotations

import torch

from cosmos_predict2._src.imaginaire.modules.camera import Camera
from cosmos_predict2._src.predict2.camera.utils import convert_camera_to_plucker_rays

from renderers.diffdrr.renderer import look_at_view_matrices
from shared.constants import NUM_LATENT_FRAMES, VOL_SIZE

# Plücker channels = 6 * PATCH_SPATIAL**2, must equal CameraMiniTrainDIT's Block.cam_dim
# (hardcoded 1536 there, not exposed as a kwarg — see dit_multiview_camera.py's Block.__init__).
PLUCKER_PATCH_SPATIAL = 16
PLUCKER_CHANNELS = 6 * PLUCKER_PATCH_SPATIAL**2  # 1536

# Our own DiT token grid: VOL_SIZE / (VAE 8x spatial) / (DiT's own patch_spatial=2) = 16x16,
# matching PLUCKER_PATCH_SPATIAL exactly (VOL_SIZE / PLUCKER_PATCH_SPATIAL = 256/16 = 16) —
# not a coincidence to preserve if VOL_SIZE or the DiT's patch_spatial ever change.
_EXPECTED_TOKEN_GRID = VOL_SIZE // PLUCKER_PATCH_SPATIAL


def _intrinsics_from_fov(
    fov_deg: torch.Tensor,
    max_depth: torch.Tensor,
    ndc_extent: float,
    image_size: int = VOL_SIZE,
) -> torch.Tensor:
    """Batched pinhole intrinsics ``K`` (B, 3, 3) matching DiffDRR's own cone-beam projection.

    DiffDRR parameterizes projection via ``sdd`` (source-to-**detector** distance — the
    renderer sets ``sdd = max_depth``, NOT ``distance``/source-to-isocenter, see
    ``renderers/diffdrr/renderer.py``'s ``set_volume``) plus a physical pixel spacing derived
    from FOV (``_fov_to_pixel_spacing``). A cone-beam source and a pinhole camera trace the
    same ray geometry (rays diverging from vs. converging to a point are the same set of
    lines run in reverse), so ``fx = fy = sdd / pixel_spacing`` recovers the pinhole focal
    length (in pixels) describing the *exact same rays* DiffDRR renders with — algebraically
    ``image_size * f_ndc / (2 * ndc_extent)``, the ``sdd`` cancelling out of
    ``_fov_to_pixel_spacing``'s own formula. The principal point is assumed centered
    (``cx = cy = image_size / 2``), matching DiffDRR's own un-offset detector.
    """
    fov_rad = torch.deg2rad(fov_deg)
    f_ndc = 1.0 / torch.tan(fov_rad / 2.0)
    pixel_spacing = max_depth * 2.0 * ndc_extent / (image_size * f_ndc)
    focal_px = max_depth / pixel_spacing

    B = fov_deg.shape[0]
    K = torch.zeros(B, 3, 3, device=fov_deg.device, dtype=torch.float32)
    K[:, 0, 0] = focal_px
    K[:, 1, 1] = focal_px
    K[:, 0, 2] = image_size / 2.0
    K[:, 1, 2] = image_size / 2.0
    K[:, 2, 2] = 1.0
    return K


def _world_to_camera(
    azimuth: torch.Tensor, elevation: torch.Tensor, distance: torch.Tensor
) -> torch.Tensor:
    """``(B, 3, 4)`` world-to-camera ``[R|t]`` (the convention `Camera` expects), derived from
    the same camera-to-world look-at matrices the renderer itself uses. `Camera.invert_pose`
    inverts a rigid SE(3) transform (``R_inv = R^T``, ``t_inv = -R^T @ t``) — that formula is
    its own inverse regardless of which direction you call "input", so feeding it a
    cam2world matrix correctly yields world2cam.
    """
    dev = azimuth.device
    cam2world = look_at_view_matrices(distance, elevation, azimuth, device=dev)  # (B, 4, 4)
    return Camera.invert_pose(cam2world[..., :3, :4])  # (B, 3, 4)


def view_plucker_tokens(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    distance: torch.Tensor,
    fov: torch.Tensor,
    max_depth: float,
    ndc_extent: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Per-sample Plücker ray token map for one camera view.

    Args:
        azimuth, elevation, distance, fov: ``(B,)`` tensors (degrees / scene units), the same
            camera parameters used to render this view.
        max_depth: ``renderer_max_depth`` hparam — the detector depth DiffDRR actually uses.
        ndc_extent: ``renderer_ndc_extent`` hparam.
        device: Target device for tensor allocation (e.g. CUDA).

    Returns:
        ``(B, 16, 16, 1536)`` float32 tensor.
    """
    if device is not None:
        azimuth = azimuth.to(device)
        elevation = elevation.to(device)
        distance = distance.to(device)
        fov = fov.to(device)

    max_depth_t = torch.full_like(fov, float(max_depth))
    w2c = _world_to_camera(azimuth, elevation, distance)
    K = _intrinsics_from_fov(fov, max_depth_t, ndc_extent)
    tokens = convert_camera_to_plucker_rays(
        w2c, K, (VOL_SIZE, VOL_SIZE),
        patch_spatial=PLUCKER_PATCH_SPATIAL, camera_patch_average=False, out_dtype=None,
    )
    if tokens.shape[-3:-1] != (_EXPECTED_TOKEN_GRID, _EXPECTED_TOKEN_GRID) or tokens.shape[-1] != PLUCKER_CHANNELS:
        raise AssertionError(
            f"Plücker token grid {tuple(tokens.shape[-3:])} != expected "
            f"({_EXPECTED_TOKEN_GRID}, {_EXPECTED_TOKEN_GRID}, {PLUCKER_CHANNELS}) — "
            "VOL_SIZE/DiT patch_spatial/PLUCKER_PATCH_SPATIAL no longer agree."
        )
    return tokens


def num_frontal_latent_frames(num_frontal_pixel_frames: int, temporal_compression: int = 4) -> int:
    """How many of the video's latent frames are entirely covered by the frontal (anchor)
    pixel frames, under the Wan-family causal VAE's temporal grouping: latent frame 0 = pixel
    frame 0 alone; latent frame ``k >= 1`` covers pixels ``[C*k-C+1, C*k+1)`` for temporal
    compression ``C`` (4 for Wan2.1).

    ``num_frontal_frames=5`` (this module's default hparam) was evidently chosen to land
    exactly on this boundary: latent frames 0-1 are purely anchor, 2-23 purely target — see
    the worked example in ``docs/cosmos-predict3/PLAN.md``-adjacent design notes. If
    ``num_frontal_pixel_frames`` is changed to a value that does NOT land on a boundary, the
    one latent frame straddling anchor/target pixel content is conservatively assigned to the
    target segment (an approximation at that single frame, not a silent full-sequence bug).
    """
    if num_frontal_pixel_frames < 1:
        raise ValueError(f"num_frontal_pixel_frames must be >= 1, got {num_frontal_pixel_frames}")

    n = 1  # latent frame 0 (single pixel frame) is always fully anchor.
    while temporal_compression * n + 1 <= num_frontal_pixel_frames:
        n += 1
    return n


def multiview_batch_camera_tensor(
    anchor_azimuth: torch.Tensor,
    anchor_elevation: torch.Tensor,
    target_azimuth: torch.Tensor,
    target_elevation: torch.Tensor,
    distance: torch.Tensor,
    fov: torch.Tensor,
    max_depth: float,
    ndc_extent: float,
    num_frontal_pixel_frames: int,
    num_latent_frames: int = NUM_LATENT_FRAMES,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Full-video Plücker camera tensor for `CameraMiniTrainDITwithConditionalMask.forward`.

    Broadcasts one constant Plücker map per segment (anchor, then target) across the video's
    latent time axis. This is exact, not an approximation of a moving camera: the video this
    task builds (`_construct_93_frame_tensor`) tiles one still anchor frame, then one still
    target frame — every pixel frame within a segment shares identical camera geometry, so
    the true per-frame Plücker map genuinely is constant within each segment.

    Returns:
        ``(B, num_latent_frames, 16, 16, 1536)`` float32 tensor.
    """
    if device is None:
        device = anchor_azimuth.device

    anchor_tokens = view_plucker_tokens(anchor_azimuth, anchor_elevation, distance, fov, max_depth, ndc_extent, device=device)
    target_tokens = view_plucker_tokens(target_azimuth, target_elevation, distance, fov, max_depth, ndc_extent, device=device)

    n_anchor = num_frontal_latent_frames(num_frontal_pixel_frames)
    n_target = num_latent_frames - n_anchor
    if n_target < 1:
        raise ValueError(
            f"num_frontal_pixel_frames={num_frontal_pixel_frames} leaves no latent frames for "
            f"the target view (num_latent_frames={num_latent_frames})."
        )

    anchor_part = anchor_tokens.unsqueeze(1).expand(-1, n_anchor, -1, -1, -1)
    target_part = target_tokens.unsqueeze(1).expand(-1, n_target, -1, -1, -1)
    return torch.cat([anchor_part, target_part], dim=1)
