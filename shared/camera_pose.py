"""Continuous camera-pose action vectors, shared by `predict2_5` and `predict3`.

Both pipelines condition a DiT on X-ray camera geometry through a continuous, per-sample
vector rather than by formatting numbers into a text prompt — `predict2_5` via
`ActionConditionedMinimalV1LVGDiT`'s AdaLN action-injection pathway, `predict3` via Cosmos
3's `camera_pose` action port (see `predict3/camera.py`). The underlying geometry and pose
representation are identical; only how each backbone consumes the resulting vector differs.
This module holds the shared half so neither pipeline depends on the other's package.

Two external conventions are reused verbatim rather than re-derived, because a silent
mismatch here would corrupt the training signal without ever raising:

* **Camera geometry** comes from ``renderers.diffdrr.renderer.look_at_view_matrices`` —
  the same look-at math the DRR renderer uses to actually render the views, so the
  conditioning poses cannot drift from rendered geometry.
* **Pose to action-vector encoding** comes from ``cosmos-framework``'s ``pose_abs_to_rel``.
  Vendored as a submodule but not pip-installed: the pose utilities it exposes need only
  numpy/scipy/torch, so its root is put on ``sys.path`` here rather than pulling in its full
  (heavy, action-training-specific) dependency tree.

Note the ``T-1``: actions describe *transitions*, so ``T`` posed frames yield ``T-1`` action
vectors — a deliberate convention (matching Cosmos 3's ``chunk_size``/``chunk_size+1``
contract), not an off-by-one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from renderers.diffdrr.renderer import look_at_view_matrices
from shared.constants import XRAY_CAMERAS, XRAY_DISTANCE_DEFAULT, XRAY_EXTRINSICS
from shared.utils import get_logger

log = get_logger(__name__)

_COSMOS_FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent / "cosmos-framework"
if _COSMOS_FRAMEWORK_ROOT.is_dir() and str(_COSMOS_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS_FRAMEWORK_ROOT))

from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel  # noqa: E402

# translation(3) + rot6d(6). Not Cosmos-3-specific — this happens to equal Cosmos 3's
# registered `camera_pose` domain width (`EMBODIMENT_TO_RAW_ACTION_DIM["camera_pose"]`,
# see predict3/camera.py), a coincidence worth keeping in sync, not a shared constant, since
# predict2_5's action head has no such external registry to stay consistent with.
RELATIVE_POSE_DIM = 9

PoseConvention = Literal["backward_anchored", "backward_framewise"]


def xray_view_matrices(
    view_names: tuple[str, ...] | list[str] = XRAY_CAMERAS,
    distance: float | torch.Tensor = XRAY_DISTANCE_DEFAULT,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Camera-to-world matrices for named X-ray views.

    Args:
        view_names: View keys, each present in ``XRAY_EXTRINSICS`` (default: all 7 views in
            canonical ``XRAY_CAMERAS`` order).
        distance: Source-to-isocenter distance in scene units. A scalar applies to every
            view; a ``(len(view_names),)`` tensor gives a per-sample distance (e.g. training
            time distance jitter, shared across a per-view batch group).

    Returns:
        ``(len(view_names), 4, 4)`` camera-to-world transforms.
    """
    unknown = [name for name in view_names if name not in XRAY_EXTRINSICS]
    if unknown:
        raise KeyError(f"Unknown X-ray view(s) {unknown}; expected keys from XRAY_EXTRINSICS.")

    azimuths = [XRAY_EXTRINSICS[name]["azimuth"] for name in view_names]
    elevations = [XRAY_EXTRINSICS[name]["elevation"] for name in view_names]
    if isinstance(distance, torch.Tensor):
        if distance.shape[0] != len(view_names):
            raise ValueError(f"distance tensor length {distance.shape[0]} != {len(view_names)} views.")
        distances = distance
    else:
        distances = [distance] * len(view_names)

    return look_at_view_matrices(distances, elevations, azimuths, device=device, dtype=dtype)


def orbit_view_matrices(
    num_frames: int,
    distance: float = XRAY_DISTANCE_DEFAULT,
    elevation: float = 0.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Camera-to-world matrices for a 360-degree azimuth sweep.

    Uses the same ``linspace(0, 360, num_frames)`` sweep as
    ``renderers.diffdrr.renderer.render_orbit_video``, so orbit conditioning matches the
    orbit videos the renderer produces.

    Returns:
        ``(num_frames, 4, 4)`` camera-to-world transforms.
    """
    if num_frames < 2:
        raise ValueError(f"`num_frames` must be >= 2 to form a transition, got {num_frames}.")

    azimuths = torch.linspace(0.0, 360.0, num_frames, dtype=dtype).tolist()
    return look_at_view_matrices(
        [distance] * num_frames, [elevation] * num_frames, azimuths, device=device, dtype=dtype
    )


def poses_to_relative_actions(
    poses_abs: torch.Tensor,
    pose_convention: PoseConvention = "backward_anchored",
) -> torch.Tensor:
    """Convert absolute camera-to-world poses into relative 9D pose-action vectors.

    Args:
        poses_abs: ``(T, 4, 4)`` camera-to-world transforms, ``T >= 2``.
        pose_convention: ``"backward_anchored"`` encodes every view relative to view 0
            (``T_0^-1 @ T_i+1``) — matches "given the anchor X-ray view, produce the view at
            this relative offset" and keeps each action independent of its neighbours.
            ``"backward_framewise"`` (``T_i^-1 @ T_i+1``) encodes consecutive deltas instead,
            appropriate for a continuous orbit sweep.

    Returns:
        ``(T - 1, RELATIVE_POSE_DIM)`` float32 tensor laid out as
        ``[translation(3), rot6d(6)]``.
    """
    if poses_abs.ndim != 3 or poses_abs.shape[-2:] != (4, 4):
        raise ValueError(f"`poses_abs` must have shape (T, 4, 4), got {tuple(poses_abs.shape)}.")
    if poses_abs.shape[0] < 2:
        raise ValueError("At least 2 poses are required to form a transition (T >= 2).")

    poses_np = poses_abs.detach().cpu().to(torch.float32).numpy().astype(np.float64)
    actions = pose_abs_to_rel(poses_np, rotation_format="rot6d", pose_convention=pose_convention)

    if actions.shape[-1] != RELATIVE_POSE_DIM:
        raise ValueError(
            f"Expected {RELATIVE_POSE_DIM}-D pose actions, got {actions.shape[-1]}D — "
            "cosmos-framework's pose layout may have changed."
        )
    return torch.from_numpy(actions).to(dtype=torch.float32)


def xray_view_actions(
    view_names: tuple[str, ...] | list[str] = XRAY_CAMERAS,
    distance: float = XRAY_DISTANCE_DEFAULT,
    pose_convention: PoseConvention = "backward_anchored",
) -> torch.Tensor:
    """Convenience: named X-ray views -> relative 9D pose-action vectors.

    Returns:
        ``(len(view_names) - 1, RELATIVE_POSE_DIM)`` float32 tensor.
    """
    poses = xray_view_matrices(view_names=view_names, distance=distance)
    return poses_to_relative_actions(poses, pose_convention=pose_convention)


def anchor_relative_action(
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    distance: torch.Tensor,
    pose_convention: PoseConvention = "backward_anchored",
) -> torch.Tensor:
    """Per-sample relative pose action from the AP anchor (azimuth=elevation=0) to a target view.

    Unlike :func:`xray_view_actions`, this takes batched per-sample tensors (azimuth/elevation
    constant within a view group, distance possibly jittered per-sample), matching how
    `predict2_5._sample_multiview_camera_params` supplies camera parameters. Looping in Python
    is deliberate: `pose_abs_to_rel` operates on one ``(T,4,4)`` trajectory at a time, batch
    sizes here are small (a handful of views per step), and the DiT forward/backward pass
    dominates step cost by orders of magnitude, so batching this further would not matter.

    Args:
        azimuth: ``(B,)`` degrees.
        elevation: ``(B,)`` degrees.
        distance: ``(B,)`` scene units.

    Returns:
        ``(B, RELATIVE_POSE_DIM)`` float32 tensor.
    """
    if not (azimuth.shape == elevation.shape == distance.shape):
        raise ValueError(
            f"azimuth/elevation/distance must share shape, got "
            f"{tuple(azimuth.shape)}/{tuple(elevation.shape)}/{tuple(distance.shape)}."
        )

    batch = azimuth.shape[0]
    actions = torch.empty(batch, RELATIVE_POSE_DIM, dtype=torch.float32)
    for i in range(batch):
        anchor = look_at_view_matrices([distance[i].item()], [0.0], [0.0])
        target = look_at_view_matrices([distance[i].item()], [elevation[i].item()], [azimuth[i].item()])
        poses = torch.cat([anchor, target], dim=0)  # (2, 4, 4)
        actions[i] = poses_to_relative_actions(poses, pose_convention=pose_convention)[0]

    return actions
