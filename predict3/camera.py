"""Camera-geometry conditioning for Cosmos 3 (`camera_pose` action domain).

Cosmos 3 carries camera geometry on its **action** port rather than in the text prompt.
The `camera_pose` embodiment domain takes a 9D vector per transition —
``[translation(3), rot6d(6)]`` — projected into the transformer's hidden space by a
continuous `DomainAwareLinear` (`action_proj_in`), so nearby viewing angles map to nearby
vectors. This is the structural reason to prefer it over `predict2_5`'s approach of
formatting ``azimuth {:.1f} deg`` into a prose prompt, where the BPE tokenizer turns
"315.0" and "314.0" into unrelated token sequences with no cyclic or metric prior.

Two external conventions are reused verbatim rather than re-derived, because a silent
mismatch here would corrupt the training signal without ever raising:

* **Camera geometry** comes from ``renderers.diffdrr.renderer.look_at_view_matrices`` —
  the same look-at math the DRR renderer uses to actually render the views, so the
  conditioning poses cannot drift from the rendered geometry.
* **Pose → action-vector encoding** comes from ``cosmos-framework``'s
  ``pose_abs_to_rel``, NVIDIA's own implementation of the layout the pretrained
  `camera_pose` head expects (``(T, 4, 4)`` camera-to-world → ``(T-1, 9)`` relative
  vectors, matching `_EMBODIMENT_TO_RAW_ACTION_DIM["camera_pose"] == 9`).

Note the ``T-1``: actions describe *transitions*, so ``T`` frames yield ``T-1`` action
tokens. That is Cosmos 3's ``chunk_size`` / ``chunk_size + 1`` frames contract, not an
off-by-one.
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

# `cosmos-framework` is vendored as a submodule but deliberately NOT pip-installed: its full
# dependency tree is heavy and irrelevant here, while the pose utilities we need import only
# numpy/scipy/torch. Put the submodule root on `sys.path` so that one module is importable
# on its own (every `__init__.py` on the path is 2-3 lines, so this pulls in nothing else).
_COSMOS_FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent / "cosmos-framework"
if _COSMOS_FRAMEWORK_ROOT.is_dir() and str(_COSMOS_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS_FRAMEWORK_ROOT))

from cosmos_framework.data.generator.action.utils.domain_utils import (  # noqa: E402
    EMBODIMENT_TO_DOMAIN_ID,
    EMBODIMENT_TO_RAW_ACTION_DIM,
)
from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel  # noqa: E402

CAMERA_POSE_DOMAIN_NAME = "camera_pose"
CAMERA_POSE_DOMAIN_ID = EMBODIMENT_TO_DOMAIN_ID[CAMERA_POSE_DOMAIN_NAME]
CAMERA_POSE_RAW_DIM = EMBODIMENT_TO_RAW_ACTION_DIM[CAMERA_POSE_DOMAIN_NAME]

PoseConvention = Literal["backward_anchored", "backward_framewise"]


def xray_view_matrices(
    view_names: tuple[str, ...] | list[str] = XRAY_CAMERAS,
    distance: float = XRAY_DISTANCE_DEFAULT,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Camera-to-world matrices for named X-ray views.

    Args:
        view_names: View keys, each present in ``XRAY_EXTRINSICS`` (default: all 7 views in
            canonical ``XRAY_CAMERAS`` order).
        distance: Source-to-isocenter distance in scene units.

    Returns:
        ``(len(view_names), 4, 4)`` camera-to-world transforms.
    """
    unknown = [name for name in view_names if name not in XRAY_EXTRINSICS]
    if unknown:
        raise KeyError(f"Unknown X-ray view(s) {unknown}; expected keys from XRAY_EXTRINSICS.")

    azimuths = [XRAY_EXTRINSICS[name]["azimuth"] for name in view_names]
    elevations = [XRAY_EXTRINSICS[name]["elevation"] for name in view_names]
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


def poses_to_camera_pose_actions(
    poses_abs: torch.Tensor,
    pose_convention: PoseConvention = "backward_anchored",
) -> torch.Tensor:
    """Convert absolute camera-to-world poses into Cosmos 3 ``camera_pose`` action vectors.

    Args:
        poses_abs: ``(T, 4, 4)`` camera-to-world transforms, ``T >= 2``.
        pose_convention: ``"backward_anchored"`` encodes every view relative to view 0
            (``T_0^-1 @ T_i+1``), which matches this task's framing — "given the anchor
            X-ray view, produce the view at this relative offset" — and keeps each action
            independent of its neighbours. ``"backward_framewise"`` (``T_i^-1 @ T_i+1``)
            encodes consecutive deltas instead, appropriate for a continuous orbit sweep.

    Returns:
        ``(T - 1, 9)`` float32 tensor laid out as ``[translation(3), rot6d(6)]``.
    """
    if poses_abs.ndim != 3 or poses_abs.shape[-2:] != (4, 4):
        raise ValueError(f"`poses_abs` must have shape (T, 4, 4), got {tuple(poses_abs.shape)}.")
    if poses_abs.shape[0] < 2:
        raise ValueError("At least 2 poses are required to form a transition (T >= 2).")

    poses_np = poses_abs.detach().cpu().to(torch.float32).numpy().astype(np.float64)
    actions = pose_abs_to_rel(poses_np, rotation_format="rot6d", pose_convention=pose_convention)

    if actions.shape[-1] != CAMERA_POSE_RAW_DIM:
        raise ValueError(
            f"Expected {CAMERA_POSE_RAW_DIM}-D camera_pose actions, got {actions.shape[-1]}D — "
            "cosmos-framework's pose layout may have changed."
        )
    return torch.from_numpy(actions).to(dtype=torch.float32)


def pad_actions_to_model_dim(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Zero-pad raw ``(T, 9)`` domain actions up to the model's ``action_dim`` (64).

    Mirrors the padding `Cosmos3OmniPipeline.prepare_latents` applies for
    ``forward_dynamics``; the domain-aware projection is trained to read the raw width from
    the leading channels and ignore the zero tail.
    """
    raw_dim = actions.shape[-1]
    if raw_dim > action_dim:
        raise ValueError(f"Raw action width {raw_dim} exceeds the model's action_dim {action_dim}.")
    if raw_dim == action_dim:
        return actions
    padding = torch.zeros(*actions.shape[:-1], action_dim - raw_dim, dtype=actions.dtype, device=actions.device)
    return torch.cat([actions, padding], dim=-1)


def xray_view_actions(
    view_names: tuple[str, ...] | list[str] = XRAY_CAMERAS,
    distance: float = XRAY_DISTANCE_DEFAULT,
    pose_convention: PoseConvention = "backward_anchored",
    action_dim: int | None = None,
) -> torch.Tensor:
    """Convenience: named X-ray views → ``camera_pose`` action vectors.

    Args:
        action_dim: When given, zero-pad to this width (the model's ``action_dim``);
            otherwise return the raw 9D vectors.

    Returns:
        ``(len(view_names) - 1, 9 or action_dim)`` float32 tensor.
    """
    poses = xray_view_matrices(view_names=view_names, distance=distance)
    actions = poses_to_camera_pose_actions(poses, pose_convention=pose_convention)
    if action_dim is not None:
        actions = pad_actions_to_model_dim(actions, action_dim)
    return actions
