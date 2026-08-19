"""Cosmos-3-specific camera conditioning: the `camera_pose` action domain.

Cosmos 3 carries camera geometry on its **action** port rather than in the text prompt.
The `camera_pose` embodiment domain takes a 9D vector per transition —
``[translation(3), rot6d(6)]`` — projected into the transformer's hidden space by a
continuous `DomainAwareLinear` (`action_proj_in`), so nearby viewing angles map to nearby
vectors. This is the structural reason to prefer it over `predict2_5`'s baseline of
formatting ``azimuth {:.1f} deg`` into a prose prompt, where the BPE tokenizer turns
"315.0" and "314.0" into unrelated token sequences with no cyclic or metric prior.

The geometry and pose-to-vector math (identical to what `predict2_5`'s own action
conditioning uses — see its `ActionConditionedMinimalV1LVGDiT` integration) live in
``shared.camera_pose``, not here, so neither pipeline depends on the other's package. This
module adds only what's specific to Cosmos 3's action port: its registered domain id/width,
and zero-padding the raw 9D vector up to the model's ``action_dim`` (64).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from shared.camera_pose import (
    RELATIVE_POSE_DIM,
    PoseConvention,
    anchor_relative_action,
    orbit_view_matrices,
    poses_to_relative_actions,
    xray_view_actions as _xray_view_actions,
    xray_view_matrices,
)
from shared.constants import XRAY_CAMERAS, XRAY_DISTANCE_DEFAULT

__all__ = [
    "CAMERA_POSE_DOMAIN_ID",
    "CAMERA_POSE_RAW_DIM",
    "PoseConvention",
    "anchor_relative_action",
    "orbit_view_matrices",
    "pad_actions_to_model_dim",
    "poses_to_relative_actions",
    "xray_view_actions",
    "xray_view_matrices",
]

# `cosmos-framework` is vendored as a submodule but deliberately NOT pip-installed: its full
# dependency tree is heavy and irrelevant here, while the domain table needed is a plain
# dict. `shared.camera_pose` already puts the submodule root on `sys.path` when imported
# above, so this import can rely on that having happened.
_COSMOS_FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent / "cosmos-framework"
if _COSMOS_FRAMEWORK_ROOT.is_dir() and str(_COSMOS_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS_FRAMEWORK_ROOT))

from cosmos_framework.data.generator.action.utils.domain_utils import (  # noqa: E402
    EMBODIMENT_TO_DOMAIN_ID,
    EMBODIMENT_TO_RAW_ACTION_DIM,
)

CAMERA_POSE_DOMAIN_NAME = "camera_pose"
CAMERA_POSE_DOMAIN_ID = EMBODIMENT_TO_DOMAIN_ID[CAMERA_POSE_DOMAIN_NAME]
CAMERA_POSE_RAW_DIM = EMBODIMENT_TO_RAW_ACTION_DIM[CAMERA_POSE_DOMAIN_NAME]

if CAMERA_POSE_RAW_DIM != RELATIVE_POSE_DIM:
    raise AssertionError(
        f"cosmos-framework's camera_pose width ({CAMERA_POSE_RAW_DIM}) no longer matches "
        f"shared.camera_pose.RELATIVE_POSE_DIM ({RELATIVE_POSE_DIM}) -- the two were assumed "
        "to coincide, not guaranteed to; re-check before relying on this module."
    )


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
    """Named X-ray views -> Cosmos 3 ``camera_pose`` action vectors.

    Args:
        action_dim: When given, zero-pad to this width (the model's ``action_dim``);
            otherwise return the raw 9D vectors.

    Returns:
        ``(len(view_names) - 1, 9 or action_dim)`` float32 tensor.
    """
    actions = _xray_view_actions(view_names=view_names, distance=distance, pose_convention=pose_convention)
    if action_dim is not None:
        actions = pad_actions_to_model_dim(actions, action_dim)
    return actions
