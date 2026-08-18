"""
Control signal generation for Transfer2.5 X-ray synthesis.

Generates spatial control signals from CT volumes / rendered DRRs,
analogous to "World Scenario Maps" (HD maps + bounding boxes) in
the AV domain.

Supported control types:
    edge_map   — Sobel edge detection on the rendered DRR
    seg_mask   — Projected organ segmentation (requires CT labels)
    depth_map  — Depth buffer from volume ray-casting
"""

import torch
import torch.nn.functional as F
from typing import Literal


def sobel_edge_map(image: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """
    Compute Sobel edge map from a grayscale image.

    Args:
        image: (B, 1, H, W) float tensor in [0, 1].
        threshold: Edge magnitude threshold for binarisation.

    Returns:
        (B, 1, H, W) float tensor in [0, 1] — edge magnitude.
    """
    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=image.dtype, device=image.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=image.dtype, device=image.device
    ).view(1, 1, 3, 3)

    gx = F.conv2d(image, sobel_x, padding=1)
    gy = F.conv2d(image, sobel_y, padding=1)
    magnitude = torch.sqrt(gx ** 2 + gy ** 2)

    # Normalize to [0,1]
    mag_max = magnitude.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    magnitude = magnitude / mag_max

    if threshold > 0:
        magnitude = (magnitude > threshold).float()

    return magnitude


def depth_map_from_volume(
    ct_volume: torch.Tensor,
    density_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Approximate depth map from a CT volume slice stack.

    Computes the first-hit depth along the Z-axis where density
    exceeds the threshold. This is a simplified version that doesn't
    require full ray-casting.

    Args:
        ct_volume: (B, 1, D, H, W) density Volume in [0,1].
        density_threshold: Threshold for "surface detection".

    Returns:
        (B, 1, H, W) depth map normalized to [0, 1].
    """
    B, _, D, H, W = ct_volume.shape
    vol = ct_volume.squeeze(1)  # (B, D, H, W)

    # Find first depth index where density > threshold along Z
    above_thresh = (vol > density_threshold).float()
    depth_indices = torch.argmax(above_thresh, dim=1, keepdim=True).float()  # (B, 1, H, W)

    # Where no voxel is above threshold, set depth to max
    any_above = above_thresh.any(dim=1, keepdim=True).float()
    depth_indices = depth_indices * any_above + D * (1 - any_above)

    # Normalize to [0, 1]
    return depth_indices / D


@torch.no_grad()
def generate_control_signal(
    drr_image: torch.Tensor,
    ct_volume: torch.Tensor = None,
    control_type: Literal["edge_map", "depth_map", "seg_mask"] = "edge_map",
    edge_threshold: float = 0.05,
    depth_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Generate control signal for ControlNet conditioning.

    Args:
        drr_image: (B, 1, H, W) rendered DRR in [0, 1].
        ct_volume: (B, 1, D, H, W) CT volume (needed for depth_map).
        control_type: Type of control signal to generate.
        edge_threshold: Threshold for Sobel edge detection.
        depth_threshold: Density threshold for depth map.

    Returns:
        (B, 1, H, W) control signal in [0, 1].
    """
    if control_type == "edge_map":
        return sobel_edge_map(drr_image, threshold=edge_threshold)
    elif control_type == "depth_map":
        if ct_volume is None:
            raise ValueError("CT volume required for depth_map control signal")
        return depth_map_from_volume(ct_volume, density_threshold=depth_threshold)
    elif control_type == "seg_mask":
        # Fallback: use thresholded DRR as pseudo-segmentation
        # (Real segmentation would require TotalSegmentator labels)
        return (drr_image > 0.3).float()
    else:
        raise ValueError(f"Unknown control_type: {control_type}")
