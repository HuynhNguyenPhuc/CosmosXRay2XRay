"""
CT volume preprocessing for the DiffDRR renderer.

This mirrors ``create_ct_transforms``, providing CT volume preprocessing
without requiring PyTorch3D at module import time.
"""

from __future__ import annotations

from monai.transforms import (
    Compose,
    DivisiblePadDict,
    EnsureChannelFirstDict,
    LoadImageDict,
    OrientationDict,
    ResizeDict,
    ScaleIntensityRangeDict,
    SpacingDict,
    ToTensorDict,
)

import torch


def create_ct_transforms(vol_shape: int = 256) -> Compose:
    """
    Create CT volume preprocessing transforms.

    Resample to 1mm isotropic spacing, reorient to "ASL", clip/scale
    intensity to [0, 1], resize to fit ``vol_shape`` on the longest side,
    then pad to a ``vol_shape`` cube.

    Args:
        vol_shape: Desired output cube side length (default: 256).

    Returns:
        A Compose object containing the sequence of transforms to apply to CT volumes.
    """
    return Compose([
        LoadImageDict(keys=["image3d"]),
        EnsureChannelFirstDict(keys=["image3d"]),
        SpacingDict(keys=["image3d"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"], align_corners=True),
        OrientationDict(keys=["image3d"], axcodes="ASL"),
        ScaleIntensityRangeDict(keys=["image3d"], a_min=-1024, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
        ResizeDict(keys=["image3d"], spatial_size=vol_shape, size_mode="longest", mode=["trilinear"], align_corners=True),
        DivisiblePadDict(keys=["image3d"], k=vol_shape, mode="constant", constant_values=0),
        ToTensorDict(keys=["image3d"]),
    ])


def load_ct_volume(ct_path: str, vol_shape: int = 256) -> torch.Tensor:
    """
    Load and preprocess a CT NIfTI volume for rendering.

    Args:
        ct_path: Path to a CT volume file (e.g. ``.nii.gz``).
        vol_shape: Cube side length to resize/pad the volume to (default: 256).

    Returns:
        Preprocessed density tensor of shape ``(1, D, H, W)`` in ``[0, 1]``.
    """
    transforms = create_ct_transforms(vol_shape=vol_shape)
    data_dict = transforms({"image3d": ct_path})
    return data_dict["image3d"]
