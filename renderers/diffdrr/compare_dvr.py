"""
Numerically verify DiffDRRXRayVolumeRenderer against the PyTorch3D DVR
(shared.renderer.ObjectCentricXRayVolumeRenderer) it is meant to be
equivalent to.

Requires both `pytorch3d` and `diffdrr` installed.

Usage:
    python -m renderers.diffdrr.compare_dvr --ct path/to/scan.nii.gz
    python -m renderers.diffdrr.compare_dvr  # uses a synthetic phantom
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def make_phantom(vol_shape: int = 128, device: str = "cuda") -> torch.Tensor:
    """A simple synthetic phantom: a soft-tissue sphere with a denser bone-like core."""
    grid = torch.linspace(-1, 1, vol_shape)
    z, y, x = torch.meshgrid(grid, grid, grid, indexing="ij")
    r = (x**2 + y**2 + z**2).sqrt()
    vol = torch.zeros(vol_shape, vol_shape, vol_shape)
    vol[r < 0.8] = 0.3
    vol[r < 0.3] = 0.9
    return vol.unsqueeze(0).to(device)  # (1, D, H, W)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=str, default=None, help="Path to a CT NIfTI volume; omit to use a synthetic phantom.")
    parser.add_argument("--vol_shape", type=int, default=128)
    parser.add_argument("--img_shape", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--dist", type=float, default=8.0)
    parser.add_argument("--elev", type=float, default=0.0)
    parser.add_argument("--fov", type=float, default=12.0)
    parser.add_argument("--min_depth", type=float, default=7.0)
    parser.add_argument("--max_depth", type=float, default=9.0)
    parser.add_argument("--n_pts_per_ray", type=int, default=320)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="dvr_vs_diffdrr.png")
    args = parser.parse_args()

    device = args.device

    try:
        from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
        from shared.renderer import ObjectCentricXRayVolumeRenderer
    except ImportError as exc:
        raise SystemExit(
            f"PyTorch3D / shared.renderer are required to run this comparison: {exc}"
        ) from exc

    from renderers.diffdrr import DiffDRRVolumeRenderer, load_ct_volume

    if args.ct is not None:
        vol = load_ct_volume(args.ct, vol_shape=args.vol_shape).to(device)
    else:
        vol = make_phantom(vol_shape=args.vol_shape, device=device)
    vol = vol.unsqueeze(0)  # (1, 1, D, H, W)

    azimuths = torch.linspace(0, 360, args.num_frames)

    # --- PyTorch3D DVR ---
    dvr = ObjectCentricXRayVolumeRenderer(
        image_width=args.img_shape,
        image_height=args.img_shape,
        n_pts_per_ray=args.n_pts_per_ray,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        ndc_extent=1.0,
    ).to(device)

    with torch.no_grad():
        R, T = look_at_view_transform(dist=args.dist, elev=args.elev, azim=azimuths)
        cam = FoVPerspectiveCameras(R=R, T=T, fov=args.fov, device=device)
        dvr_frames = dvr(
            volume=vol.repeat(len(azimuths), 1, 1, 1, 1),
            cameras=cam,
            opacity=None,
            norm_type="standardized",
            scaling_factor=1.0,
            is_grayscale=True,
            return_bundle=False,
        )

    # --- DiffDRR ---
    diffdrr_renderer = DiffDRRVolumeRenderer(
        image_width=args.img_shape,
        image_height=args.img_shape,
        n_pts_per_ray=args.n_pts_per_ray,
        ndc_extent=1.0,
        device=device,
    )
    diffdrr_renderer.set_volume(vol[0])

    diffdrr_frames = diffdrr_renderer.render(
        azimuth=azimuths,
        elev=args.elev,
        dist=args.dist,
        fov=args.fov,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        norm_type="standardized",
    )

    dvr_np = dvr_frames.cpu().numpy()
    diffdrr_np = diffdrr_frames.cpu().numpy()

    mse = float(np.mean((dvr_np - diffdrr_np) ** 2))
    print(f"Comparison DVR vs DiffDRR: MSE = {mse:.6f}")


if __name__ == "__main__":
    main()
