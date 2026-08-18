"""
Pre-render CT volumes to 2D X-Ray projections (PA, LAT, and 360-degree views)
using the physically-rigorous DiffDRR renderer.

This populates the datasets/pre_rendered/ directory for training and evaluating
multiview synthesis models, with no PyTorch3D dependency.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from renderers.diffdrr.renderer import FIXED_LINE_INTEGRAL_MAX, create_diffdrr_renderer
from renderers.diffdrr.data import load_ct_volume

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_patient_id(file_path: str) -> str:
    """Extract a clean patient/scan identifier from file path."""
    name = Path(file_path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    elif name.endswith(".nii"):
        return name[:-4]
    return name


def get_dataset_optimal_fov(file_path: str) -> float:
    """Return dataset-matched optimal FOV for thoracic framing."""
    p_str = str(file_path).upper()
    if "NSCLC" in p_str:
        return 10.2
    elif "TCIA" in p_str:
        return 11.9
    elif "MELA" in p_str:
        return 12.0
    return 12.0


def pre_render_ct(
    ct_path: str,
    output_patient_dir: Path,
    img_shape: int = 256,
    vol_shape: int = 256,
    num_frames: int = 93,
    dist: float = 8.0,
    elev: float = 0.0,
    fov: float = 12.0,
    min_depth: float = 7.0,
    max_depth: float = 9.0,
    device: str = "cuda",
) -> bool:
    """Render projections for a single CT scan and save them."""
    try:
        output_patient_dir.mkdir(parents=True, exist_ok=True)
        views_dir = output_patient_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        pa_path = output_patient_dir / "pa.png"
        lat_path = output_patient_dir / "lat.png"

        logger.info(f"Loading CT volume from {ct_path}...")
        vol = load_ct_volume(ct_path, vol_shape=vol_shape)

        logger.info("Initializing DiffDRR renderer...")
        renderer = create_diffdrr_renderer(img_shape=img_shape, device=device)
        renderer.set_volume(vol)

        logger.info(f"Rendering 360° sweep ({num_frames} frames raw line integrals)...")
        azimuths = torch.linspace(0.0, 360.0, num_frames, device=device)
        
        raw_frames_list = []
        chunk_size = 1
        for i in range(0, num_frames, chunk_size):
            chunk_azimuths = azimuths[i : i + chunk_size]
            chunk_frames = renderer.render(
                azimuth=chunk_azimuths,
                elev=elev,
                dist=dist,
                fov=fov,
                min_depth=min_depth,
                max_depth=max_depth,
                norm_type=None,
                batch_size=chunk_size,
            )
            raw_frames_list.append(chunk_frames.cpu())
            
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        all_raw = torch.cat(raw_frames_list, dim=0)
        norm_frames = torch.clamp(all_raw / FIXED_LINE_INTEGRAL_MAX, 0.0, 1.0)

        # 1. Save PA view (0.0°) and LAT view (90.0°)
        pa_np = (norm_frames[0, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(pa_np, mode="L").save(pa_path)

        lat_idx = int(round(90.0 / (360.0 / (num_frames - 1))))
        lat_np = (norm_frames[lat_idx, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(lat_np, mode="L").save(lat_path)

        # 2. Save individual PNG views
        for frame_idx in range(num_frames):
            frame_np = (norm_frames[frame_idx, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
            frame_path = views_dir / f"{frame_idx:03d}.png"
            Image.fromarray(frame_np, mode="L").save(frame_path)

        # 3. Save single binary float32 tensor container (views.pt)
        pt_tensor = norm_frames.squeeze(1)
        if hasattr(pt_tensor, "as_tensor"):
            pt_tensor = pt_tensor.as_tensor()
        pt_path = output_patient_dir / "views.pt"
        torch.save(pt_tensor, pt_path)

        logger.info(f"Successfully processed and pre-rendered CT volume: {ct_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process CT volume {ct_path}: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="DiffDRR-based Medical Dataset Pre-rendering Pipeline")
    parser.add_argument(
        "--dest_dir",
        type=str,
        default=str(BASE_DIR / "datasets" / "pre_rendered"),
        help="Destination directory for processed pre-rendered outputs"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process per split"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run renderer on"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pre-rendered files"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets to process"
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    allowed_ds = [d.strip().upper() for d in args.datasets.split(",")] if args.datasets else None

    # Collect raw CT files for train and test splits
    train_dirs = []
    if not allowed_ds or "TCIA" in allowed_ds:
        train_dirs.append(BASE_DIR / "data" / "TCIA")
    if not allowed_ds or "MELA2022" in allowed_ds or "MELA" in allowed_ds:
        train_dirs.append(BASE_DIR / "data" / "MELA2022" / "raw")

    test_dirs = []
    if not allowed_ds or "NSCLC" in allowed_ds:
        test_dirs.append(BASE_DIR / "data" / "NSCLC" / "processed")

    for split_name, source_dirs in [("train", train_dirs), ("test", test_dirs)]:
        split_dest = dest_dir / split_name
        split_dest.mkdir(parents=True, exist_ok=True)

        ct_files = []
        for sdir in source_dirs:
            if sdir.exists():
                ct_files.extend(glob.glob(str(sdir / "**" / "*.nii.gz"), recursive=True))
                ct_files.extend(glob.glob(str(sdir / "**" / "*.nii"), recursive=True))

        ct_files = sorted(list(set(ct_files)))
        if args.max_files:
            ct_files = ct_files[:args.max_files]

        logger.info(f"Processing split '{split_name}': {len(ct_files)} files found.")

        for idx, ct_path in enumerate(ct_files, 1):
            pid = get_patient_id(ct_path)
            patient_out_dir = split_dest / pid

            if (patient_out_dir / "views.pt").exists() and not args.overwrite:
                logger.info(f"[{idx}/{len(ct_files)}] Skipping {pid} (already pre-rendered).")
                continue

            fov = get_dataset_optimal_fov(ct_path)
            logger.info(f"[{idx}/{len(ct_files)}] Pre-rendering {pid} with FOV={fov}°...")
            pre_render_ct(
                ct_path=ct_path,
                output_patient_dir=patient_out_dir,
                fov=fov,
                device=args.device,
            )


if __name__ == "__main__":
    main()
