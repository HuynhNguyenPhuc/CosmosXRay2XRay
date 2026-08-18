"""Visualizer for cached CT volumes and X-ray images."""

import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from shared.utils import get_logger

log = get_logger(__name__)


class CacheVisualizer:
    """Visualize cached CT volumes and rendered X-rays."""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)

    def load_volume(self, hash_key: str, split: str = "train") -> Optional[np.ndarray]:
        """Load a CT volume (.npy file)."""
        vol_path = self.cache_dir / split / "ct" / f"{hash_key}.npy"
        if not vol_path.exists():
            log.warning(f"Volume not found: {vol_path}")
            return None
        return np.load(vol_path)

    def load_xray(self, hash_key: str, split: str = "train") -> Optional[np.ndarray]:
        """Load a rendered X-ray image (.npy file)."""
        xray_path = self.cache_dir / split / "xr" / f"{hash_key}.npy"
        if not xray_path.exists():
            log.warning(f"X-ray not found: {xray_path}")
            return None
        return np.load(xray_path)

    def get_sample_hashes(self, split: str = "train", limit: int = 5) -> List[str]:
        """Get list of available sample hashes."""
        ct_dir = self.cache_dir / split / "ct"
        if not ct_dir.exists():
            log.warning(f"Split directory not found: {ct_dir}")
            return []
        
        hashes = [f.stem for f in ct_dir.glob("*.npy")]
        return sorted(hashes)[:limit]

    def visualize_sample(
        self,
        hash_key: str,
        split: str = "train",
        slice_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[str] = None,
    ):
        """Visualize a single sample with CT volume slices and X-ray."""
        vol = self.load_volume(hash_key, split)
        xray = self.load_xray(hash_key, split)

        if vol is None or xray is None:
            log.error(f"Failed to load data for {hash_key}")
            return

        # Default to middle slice if not specified
        if slice_idx is None:
            slice_idx = vol.shape[0] // 2

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)

        # Axial, Coronal, Sagittal slices
        ax_axial = fig.add_subplot(gs[0, 0])
        ax_coronal = fig.add_subplot(gs[0, 1])
        ax_xray = fig.add_subplot(gs[0, 2])

        # Axial slice (Z-axis)
        axial_slice = vol[slice_idx, :, :]
        ax_axial.imshow(axial_slice, cmap="gray")
        ax_axial.set_title(f"Axial (slice {slice_idx}/{vol.shape[0]-1})")
        ax_axial.axis("off")

        # Coronal slice (Y-axis)
        coronal_slice = vol[:, slice_idx, :]
        ax_coronal.imshow(coronal_slice, cmap="gray")
        ax_coronal.set_title(f"Coronal (slice {slice_idx}/{vol.shape[1]-1})")
        ax_coronal.axis("off")

        # X-ray (frontal projection)
        if xray.ndim == 3:
            xray_display = xray[0]
        else:
            xray_display = xray
        ax_xray.imshow(xray_display, cmap="gray")
        ax_xray.set_title("Frontal X-ray")
        ax_xray.axis("off")

        fig.suptitle(f"Sample: {hash_key[:8]}... [{split}]", fontsize=14, fontweight="bold")

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            log.info(f"Visualization saved to {save_path}")

        plt.show()

    def visualize_grid(
        self,
        split: str = "train",
        n_samples: int = 9,
        figsize: Tuple[int, int] = (15, 12),
        save_path: Optional[str] = None,
    ):
        """Visualize a grid of samples."""
        hashes = self.get_sample_hashes(split, limit=n_samples)

        if not hashes:
            log.warning(f"No samples found in {split} split")
            return

        n_cols = 3
        n_rows = (len(hashes) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if len(hashes) > 1 else [axes]

        for idx, hash_key in enumerate(hashes):
            ax = axes[idx]
            xray = self.load_xray(hash_key, split)

            if xray is not None:
                if xray.ndim == 3:
                    xray = xray[0]
                ax.imshow(xray, cmap="gray")
                ax.set_title(f"{hash_key[:12]}...", fontsize=10)
            else:
                ax.text(0.5, 0.5, "Failed to load", ha="center", va="center")

            ax.axis("off")

        # Hide unused subplots
        for idx in range(len(hashes), len(axes)):
            axes[idx].axis("off")

        fig.suptitle(f"X-ray Grid [{split}] - {len(hashes)} samples", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            log.info(f"Grid saved to {save_path}")

        plt.show()

    def get_cache_stats(self) -> dict:
        """Get statistics about the cached dataset."""
        stats = {}

        for split in ["train", "val", "test"]:
            ct_dir = self.cache_dir / split / "ct"
            xr_dir = self.cache_dir / split / "xr"

            ct_count = len(list(ct_dir.glob("*.npy"))) if ct_dir.exists() else 0
            xr_count = len(list(xr_dir.glob("*.npy"))) if xr_dir.exists() else 0

            stats[split] = {
                "ct_count": ct_count,
                "xr_count": xr_count,
                "complete": ct_count == xr_count,
            }

        return stats

    def print_cache_info(self):
        """Print information about the cache."""
        stats = self.get_cache_stats()
        total = sum(s["ct_count"] for s in stats.values())

        log.info(f"Cache directory: {self.cache_dir}")
        log.info(f"Total volumes: {total}")

        for split, s in stats.items():
            status = "✓" if s["complete"] else "✗"
            log.info(
                f"  [{status}] {split}: CT={s['ct_count']}, XR={s['xr_count']}"
            )


# Example usage
if __name__ == "__main__":
    viz = CacheVisualizer(cache_dir="./cache")
    viz.print_cache_info()

    # Visualize a single sample
    hashes = viz.get_sample_hashes("train", limit=1)
    if hashes:
        viz.visualize_sample(hashes[0], "train")

    # Visualize grid
    viz.visualize_grid("train", n_samples=9)
