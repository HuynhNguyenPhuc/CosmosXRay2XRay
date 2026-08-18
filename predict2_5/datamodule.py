"""DataModule for pre-rendered 93-view 360° rotational projections."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from lightning import LightningDataModule, seed_everything

from shared.constants import NUM_FRAMES, XRAY_PROMPT_TEMPLATE
from shared.utils import get_logger

logger = get_logger(__name__)

DEFAULT_PROMPT = "A 360-degree rotational view of a chest CT scan."


# ============================================================
# Dataset Class
# ============================================================

class PreRendered360Dataset(Dataset):
    """
    Dataset for loading pre-rendered 93-view 360° rotation projections.

    Expected directory structure:
        data_dir/
        ├── patient_001/
        │   ├── pa.png          # Frontal PA radiograph [256x256]
        │   └── views/          # 93 rotational frames
        │       ├── 000.png
        │       ├── 001.png
        │       └── ... (092.png)
        └── ...
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        patient_dirs: Optional[list[str]] = None,
        num_frames: int = NUM_FRAMES,
        img_size: int = 256,
        prompt: str = DEFAULT_PROMPT,
    ) -> None:
        """
        Initialize PreRendered360Dataset.

        Args:
            data_dir: Root directory containing patient subdirectories.
            patient_dirs: Optional list of specific patient directory names. If None, scans all in data_dir.
            num_frames: Total expected frames per rotation video (default: 93).
            img_size: Target image resolution (H, W). Default: 256.
            prompt: Default text prompt string for conditioning.
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.img_size = (img_size, img_size)
        self.prompt = prompt

        if patient_dirs is not None:
            self.patient_paths = [
                self.data_dir / p for p in patient_dirs if (self.data_dir / p).is_dir()
            ]
        else:
            if self.data_dir.exists():
                self.patient_paths = sorted([
                    p for p in self.data_dir.iterdir()
                    if p.is_dir() and (p / "views").exists()
                ])
            else:
                self.patient_paths = []

        logger.info(
            "Initialized PreRendered360Dataset with %d patient cases from %s",
            len(self.patient_paths),
            self.data_dir,
        )

    def __len__(self) -> int:
        return len(self.patient_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patient_path = self.patient_paths[idx]
        patient_id = patient_path.name

        # 1. Load frontal PA X-ray image (pa.png)
        pa_path = patient_path / "pa.png"

        if pa_path.exists():
            pa_img = Image.open(pa_path).convert("L")
        else:
            pa_img = Image.open(patient_path / "views" / "000.png").convert("L")

        pa_tensor = TF.to_tensor(pa_img)  # Shape: [1, H, W], range [0.0, 1.0]

        if pa_tensor.shape[-2:] != self.img_size:
            pa_tensor = TF.resize(pa_tensor, list(self.img_size))

        pa_tensor = pa_tensor.repeat(3, 1, 1)  # Expand to 3 channels: [3, H, W]

        # 2. Load 93 rotation views directly from views/
        views_dir = patient_path / "views"
        view_files = sorted([f for f in os.listdir(views_dir) if f.endswith(".png")])

        if len(view_files) < self.num_frames:
            logger.warning(
                "Patient %s has %d views, expected %d.",
                patient_id,
                len(view_files),
                self.num_frames,
            )

        selected_files = view_files[: self.num_frames]
        frame_tensors: list[torch.Tensor] = []

        for vf in selected_files:
            v_path = views_dir / vf
            v_img = Image.open(v_path).convert("L")

            v_tensor = TF.to_tensor(v_img)

            if v_tensor.shape[-2:] != self.img_size:
                v_tensor = TF.resize(v_tensor, list(self.img_size))

            v_tensor = v_tensor.repeat(3, 1, 1)
            frame_tensors.append(v_tensor)

        while len(frame_tensors) < self.num_frames:
            frame_tensors.append(frame_tensors[-1].clone())

        video_tensor = torch.stack(frame_tensors, dim=1)

        prompt_txt_path = patient_path / "prompt.txt"
        prompt_str = self.prompt

        if prompt_txt_path.exists():
            try:
                prompt_str = prompt_txt_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning("Failed to read prompt file %s: %s", prompt_txt_path, e)

        return {
            "video": video_tensor,  # FloatTensor [3, 93, 256, 256] in [0, 1]
            "image": pa_tensor,     # FloatTensor [3, 256, 256] in [0, 1]
            "prompt": prompt_str,
            "patient_id": patient_id,
        }


# ============================================================
# DataModule Class
# ============================================================

class PreRenderedDataModule(LightningDataModule):
    """Lightning DataModule for loading pre-rendered 93-view 360° rotation dataset."""

    def __init__(
        self,
        dataset_path: str = "datasets/pre_rendered",
        val_split_ratio: float = 0.1,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialize PreRenderedDataModule.

        Args:
            dataset_path: Path to dataset root (containing `train` and `test` subdirs).
            val_split_ratio: Fraction of train dataset used for validation.
            batch_size: Batch size for dataloaders.
            num_workers: DataLoader worker count.
            pin_memory: Pin memory for GPU transfer efficiency.
            persistent_workers: Keep dataloader workers active across epochs.
            seed: RNG seed for reproducible train/val splits.
            **kwargs: Extra arguments for compatibility with TrainingConfig.
        """
        super().__init__()

        self.save_hyperparameters()

        self.dataset_path = Path(dataset_path)
        self.train_dir = (
            self.dataset_path / "train"
            if (self.dataset_path / "train").exists()
            else self.dataset_path
        )
        self.test_dir = self.dataset_path / "test"

        self.train_dataset: Optional[PreRendered360Dataset] = None
        self.val_dataset: Optional[PreRendered360Dataset] = None
        self.test_dataset: Optional[PreRendered360Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for train, val, and test splits.

        Args:
            stage: Stage hint ('fit', 'test', or None).
        """
        seed_everything(self.hparams.seed)

        if stage in (None, "fit"):
            if self.train_dir.exists():
                all_patients = sorted([
                    p.name
                    for p in self.train_dir.iterdir()
                    if p.is_dir() and (p / "views").exists()
                ])
            else:
                all_patients = []

            n_val = int(len(all_patients) * self.hparams.val_split_ratio)

            g = torch.Generator().manual_seed(self.hparams.seed)
            perm = (
                torch.randperm(len(all_patients), generator=g).tolist()
                if len(all_patients) > 0
                else []
            )
            shuffled_patients = [all_patients[i] for i in perm]

            val_patients = shuffled_patients[:n_val]
            train_patients = shuffled_patients[n_val:]

            self.train_dataset = PreRendered360Dataset(
                data_dir=self.train_dir,
                patient_dirs=train_patients,
            )
            self.val_dataset = PreRendered360Dataset(
                data_dir=self.train_dir,
                patient_dirs=val_patients,
            )

            logger.info(
                "DataModule setup (fit): %d train cases, %d val cases",
                len(train_patients),
                len(val_patients),
            )

        if stage in (None, "test"):
            if self.test_dir.exists():
                self.test_dataset = PreRendered360Dataset(data_dir=self.test_dir)
                logger.info(
                    "DataModule setup (test): %d test cases",
                    len(self.test_dataset),
                )
            else:
                self.test_dataset = None

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise RuntimeError("Train dataset is empty or not initialized.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers
                if self.hparams.num_workers > 0
                else False
            ),
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers
                if self.hparams.num_workers > 0
                else False
            ),
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None or len(self.test_dataset) == 0:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers
                if self.hparams.num_workers > 0
                else False
            ),
        )
