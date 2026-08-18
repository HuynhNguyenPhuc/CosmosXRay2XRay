"""
Chest CT DataModule for Novel-View X-Ray Synthesis

Load CT volumes from a pre-built cache. Frontal X-ray images are generated during training.
The cache is expected to be created by `cache_builder.py`.

Cache structure:
    cache_dir/
    └── ct/{hash}.npy      # Preprocessed 256³ CT volume

Each dataset sample returns:
    {
        "hash": str,
        "ct": Tensor (1, 256, 256, 256),
    }
"""

import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from lightning import LightningDataModule, seed_everything

from shared.utils import get_rank, get_world_size, is_distributed, get_logger

log = get_logger(__name__)


class CTVolumeDataset(Dataset):
    """Dataset for CT volumes (.npy) and X-ray images (.png)."""

    def __init__(self, data_list: List[Dict[str, Any]], img_shape: int = 256):
        self.data_list = data_list
        self.img_shape = img_shape

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data_list[idx]

        ct_arr = np.load(sample["ct_path"], mmap_mode="r")
        if ct_arr.ndim == 3:
            ct_arr = ct_arr[np.newaxis, ...]
            
        # Copy to make writable; mmap files are read-only
        ct_arr = ct_arr.copy()

        return {
            "hash": sample["hash"],
            "ct": torch.from_numpy(ct_arr),
        }


class ChestCTDataModule(LightningDataModule):
    """DataModule for loading Chest CT volumes and X-ray images."""

    def __init__(
        self,
        dataset_dirs: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        img_shape: int = 256,
        batch_size: int = 1,
        num_workers: int = 12,
        prefetch_factor: int = 2,
        seed: int = 2222,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        if cache_dir is None:
            cache_dir = "./cache"
            if dataset_dirs is not None and len(dataset_dirs) > 0:
                log.warning("[data] Automatic cache building from dataset_dirs is deprecated; please pre-render or cache dataset prior to training.")
            else:
                log.warning("[data] cache_dir is None and no dataset_dirs provided. Using default './cache'")

        self.cache_dir = Path(cache_dir)

    def _create_data_list(self, split: str) -> List[Dict[str, Any]]:
        ct_cache_dir = self.cache_dir / split / "ct"

        if not ct_cache_dir.exists():
            log.warning(f"[data] Cache not found for split '{split}'")
            return []

        ct_files = sorted(glob.glob(str(ct_cache_dir / "*.npy")))
        data_list = []

        for ct_path in ct_files:
            hash_key = Path(ct_path).stem
            data_list.append({
                "hash": hash_key,
                "ct_path": str(ct_path),
            })

        return data_list

    def setup(self, stage: Optional[str] = None):
        seed_everything(self.hparams.seed)

        if stage == "test":
            self.train_dataset = None
            self.val_dataset = None
            test_list = self._create_data_list("test")
            self.test_dataset = CTVolumeDataset(test_list, self.hparams.img_shape) if test_list else None
        else:
            train_list = self._create_data_list("train")
            val_list = self._create_data_list("val")
            self.train_dataset = CTVolumeDataset(train_list, self.hparams.img_shape) if train_list else None
            self.val_dataset = CTVolumeDataset(val_list, self.hparams.img_shape) if val_list else None
            self.test_dataset = None

        log.info(
            f"[data] Datasets ready: "
            f"train={len(self.train_dataset) if self.train_dataset else 0}, "
            f"val={len(self.val_dataset) if self.val_dataset else 0}, "
            f"test={len(self.test_dataset) if self.test_dataset else 0}"
        )

    def _make_dataloader(self, dataset: Dataset, shuffle: bool = False, drop_last: bool = False):
        if dataset is None:
            raise ValueError("Dataset not available. Did you run setup()?")

        is_ddp = is_distributed()

        kwargs = dict(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False,
            drop_last=drop_last,
        )

        if is_ddp:
            sampler = DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=shuffle,
                drop_last=drop_last,
            )
            return DataLoader(dataset, sampler=sampler, **kwargs)
        else:
            return DataLoader(dataset, shuffle=shuffle, **kwargs)

    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._make_dataloader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._make_dataloader(self.test_dataset, shuffle=False, drop_last=False)
