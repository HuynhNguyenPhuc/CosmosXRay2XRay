"""Build the CT volume cache consumed by :class:`shared.datamodule.ChestCTDataModule`.

Converts raw NIfTI CT scans into preprocessed ``.npy`` volumes laid out as::

    cache_dir/
    +-- train/ct/{patient_id}.npy
    +-- val/ct/{patient_id}.npy
    +-- test/ct/{patient_id}.npy

Only CT volumes are cached. The 7-view X-ray projections are rendered **on the fly** during
training by ``renderers.diffdrr.renderer.DiffDRRVolumeRenderer`` (see
``predict2_5.module._setup_renderer``), so there is deliberately no pre-rendered image cache
here — that keeps the cache ~42GB instead of hundreds, and lets camera parameters (FOV,
distance, jitter) vary per step instead of being frozen at cache time.
``datasets/pre_render_diffdrr.py`` is a separate tool that pre-renders 360-degree sweeps for
baselines/evaluation; it does not feed this cache.

Preprocessing is delegated wholly to ``renderers.diffdrr.data.load_ct_volume`` — the same
loader the renderer itself uses — so cached volumes are identical to what on-the-fly
rendering would consume: 1mm isotropic, ASL-oriented, HU windowed [-1024, 1500] -> [0, 1],
resized on the longest side and padded to a ``vol_shape`` cube.

Split protocol follows ``cross_dataset_split.json``: TCIA + MELA2022 for train (with a
deterministic validation holdout), NSCLC held out entirely as a cross-dataset test set, so
test performance measures generalization across acquisition sites rather than within one.

Example::

    python -m shared.cache_builder \\
        --train-dirs /path/TCIA /path/MELA2022/raw \\
        --test-dirs /path/NSCLC/processed \\
        --cache-dir ./cache
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from renderers.diffdrr.data import load_ct_volume
from shared.constants import VOL_SIZE
from shared.utils import get_logger

log = get_logger(__name__)

NIFTI_SUFFIXES = (".nii", ".nii.gz")


def patient_id_from_path(path: Path) -> str:
    """Strip NIfTI suffixes to get a stable per-scan id.

    This becomes the ``.npy`` stem, which ``ChestCTDataModule`` reads back as the sample
    ``hash`` (see its ``_create_data_list``).
    """
    name = path.name
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def find_nifti_files(directories: list[str | Path]) -> list[Path]:
    """Recursively collect NIfTI scans from *directories*, de-duplicated and sorted."""
    found: set[Path] = set()
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            log.warning(f"[cache] Source directory does not exist, skipping: {root}")
            continue
        for suffix in NIFTI_SUFFIXES:
            found.update(root.rglob(f"*{suffix}"))
    return sorted(found)


def is_validation_case(patient_id: str, val_fraction: float) -> bool:
    """Deterministically assign a scan to validation by hashing its id.

    Hashing the id rather than slicing a shuffled list keeps the assignment stable when new
    scans are added or files are enumerated in a different order, so a cache rebuilt later
    cannot silently leak a previous validation case into training.
    """
    if val_fraction <= 0.0:
        return False
    digest = hashlib.sha256(patient_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF) < val_fraction


def cache_one_volume(ct_path: Path, out_path: Path, vol_shape: int = VOL_SIZE) -> bool:
    """Preprocess one CT scan and write it as ``.npy``. Returns True on success.

    Writes via a temporary file and atomically renames, so an interrupted run cannot leave a
    truncated ``.npy`` that a later run would treat as already cached.
    """
    try:
        volume = load_ct_volume(str(ct_path), vol_shape=vol_shape)
        array = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))

        expected = (1, vol_shape, vol_shape, vol_shape)
        if array.shape != expected:
            log.error(f"[cache] Unexpected shape {array.shape} (expected {expected}) for {ct_path}")
            return False
        if not np.isfinite(array).all():
            log.error(f"[cache] Non-finite values in preprocessed volume for {ct_path}")
            return False

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".npy.tmp")
        # Write through a file handle: np.save() silently appends ".npy" to a path that does
        # not already end in it, which would produce "<id>.npy.tmp.npy" and break the rename.
        with open(tmp_path, "wb") as handle:
            np.save(handle, array)
        tmp_path.replace(out_path)
        return True

    except Exception as exc:  # noqa: BLE001 - one bad scan must not abort the whole build
        log.error(f"[cache] Failed to process {ct_path}: {exc}")
        return False


def build_split(
    ct_files: list[Path],
    cache_dir: Path,
    split: str,
    vol_shape: int = VOL_SIZE,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """Cache *ct_files* into ``cache_dir/split/ct``. Returns ``(written, skipped, failed)``."""
    out_dir = cache_dir / split / "ct"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = failed = 0
    total = len(ct_files)

    for index, ct_path in enumerate(ct_files, start=1):
        out_path = out_dir / f"{patient_id_from_path(ct_path)}.npy"

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        if cache_one_volume(ct_path, out_path, vol_shape=vol_shape):
            written += 1
        else:
            failed += 1

        if index % 25 == 0 or index == total:
            log.info(f"[cache] {split}: {index}/{total} (written={written} skipped={skipped} failed={failed})")

    return written, skipped, failed


def build_cache(
    train_dirs: list[str],
    test_dirs: list[str],
    cache_dir: str = "./cache",
    val_fraction: float = 0.05,
    vol_shape: int = VOL_SIZE,
    max_files: int | None = None,
    overwrite: bool = False,
) -> dict[str, tuple[int, int, int]]:
    """Build train/val/test caches. Returns per-split ``(written, skipped, failed)``."""
    cache_root = Path(cache_dir)

    train_pool = find_nifti_files(train_dirs)
    test_files = find_nifti_files(test_dirs)

    if max_files is not None:
        train_pool = train_pool[:max_files]
        test_files = test_files[:max_files]

    train_files = [p for p in train_pool if not is_validation_case(patient_id_from_path(p), val_fraction)]
    val_files = [p for p in train_pool if is_validation_case(patient_id_from_path(p), val_fraction)]

    log.info(
        f"[cache] Discovered {len(train_pool)} train-pool and {len(test_files)} test scans -> "
        f"train={len(train_files)} val={len(val_files)} test={len(test_files)}"
    )

    results: dict[str, tuple[int, int, int]] = {}
    for split, files in (("train", train_files), ("val", val_files), ("test", test_files)):
        if not files:
            log.warning(f"[cache] No files for split '{split}'")
            results[split] = (0, 0, 0)
            continue
        results[split] = build_split(files, cache_root, split, vol_shape=vol_shape, overwrite=overwrite)

    for split, (written, skipped, failed) in results.items():
        log.info(f"[cache] {split}: written={written} skipped={skipped} failed={failed}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CT volume cache for CosmosXRay2XRay")
    parser.add_argument(
        "--train-dirs", type=str, nargs="+", required=True,
        help="Directories searched recursively for training NIfTI scans (e.g. TCIA, MELA2022/raw).",
    )
    parser.add_argument(
        "--test-dirs", type=str, nargs="+", default=[],
        help="Directories for the held-out cross-dataset test split (e.g. NSCLC/processed).",
    )
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument(
        "--val-fraction", type=float, default=0.05,
        help="Fraction of the training pool held out for validation (deterministic, hashed by id).",
    )
    parser.add_argument("--vol-shape", type=int, default=VOL_SIZE)
    parser.add_argument("--max-files", type=int, default=None, help="Cap files per split (for smoke runs).")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild volumes that are already cached.")
    args = parser.parse_args()

    build_cache(
        train_dirs=args.train_dirs,
        test_dirs=args.test_dirs,
        cache_dir=args.cache_dir,
        val_fraction=args.val_fraction,
        vol_shape=args.vol_shape,
        max_files=args.max_files,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
