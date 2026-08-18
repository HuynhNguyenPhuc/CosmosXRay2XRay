from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.cache_builder import (
    build_split,
    cache_one_volume,
    find_nifti_files,
    is_validation_case,
    patient_id_from_path,
)


def _write_synthetic_ct(path: Path, size: int = 32) -> Path:
    """Write a small synthetic CT NIfTI with a realistic HU range."""
    nib = pytest.importorskip("nibabel")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Air background (-1000 HU) with a dense cube (+800 HU) so preprocessing has real
    # structure to window and rescale rather than a constant field.
    volume = np.full((size, size, size), -1000.0, dtype=np.float32)
    volume[8:24, 8:24, 8:24] = 800.0
    nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), str(path))
    return path


def test_patient_id_strips_nifti_suffixes():
    assert patient_id_from_path(Path("/data/mela_0001.nii.gz")) == "mela_0001"
    assert patient_id_from_path(Path("/data/LUNG1-001_0000.nii")) == "LUNG1-001_0000"
    # ".nii.gz" must be stripped whole — Path.stem alone would leave a trailing ".nii".
    assert not patient_id_from_path(Path("/data/x.nii.gz")).endswith(".nii")


def test_validation_split_is_deterministic_and_proportional():
    ids = [f"case_{i:04d}" for i in range(2000)]

    # Same input must always land in the same split, so a rebuilt cache cannot leak a
    # previous validation case into training.
    assert [is_validation_case(i, 0.05) for i in ids] == [is_validation_case(i, 0.05) for i in ids]

    fraction = sum(is_validation_case(i, 0.05) for i in ids) / len(ids)
    assert fraction == pytest.approx(0.05, abs=0.02)

    assert not any(is_validation_case(i, 0.0) for i in ids)


def test_find_nifti_files_recurses_and_ignores_missing(tmp_path):
    _write_synthetic_ct(tmp_path / "a" / "one.nii.gz")
    _write_synthetic_ct(tmp_path / "b" / "nested" / "two.nii.gz")
    (tmp_path / "a" / "notes.txt").write_text("ignore me")

    found = find_nifti_files([tmp_path, tmp_path / "does_not_exist"])

    assert [p.name for p in found] == ["one.nii.gz", "two.nii.gz"]


def test_cache_one_volume_writes_loadable_npy(tmp_path):
    """Regression: np.save() appends '.npy' to a path not already ending in it, which sent the
    atomic-write temp file to '<id>.npy.tmp.npy' and made every rename fail."""
    ct_path = _write_synthetic_ct(tmp_path / "src" / "case_0001.nii.gz")
    out_path = tmp_path / "cache" / "train" / "ct" / "case_0001.npy"

    assert cache_one_volume(ct_path, out_path, vol_shape=32) is True
    assert out_path.exists()
    assert not list(out_path.parent.glob("*.tmp*")), "temp file left behind"

    array = np.load(out_path, mmap_mode="r")
    assert array.shape == (1, 32, 32, 32)
    assert array.dtype == np.float32
    assert float(array.min()) >= 0.0 and float(array.max()) <= 1.0
    assert np.isfinite(array).all()


def test_cache_one_volume_reports_failure_without_raising(tmp_path):
    """A single unreadable scan must not abort a multi-hundred-volume build."""
    bad = tmp_path / "broken.nii.gz"
    bad.write_bytes(b"not a nifti")

    assert cache_one_volume(bad, tmp_path / "out" / "broken.npy", vol_shape=32) is False


def test_build_split_skips_existing_unless_overwrite(tmp_path):
    ct_files = [_write_synthetic_ct(tmp_path / "src" / f"case_{i}.nii.gz") for i in range(2)]
    cache_dir = tmp_path / "cache"

    assert build_split(ct_files, cache_dir, "train", vol_shape=32) == (2, 0, 0)
    # Re-running must be a cheap no-op, so an interrupted build can resume.
    assert build_split(ct_files, cache_dir, "train", vol_shape=32) == (0, 2, 0)
    assert build_split(ct_files, cache_dir, "train", vol_shape=32, overwrite=True) == (2, 0, 0)
