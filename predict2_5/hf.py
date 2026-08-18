"""HuggingFace download helpers for CosmosXRay2XRay."""

import contextlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE

from shared.utils import get_logger


# -- Logger -- #
log = get_logger(__name__)


# ============================================================
# Text Encoder Snapshot (nvidia/Cosmos-Reason1-7B)
# ============================================================

def _resolve_hf_token() -> str:
    """Resolve HuggingFace authentication token from environment/config."""
    # Check environment variables
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    )
    if token:
        return token

    # Check config files
    for token_path in [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]:
        if token_path.exists():
            token = token_path.read_text().strip()
            if token:
                return token

    # Try library API
    try:
        from huggingface_hub.utils import get_token
        token = get_token() or ""
        if token:
            return token
    except Exception:
        pass

    return ""


def _get_lock_manager():
    """Return a filelock context manager, or a null context if unavailable."""
    try:
        from filelock import FileLock
        return FileLock
    except ImportError:
        return lambda _p: contextlib.nullcontext()


def _curl_download(url: str, dest: Path, token: str, attempt: int) -> None:
    """
    Execute a single curl download attempt.
    
    Args:
        url:     Full download URL
        dest:    Destination file path
        token:   HuggingFace auth token (may be empty)
        attempt: Attempt number (1-5) for logging
    """
    cmd = [
        "curl",
        "-fL",              # fail on HTTP error, follow redirects
        "-C", "-",          # resume partial downloads
        "--speed-limit", "102400",   # min speed: 100 KB/s
        "--speed-time", "30",        # for 30 consecutive seconds
        "-o", str(dest),
        url,
    ]
    
    if token:
        cmd.extend(["-H", f"Authorization: Bearer {token}"])

    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.warning(
            f"[HF] curl attempt {attempt}/5 failed "
            f"(exit code={result.returncode}); retrying …"
        )


def hf_download(
    repo_id: str,
    filename: str,
    revision: str = "main",
    override_dest: str = None,
) -> str:
    """
    Download a large HuggingFace file reliably via curl.
    
    hf_hub_download (requests-based) stalls silently at CDN chunk boundaries.
    curl with --speed-limit/--speed-time detects and recovers from stalls
    within 30 seconds, and -C - resumes without re-downloading fetched bytes.

    Args:
        repo_id:       HuggingFace repo, e.g. "nvidia/Cosmos-Predict2.5-2B"
        filename:      Path within repo, e.g. "tokenizer.pth"
        revision:      Git revision/commit hash (default: "main")
        override_dest: Custom destination path; if None, use default HF cache

    Returns:
        Absolute path to the downloaded file
    """
    # Fast path: already in local HF cache (only for default revision/dest)
    if revision == "main" and override_dest is None:
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_files_only=True,
            )
        except Exception:
            pass

    # Determine destination path
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    if override_dest is not None:
        dest = Path(override_dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = (
            Path(HF_HUB_CACHE) / "manual" / repo_id.replace("/", "--")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / Path(filename).name

    # Skip if already downloaded (>1 MB threshold)
    MIN_SIZE_TO_REUSE = 1_000_000
    if dest.exists() and dest.stat().st_size > MIN_SIZE_TO_REUSE:
        log.info(f"[HF] Reusing previously downloaded: {dest}")
        return str(dest)

    # Resolve authentication token from multiple sources
    token = _resolve_hf_token()

    # Lock: serialize concurrent DDP processes
    lock_manager = _get_lock_manager()
    lock_path = dest.parent / f".lock_{dest.name}"
    
    with lock_manager(str(lock_path)):
        # Re-check after acquiring lock
        if dest.exists() and dest.stat().st_size > MIN_SIZE_TO_REUSE:
            log.info(f"[HF] Reusing file (downloaded by another process): {dest}")
            return str(dest)

        log.info(f"[HF] Starting curl download: {repo_id}/{filename}")
        for attempt in range(1, 6):
            _curl_download(url, dest, token, attempt)
            if dest.exists() and dest.stat().st_size > MIN_SIZE_TO_REUSE:
                log.info(
                    f"[HF] Download complete: {dest} "
                    f"({dest.stat().st_size / 1e9:.2f} GB)"
                )
                return str(dest)

    raise RuntimeError(
        f"[HF] Failed to download {repo_id}/{filename!r} "
        f"after 5 curl attempts"
    )


# ============================================================
# Text Encoder Snapshot (nvidia/Cosmos-Reason1-7B)
# ============================================================

# Repository and revision identifiers
_TE_REPO_ID = "nvidia/Cosmos-Reason1-7B"
_TE_REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"

# Small files (<100 MB) that hf_hub_download handles reliably
_TE_SMALL_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
    "chat_template.json",
]

# Thresholds for shard validation
_TE_SHARD_MIN_SIZE = 1_500_000_000  # 1.5 GB


def text_encoder_snapshot() -> str:
    """
    Download and cache the text encoder snapshot.
    
    Strategy:
    1. Check standard HF snapshot cache (fast if complete)
    2. Check our manual cache sentinel (fast if already done)
    3. Within a coarse lock (for DDP): download missing shards via curl,
       hard-link them into the HF snapshot dir for checkpoint_db compatibility
    
    Returns:
        Absolute path to the cached snapshot directory
    """
    snap_dir = _get_manual_snapshot_dir()
    
    # Fast path 1: already in standard HF snapshot cache
    cached = _check_hf_snapshot_cache()
    if cached:
        log.info(f"[HF] Text encoder already in HF snapshot cache: {cached}")
        return cached
    
    # Fast path 2: manual cache sentinel exists
    done_file = snap_dir / ".download_complete"
    if done_file.exists():
        log.info(f"[HF] Text encoder manual snapshot already complete: {snap_dir}")
        return str(snap_dir)
    
    # Slow path: download with DDP concurrency control
    return _download_text_encoder_with_lock(snap_dir, done_file)


def _get_manual_snapshot_dir() -> Path:
    """Get the manual snapshot cache directory."""
    cache_dir = (
        Path(HF_HUB_CACHE)
        / "manual"
        / _TE_REPO_ID.replace("/", "--")
        / _TE_REVISION[:8]
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _check_hf_snapshot_cache() -> str:
    """Check if text encoder is complete in standard HF snapshot cache.
    
    Returns:
        Path to snapshot if complete, None otherwise
    """
    try:
        from huggingface_hub import snapshot_download
        
        cached = snapshot_download(
            repo_id=_TE_REPO_ID,
            revision=_TE_REVISION,
            local_files_only=True,
        )
        
        # Verify all shards are complete (≥1.5 GB)
        cached_path = Path(cached)
        index_p = cached_path / "model.safetensors.index.json"
        
        if not index_p.exists():
            return None
        
        with open(index_p) as f:
            index = json.load(f)
        
        shards = sorted(set(index["weight_map"].values()))
        all_complete = all(
            (cached_path / s).exists()
            and (cached_path / s).stat().st_size >= _TE_SHARD_MIN_SIZE
            for s in shards
        )
        
        return cached if all_complete else None
    
    except Exception:
        return None


def _download_text_encoder_with_lock(snap_dir: Path, done_file: Path) -> str:
    """Download text encoder with DDP-safe locking.
    
    Args:
        snap_dir: Manual snapshot cache directory
        done_file: Sentinel file marking completion
    
    Returns:
        Path to the downloaded snapshot
    """
    lock_manager = _get_lock_manager()
    lock_path = snap_dir / ".snapshot.lock"
    
    with lock_manager(str(lock_path)):
        # Double-check after acquiring lock
        if done_file.exists():
            log.info(
                f"[HF] Text encoder manual snapshot already complete "
                f"(locked re-check): {snap_dir}"
            )
            return str(snap_dir)
        
        # Download small files (config, tokenizer)
        _download_small_files(snap_dir)
        
        # Parse shard list from index
        index_path = snap_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        
        # Download large shards via curl
        _download_large_shards(snap_dir, shard_files)
        
        # Hard-link into HF snapshot dir for checkpoint_db
        _hardlink_to_hf_snapshot(snap_dir, shard_files)
        
        # Write completion sentinel
        done_file.write_text("done\n")
        log.info(f"[HF] Text encoder snapshot complete: {snap_dir}")
        
        return str(snap_dir)


def _download_small_files(snap_dir: Path) -> None:
    """Download config/tokenizer files via hf_hub_download."""
    for fname in _TE_SMALL_FILES:
        dest_f = snap_dir / fname
        if not dest_f.exists():
            log.info(f"[HF] Fetching small file: {fname}")
            src = hf_hub_download(
                repo_id=_TE_REPO_ID,
                filename=fname,
                revision=_TE_REVISION,
            )
            shutil.copy2(src, dest_f)


def _download_large_shards(snap_dir: Path, shard_files: list) -> None:
    """Download .safetensors shards via curl with resume + skip-if-complete.
    
    Args:
        snap_dir: Directory to download shards into
        shard_files: List of shard filenames to download
    """
    for shard in shard_files:
        dest_f = snap_dir / shard
        
        # Skip if already complete: >= 1.5 GB threshold
        # (generic 1 MB threshold is too low for 4–5 GB files)
        if dest_f.exists() and dest_f.stat().st_size >= _TE_SHARD_MIN_SIZE:
            size_gb = dest_f.stat().st_size / 1e9
            log.info(
                f"[HF] Shard already complete, skipping: {shard} "
                f"({size_gb:.2f} GB)"
            )
            continue
        
        hf_download(
            _TE_REPO_ID,
            shard,
            revision=_TE_REVISION,
            override_dest=str(dest_f),
        )


def _hardlink_to_hf_snapshot(snap_dir: Path, shard_files: list) -> None:
    """Hard-link completed shards into standard HF snapshot dir.
    
    This allows checkpoint_db.download() (uvx hf download) to find them
    complete and skip the expensive re-download.
    
    Args:
        snap_dir: Source directory with completed shards
        shard_files: List of shard filenames
    """
    try:
        hf_snap = (
            Path(HF_HUB_CACHE)
            / f"models--{_TE_REPO_ID.replace('/', '--')}"
            / "snapshots"
            / _TE_REVISION
        )
        
        if not hf_snap.exists():
            log.info("[HF] HF snapshot dir does not exist; checkpoint_db may re-download")
            return
        
        for shard in shard_files:
            hf_shard = hf_snap / shard
            src = snap_dir / shard
            
            # Skip if already complete
            if hf_shard.exists() and hf_shard.stat().st_size >= _TE_SHARD_MIN_SIZE:
                continue
            
            try:
                if hf_shard.exists():
                    hf_shard.unlink()
                hf_shard.hardlink_to(src)
                log.info(f"[HF] Hard-linked {shard} into HF snapshot dir")
            except OSError:
                # Fallback to copy if hardlink fails (e.g., different filesystems)
                shutil.copy2(str(src), str(hf_shard))
                log.info(f"[HF] Copied {shard} into HF snapshot dir")
    
    except Exception as e:
        log.warning(
            f"[HF] Could not populate HF snapshot dir ({e}); "
            f"checkpoint_db may re-download"
        )
