#!/usr/bin/env python3
"""
Download directory from Google Cloud Storage (GCS) to local path.

Usage:
    python scripts/download_from_gcs.py --bucket graphicsminer-data-science-bucket --gcs-folder data/ChestCT/ --local-dir data/ --num-workers 8

Environment:
    Requires google-cloud-storage and valid GCS authentication.
    Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
"""

import os
import time
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError

# === DEFAULTS ===
DEFAULT_NUM_WORKERS = 8
DEFAULT_CHUNK_SIZE = 32 * 1024 * 1024  # 32 MB
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1  # seconds

log = logging.getLogger(__name__)


class DownloadStats:
    """Track download statistics."""
    
    def __init__(self):
        self.total_files = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.total_bytes = 0
        self.start_time = None
        self.failed_files = []
    
    def elapsed(self):
        """Elapsed time in seconds."""
        if self.start_time:
            return time.time() - self.start_time
        return 0
    
    def bandwidth_mbps(self):
        """Average bandwidth in MB/s."""
        elapsed = self.elapsed()
        if elapsed > 0:
            return (self.total_bytes / (1024**2)) / elapsed
        return 0
    
    def print_summary(self):
        """Print download summary."""
        print(f"\n{'='*60}")
        print(f"Download Summary")
        print(f"{'='*60}")
        print(f"Total files processed: {self.total_files}")
        print(f"Successful: {self.successful}")
        print(f"Failed: {self.failed}")
        print(f"Skipped: {self.skipped}")
        print(f"Total data: {self.total_bytes / (1024**3):.2f} GB")
        print(f"Time elapsed: {self.elapsed():.1f} seconds")
        print(f"Average bandwidth: {self.bandwidth_mbps():.2f} MB/s")
        
        if self.failed_files:
            print(f"\nFailed files:")
            for fname, error in self.failed_files:
                print(f"  - {fname}: {error}")
        print(f"{'='*60}\n")


def download_single_file(
    bucket,
    blob_name,
    local_path,
    stats,
    chunk_size=DEFAULT_CHUNK_SIZE,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_delay=DEFAULT_RETRY_DELAY,
):
    """Download a single file from GCS with retry logic."""
    try:
        # Create local directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download with retry logic
        for attempt in range(max_retries):
            try:
                blob = bucket.blob(blob_name)
                blob.chunk_size = chunk_size
                
                # Check if file exists
                if not blob.exists():
                    stats.skipped += 1
                    return True
                
                blob.download_to_filename(local_path)
                
                # Update stats
                file_size = os.path.getsize(local_path)
                stats.total_bytes += file_size
                stats.successful += 1
                return True
                
            except GoogleAPIError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    raise
        
    except Exception as e:
        stats.failed += 1
        stats.failed_files.append((blob_name, str(e)))
        return False


def list_blobs(bucket, prefix):
    """List all blobs in bucket with given prefix."""
    blobs = []
    for blob in bucket.list_blobs(prefix=prefix):
        # Skip directories (blobs ending with /)
        if not blob.name.endswith("/"):
            blobs.append(blob)
    return blobs


def download_from_gcs_parallel(
    bucket_name,
    gcs_folder,
    local_dir,
    num_workers=DEFAULT_NUM_WORKERS,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    """Download entire folder from GCS using parallel downloads."""
    print(f"\n{'='*60}")
    print(f"Google Cloud Storage Download (Parallel)")
    print(f"{'='*60}")
    print(f"Source bucket: {bucket_name}")
    print(f"Source folder: {gcs_folder}")
    print(f"Destination directory: {local_dir}")
    print(f"Parallel workers: {num_workers}")
    print(f"Chunk size: {chunk_size / (1024**2):.0f} MB")
    print(f"{'='*60}\n")
    
    # Initialize GCS client
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        print(f"Error initializing GCS client. Is your authentication set up correctly? Error: {e}")
        return False

    # List blobs to download
    blobs = list_blobs(bucket, prefix=gcs_folder)
    if not blobs:
        print(f"No files found in '{gcs_folder}' to download.")
        return True

    stats = DownloadStats()
    stats.total_files = len(blobs)
    stats.start_time = time.time()
    
    print(f"Found {len(blobs)} files to download\n")

    # Download with thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        
        # Submit all download tasks
        for blob in blobs:
            # Compute local path by removing GCS folder prefix
            relative_path = blob.name[len(gcs_folder):].lstrip("/")
            local_path = os.path.join(local_dir, relative_path)
            
            future = executor.submit(
                download_single_file,
                bucket,
                blob.name,
                local_path,
                stats,
                chunk_size=chunk_size,
            )
            futures[future] = (blob.name, local_path)
        
        # Process completed downloads with progress bar
        with tqdm(as_completed(futures), total=len(futures), desc="Downloading from GCS") as pbar:
            for future in pbar:
                try:
                    future.result()
                except Exception as e:
                    blob_name, local_path = futures[future]
                    stats.failed += 1
                    stats.failed_files.append((blob_name, str(e)))
                
                pbar.update(1)
    
    stats.print_summary()
    return stats.failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download directory from Google Cloud Storage to local path."
    )
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="GCS bucket name (e.g., edusi-data-science-bucket)",
    )
    parser.add_argument(
        "--gcs-folder",
        type=str,
        required=True,
        help="Source folder in GCS (e.g., data/ChestXR/)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        required=True,
        help="Path to local directory for downloads",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel download threads (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for downloads in bytes (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retry attempts for failed downloads (default: {DEFAULT_MAX_RETRIES})",
    )
    
    args = parser.parse_args()
    
    # Create local directory if needed
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Run download
    success = download_from_gcs_parallel(
        bucket_name=args.bucket,
        gcs_folder=args.gcs_folder,
        local_dir=args.local_dir,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
    
    exit(0 if success else 1)
