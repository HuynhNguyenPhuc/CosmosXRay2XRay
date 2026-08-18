#!/usr/bin/env python3
"""
Upload local directory to Google Cloud Storage (GCS).

Usage:
    python scripts/upload_to_gcs.py \\
        --source-dir data/ \\
        --bucket graphicsminer-data-science-bucket \\
        --gcs-folder data/ChestCT/ \\
        --num-workers 8 \\
        --compress

Environment:
    Requires google-cloud-storage and valid GCS authentication.
    Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
"""

import os
import time
import gzip
import shutil
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


class UploadStats:
    """Track upload statistics."""
    
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
        """Print upload summary."""
        print(f"\n{'='*60}")
        print(f"Upload Summary")
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


def compress_file(local_path, temp_path):
    """Compress a file using gzip."""
    try:
        with open(local_path, 'rb') as f_in:
            with gzip.open(temp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return temp_path
    except Exception as e:
        log.error(f"Compression failed for {local_path}: {e}")
        return local_path


def upload_single_file(
    bucket,
    local_path,
    gcs_path,
    stats,
    compress=False,
    chunk_size=DEFAULT_CHUNK_SIZE,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_delay=DEFAULT_RETRY_DELAY,
):
    """Upload a single file to GCS with retry logic."""
    temp_compressed = None
    upload_path = local_path
    
    try:
        # Compress if enabled
        if compress:
            temp_compressed = f"{local_path}.gz"
            upload_path = compress_file(local_path, temp_compressed)
            gcs_path = f"{gcs_path}.gz"
        
        file_size = os.path.getsize(upload_path)
        stats.total_bytes += file_size
        
        # Upload with retry logic
        for attempt in range(max_retries):
            try:
                blob = bucket.blob(gcs_path)
                blob.chunk_size = chunk_size
                blob.upload_from_filename(upload_path)
                
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
        stats.failed_files.append((local_path, str(e)))
        return False
    
    finally:
        # Clean up temporary compressed file
        if temp_compressed and os.path.exists(temp_compressed):
            try:
                os.remove(temp_compressed)
            except:
                pass


def collect_files(local_dir):
    """Recursively collect all files from directory."""
    filepaths = []
    for root, _, files in os.walk(local_dir):
        for filename in files:
            filepaths.append(os.path.join(root, filename))
    return filepaths


def upload_to_gcs_parallel(
    local_dir,
    bucket_name,
    gcs_folder,
    num_workers=DEFAULT_NUM_WORKERS,
    compress=False,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    """Upload entire directory to GCS using parallel uploads."""
    print(f"\n{'='*60}")
    print(f"Google Cloud Storage Upload (Parallel)")
    print(f"{'='*60}")
    print(f"Source directory: {local_dir}")
    print(f"Destination bucket: {bucket_name}")
    print(f"Destination folder: {gcs_folder}")
    print(f"Parallel workers: {num_workers}")
    print(f"Compression: {'enabled' if compress else 'disabled'}")
    print(f"Chunk size: {chunk_size / (1024**2):.0f} MB")
    print(f"{'='*60}\n")
    
    # Initialize GCS client
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        print(f"Error initializing GCS client. Is your authentication set up correctly? Error: {e}")
        return False

    # Collect files
    filepaths = collect_files(local_dir)
    if not filepaths:
        print(f"No files found in '{local_dir}' to upload.")
        return True

    stats = UploadStats()
    stats.total_files = len(filepaths)
    stats.start_time = time.time()
    
    print(f"Found {len(filepaths)} files to upload\n")

    # Upload with thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        
        # Submit all upload tasks
        for local_path in filepaths:
            relative_path = os.path.relpath(local_path, local_dir)
            gcs_path = os.path.join(gcs_folder, relative_path).replace("\\", "/")
            
            future = executor.submit(
                upload_single_file, 
                bucket, 
                local_path, 
                gcs_path, 
                stats,
                compress=compress,
                chunk_size=chunk_size,
            )
            futures[future] = (local_path, gcs_path)
        
        # Process completed uploads with progress bar
        with tqdm(as_completed(futures), total=len(futures), desc="Uploading to GCS") as pbar:
            for future in pbar:
                try:
                    future.result()
                except Exception as e:
                    local_path, gcs_path = futures[future]
                    stats.failed += 1
                    stats.failed_files.append((local_path, str(e)))
                
                pbar.update(1)
    
    stats.print_summary()
    return stats.failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload local directory to Google Cloud Storage."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Path to local directory to upload",
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
        help="Destination folder in GCS (e.g., data/ChestXR/)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel upload threads (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for multipart uploads in bytes (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress files with gzip before uploading",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retry attempts for failed uploads (default: {DEFAULT_MAX_RETRIES})",
    )
    
    args = parser.parse_args()
    
    # Validate source directory
    if not os.path.isdir(args.source_dir):
        print(f"Error: Source directory '{args.source_dir}' does not exist.")
        exit(1)
    
    # Run upload
    success = upload_to_gcs_parallel(
        local_dir=args.source_dir,
        bucket_name=args.bucket,
        gcs_folder=args.gcs_folder,
        num_workers=args.num_workers,
        compress=args.compress,
        chunk_size=args.chunk_size,
    )
    
    exit(0 if success else 1)
