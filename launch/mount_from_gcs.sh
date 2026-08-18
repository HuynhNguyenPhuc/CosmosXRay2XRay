#!/bin/bash
# ============================================================
# Mount GCS Folder (Fuse)
#
# Prerequisites:
#   - Install gcsfuse: https://github.com/GoogleCloudStorage/gcsfuse/blob/master/docs/installing.md
#   - Setup GCS credentials
#
# Usage:
#   GCS_BUCKET=my-bucket ./launch/mount_from_gcs.sh \
#     --source data/raw \
#     --target /mnt/gcs-data
#
#   Or pass bucket as argument:
#   ./launch/mount_from_gcs.sh \
#     --bucket my-bucket \
#     --source data/raw \
#     --target /mnt/gcs-data
# ============================================================

set -e

# Default bucket (can be overridden)
GCS_BUCKET="${GCS_BUCKET:-graphicsminer-data-science-bucket}"
SOURCE_FOLDER=""
TARGET_PATH=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --bucket) GCS_BUCKET="$2"; shift 2 ;;
    --source) SOURCE_FOLDER="$2"; shift 2 ;;
    --target) TARGET_PATH="$2"; shift 2 ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --bucket BUCKET       GCS bucket name"
      echo "  --source FOLDER       Source folder in GCS (e.g., data/raw)"
      echo "  --target PATH         Local mount path (e.g., /mnt/gcs-data)"
      echo "  --help                Show this help message"
      echo ""
      echo "Examples:"
      echo "  ./mount_from_gcs.sh --source data/raw --target /mnt/gcs-data"
      echo "  GCS_BUCKET=my-bucket ./mount_from_gcs.sh --source datasets --target /mnt/data"
      exit 0
      ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Validate inputs
if [[ -z "$SOURCE_FOLDER" || -z "$TARGET_PATH" ]]; then
  echo "Error: Missing required arguments"
  echo "Usage: $0 --source <GCS_FOLDER> --target <LOCAL_MOUNT_PATH>"
  echo "Run with --help for examples"
  exit 1
fi

# Create mount point
mkdir -p "$TARGET_PATH"

# Mount GCS folder
echo "Mounting $SOURCE_FOLDER from bucket $GCS_BUCKET to $TARGET_PATH..."
gcsfuse \
  --only-dir "$SOURCE_FOLDER" \
  --implicit-dirs \
  --rename-dir-limit=100 \
  --client-protocol=http1 \
  --max-conns-per-host=100 \
  "$GCS_BUCKET" "$TARGET_PATH"

echo "✓ Successfully mounted!"
echo "  Bucket: $GCS_BUCKET"
echo "  Source: $SOURCE_FOLDER"
echo "  Mount point: $TARGET_PATH"
echo ""
echo "To unmount later, run:"
echo "  fusermount -u $TARGET_PATH"