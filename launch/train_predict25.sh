#!/bin/bash
# ============================================================
# CosmosXRay2XRay Training Script
#
# Trains Cosmos-Predict 2.5 for CT volume → novel-view X-ray synthesis
# Supports single-GPU and multi-GPU (DDP) training
#
# Usage:
#   ./launch/train.sh --single --steps 10000       # Quick test
#   ./launch/train.sh --ddp --steps 100000         # Full training (4 GPUs)
#   ./launch/train.sh --data ./data1 ./data2       # Multiple datasets
#   ./launch/train.sh --resume path/to/last.ckpt   # Resume from checkpoint
# ============================================================

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_header() { echo -e "\n${BLUE}▶ $1${NC}"; }
log_info() { echo -e "  ${CYAN}ℹ${NC} $1"; }
log_success() { echo -e "  ${GREEN}✓${NC} $1"; }
log_error() { echo -e "  ${RED}✗${NC} $1"; }

# Add cosmos-predict2.5 to path
export PYTHONPATH="cosmos-predict2.5:cosmos-predict2.5/packages/cosmos-oss:${PYTHONPATH:-}"

# Support for multiple dataset directories
declare -a DATASET_DIRS=()

# NCCL Configuration (prevent hangs in distributed training)
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800        # 30 minutes
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900  # 15 minutes
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1        # Use TCP (for cloud VMs)

# CUDA Configuration
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
# Keep all 4 GPUs visible: GPUs 0-2 are used by DDP workers, GPU 3 is dedicated to the text encoder
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Disable tokenizers parallelism after fork
export TOKENIZERS_PARALLELISM=false

# ============================================================
# Defaults
# ============================================================
DATASET_DIRS=("${DATASET_PATH:-./data}")
CACHE_DIR="${CACHE_DIR:-./cache}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EXPERIMENT_NAME="cosmos-xray2xray-$(date +%Y%m%d-%H%M%S)"
# Use 3 DDP workers (GPUs 0-2); GPU 3 is reserved for the text encoder to avoid OOM
NUM_GPUS=3
BATCH_SIZE=1
ACCUMULATE=4
MAX_STEPS=100000
LEARNING_RATE="4.3e-5"
WARMUP_STEPS=5000
NUM_WORKERS=12
PREFETCH_FACTOR=2
ENABLE_EMA=true
EMA_RATE=0.05
PRECISION="bf16-mixed"
RESUME_CHECKPOINT=""

# ============================================================
# Parse Arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --single)
            NUM_GPUS=1
            log_info "Single GPU mode"
            shift
            ;;
        --ddp)
            NUM_GPUS=3
            log_info "DDP mode (3 GPUs for DiT, GPU 3 reserved for text encoder)"
            shift
            ;;
        --gpus)
            NUM_GPUS=$2
            shift 2
            ;;
        --batch)
            BATCH_SIZE=$2
            shift 2
            ;;
        --accumulate)
            ACCUMULATE=$2
            shift 2
            ;;
        --steps)
            MAX_STEPS=$2
            shift 2
            ;;
        --data)
            # Support multiple paths: --data dir1 dir2 dir3
            DATASET_DIRS=()
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATASET_DIRS+=("$1")
                shift
            done
            ;;
        --cache)
            CACHE_DIR=$2
            shift 2
            ;;
        --output)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --name)
            EXPERIMENT_NAME=$2
            shift 2
            ;;
        --lr)
            LEARNING_RATE=$2
            shift 2
            ;;
        --warmup)
            WARMUP_STEPS=$2
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS=$2
            shift 2
            ;;
        --prefetch-factor)
            PREFETCH_FACTOR=$2
            shift 2
            ;;
        --precision)
            PRECISION=$2
            shift 2
            ;;
        --no-ema)
            ENABLE_EMA=false
            shift
            ;;
        --ema-rate)
            EMA_RATE=$2
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT=$2
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "GPU Configuration:"
            echo "  --single              Train on single GPU"
            echo "  --ddp                 Train on 4 GPUs (DDP, default)"
            echo "  --gpus N              Train on N GPUs (DDP)"
            echo ""
            echo "Training Configuration:"
            echo "  --steps N             Max training steps (default: 100000)"
            echo "  --batch N             Batch size per GPU (default: 1)"
            echo "  --accumulate N        Gradient accumulation steps (default: 4)"
            echo "  --lr LR               Learning rate (default: 4.3e-5)"
            echo "  --warmup N            Warmup steps (default: 5000)"
            echo "  --num-workers N       Data loading workers per GPU (default: 12 → 48 total)"
            echo "  --prefetch-factor N   Prefetch buffer size (default: 2)"
            echo "  --precision PREC      Precision: bf16-mixed, fp32 (default: bf16-mixed)"
            echo "  --no-ema              Disable EMA"
            echo "  --ema-rate RATE       EMA update rate (default: 0.05)"
            echo ""
            echo "Data Configuration:"
            echo "  --data PATH [PATH...]  One or more dataset directories (default: ./data)"
            echo "  --cache PATH          Cache directory (default: ./cache)"
            echo "  --output PATH         Output directory (default: outputs)"
            echo "  --name NAME           Experiment name"
            echo ""
            echo "Resume Training:"
            echo "  --resume CKPT         Resume from checkpoint"
            echo ""
            echo "Examples:"
            echo "  Quick test (1 GPU, 10K steps):"
            echo "    ./launch/train.sh --single --steps 10000"
            echo ""
            echo "  Full training (4 GPUs, 100K steps):"
            echo "    ./launch/train.sh --ddp --steps 100000"
            echo ""
            echo "  With custom dataset (single or multiple):"
            echo "    ./launch/train.sh --data ./data1"
            echo "    ./launch/train.sh --data ./data1 ./data2 ./data3"
            echo ""
            echo "  8 GPUs with custom batch:"
            echo "    ./launch/train.sh --gpus 8 --batch 2 --accumulate 2"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# Validation
# ============================================================
log_header "Pre-flight Checks"

if [ ${#DATASET_DIRS[@]} -eq 0 ]; then
    log_info "No dataset directories specified (will use cache from --cache-dir)"
fi

for dir in "${DATASET_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        log_error "Dataset directory not found: $dir"
        exit 1
    fi
    log_success "Dataset found: $dir"
done

if [ ! -d "predict2_5" ]; then
    log_error "Trainer not found at predict2_5/"
    exit 1
fi
log_success "Trainer found: predict2_5/"

# ============================================================
# Calculate Effective Batch Size
# ============================================================
EFFECTIVE_BATCH=$((BATCH_SIZE * ACCUMULATE * NUM_GPUS))

# ============================================================
# Create Output Directory
# ============================================================
mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME}"

# ============================================================
# Launch Training
# ============================================================
log_header "Launching Training"
log_info "Experiment: $EXPERIMENT_NAME"
log_info "GPUs: $NUM_GPUS | Steps: $MAX_STEPS | Batch: $BATCH_SIZE"
echo ""

# Build torchrun command for multi-GPU
if [ $NUM_GPUS -gt 1 ]; then
    CMD="torchrun --nproc_per_node=$NUM_GPUS -m predict2_5.trainer"
else
    CMD="python -m predict2_5.trainer"
fi

CMD="$CMD --cache-dir '$CACHE_DIR'"
CMD="$CMD --output-dir '${OUTPUT_DIR}/${EXPERIMENT_NAME}'"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --num-gpus $NUM_GPUS"
CMD="$CMD --max-iters $MAX_STEPS"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --warmup $WARMUP_STEPS"
CMD="$CMD --strategy $([[ $NUM_GPUS -gt 1 ]] && echo 'ddp' || echo 'ddp')"
CMD="$CMD --precision $PRECISION"
CMD="$CMD --text-encoder-device cuda:3"
CMD="$CMD --wandb-project '$EXPERIMENT_NAME'"

if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume-ckpt '$RESUME_CHECKPOINT'"
fi

# Execute
eval "$CMD"

# ============================================================
# Post-Training
# ============================================================
log_header "Training Complete"

echo ""
echo -e "${GREEN}Results saved to:${NC}"
echo "  ${OUTPUT_DIR}/${EXPERIMENT_NAME}/"
echo ""
echo -e "${GREEN}Monitor with Weights & Biases:${NC}"
echo "  https://wandb.ai/[your-username]"
echo ""
echo -e "${GREEN}View checkpoints:${NC}"
echo "  ls -lah ${OUTPUT_DIR}/${EXPERIMENT_NAME}/checkpoints/"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Resume training: ./launch/train.sh --resume checkpoints/last.ckpt"
echo "  2. Monitor training: tail -f ${OUTPUT_DIR}/${EXPERIMENT_NAME}/logs/*.log"
echo "  3. Generate novel-view X-rays for publication"
echo ""
echo -e "${GREEN}============================================================${NC}"
