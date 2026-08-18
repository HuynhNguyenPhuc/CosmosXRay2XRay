#!/bin/bash
# ============================================================
# CosmosXRay2XRay Transfer Learning Script
#
# Trains Cosmos-Transfer 2.5 for CT volume → novel-view X-ray synthesis
# with ControlNet-style fine-tuning (edge maps, depth maps, etc.)
# Supports single-GPU and multi-GPU (DDP) training
#
# Usage:
#   ./launch/train_transfer25.sh --single --steps 10000       # Quick test
#   ./launch/train_transfer25.sh --ddp --control edge_map     # 4 GPUs, edge control
#   ./launch/train_transfer25.sh --gpus 8 --steps 100000      # 8 GPUs
#   ./launch/train_transfer25.sh --resume path/to/last.ckpt   # Resume from checkpoint
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

# Add cosmos-transfer2.5 to path
export PYTHONPATH="cosmos-predict2.5:cosmos-predict2.5/packages/cosmos-oss:cosmos-transfer2.5:${PYTHONPATH:-}"

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
CACHE_DIR="${CACHE_DIR:-./cache}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EXPERIMENT_NAME="cosmos-xray2xray-transfer-$(date +%Y%m%d-%H%M%S)"
# Use 3 DDP workers (GPUs 0-2); GPU 3 is reserved for the text encoder to avoid OOM
NUM_GPUS=3
BATCH_SIZE=1
MAX_STEPS=100000
LEARNING_RATE="1e-5"
WARMUP_STEPS=1000
PRECISION="bf16-mixed"
CONTROL_TYPE="edge_map"
CONTROL_CONTEXT_SCALE=1.0
FREEZE_BASE=true
CONDITION_STRATEGY="spaced"
COPY_WEIGHT_STRATEGY="spaced_n"
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
        --steps)
            MAX_STEPS=$2
            shift 2
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
        --control)
            CONTROL_TYPE=$2
            shift 2
            ;;
        --control-scale)
            CONTROL_CONTEXT_SCALE=$2
            shift 2
            ;;
        --no-freeze)
            FREEZE_BASE=false
            shift
            ;;
        --freeze)
            FREEZE_BASE=true
            shift
            ;;
        --condition-strategy)
            CONDITION_STRATEGY=$2
            shift 2
            ;;
        --copy-weight-strategy)
            COPY_WEIGHT_STRATEGY=$2
            shift 2
            ;;
        --precision)
            PRECISION=$2
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
            echo "  --lr LR               Learning rate (default: 1e-5)"
            echo "  --warmup N            Warmup steps (default: 1000)"
            echo "  --precision PREC      Precision: bf16-mixed, fp32 (default: bf16-mixed)"
            echo ""
            echo "Control Configuration:"
            echo "  --control TYPE        Control signal: edge_map, depth_map, seg_mask (default: edge_map)"
            echo "  --control-scale SCALE Control context scale (default: 1.0)"
            echo "  --no-freeze           Train entire model (including base DiT)"
            echo "  --freeze              Freeze base DiT, train only control (default)"
            echo "  --condition-strategy S Condition distribution: spaced, ... (default: spaced)"
            echo "  --copy-weight-strategy S Weight init: spaced_n, ... (default: spaced_n)"
            echo ""
            echo "Data Configuration:"
            echo "  --cache PATH          Cache directory (default: ./cache)"
            echo "  --output PATH         Output directory (default: outputs)"
            echo "  --name NAME           Experiment name"
            echo ""
            echo "Resume Training:"
            echo "  --resume CKPT         Resume from checkpoint"
            echo ""
            echo "Examples:"
            echo "  Quick test (1 GPU, edge maps, 10K steps):"
            echo "    ./launch/train_transfer25.sh --single --steps 10000 --control edge_map"
            echo ""
            echo "  Full training (4 GPUs, depth maps, frozen base):"
            echo "    ./launch/train_transfer25.sh --ddp --control depth_map"
            echo ""
            echo "  Fine-tune entire model (8 GPUs, no freeze):"
            echo "    ./launch/train_transfer25.sh --gpus 8 --no-freeze"
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

if [ ! -d "transfer2_5" ]; then
    log_error "Trainer not found at transfer2_5/"
    exit 1
fi
log_success "Trainer found: transfer2_5/"

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
log_info "Control: $CONTROL_TYPE | Freeze Base: $FREEZE_BASE"
echo ""

# Build torchrun command for multi-GPU
if [ $NUM_GPUS -gt 1 ]; then
    CMD="torchrun --nproc_per_node=$NUM_GPUS -m transfer2_5.trainer"
else
    CMD="python -m transfer2_5.trainer"
fi

CMD="$CMD --cache-dir '$CACHE_DIR'"
CMD="$CMD --output-dir '${OUTPUT_DIR}/${EXPERIMENT_NAME}'"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --num-gpus $NUM_GPUS"
CMD="$CMD --max-iters $MAX_STEPS"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --warmup $WARMUP_STEPS"
CMD="$CMD --strategy ddp"
CMD="$CMD --precision $PRECISION"
CMD="$CMD --text-encoder-device cuda:3"
CMD="$CMD --wandb-project '$EXPERIMENT_NAME'"
CMD="$CMD --control-type $CONTROL_TYPE"
CMD="$CMD --control-context-scale $CONTROL_CONTEXT_SCALE"
CMD="$CMD --condition-strategy $CONDITION_STRATEGY"
CMD="$CMD --copy-weight-strategy $COPY_WEIGHT_STRATEGY"

if [ "$FREEZE_BASE" = "true" ]; then
    CMD="$CMD --freeze-base"
fi

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
echo "  1. Resume training: ./launch/train_transfer25.sh --resume checkpoints/last.ckpt"
echo "  2. Monitor training: tail -f ${OUTPUT_DIR}/${EXPERIMENT_NAME}/logs/*.log"
echo "  3. Generate novel-view X-rays for publication"
echo ""
echo -e "${GREEN}============================================================${NC}"
