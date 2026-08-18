# Master Execution & Scaling Protocol — CosmosXRay2XRay

This document provides the operational guidelines for multi-GPU training, precision configuration, distributed data parallel (DDP) scaling, and checkpoint management in **CosmosXRay2XRay**.

---

## 1. Multi-GPU Distributed Scaling (DDP)

**CosmosXRay2XRay** leverages PyTorch Lightning with Distributed Data Parallel (`ddp`) strategy for scaling training across multiple GPUs.

### Single Node Multi-GPU Launch Syntax
Use `torchrun` to instantiate process groups:

```bash
# Launch Predict 2.5 on 4 GPUs
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    -m predict2_5.trainer \
    --cache-dir ./cache \
    --batch-size 1 \
    --num-gpus 4 \
    --max-iters 100000 \
    --warmup 5000 \
    --strategy ddp \
    --precision bf16-mixed

# Launch Transfer 2.5 on 8 GPUs
torchrun \
    --nproc_per_node=8 \
    --master_port=29501 \
    -m transfer2_5.trainer \
    --cache-dir ./cache \
    --batch-size 1 \
    --num-gpus 8 \
    --max-iters 100000 \
    --warmup 2000 \
    --control-type edge_map \
    --freeze-base \
    --strategy ddp \
    --precision bf16-mixed
```

---

## 2. Mixed Precision Configuration

To optimize VRAM consumption and throughput on NVIDIA Ampere (A100) and Hopper (H100) GPUs:
- **Precision Flag**: `--precision bf16-mixed` (Bfloat16 mixed precision).
- **Benefits**: Cuts VRAM requirements by ~40% while preserving dynamic range for diffusion flow-matching loss computation.

---

## 3. Exponential Moving Average (EMA) & Synchronization

`shared/callbacks.py` implements an in-place Exponential Moving Average (`EMACallback`) with rate $\alpha = 0.10$ (`ema_rate`):
- **CPU Offloading**: Maintains EMA weights on host CPU RAM when GPU memory is constrained.
- **DDP Sync**: Synchronizes EMA buffers across rank 0 before checkpoint save calls.

---

## 4. Checkpoint Management & DCP Conversion

PyTorch Lightning saves Distributed Checkpoint (DCP) directories during DDP runs. To convert DCP checkpoints into standalone `.pt` state dicts for deployment or inference:

```python
import torch
from torch.distributed.checkpoint import load

state_dict = {}
load(state_dict, checkpoint_id="outputs/predict2.5_multiview/checkpoints/last.ckpt/")
torch.save(state_dict, "outputs/predict2.5_multiview/model_ema.pt")
```
