# Execution Walkthrough — CosmosXRay2XRay

This walkthrough provides step-by-step instructions for running data preprocessing, training Predict 2.5, training Transfer 2.5 (ControlNet), and running inference/visualization in **CosmosXRay2XRay**.

---

## Step 1: Data Preparation & Caching

The pipeline requires 3D CT volumes (NIfTI format `.nii` / `.nii.gz`). The `CacheBuilder` converts CT volumes into cached `.npy` volumes and pre-rendered frontal X-rays.

```python
from shared.cache_builder import build_cache_from_directory

# Preprocess raw CT datasets into standard cache format
build_cache_from_directory(
    dataset_dirs=[
        "data/NSCLC/processed/",
        "data/MELA2022/raw/",
        "data/TCIA/",
        "data/MOSMED/processed/",
        "data/VinDr/v1/processed/",
    ],
    cache_dir="./cache",
)
```

Expected cache structure:
```
cache/
├── train/
│   ├── ct/    # [1, D, H, W] preprocessed CT volume tensors
│   └── xr/    # [1, H, W] rendered frontal X-ray tensors
├── val/
│   ├── ct/
│   └── xr/
└── test/
    ├── ct/
    └── xr/
```

---

## Step 2: Training Predict 2.5 (Multiview Synthesis)

Fine-tune Cosmos Predict 2.5 (`MinimalV1LVGDiT`) on 7-view X-ray sequence generation.

### Single-GPU Launch
```bash
./launch/train_predict25.sh --single --steps 50000 --warmup 2500
```

### Multi-GPU DDP Launch (4 GPUs)
```bash
torchrun --nproc_per_node=4 -m predict2_5.trainer \
    --cache-dir ./cache \
    --batch-size 1 \
    --num-gpus 4 \
    --max-iters 100000 \
    --warmup 5000 \
    --strategy ddp \
    --precision bf16-mixed
```

---

## Step 3: Training Transfer 2.5 (ControlNet Conditioning)

Train Cosmos Transfer 2.5 (`MinimalV4LVGControlVaceDiT`) with structural control signals (edge maps or depth maps).

### Edge Map Control (Single GPU)
```bash
./launch/train_transfer25.sh --single --steps 50000 --control edge_map
```

### Depth Map Control (8 GPUs DDP)
```bash
torchrun --nproc_per_node=8 -m transfer2_5.trainer \
    --cache-dir ./cache \
    --batch-size 1 \
    --num-gpus 8 \
    --max-iters 100000 \
    --warmup 2000 \
    --control-type depth_map \
    --freeze-base \
    --strategy ddp \
    --precision bf16-mixed
```

---

## Step 4: Monitoring Training

### TensorBoard
```bash
# Predict 2.5 logs
tensorboard --logdir outputs/predict2.5_multiview/

# Transfer 2.5 logs
tensorboard --logdir outputs/transfer2.5_multiview/
```

### WandB Integration
Ensure you are logged in via `wandb login`. WandB will automatically log loss curves, learning rate schedules, and 7-view ground-truth vs. generated X-ray comparison grids.

---

## Step 5: Interactive Web App & Inference

Launch the Gradio web application for interactive inference:

```bash
python app.py --share
```

The Gradio web app allows you to:
1. Upload a single frontal X-ray or select a sample CT volume.
2. Select pipeline mode: **Predict 2.5** (Multiview Direct) or **Transfer 2.5** (ControlNet Edge/Depth).
3. Synthesize and visualize 7 anatomical X-ray projections (AP, PA, LAT-L, LAT-R, LAO, RAO, Cranial).
