# CosmosXRay2XRay

Multiview X-ray synthesis from CT volumes using NVIDIA Cosmos 2.5 video world models.

Two training pipelines:
- **Predict 2.5** — Fine-tunes a pre-trained Cosmos video DiT (`MinimalV1LVGDiT`) for X-ray multiview generation.
- **Transfer 2.5** — ControlNet-style training via `MinimalV4LVGControlVaceDiT`, adding a control branch (e.g. edge maps) while freezing the base model.

Both share a common data preparation pipeline (CT → rendered X-rays → cached .npy tensors) and 7-view camera configuration (AP, PA, LAT-L, LAT-R, LAO, RAO, Cranial).

---

## Project Structure

```
CosmosXRay2XRay/
├── renderers/             # DiffDRR Siddon-Jacob ray-tracing renderer backend
│   └── diffdrr/           # DiffDRRVolumeRenderer & data loader
├── shared/                # Shared modules (data, callbacks, constants)
│   ├── constants.py       # UUIDs, camera config, prompts, dimensions
│   ├── datamodule.py      # ChestCTDataModule (Lightning DataModule)
│   ├── cache_builder.py   # NIfTI CT → preprocessed .npy cache
│   ├── transforms.py      # MONAI/intensity transforms
│   ├── callbacks.py       # EMA, gradient clipping, DDP, monitoring
│   └── utils.py           # DDP utilities, logging, EMA sync
├── predict2_5/            # Cosmos-Predict 2.5 pipeline
│   ├── module.py          # CosmosXRay2XRayMultiview (LightningModule)
│   ├── callback.py        # WandB image/generation logging
│   └── trainer.py         # Config + training entrypoint
├── transfer2_5/           # Cosmos-Transfer 2.5 pipeline (ControlNet)
│   ├── module.py          # CosmosXRay2XRayTransferMultiview (LightningModule)
│   ├── callback.py        # WandB logging with control images
│   ├── control_signal.py  # Edge map / depth map generation
│   └── trainer.py         # Config + training entrypoint
├── cosmos-predict2.5/     # Git submodule — NVIDIA Cosmos Predict 2.5
├── cosmos-transfer2.5/    # Git submodule — NVIDIA Cosmos Transfer 2.5
└── requirements.txt
```

---

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repo-url>
cd CosmosXRay2XRay
```

If already cloned without submodules:
```bash
git submodule update --init --recursive
```

### 2. Choose environment manager

#### Option A: Conda

```bash
# Use Python 3.10 for cu128 extra (flash-attn wheels only available for cp310)
# For Python 3.11, use cu130 extra instead
conda create -n cosmos-xray python=3.10 -y
conda activate cosmos-xray
```

Install dependencies in the activated conda env:

```bash
# Ensure submodules are present
git submodule update --init --recursive

# Check your CUDA version
nvcc --version

# Install cosmos-predict2.5 with CUDA extra (cu128 for CUDA 12.x, cu130 for CUDA 13.x)
cd cosmos-predict2.5
pip install -e ".[cu128]"
cd ..

# Install cosmos-transfer2.5 with CUDA extra
cd cosmos-transfer2.5
pip install -e ".[cu128]"
cd ..

# Install project dependencies
pip install -r requirements.txt

# Install pytorch3d for X-ray volume rendering
# (install with --no-build-isolation to ensure torch is available during build)
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

#### Option B: uv

Install `uv` first (if not installed).

**On Linux / cluster environments**, install directly from the installer:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh

# Add to PATH (or restart shell to pick up ~/.bashrc)
source ~/.bashrc  # or source ~/.local/bin/env

# Verify installation
uv --version
```

**On Windows** or if you prefer pip:

```bash
pip install uv
```

Create and activate virtual environment:

```bash
# Use Python 3.10 for cu128 extra (flash-attn wheels only available for cp310)
# For Python 3.11, use cu130 extra instead
uv venv .venv --python 3.10
```

Activation:

```bash
# Linux/macOS
source .venv/bin/activate

# PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies with `uv pip`:

```bash
# Ensure submodules are present
git submodule update --init --recursive

# Check your CUDA version
nvcc --version

# Install cosmos-predict2.5 with CUDA extra (cu128 for CUDA 12.x, cu130 for CUDA 13.x)
cd cosmos-predict2.5
uv pip install -e ".[cu128]"
cd ..

# Install cosmos-transfer2.5 with CUDA extra
cd cosmos-transfer2.5
uv pip install -e ".[cu128]"
cd ..

# Install project dependencies
uv pip install -r requirements.txt

# Install pytorch3d for X-ray volume rendering
# (install with --no-build-isolation to ensure torch is available during build)
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

export CCACHE_DIR=~/.ccache
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

export TORCH_CUDA_ARCH_LIST="8.0"

sudo apt-get install -y ninja-build ccache

uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

**CUDA Extra Options:**
- `cu128`: CUDA 12.x + PyTorch 2.7 + **Python 3.10** (recommended for A100)
- `cu130`: CUDA 13.x + PyTorch 2.9 + Python 3.11 compatible

**Python Version Compatibility:**
- `cu128` requires Python **3.10** (flash-attn wheels only available for cp310)
- `cu130` supports Python **3.11**

If installing failed with "no wheels with a matching Python ABI tag", recreate your venv with the correct Python version.

### 3. Set `PYTHONPATH`

Both pipelines import from `shared/`, `predict2_5/`, and `transfer2_5/` as top-level packages. Set the project root on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
```

> **Note**: The training entrypoints should be launched as Python modules from the repository root.

### 4. WandB login (optional, for logging)

```bash
wandb login
```

### 5. TensorBoard (optional, for real-time monitoring)

Training logs are saved to TensorBoard by default. To view them:

```bash
# Monitor Predict 2.5 training
tensorboard --logdir outputs/predict2.5_multiview/

# Monitor Transfer 2.5 training
tensorboard --logdir outputs/transfer2.5_multiview/
```

Then open http://localhost:6006/ in your browser.

---

## Data Preparation

### Input format

Raw NIfTI CT volumes (`.nii` or `.nii.gz`) organized in directories.

### Build the cache

The `CacheBuilder` preprocesses CT volumes into cached `.npy` files with rendered frontal X-rays:

```python
from shared.cache_builder import build_cache_from_directory

build_cache_from_directory(
    dataset_dirs=["data/NSCLC/processed/", "data/MELA2022/raw/", "data/TCIA/"],
    cache_dir="./cache",
)
```

This creates:
```
cache/
├── train/
│   ├── ct/     # Preprocessed CT volumes (.npy, shape: [1, D, H, W])
│   └── xr/     # Rendered frontal X-rays (.npy, shape: [1, H, W])
├── val/
│   ├── ct/
│   └── xr/
└── test/
    ├── ct/
    └── xr/
```

Alternatively, pass `--data-dirs` to the trainer and it will auto-build the cache.

---

## Training

### Predict 2.5 — Direct multiview fine-tuning

Using launch script:
```bash
# Single GPU
./launch/train_predict25.sh --single --steps 50000 --warmup 2500

# 4 GPUs (DDP)
./launch/train_predict25.sh --ddp --steps 100000 --warmup 5000
```

Direct command:
```bash
# Single GPU
python -m predict2_5.trainer \
    --cache-dir ./cache \
    --batch-size 1 \
    --num-gpus 1 \
    --max-iters 50000 \
    --warmup 2500 \
    --strategy ddp \
    --precision bf16-mixed

# Multi-GPU (torchrun)
torchrun --nproc_per_node=4 -m predict2_5.trainer \
    --cache-dir ./cache \
    --batch-size 1 \
    --num-gpus 4 \
    --max-iters 100000 \
    --warmup 5000 \
    --strategy ddp
```

### Transfer 2.5 — ControlNet-style training

Using launch script:
```bash
# Single GPU with edge maps
./launch/train_transfer25.sh --single --steps 50000 --control edge_map

# 4 GPUs with depth maps
./launch/train_transfer25.sh --ddp --steps 100000 --control depth_map --freeze

# 8 GPUs, fine-tune entire model (no freeze)
./launch/train_transfer25.sh --gpus 8 --no-freeze --control edge_map
```

Direct command:
```bash
# Single GPU
python -m transfer2_5.trainer \
    --cache-dir ./cache \
    --batch-size 1 \
    --num-gpus 1 \
    --max-iters 50000 \
    --warmup 1000 \
    --control-type edge_map \
    --freeze-base \
    --strategy ddp \
    --precision bf16-mixed

# Multi-GPU (torchrun)
torchrun --nproc_per_node=8 -m transfer2_5.trainer \
    --cache-dir ./cache \
    --batch-size 1 \
    --num-gpus 8 \
    --max-iters 100000 \
    --warmup 2000 \
    --control-type edge_map \
    --freeze-base \
    --strategy ddp
```

### Resume from checkpoint

```bash
# Using launch script
./launch/train_predict25.sh --resume outputs/cosmos-xray2xray-20260401-123456/checkpoints/last.ckpt

# Direct command
python -m predict2_5.trainer \
    --cache-dir ./cache \
    --resume-ckpt outputs/predict2.5_multiview/checkpoints/last.ckpt \
    --batch-size 1 \
    --num-gpus 4 \
    --strategy ddp
```

---

## Checkpoints

Pre-trained Cosmos checkpoints are identified by UUID and downloaded automatically via `download_checkpoint()`:

| Component | UUID |
|-----------|------|
| Tokenizer (Wan2.1 VAE) | `685afcaa-4de2-42fe-b7b9-69f7a2dee4d8` |
| Predict 2.5 DiT (2B) | `d20b7120-df3e-4911-919d-db6e08bad31c` |
| Cosmos Reason | `cb3e3ffa-7b08-4c34-822d-61c7aa31a14f` |
| T5-11B Text Encoder | `4dbf13c6-1d30-4b02-99d6-75780dd8b744` |

You can either:
- Pass `--checkpoint-path` / `--tokenizer-path` with local `.pt` file paths, or
- Set the UUIDs in `shared/constants.py` and the module will download them automatically.

### Converting DCP checkpoints to PyTorch format

If training produces DCP (Distributed Checkpoint) format, convert to `.pt` for inference:

```python
import torch
from torch.distributed.checkpoint import load

state_dict = {}
load(state_dict, checkpoint_id="path/to/dcp_checkpoint/")
torch.save(state_dict, "model.pt")
```

---

## Key Configuration Parameters

### Predict 2.5

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 2^(-14.5) ≈ 5.5e-5 | Learning rate |
| `warmup_steps` | 2000 | Linear warmup steps |
| `max_iters` | 100000 | Maximum training iterations |
| `cfg_dropout_rate` | 0.2 | CFG (classifier-free guidance) dropout |
| `guidance_scale` | 1.5 | Guidance scale at inference |
| `num_inference_steps` | 35 | Denoising steps for generation |
| `ema_rate` | 0.10 | EMA update rate |
| `gradient_clip_val` | 1.0 | Gradient clipping norm |

### Transfer 2.5 (additional)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `control_type` | `edge_map` | Control signal: `edge_map`, `depth_map`, `seg_mask` |
| `control_context_scale` | 1.0 | Strength of control conditioning |
| `freeze_base` | `True` | Freeze base DiT, train only control branch |
| `num_max_modalities` | 1 | Number of control modalities |
| `vace_block_every_n` | 2 | Insert VACE control block every N DiT blocks |
| `condition_strategy` | `spaced` | How control blocks are distributed |
| `copy_weight_strategy` | `spaced_n` | Weight initialization strategy for control branch |

---

## Architecture Overview

### Predict 2.5

1. CT volume → 7-view X-ray rendering (absorption-emission ray marching)
2. Frontal frames (repeated) + target view frames → 93-frame video tensor
3. Encode to latent space via Wan2.1 VAE (temporal compression 4× → 24 latent frames)
4. Text prompt → text encoder → cross-attention embeddings
5. Rectified flow training: sample timestep, add noise, predict velocity
6. FRAME_REPLACE: first `num_frontal_frames` latent frames replaced with clean frontal encoding (conditioning)

### Transfer 2.5

Same as Predict 2.5, plus:
1. Generate control signal (e.g. Sobel edge map) from target X-ray
2. Encode control signal to latent → build control video
3. ControlNet branch processes control input alongside base DiT
4. Base model frozen; only control blocks, control embedder, and VACE components are trained

---

## 7-View Camera Configuration

| View | Azimuth | Elevation | Description |
|------|---------|-----------|-------------|
| AP | 0° | 0° | Anterior-Posterior |
| PA | 180° | 0° | Posterior-Anterior |
| LAT-L | 90° | 0° | Left Lateral |
| LAT-R | 270° | 0° | Right Lateral |
| LAO | 135° | 0° | Left Anterior Oblique |
| RAO | 225° | 0° | Right Anterior Oblique |
| Cranial | 0° | -30° | Cranio-caudal |

Camera distance range: 7.0–9.0 units, FOV: 25°–35°.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

It also integrates NVIDIA Cosmos 2.5 components. See [cosmos-predict2.5/LICENSE](cosmos-predict2.5/LICENSE) and [cosmos-transfer2.5/LICENSE](cosmos-transfer2.5/LICENSE) for third-party component licenses.