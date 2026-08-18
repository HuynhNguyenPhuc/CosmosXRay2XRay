# Repository Structure — CosmosXRay2XRay

This document details the exact repository structure and file-by-file contents of the **CosmosXRay2XRay** codebase, mapping configurations, core predictive engines, ControlNet transfer modules, rendering pipelines, and utility scripts.

---

## 📂 Visual Repository Tree

```markdown
CosmosXRay2XRay/             # Repository Root
├── app.py                  # Gradio Web Interface (Interactive Inference CLI)
├── predict2_5/             # Core Predict 2.5 Fine-Tuning Package
│   ├── callback.py         # WandB & TensorBoard 7-view image logging callback
│   ├── hf.py               # Hugging Face checkpoint loading and management
│   ├── module.py           # CosmosXRay2XRayMultiview (LightningModule for Predict 2.5)
│   └── trainer.py          # Predict 2.5 CLI configuration and training entrypoint
├── transfer2_5/            # Core Transfer 2.5 ControlNet Package
│   ├── callback.py         # WandB & TensorBoard logging with control signal overlays
│   ├── control_signal.py   # Control signal generator (Sobel edge map, depth map, seg mask)
│   ├── module.py           # CosmosXRay2XRayTransferMultiview (ControlNet LightningModule)
│   └── trainer.py          # Transfer 2.5 CLI configuration and training entrypoint
├── predict3/               # Cosmos 3 Package — SCAFFOLDING (see docs/cosmos-predict3/PLAN.md §7-9)
│   ├── camera.py            # camera_pose action conditioning (9D poses, not text-encoded geometry)
│   ├── constants.py        # Real nvidia/Cosmos3-Edge architecture constants (from published config)
│   ├── hf.py                # Config-file resolution for nvidia/Cosmos3-Edge (weights not yet downloaded)
│   ├── module.py            # CosmosXRay2XRayPredict3Multiview (native I2V + action, real Cosmos3OmniTransformer)
│   └── trainer.py           # `--smoke-test` wiring verification; real training not wired up yet
├── renderers/             # DiffDRR Siddon-Jacob Ray-Tracing Renderer Package
│   └── diffdrr/           # DiffDRRVolumeRenderer & dataset pre-rendering
├── shared/                 # Shared Infrastructure & Utilities
│   ├── cache_builder.py    # Preprocesses NIfTI CT volumes into cached .npy files
│   ├── callbacks.py        # Lightning callbacks (EMA sync, DDP, gradient clipping)
│   ├── constants.py        # Single source of truth (UUIDs, 7-view cameras, captions, dimensions)
│   ├── datamodule.py       # ChestCTDataModule (Lightning DataModule)
│   ├── transforms.py       # MONAI & intensity normalization pipeline
│   ├── utils.py            # DDP utilities, logging, and state dict helpers
│   └── visualizer.py       # Matplotlib & TensorBoard 7-view layout visualizer
├── launch/                 # Automation Shell Scripts
│   ├── mount_from_gcs.sh   # GCS bucket mounting script
│   ├── setup_venv.sh       # Environment setup script
│   ├── train_predict25.sh  # Predict 2.5 training launcher (single-GPU / DDP)
│   └── train_transfer25.sh # Transfer 2.5 training launcher (single-GPU / DDP)
├── scripts/                # Utility & Data Transfer Scripts
│   ├── download_from_gcs.py # Download datasets/checkpoints from Google Cloud Storage
│   └── upload_to_gcs.py   # Upload training outputs/checkpoints to GCS
├── cosmos-predict2.5/      # Submodule (NVIDIA Cosmos Predict 2.5 Backbone)
├── cosmos-transfer2.5/     # Submodule (NVIDIA Cosmos Transfer 2.5 Backbone)
├── cosmos-framework/       # Submodule (NVIDIA Cosmos 3 — action/pose utils + SFT recipes; not pip-installed)
├── docs/                   # Documentation Suite
│   ├── INDEX.md            # Navigation Matrix
│   ├── STRUCTURE.md        # Repository Map (This file)
│   ├── WALKTHROUGH.md      # Step-by-step execution walkthrough
│   ├── DATASET.md          # Dataset specs, CT preprocessing & HU windowing
│   ├── RENDERER.md         # 7-view camera setup & raymarching physics
│   ├── BENCHMARK.md        # Quantitative evaluation protocol
│   ├── EXECUTION.md        # Master training & multi-GPU scaling protocol
│   ├── LITERATURE.md       # Academic context on Cosmos 2.5 & ControlNet
│   ├── GOTCHAS.md          # Technical & physical pitfalls
│   ├── PROPOSAL.md         # Project proposal & milestone roadmap
│   └── REVIEWS.md          # Submission readiness & review criteria
├── tests/                  # Unit and Integration Test Suite
│   ├── test_shared.py      # Tests for shared constants, transforms, renderer
│   ├── test_predict25.py   # Tests for Predict 2.5 Lightning module
│   └── test_transfer25.py  # Tests for Transfer 2.5 ControlNet module
├── run_all_tests_isolate.py # Isolated test execution runner
├── AGENTS.md               # Contributor & AI Agent Guide
├── CLAUDE.md               # Development instructions map
├── WORKFLOW.md             # Research strategy guidelines
├── CHAT.md                 # Session history tracking
├── research-state.yaml     # Machine-readable project state
├── research-log.md         # Chronological experiment log
├── .mcp.json               # MCP server configuration
├── requirements.txt        # Python dependencies list
└── README.md               # Project overview & quickstart guide
```

---

## 🛠️ Module Specifications

### 1. `predict2_5/` Pipeline
- `module.py`: Implements `CosmosXRay2XRayMultiview`, extending PyTorch Lightning. Initializes `MinimalV1LVGDiT`, Wan2.1 VAE tokenizer, and T5/Reason text encoder. Computes flow-matching loss over 7-view latent sequences.
- `trainer.py`: Command-line parser and entrypoint for Predict 2.5 training. Supports `--cache-dir`, `--batch-size`, `--num-gpus`, `--strategy ddp`, `--precision bf16-mixed`, `--max-iters`.
- `callback.py`: Intercepts validation steps to render 7-view ground-truth X-rays alongside generated outputs in WandB and TensorBoard.

### 2. `transfer2_5/` Pipeline
- `module.py`: Implements `CosmosXRay2XRayTransferMultiview`. Wraps `MinimalV4LVGControlVaceDiT`. Supports base model freezing while training the control branch with ControlNet conditioning.
- `control_signal.py`: Generates structural control signals (Sobel edge maps, depth projections, anatomical masks).
- `trainer.py`: Command-line entrypoint for Transfer 2.5 training. Accepts `--control-type` (`edge_map`, `depth_map`, `seg_mask`), `--control-context-scale`, `--freeze-base`.
- `callback.py`: Logs control images alongside input X-rays and generated multiview frames.

### 3. `shared/` & `renderers/` Modules
- `constants.py`: Holds exact configurations: `NUM_FRAMES=93`, `VOL_SIZE=256`, Cosmos UUIDs, 7 camera extrinsics (AP, PA, LAT-L, LAT-R, LAO, RAO, Cranial), and clinical captions.
- `renderers/diffdrr/`: DiffDRR Siddon-Jacob ray-tracing renderer simulating physical X-ray attenuation ($I = I_0 e^{-\int \mu(x) dx}$).
- `datamodule.py` & `cache_builder.py`: Manages NIfTI volume processing, train/val/test splits, and cached `.npy` tensor loading.
