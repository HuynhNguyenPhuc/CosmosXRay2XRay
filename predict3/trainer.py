"""Trainer entrypoint for Cosmos 3 multiview X-ray synthesis (predict3) — SCAFFOLDING.

Only `--smoke-test` is implemented: it builds the real `Cosmos3OmniTransformer` +
`AutoencoderKLWan` architecture (randomly initialized — see predict3/module.py's docstring)
and runs one real forward + backward pass against a synthetic X-ray-shaped video, to verify
the joint-sequence packing plus native-I2V and `camera_pose` action wiring end to end
without needing real checkpoint weights or a real dataset.

Real training (`train()`) is NOT implemented yet: it needs (1) real `nvidia/Cosmos3-Edge`
weight loading and (2) a datamodule producing the (video, prompt, camera) batches
`training_step` expects, which in turn needs the DRR dataset converted to captions per
docs/cosmos-predict3/PLAN.md §5. Calling `train()` raises `NotImplementedError` pointing here
rather than silently doing something partial.

Must run in a separate venv from predict2_5/transfer2_5's — see predict3/module.py's
docstring for why.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch

torch.set_float32_matmul_precision("high")

from predict3.constants import COSMOS3_EDGE_REPO_ID
from predict3.module import CosmosXRay2XRayPredict3Multiview
from shared.utils import get_logger

log = get_logger(__name__)


@dataclass
class Predict3Config:
    """Configuration for Cosmos 3 multiview X-ray scaffolding."""

    model_repo_id: str = COSMOS3_EDGE_REPO_ID
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_camera_action: bool = True
    # Small synthetic clip; real X-ray clips use NUM_FRAMES=93 at VOL_SIZE=256.
    smoke_test_height: int = 64
    smoke_test_width: int = 64
    smoke_test_num_frames: int = 9
    # Depth-only scale-down so the wiring is testable without ~27 GB for the full 28-layer
    # model. See CosmosXRay2XRayPredict3Multiview._create_transformer. None = real depth.
    num_hidden_layers_override: int | None = 2


def smoke_test(config: Predict3Config) -> float:
    """Build the real Cosmos 3 architecture and run one real forward + backward pass on a
    synthetic X-ray-shaped clip, verifying the native-I2V + `camera_pose` action wiring.

    Returns the scalar training loss (meaningless numerically — weights are random — but a
    finite value confirms shapes, dtypes, mRoPE packing and loss indices all line up).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = CosmosXRay2XRayPredict3Multiview(
        model_repo_id=config.model_repo_id,
        use_camera_action=config.use_camera_action,
        num_hidden_layers_override=config.num_hidden_layers_override,
    )
    model.to(config.device)
    model.setup()

    video = torch.rand(
        1, 3, config.smoke_test_num_frames, config.smoke_test_height, config.smoke_test_width,
        device=config.device,
    )
    batch = {"video": video, "prompt": "A 360-degree rotational view of a chest CT scan."}

    output = model.training_step(batch, batch_idx=0)
    loss = output["loss"]
    loss.backward()

    num_grad = sum(1 for p in model.transformer.parameters() if p.grad is not None)
    num_params = sum(1 for _ in model.transformer.parameters())
    log.info(
        f"[predict3 smoke test] loss={loss.item():.6f}, "
        f"transformer_params_with_grad={num_grad}/{num_params}, "
        f"camera_action={'on' if config.use_camera_action else 'off'}, "
        f"layers={config.num_hidden_layers_override or 'real'}"
    )
    return loss.item()


def train(config: Predict3Config) -> None:
    raise NotImplementedError(
        "Real predict3 training is not wired up yet — needs real nvidia/Cosmos3-Edge weight "
        "loading and a captioned-dataset datamodule (docs/cosmos-predict3/PLAN.md §5). "
        "Use `--smoke-test` to verify the architecture/packing wiring in the meantime."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cosmos 3 Multiview X-ray Scaffolding")
    parser.add_argument("--smoke-test", action="store_true", help="Run the synthetic wiring smoke test")
    parser.add_argument("--model-repo-id", type=str, default=COSMOS3_EDGE_REPO_ID)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-camera-action", action="store_true", help="Disable camera_pose action conditioning")
    parser.add_argument(
        "--layers", type=int, default=2,
        help="Transformer depth for the smoke test; 0 uses the real published depth (needs ~27GB fp32).",
    )
    args = parser.parse_args()

    cfg = Predict3Config(
        model_repo_id=args.model_repo_id,
        use_camera_action=not args.no_camera_action,
        num_hidden_layers_override=None if args.layers == 0 else args.layers,
    )
    if args.device:
        cfg.device = args.device

    if args.smoke_test:
        smoke_test(cfg)
    else:
        train(cfg)
