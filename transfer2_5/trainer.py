"""Trainer entrypoint for Cosmos-Transfer 2.5 multiview X-ray synthesis."""

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy
import torch

torch.set_float32_matmul_precision("high")

# PyTorch 2.6+ defaults weights_only=True; allowlist numpy types so legacy
# checkpoints that contain numpy scalars/dtypes can be resumed.
_numpy_safe = [numpy._core.multiarray.scalar]
try:
    _numpy_safe += [
        numpy.dtypes.Float64DType,
        numpy.dtypes.Float32DType,
        numpy.dtypes.Int64DType,
        numpy.dtypes.Int32DType,
        numpy.dtypes.Int16DType,
        numpy.dtypes.Int8DType,
        numpy.dtypes.BoolDType,
    ]
except AttributeError:
    pass
torch.serialization.add_safe_globals(_numpy_safe)

from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy

from transfer2_5.module import CosmosXRay2XRayTransferMultiview
from transfer2_5.callback import TransferMultiviewTensorBoardCallback, TransferMultiviewWanDBCallback
from shared.datamodule import ChestCTDataModule
from shared.callbacks import (
    EMAMonitorCallback,
    GradClipCallback,
    DDPSamplerEpochCallback,
    GradientMonitorCallback,
)
from shared.utils import get_logger

log = get_logger(__name__)


@dataclass
class TransferTrainingConfig:
    """Configuration for Transfer2.5 multiview X-ray training."""

    # ── Paths ──
    data_dirs: str = ""  # Comma-separated NIfTI source directories (for cache building)
    cache_dir: str = "./cache"
    output_dir: str = "outputs/transfer2.5_multiview"

    # ── Checkpoint ──
    checkpoint_uuid: str = ""
    tokenizer_uuid: str = ""
    checkpoint_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    resume_ckpt: Optional[str] = None

    # ── Model Size ──
    model_size: str = "2B"

    # ── Training ──
    learning_rate: float = 1e-5
    weight_decay: float = 0.001
    warmup_steps: int = 1000
    max_iters: int = 100000
    max_epochs: int = 200
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    loss_scale: float = 1.0

    # ── Batch ──
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True

    # ── EMA ──
    enable_ema: bool = True
    ema_rate: float = 0.10
    ema_offload_cpu: bool = True

    # ── CFG ──
    cfg_dropout_rate: float = 0.2

    # ── Inference ──
    num_inference_steps: int = 35
    guidance_scale: float = 1.5

    # ── Control ──
    control_type: str = "edge_map"
    control_context_scale: float = 1.0
    freeze_base: bool = True
    num_max_modalities: int = 1
    vace_block_every_n: int = 2
    condition_strategy: str = "spaced"
    copy_weight_strategy: str = "spaced_n"

    # ── Distributed ──
    strategy: str = "ddp"
    num_gpus: int = 1
    precision: str = "bf16-mixed"

    # ── Logging ──
    logger_type: str = "tensorboard"  # "wandb" or "tensorboard"
    wandb_project: str = "cosmos-xray2xray"
    wandb_name: Optional[str] = None
    log_every_n_steps: int = 50
    wandb_log_images_every: int = 500
    wandb_generate_every: int = 2000

    # ── Checkpointing ──
    save_top_k: int = 3
    save_every_n_steps: int = 500

    # ── X-Ray Rendering ──
    renderer_n_pts_per_ray: int = 1000
    renderer_min_depth: float = 7.0
    renderer_max_depth: float = 9.0
    num_frontal_frames: int = 5

    # ── Text Encoder ──
    text_encoder_device: Optional[str] = None
    text_encoder_ckpt: Optional[str] = None

    # ── Multiview ──
    views_per_batch: int = 1


def get_callbacks(config: TransferTrainingConfig) -> List:
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=os.path.join(config.output_dir, "checkpoints"),
            filename="step={step:06d}-val_loss={val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=config.save_top_k,
            every_n_train_steps=config.save_every_n_steps,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        GradClipCallback(clip_norm=config.gradient_clip_val),
        DDPSamplerEpochCallback(),
        GradientMonitorCallback(log_every_n_steps=config.log_every_n_steps),
    ]
    if config.logger_type == "wandb":
        callbacks.append(TransferMultiviewWanDBCallback(
            log_every_n_steps=config.wandb_log_images_every,
            log_generation_every_n_steps=config.wandb_generate_every,
        ))
    else:
        callbacks.append(TransferMultiviewTensorBoardCallback(
            log_every_n_steps=config.wandb_log_images_every,
        ))
    if config.enable_ema:
        callbacks.append(EMAMonitorCallback())
    return callbacks


def _build_logger(config: "TransferTrainingConfig"):
    if config.logger_type == "wandb":
        return WandbLogger(
            project=config.wandb_project,
            name=config.wandb_name or f"transfer2.5_mv_{config.model_size}",
            save_dir=config.output_dir,
            log_model=False,
        )
    return TensorBoardLogger(
        save_dir=config.output_dir,
        name="tensorboard",
        default_hp_metric=False,
    )


def get_strategy(config: TransferTrainingConfig):
    if config.strategy == "fsdp":
        return FSDPStrategy(
            auto_wrap_policy=None,
            activation_checkpointing_policy=None,
            sharding_strategy="FULL_SHARD",
        )
    elif config.strategy == "ddp":
        return DDPStrategy(
            find_unused_parameters=True,  # Control branch may have unused params initially
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
    return config.strategy


def train(config: TransferTrainingConfig):
    """Run training."""
    os.makedirs(config.output_dir, exist_ok=True)

    # ── Model ──
    model_kwargs = {
        "model_size": config.model_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_steps": config.warmup_steps,
        "max_iters": config.max_iters,
        "gradient_clip_val": config.gradient_clip_val,
        "loss_scale": config.loss_scale,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "cfg_dropout_rate": config.cfg_dropout_rate,
        "enable_ema": config.enable_ema,
        "ema_rate": config.ema_rate,
        "ema_offload_cpu": config.ema_offload_cpu,
        "distributed_strategy": config.strategy,
        "renderer_n_pts_per_ray": config.renderer_n_pts_per_ray,
        "renderer_min_depth": config.renderer_min_depth,
        "renderer_max_depth": config.renderer_max_depth,
        "num_frontal_frames": config.num_frontal_frames,
        # Control-specific params
        "control_type": config.control_type,
        "control_context_scale": config.control_context_scale,
        "freeze_base": config.freeze_base,
        "num_max_modalities": config.num_max_modalities,
        "vace_block_every_n": config.vace_block_every_n,
        "condition_strategy": config.condition_strategy,
        "copy_weight_strategy": config.copy_weight_strategy,
        "views_per_batch": config.views_per_batch,
    }
    if config.checkpoint_path:
        model_kwargs["checkpoint_path"] = config.checkpoint_path
    elif config.checkpoint_uuid:
        model_kwargs["checkpoint_uuid"] = config.checkpoint_uuid
    if config.tokenizer_path:
        model_kwargs["tokenizer_path"] = config.tokenizer_path
    elif config.tokenizer_uuid:
        model_kwargs["tokenizer_uuid"] = config.tokenizer_uuid
    if config.text_encoder_device:
        model_kwargs["text_encoder_device"] = config.text_encoder_device
    if config.text_encoder_ckpt:
        model_kwargs["text_encoder_ckpt"] = config.text_encoder_ckpt

    model = CosmosXRay2XRayTransferMultiview(**model_kwargs)

    # ── Data ──
    dataset_dirs = [d.strip() for d in config.data_dirs.split(",") if d.strip()] if config.data_dirs else None
    datamodule = ChestCTDataModule(
        dataset_dirs=dataset_dirs,
        cache_dir=config.cache_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # ── Logger ──
    logger = _build_logger(config)

    # ── Trainer ──
    # When launched via torchrun, LOCAL_WORLD_SIZE overrides the config value
    # so Lightning's device count matches the actual world size.
    num_devices = int(os.environ.get("LOCAL_WORLD_SIZE", config.num_gpus))
    trainer = Trainer(
        max_epochs=config.max_epochs,
        max_steps=config.max_iters,
        accelerator="gpu",
        devices=num_devices,
        strategy=get_strategy(config),
        precision=config.precision,
        callbacks=get_callbacks(config),
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=None,
        check_val_every_n_epoch=1,
        limit_val_batches=1.0,
        enable_checkpointing=True,
        default_root_dir=config.output_dir,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=config.resume_ckpt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cosmos-Transfer 2.5 Multiview X-ray Training")
    parser.add_argument("--data-dirs", type=str, default="", help="Comma-separated NIfTI source dirs")
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, default="outputs/transfer2.5_multiview")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--model-size", type=str, default="2B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--max-iters", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--logger-type", type=str, default="tensorboard", choices=["wandb", "tensorboard"])
    parser.add_argument("--wandb-project", type=str, default="cosmos-xray2xray")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--text-encoder-device", type=str, default=None)
    parser.add_argument("--text-encoder-ckpt", type=str, default=None)
    parser.add_argument("--views-per-batch", type=int, default=2)
    # Control-specific args
    parser.add_argument("--control-type", type=str, default="edge_map")
    parser.add_argument("--control-context-scale", type=float, default=1.0)
    parser.add_argument("--freeze-base", action="store_true", default=True)
    parser.add_argument("--no-freeze-base", dest="freeze_base", action="store_false")
    parser.add_argument("--condition-strategy", type=str, default="spaced")
    parser.add_argument("--copy-weight-strategy", type=str, default="spaced_n")

    args = parser.parse_args()

    cfg = TransferTrainingConfig(
        data_dirs=args.data_dirs,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        model_size=args.model_size,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        max_epochs=args.max_epochs,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        strategy=args.strategy,
        precision=args.precision,
        logger_type=args.logger_type,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        resume_ckpt=args.resume_ckpt,
        text_encoder_device=args.text_encoder_device,
        text_encoder_ckpt=args.text_encoder_ckpt,
        control_type=args.control_type,
        control_context_scale=args.control_context_scale,
        freeze_base=args.freeze_base,
        condition_strategy=args.condition_strategy,
        copy_weight_strategy=args.copy_weight_strategy,
        views_per_batch=args.views_per_batch,
    )
    train(cfg)
