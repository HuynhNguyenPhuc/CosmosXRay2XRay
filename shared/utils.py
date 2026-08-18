"""Utilities"""

import os
import sys
import logging
import itertools

import numpy as np
import torch
import torch.distributed as dist


# ============================================================
# DDP Utilities
# ============================================================

def get_local_rank() -> int:
    """Get local rank for DDP (which GPU on this node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Get total number of processes in DDP."""
    return int(os.environ.get("WORLD_SIZE", 1))


def get_global_rank() -> int:
    """Get global rank across all nodes."""
    return int(os.environ.get("RANK", 0))


def is_ddp_mode() -> bool:
    """Check if running under DDP."""
    return get_world_size() > 1


def ddp_barrier():
    """Synchronize all DDP ranks."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current rank in distributed setting."""
    return dist.get_rank() if is_distributed() else 0


def rank_print(*args, **kwargs):
    """Print from rank 0 only."""
    if get_local_rank() == 0:
        print(*args, **kwargs)


def broadcast_model_states(model: torch.nn.Module, src: int = 0, logger=None) -> None:
    """Broadcast model states from source rank to all others."""
    if not is_distributed() or get_world_size() == 1:
        return
    for _, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        dist.broadcast(tensor.data, src=src)


def broadcast_model_states_packed(
    model: torch.nn.Module,
    src: int = 0,
    logger=None,
) -> None:
    """Broadcast model parameters and buffers using a single packed tensor."""
    if not is_distributed() or get_world_size() == 1:
        return
    
    try:
        tensors = []
        shapes = []
        dtypes = []
        
        for tensor in itertools.chain(
            (p.data for p in model.parameters()),
            (b for b in model.buffers()),
        ):
            if tensor.numel() == 0:
                continue
            tensors.append(tensor)
            shapes.append(tensor.shape)
            dtypes.append(tensor.dtype)
        
        if not tensors:
            return
        
        device = tensors[0].device
        if device.type != 'cuda':
            if logger:
                logger.warning(f"[EMA Sync] Tensors not on CUDA ({device}), skipping packed broadcast")
            return
        
        flat_tensors = []
        for t in tensors:
            if t.device != device:
                t = t.to(device=device)
            flat_tensors.append(t.flatten().to(dtype=torch.float32))
        
        packed = torch.cat(flat_tensors)
        dist.broadcast(packed, src=src, async_op=False)
        
        offset = 0
        for tensor, shape, dtype in zip(tensors, shapes, dtypes):
            numel = tensor.numel()
            unpacked = packed[offset:offset + numel].view(shape).to(dtype=dtype)
            tensor.data.copy_(unpacked)
            offset += numel
            
    except Exception as e:
        if logger:
            logger.error(f"[EMA Sync] Packed broadcast failed: {e}, falling back to naive broadcast")
        broadcast_model_states(model, src=src, logger=logger)


def sync_ema_ddp(net_ema: torch.nn.Module, sync_every_n_steps: int = 1, current_step: int = 0, logger=None) -> None:
    """Synchronize EMA model across DDP ranks."""
    if not is_distributed() or get_world_size() == 1:
        return
    if current_step % sync_every_n_steps != 0:
        return
    
    try:
        dist.barrier()
        broadcast_model_states_packed(net_ema, src=0, logger=logger)
        dist.barrier()
    except Exception as e:
        if logger:
            logger.error(f"[EMA Sync] Failed at step {current_step}: {e}")


# ============================================================
# Logging
# ============================================================

def get_logger(name: str = __name__) -> logging.Logger:
    """Get a DDP-aware logger (only rank-0 logs)."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        rank = get_local_rank()
        
        if rank != 0:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
        
        if rank == 0:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.propagate = False
    
    return logger


def setup_early_logging():
    """Suppress non-rank-0 logs and filter warnings."""
    import warnings

    warnings.filterwarnings("ignore", message=".*_extra_state.*")
    warnings.filterwarnings("ignore", message=".*FP8.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformer_engine.*")
    # HuggingFace: "use_fast is unset and a slow processor was saved…"
    warnings.filterwarnings("ignore", message=".*use_fast is unset.*")
    # PyTorch Lightning lr_monitor: "To copy construct from a tensor…"
    warnings.filterwarnings("ignore", message=".*copy construct from a tensor.*")

    for logger_name in ["torch.distributed", "torch.nn.parallel", "PIL", "matplotlib"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if get_local_rank() != 0:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

        # Cosmos uses a *custom* loguru Logger instance (not the global one),
        # so we must silence it directly for non-rank-0 processes.
        try:
            from cosmos_predict2._src.imaginaire.utils import log as _cosmos_log
            _cosmos_log.logger.disable("")          # "" matches every module name
        except Exception:
            pass

        try:
            from loguru import logger as loguru_logger
            loguru_logger.disable("")
        except ImportError:
            pass


# ============================================================
# PyTorch & Format Utilities
# ============================================================

import re

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


def is_uuid_format(text: str) -> bool:
    """Check whether text matches canonical UUID format."""
    if not text:
        return False
    return bool(_UUID_RE.fullmatch(text.strip()))


def safe_torch_load(path: str, map_location: str = "cpu") -> dict:
    """Load checkpoint safely with fallback for weights_only errors."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        if "weights_only" in str(exc) or "Unsupported global" in str(exc):
            logger = get_logger(__name__)
            logger.warning("weights_only=True failed; retrying with weights_only=False")
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


def fix_rope_buffers(module: torch.nn.Module) -> None:
    """Fix RoPE buffers for VideoRopePosition3DEmb layers after meta-device initialization."""
    for _, child in module.named_children():
        if child.__class__.__name__ == "VideoRopePosition3DEmb":
            target_device = child._buffers["dim_spatial_range"].device
            max_len = max(child.max_h, child.max_w, child.max_t)
            child.seq = torch.arange(max_len, device=target_device, dtype=torch.float32)

            dim_h = child._dim_h
            dim_t = child._dim_t

            child.dim_spatial_range = (
                torch.arange(0, dim_h, 2, device=target_device, dtype=torch.float32)[
                    : (dim_h // 2)
                ]
                / dim_h
            )
            child.dim_temporal_range = (
                torch.arange(0, dim_t, 2, device=target_device, dtype=torch.float32)[
                    : (dim_t // 2)
                ]
                / dim_t
            )
        else:
            fix_rope_buffers(child)


def move_tokenizer_to_device(tokenizer: object, target_device: str) -> None:
    """Move tokenizer buffers and normalization tensors to target device."""
    if not hasattr(tokenizer, "model") or not hasattr(tokenizer.model, "model"):
        return

    vae_module = tokenizer.model.model
    vae_module.to(target_device)
    tokenizer.model.device = target_device

    for attr in ["mean", "std", "img_mean", "img_std", "video_mean", "video_std"]:
        if hasattr(tokenizer.model, attr):
            val = getattr(tokenizer.model, attr)
            if isinstance(val, torch.Tensor):
                setattr(tokenizer.model, attr, val.to(target_device))


# ============================================================
# Random
# ============================================================

def arch_invariant_rand(
    shape: tuple,
    dtype: torch.dtype,
    device: str | torch.device,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate deterministic random tensor (GPU-agnostic)."""
    rng = np.random.RandomState(seed)
    random_array = rng.standard_normal(shape).astype(np.float32)
    return torch.from_numpy(random_array).to(dtype=dtype, device=device)
