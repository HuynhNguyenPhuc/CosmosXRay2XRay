"""
Cosmos-Transfer 2.5 Multiview X-Ray Synthesis with ControlNet (VACE DiT).

Uses MinimalV4LVGControlVaceDiT from cosmos_transfer2 — the ControlNet
branch is INSIDE the DiT model (control_blocks + control_embedder).

Training workflow:
    1. Load predict2.5 checkpoint → non_strict_load into MinimalV4LVGControlVaceDiT
       (base blocks match; control blocks are new)
    2. copy_weights_to_control_branch() — seed control blocks from base blocks
    3. freeze_base_model() — freeze base blocks, only train control_blocks + control_embedder
    4. Control signal: pixel-space Sobel edge → 93-frame video → 3ch → VAE encode → 16ch latent
       → pass as latent_control_input to forward()
"""

from shared.utils import setup_early_logging

setup_early_logging()

import contextlib
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

torch.set_float32_matmul_precision("high")

from lightning import LightningModule

from cosmos_predict2._src.imaginaire.utils.ema import FastEmaModelUpdater
from shared.utils import get_logger

log = get_logger(__name__)
from cosmos_predict2._src.imaginaire.utils.checkpointer import non_strict_load_model
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import download_checkpoint
from cosmos_oss.checkpoints_predict2 import register_checkpoints as _register_cosmos_checkpoints
_register_cosmos_checkpoints()

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_predict2._src.predict2.schedulers.rectified_flow import RectifiedFlow
from cosmos_predict2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
from cosmos_predict2._src.predict2.utils.optim_instantiate import get_base_optimizer

# cosmos_predict2 and cosmos_transfer2 each vendor their own copy of lazy_config.lazy,
# which unconditionally calls OmegaConf.register_new_resolver("add"/"subtract", ...) at
# import time; having imported cosmos_predict2 above already registered them, so the
# transfer2 import below would raise ValueError on the duplicate registration.
from omegaconf import OmegaConf

_omegaconf_register_new_resolver = OmegaConf.register_new_resolver


def _register_new_resolver_if_absent(name, *args, **kwargs):
    if OmegaConf.has_resolver(name):
        return None
    return _omegaconf_register_new_resolver(name, *args, **kwargs)


OmegaConf.register_new_resolver = _register_new_resolver_if_absent
try:
    from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import (
        MinimalV4LVGControlVaceDiT,
    )
finally:
    OmegaConf.register_new_resolver = _omegaconf_register_new_resolver

from shared.constants import (
    NUM_LATENT_FRAMES,
    COSMOS_TOKENIZER_UUID, COSMOS_2B_PRETRAINED_UUID,
    CR1_EMBEDDING_DIM, CR1_MAX_LENGTH,
    CROSSATTN_EMB_CHANNELS, CROSSATTN_PROJ_IN_CHANNELS,
    NUM_FRAMES,
    XRAY_CAMERAS,
    XRAY_EXTRINSICS,
    XRAY_FOV_RANGE, XRAY_FOV_DEFAULT,
    XRAY_DISTANCE_RANGE, XRAY_DISTANCE_DEFAULT,
    XRAY_CAPTION_PREFIXES,
    XRAY_PROMPT_TEMPLATE,
    NUM_XRAY_VIEWS,
)

from shared.utils import get_local_rank, get_rank, get_world_size, sync_ema_ddp, arch_invariant_rand
from renderers.diffdrr.renderer import DiffDRRVolumeRenderer

from transfer2_5.control_signal import generate_control_signal


_CHECKPOINT_RESOLVED = {}
_TOKENIZER_RESOLVED = {}


def _hf_download(repo_id: str, filename: str) -> str:
    """Download a large HuggingFace file reliably via curl.

    hf_hub_download (requests-based) stalls silently at CDN chunk boundaries.
    curl --speed-limit/--speed-time detects the stall within 30 s and -C - resumes
    without re-downloading already-fetched bytes.
    """
    import os
    import subprocess
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE

    # Fast path: already in local HF cache
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
    except Exception:
        pass

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    cache_dir = Path(HF_HUB_CACHE) / "manual" / repo_id.replace("/", "--")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / Path(filename).name

    if dest.exists() and dest.stat().st_size > 1_000_000:
        log.info(f"[HF] Reusing previously downloaded: {dest}")
        return str(dest)

    # Resolve auth token from all possible sources
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if not token:
        for p in [
            Path.home() / ".cache" / "huggingface" / "token",
            Path.home() / ".huggingface" / "token",
        ]:
            if p.exists():
                token = p.read_text().strip()
                break
    if not token:
        try:
            from huggingface_hub.utils import get_token as _get_token
            token = _get_token() or ""
        except Exception:
            pass

    log.info(f"[HF] curl download: {repo_id}/{filename}  →  {dest}")
    for attempt in range(1, 6):
        cmd = [
            "curl", "-fL",
            "-C", "-",               # resume partial download
            "--speed-limit", "102400",   # restart if < 100 KB/s ...
            "--speed-time",  "30",       # ... for 30 consecutive seconds
            "-o", str(dest), url,
        ]
        if token:
            cmd += ["-H", f"Authorization: Bearer {token}"]
        result = subprocess.run(cmd)
        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1_000_000:
            log.info(f"[HF] Download complete: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
            return str(dest)
        log.warning(f"[HF] curl attempt {attempt}/5 failed (code={result.returncode}); retrying …")

    raise RuntimeError(f"[HF] Failed to download {repo_id}/{filename!r} after 5 curl attempts")


class CosmosXRay2XRayTransferMultiview(LightningModule):
    """
    Cosmos-Transfer 2.5 for multiview X-ray synthesis.

    Uses MinimalV4LVGControlVaceDiT which integrates ControlNet control_blocks
    directly inside the DiT architecture.
    """

    _setup_complete: bool = False

    def __init__(
        self,
        # ── Checkpoint & Tokenizer ──
        checkpoint_uuid: str = COSMOS_2B_PRETRAINED_UUID,
        tokenizer_uuid: str = COSMOS_TOKENIZER_UUID,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        # ── Tokenizer Parameters ──
        tokenizer_chunk_duration: int = 93,
        tokenizer_temporal_window: int = 16,
        # ── Model Architecture ──
        model_size: str = "2B",
        state_t: int = NUM_LATENT_FRAMES,
        state_ch: int = 16,
        # ── Training Hyperparameters ──
        learning_rate: float = 2 ** (-14.5),
        weight_decay: float = 0.001,
        warmup_steps: int = 2000,
        max_iters: int = 100000,
        gradient_clip_val: float = 1.0,
        # ── Loss & Sampling ──
        loss_scale: float = 1.0,
        num_inference_steps: int = 35,
        guidance_scale: float = 1.5,
        rf_shift: float = 5.0,
        # ── CFG ──
        cfg_dropout_rate: float = 0.2,
        # ── Conditional Frames ──
        min_num_conditional_frames: int = 1,
        max_num_conditional_frames: int = 2,
        conditional_frame_timestep: float = -1.0,
        # ── EMA ──
        enable_ema: bool = True,
        ema_rate: float = 0.10,
        ema_offload_cpu: bool = True,
        ema_iteration_shift: int = 0,
        ema_sync_every_n_steps: int = 1,
        # ── Distributed ──
        distributed_strategy: Literal["auto", "ddp", "fsdp"] = "auto",
        # ── Renderer ──
        renderer_n_pts_per_ray: int = 1000,
        renderer_min_depth: float = 7.0,
        renderer_max_depth: float = 9.0,
        renderer_ndc_extent: float = 1.0,
        num_frontal_frames: int = 5,
        # ── Text Encoder ──
        text_encoder_device: Optional[str] = None,
        text_encoder_ckpt: Optional[str] = None,
        # ── Multiview ──
        views_per_batch: int = 2,
        # ── ControlNet (VACE) ──
        control_type: str = "edge_map",
        control_context_scale: float = 1.0,
        num_max_modalities: int = 1,
        vace_block_every_n: int = 2,
        condition_strategy: str = "spaced",
        copy_weight_strategy: str = "first_n",
        freeze_base: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tensor_kwargs = {"device": "cuda", "dtype": torch.float32}

        self._setup_rectified_flow()
        self._setup_tokenizer()
        self._setup_network()
        self._setup_ema()
        self._setup_renderer()
        self._setup_text_encoder()

    def _setup_rectified_flow(self):
        self.rectified_flow = RectifiedFlow(
            velocity_field=lambda *a, **k: None,
            train_time_distribution="logitnormal",
            train_time_weight_method="uniform",
            use_dynamic_shift=False,
            shift=self.hparams.rf_shift,
            device=self.device if hasattr(self, "device") else torch.device("cpu"),
            dtype=torch.float32,
        )
        self._num_train_timesteps = self.rectified_flow.num_train_timesteps
        self.sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False,
        )

    def _setup_tokenizer(self):
        global _TOKENIZER_RESOLVED
        if self.hparams.tokenizer_path:
            tokenizer_path = self.hparams.tokenizer_path
        else:
            uuid = self.hparams.tokenizer_uuid
            if uuid not in _TOKENIZER_RESOLVED:
                from cosmos_predict2._src.imaginaire.flags import INTERNAL
                from huggingface_hub import hf_hub_download
                
                # Only rank 0 downloads to avoid DDP file lock contention.
                # Other ranks wait for rank 0 to set _TOKENIZER_RESOLVED, then reuse.
                if get_local_rank() == 0:
                    if INTERNAL:
                        # NVidia cluster: S3 path accessible via checkpoint_db.
                        try:
                            tokenizer_path = download_checkpoint(uuid)
                        except Exception as e:
                            # Fallback to HuggingFace even on cluster (in case S3 is temporarily down)
                            log.info(
                                f"[Setup] Tokenizer checkpoint_db download failed ({type(e).__name__}); "
                                f"falling back to HuggingFace download …"
                            )
                            try:
                                tokenizer_path = _hf_download(
                                    repo_id="nvidia/Cosmos-Predict2.5-2B",
                                    filename="tokenizer.pth",
                                )
                            except Exception as e2:
                                log.error(
                                    f"[Setup] Both checkpoint_db and HuggingFace downloads failed. "
                                    f"Error: {e2}"
                                )
                                raise
                    else:
                        # External machine: go straight to HuggingFace (avoid S3 timeout).
                        log.info("[Setup] External mode (INTERNAL=False): downloading tokenizer from HuggingFace …")
                        try:
                            tokenizer_path = _hf_download(
                                repo_id="nvidia/Cosmos-Predict2.5-2B",
                                filename="tokenizer.pth",
                            )
                        except Exception as e:
                            log.error(f"[Setup] HuggingFace tokenizer download failed. Error: {e}")
                            raise
                    _TOKENIZER_RESOLVED[uuid] = tokenizer_path
                else:
                    # Non-rank-0 processes: wait for rank 0 to download and cache globally.
                    # Poll until _TOKENIZER_RESOLVED[uuid] is set by rank 0 (max 10min).
                    import time
                    max_wait = 600  # seconds
                    start_time = time.time()
                    while uuid not in _TOKENIZER_RESOLVED:
                        if time.time() - start_time > max_wait:
                            log.error(f"[Setup] Rank {get_local_rank()} timeout waiting for rank 0 to download tokenizer")
                            raise TimeoutError("Tokenizer download timeout on rank 0")
                        time.sleep(0.5)
                    tokenizer_path = _TOKENIZER_RESOLVED[uuid]
                
                if uuid not in _TOKENIZER_RESOLVED:
                    _TOKENIZER_RESOLVED[uuid] = tokenizer_path
            else:
                tokenizer_path = _TOKENIZER_RESOLVED[uuid]
        self.tokenizer = Wan2pt1VAEInterface(
            chunk_duration=self.hparams.tokenizer_chunk_duration,
            load_mean_std=False, vae_pth=tokenizer_path,
            temporal_window=self.hparams.tokenizer_temporal_window,
            keep_decoder_cache=False, keep_encoder_cache=False,
        )
        assert self.tokenizer.latent_ch == self.hparams.state_ch
        if get_local_rank() == 0:
            log.info(
                f"[Setup] VAE tokenizer loaded — path={tokenizer_path}, "
                f"latent_ch={self.tokenizer.latent_ch}, "
                f"chunk_dur={self.hparams.tokenizer_chunk_duration}, "
                f"temporal_window={self.hparams.tokenizer_temporal_window}"
            )
        configs = {
            "2B": {"model_channels": 2048, "num_heads": 16, "num_blocks": 28},
            "7B": {"model_channels": 4096, "num_heads": 32, "num_blocks": 28},
            "14B": {"model_channels": 5120, "num_heads": 40, "num_blocks": 36},
        }
        return configs[self.hparams.model_size]

    def _create_control_vace_dit(self, device: str = "meta") -> MinimalV4LVGControlVaceDiT:
        config = self._get_model_config()
        with torch.device(device):
            net = MinimalV4LVGControlVaceDiT(
                max_img_h=240, max_img_w=240, max_frames=128,
                in_channels=self.hparams.state_ch,
                out_channels=self.hparams.state_ch,
                patch_spatial=2, patch_temporal=1,
                concat_padding_mask=True,
                model_channels=config["model_channels"],
                num_blocks=config["num_blocks"],
                num_heads=config["num_heads"],
                atten_backend="minimal_a2a",
                pos_emb_cls="rope3d",
                pos_emb_learnable=True,
                pos_emb_interpolation="crop",
                use_adaln_lora=True, adaln_lora_dim=256,
                rope_h_extrapolation_ratio=3.0,
                rope_w_extrapolation_ratio=3.0,
                rope_t_extrapolation_ratio=1.0,
                crossattn_emb_channels=CROSSATTN_EMB_CHANNELS,
                use_crossattn_projection=True,
                crossattn_proj_in_channels=CROSSATTN_PROJ_IN_CHANNELS,
                sac_config=SACConfig(mode="mm_only"),
                timestep_scale=0.001,
                # ControlNet-specific
                num_max_modalities=self.hparams.num_max_modalities,
                vace_block_every_n=self.hparams.vace_block_every_n,
                condition_strategy=self.hparams.condition_strategy,
            )
        return net

    def _fix_rope_buffers(self, module: torch.nn.Module, model_name: str = "net"):
        for name, child in module.named_children():
            if child.__class__.__name__ == "VideoRopePosition3DEmb":
                target_device = child._buffers["dim_spatial_range"].device
                max_len = max(child.max_h, child.max_w, child.max_t)
                child.seq = torch.arange(max_len, device=target_device, dtype=torch.float32)
                dim_h = child._dim_h
                dim_t = child._dim_t
                child.dim_spatial_range = (
                    torch.arange(0, dim_h, 2, device=target_device, dtype=torch.float32)[: (dim_h // 2)] / dim_h
                )
                child.dim_temporal_range = (
                    torch.arange(0, dim_t, 2, device=target_device, dtype=torch.float32)[: (dim_t // 2)] / dim_t
                )
            else:
                self._fix_rope_buffers(child, model_name)

    def _copy_weights_to_control_branch(self):
        """Copy base block weights to control blocks (following cosmos-transfer2 pattern)."""
        net = self.net
        control_blocks = net.control_blocks
        strategy = self.hparams.copy_weight_strategy

        if strategy == "first_n":
            mapping = {i: i for i in range(len(control_blocks))}
        elif strategy == "spaced_n":
            mapping = {v: k for k, v in net.control_layers_mapping.items()}
        else:
            raise ValueError(f"Unknown copy_weight_strategy: {strategy}")

        for ctrl_idx, base_idx in mapping.items():
            if get_local_rank() == 0:
                log.info(f"[Transfer] Copying base block {base_idx} → control block {ctrl_idx}")
            missing, unexpected = control_blocks[ctrl_idx].load_state_dict(
                net.blocks[base_idx].state_dict(), strict=False
            )
            assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
            expected_missing = {
                "before_proj.weight", "before_proj.bias",
                "after_proj.weight", "after_proj.bias",
                "_checkpoint_wrapped_module.before_proj.weight",
                "_checkpoint_wrapped_module.before_proj.bias",
                "_checkpoint_wrapped_module.after_proj.weight",
                "_checkpoint_wrapped_module.after_proj.bias",
            }
            assert set(missing).issubset(expected_missing), f"Unexpected missing keys: {missing}"

    def _freeze_base_model(self):
        """Freeze base blocks, keep control_blocks + control_embedder trainable."""
        net = self.net

        # 1. Freeze everything
        for param in net.parameters():
            param.requires_grad = False

        # 2. Unfreeze control blocks + embedder
        for block in net.control_blocks:
            for param in block.parameters():
                param.requires_grad = True
        for param in net.control_embedder.parameters():
            param.requires_grad = True

        if hasattr(net, "input_hint_block"):
            for param in net.input_hint_block.parameters():
                param.requires_grad = True

        n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in net.parameters())
        if get_local_rank() == 0:
            log.info(
                f"[Transfer] Frozen: {n_total - n_trainable:,} params | "
                f"Trainable: {n_trainable:,} params ({100 * n_trainable / n_total:.1f}%)"
            )

    def _setup_network(self):
        self.net = self._create_control_vace_dit(device="meta")
        self.net.to_empty(device="cuda")
        self.net.init_weights()
        self._fix_rope_buffers(self.net, "net")

        # Load predict2.5 checkpoint into base blocks
        checkpoint_path = None
        if self.hparams.checkpoint_path:
            checkpoint_path = self.hparams.checkpoint_path
        elif self.hparams.checkpoint_uuid:
            uuid = self.hparams.checkpoint_uuid
            if uuid not in _CHECKPOINT_RESOLVED:
                # Only rank 0 downloads to avoid DDP bandwidth contention and file lock issues.
                # Other ranks wait for rank 0 to cache globally, then reuse.
                if get_local_rank() == 0:
                    from cosmos_predict2._src.imaginaire.flags import INTERNAL
                    from huggingface_hub import hf_hub_download
                    
                    if INTERNAL:
                        # NVidia cluster: try S3 via checkpoint_db first, fallback to HuggingFace.
                        try:
                            _CHECKPOINT_RESOLVED[uuid] = download_checkpoint(uuid)
                        except Exception as e:
                            log.info(
                                f"[Setup] Checkpoint checkpoint_db download failed ({type(e).__name__}); "
                                f"falling back to HuggingFace download …"
                            )
                            try:
                                checkpoint_path = _hf_download(
                                    repo_id="nvidia/Cosmos-Predict2.5-2B",
                                    filename="base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",
                                )
                                _CHECKPOINT_RESOLVED[uuid] = checkpoint_path
                            except Exception as e2:
                                log.error(f"[Setup] Both checkpoint_db and HuggingFace downloads failed. Error: {e2}")
                                raise
                    else:
                        # External machine: go straight to HuggingFace (avoid S3 timeout).
                        log.info("[Setup] External mode (INTERNAL=False): downloading checkpoint from HuggingFace …")
                        try:
                            checkpoint_path = _hf_download(
                                repo_id="nvidia/Cosmos-Predict2.5-2B",
                                filename="base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",
                            )
                            _CHECKPOINT_RESOLVED[uuid] = checkpoint_path
                        except Exception as e:
                            log.error(f"[Setup] HuggingFace checkpoint download failed. Error: {e}")
                            raise
                else:
                    # Non-rank-0 processes: wait for rank 0 to download (max 30min for large checkpoints).
                    import time
                    max_wait = 1800  # seconds
                    start_time = time.time()
                    while uuid not in _CHECKPOINT_RESOLVED:
                        if time.time() - start_time > max_wait:
                            log.error(f"[Setup] Rank {get_local_rank()} timeout waiting for rank 0 to download checkpoint")
                            raise TimeoutError("Checkpoint download timeout on rank 0")
                        time.sleep(1.0)
            checkpoint_path = _CHECKPOINT_RESOLVED[uuid]

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "net." in list(state_dict.keys())[0]:
                state_dict = {k.replace("net.", ""): v for k, v in state_dict.items() if k.startswith("net.")}
            _ = non_strict_load_model(self.net, state_dict)
            if get_local_rank() == 0:
                log.info(f"[Transfer] Loaded base checkpoint: {checkpoint_path}")
        else:
            if get_local_rank() == 0:
                log.warning("[Transfer] No checkpoint loaded!")

        # Copy base block weights to control blocks
        self._copy_weights_to_control_branch()

        # Freeze base model if requested
        if self.hparams.freeze_base:
            self._freeze_base_model()
        else:
            self.net.train()
            self.net.requires_grad_(True)

    def _setup_ema(self):
        if self.hparams.enable_ema and get_rank() == 0:
            # Only rank 0 maintains EMA — DDP guarantees identical weights.
            self.net_ema = self._create_control_vace_dit(device="meta")
            self.net_ema.to_empty(device="cpu")
            self.net_ema.init_weights()
            self._fix_rope_buffers(self.net_ema, "net_ema")
            self.net_ema.to(dtype=torch.float32)
            self.net_ema.eval()
            self.net_ema.requires_grad_(False)

            self.ema_updater = FastEmaModelUpdater()
            s = self.hparams.ema_rate
            self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

            with torch.no_grad():
                for p_ema, p_net in zip(self.net_ema.parameters(), self.net.parameters()):
                    p_ema.data.copy_(p_net.data.to("cpu"))
        else:
            self.net_ema = None
            self.ema_updater = None
            self.ema_exp_coefficient = None

    def _setup_renderer(self):
        """DiffDRR renderer setup (instantiated on demand per volume)."""
        pass

    def _setup_text_encoder(self):
        encoder = None
        try:
            from cosmos_predict2._src.predict2.text_encoders.text_encoder import (
                TextEncoder, TextEncoderConfig,
            )
            ckpt = self.hparams.text_encoder_ckpt or (
                "s3://bucket/cosmos_reasoning1/sft_exp700/"
                "sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/"
                "checkpoints/iter_000016000/model/"
            )
            dev = self.hparams.text_encoder_device or "cuda:3"
            cfg = TextEncoderConfig(ckpt_path=ckpt, compute_online=True,
                                    embedding_concat_strategy="full_concat")
            # Mask torch.distributed.is_initialized so QwenModel.__init__ skips
            # init_mesh/parallelize_qwen — those wrap layers with DTensor sharding
            # across all DDP ranks, causing collective hangs at inference time.
            import torch.distributed as _dist
            _orig_is_init = _dist.is_initialized
            _dist.is_initialized = lambda: False
            try:
                encoder = TextEncoder(config=cfg, device=dev)
            finally:
                _dist.is_initialized = _orig_is_init
            # Provide a null world_mesh so cp_mesh / tp_mesh properties return None
            # instead of raising AttributeError when called during forward.
            class _NullMesh:
                mesh_dim_names = None
                def __getitem__(self, key): return None
            encoder.model.world_mesh = _NullMesh()
            if get_local_rank() == 0:
                log.info(f"[Transfer] Text encoder loaded on {dev}")
        except Exception as e:
            if get_local_rank() == 0:
                log.warning(f"[Transfer] Text encoder unavailable ({e})")
        object.__setattr__(self, "_text_encoder", encoder)

    def _ensure_tokenizer_device(self):
        if not hasattr(self, "_tokenizer_device_set"):
            self._tokenizer_device_set = False
        target_device = self.device
        if hasattr(self.tokenizer, "model") and hasattr(self.tokenizer.model, "model"):
            vae_module = self.tokenizer.model.model
            try:
                first_param = next(vae_module.parameters())
                if first_param.device != target_device:
                    vae_module.to(target_device)
                    self.tokenizer.model.device = target_device
                    for attr in ("mean", "std", "img_mean", "img_std", "video_mean", "video_std"):
                        if hasattr(self.tokenizer.model, attr):
                            val = getattr(self.tokenizer.model, attr)
                            if isinstance(val, torch.Tensor):
                                setattr(self.tokenizer.model, attr, val.to(target_device))
                    if hasattr(self.tokenizer.model, "scale") and isinstance(self.tokenizer.model.scale, list):
                        self.tokenizer.model.scale = [
                            s.to(target_device) if isinstance(s, torch.Tensor) else s
                            for s in self.tokenizer.model.scale
                        ]
                    self._tokenizer_device_set = True
            except StopIteration:
                pass

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        self._ensure_tokenizer_device()
        return self.tokenizer.encode(video.float()).float()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self._ensure_tokenizer_device()
        video = self.tokenizer.decode(latent.float()).float()
        if not self.training and not torch.is_grad_enabled():
            v_min, v_max = video.min().item(), video.max().item()
            assert v_min >= -1.5 and v_max <= 1.5, f"VAE range: [{v_min:.3f}, {v_max:.3f}]"
        return video

    def _sample_shared_camera_params(self, batch_size: int, stage: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample shared camera parameters (FOV and camera-to-object distance once per batch.
        
        These shared values are used by both frontal and target views
        to ensure multiview consistency.

        Args:
            batch_size: Number of samples per batch.
            stage: "train" or "val" - determines sampling strategy.

        Returns:
            Tuple of (fov, distance) tensors, each of shape (batch_size,).
        """
        device = self.device
        
        fov = XRAY_FOV_DEFAULT * torch.ones(batch_size, device=device)
        distance = XRAY_DISTANCE_DEFAULT * torch.ones(batch_size, device=device)

        if stage == "train":
            fov += torch.randint_like(fov, low=-2, high=3, device=device)
            distance += torch.rand_like(distance, device=device) / 4.0 - 0.125

        return fov, distance

    def _sample_multiview_camera_params(
        self, batch_size: int, stage: str = "train", fov: Optional[torch.Tensor] = None, distance: Optional[torch.Tensor] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate per-view camera parameters by combining shared intrinsics with view-specific extrinsics.

        Args:
            batch_size: Number of samples per batch.
            stage: "train" or "val" - determines sampling strategy.
            fov: Pre-sampled FOV tensor. If None, samples new values.
            distance: Pre-sampled distance tensor. If None, samples new values.

        Returns:
            List of dicts, one per view, each containing azimuth, elevation, distance, fov tensors.
        """
        device = self.device
        n_views = self.hparams.views_per_batch
        
        # If not provided, sample new FOV/distance
        if fov is None or distance is None:
            fov, distance = self._sample_shared_camera_params(batch_size, stage)
        
        # Randomly sample views when views_per_batch < total views (training)
        if n_views < NUM_XRAY_VIEWS and stage == "train":
            selected = torch.randperm(NUM_XRAY_VIEWS)[:n_views].sort().values.tolist()
        else:
            selected = list(range(min(n_views, NUM_XRAY_VIEWS)))

        all_view_params = []
        for v_idx in selected:
            view_name = XRAY_CAMERAS[v_idx]
            ext = XRAY_EXTRINSICS[view_name]
            all_view_params.append({
                "azimuth": torch.full((batch_size,), ext["azimuth"], device=device),
                "elevation": torch.full((batch_size,), ext["elevation"], device=device),
                "distance": distance.clone(),
                "fov": fov.clone(),
                "view_name": view_name,
                "view_index": v_idx,
            })
        return all_view_params

    @torch.no_grad()
    def _render_xray(self, ct_volume, camera_params):
        B = ct_volume.shape[0]
        xrays = []
        for i in range(B):
            renderer = DiffDRRVolumeRenderer(
                image_width=256,
                image_height=256,
                n_pts_per_ray=self.hparams.renderer_n_pts_per_ray,
                ndc_extent=self.hparams.renderer_ndc_extent,
                device=self.device,
            )
            renderer.set_volume(ct_volume[i:i + 1])
            proj = renderer.render(
                azimuth=camera_params["azimuth"][i],
                elev=camera_params["elevation"][i],
                dist=camera_params["distance"][i],
                fov=camera_params["fov"][i],
                min_depth=self.hparams.renderer_min_depth,
                max_depth=self.hparams.renderer_max_depth,
                norm_type="minimized",
            )
            xrays.append(proj)
        return torch.cat(xrays, dim=0)

    @torch.no_grad()
    def _render_multiview_xrays(self, ct_volume, all_view_params):
        return [self._render_xray(ct_volume, vp) for vp in all_view_params]

    def _construct_93_frame_tensor(self, frontal_xray, target_xray):
        nf = self.hparams.num_frontal_frames
        nt = NUM_FRAMES - nf
        f3 = frontal_xray.expand(-1, 3, -1, -1)
        t3 = target_xray.expand(-1, 3, -1, -1)
        return torch.cat([
            f3.unsqueeze(2).expand(-1, -1, nf, -1, -1),
            t3.unsqueeze(2).expand(-1, -1, nt, -1, -1),
        ], dim=2)

    @torch.no_grad()
    def _generate_control_signals(
        self,
        all_target_xrays: List[torch.Tensor],
        ct_volume: torch.Tensor = None,
    ) -> List[torch.Tensor]:
        controls = []
        for target_xr in all_target_xrays:
            ctrl = generate_control_signal(
                drr_image=target_xr,
                ct_volume=ct_volume,
                control_type=self.hparams.control_type,
            )
            controls.append(ctrl)
        return controls

    def _build_control_video(self, control_signal: torch.Tensor) -> torch.Tensor:
        """
        Build 93-frame control video and encode to latent space.

        Control signal (B,1,H,W) → expand to 3ch → repeat 93 frames
        → encode through VAE → 16ch latent control input.
        """
        B, C, H, W = control_signal.shape
        nf = self.hparams.num_frontal_frames
        nt = NUM_FRAMES - nf

        # Expand to 3 channels for VAE
        ctrl_3ch = control_signal.expand(-1, 3, -1, -1)

        # Zero frames for frontal conditioning region + control for target region
        frontal_zeros = torch.zeros(B, 3, nf, H, W, device=control_signal.device,
                                     dtype=control_signal.dtype)
        target_ctrl = ctrl_3ch.unsqueeze(2).expand(-1, -1, nt, -1, -1)
        ctrl_video = torch.cat([frontal_zeros, target_ctrl], dim=2)  # (B, 3, 93, H, W)

        # Encode to latent space: (B, 3, 93, 256, 256) → (B, 16, T_lat, H_lat, W_lat)
        ctrl_video_norm = ctrl_video * 2.0 - 1.0  # to [-1, 1]
        latent_control = self.encode(ctrl_video_norm)  # (B, 16, T_lat, H_lat, W_lat)
        return latent_control

    def _generate_multiview_prompts(self, all_view_params):
        """
        Generate detailed text prompts describing camera pose and rendering parameters for each view.

        Args:
            all_view_params: List of dicts from _sample_multiview_camera_params, one per view.

        Returns:
            Nested list: outer list per view, inner list per batch sample within that view.
        """
        all_prompts = []
        for vp in all_view_params:
            view_name = vp["view_name"]
            prefix = XRAY_CAPTION_PREFIXES[view_name]
            B = vp["azimuth"].shape[0]
            view_prompts = []
            for i in range(B):
                prompt = XRAY_PROMPT_TEMPLATE.format(
                    prefix=prefix,
                    azimuth=vp["azimuth"][i].item(),
                    elevation=vp["elevation"][i].item(),
                    distance=vp["distance"][i].item(),
                    fov=vp["fov"][i].item(),
                    znear=self.hparams.renderer_min_depth,
                    zfar=self.hparams.renderer_max_depth,
                )
                view_prompts.append(prompt)
            all_prompts.append(view_prompts)
        return all_prompts

    @torch.no_grad()
    def _encode_prompts(self, prompts):
        B = len(prompts)
        encoder = getattr(self, "_text_encoder", None)
        if encoder is not None:
            try:
                te_device = self.hparams.text_encoder_device or "cuda:3"
                with torch.cuda.device(te_device):
                    emb = encoder.compute_text_embeddings_online(
                        {"text": prompts}, input_caption_key="text",
                    ).detach().to(dtype=torch.float32, device=self.device).contiguous()
                seq = emb.shape[1]
                if seq < CR1_MAX_LENGTH:
                    pad = torch.zeros(B, CR1_MAX_LENGTH - seq, CR1_EMBEDDING_DIM,
                                      device=self.device, dtype=torch.float32)
                    emb = torch.cat([emb, pad], dim=1)
                elif seq > CR1_MAX_LENGTH:
                    emb = emb[:, :CR1_MAX_LENGTH, :]
                return emb
            except Exception as e:
                if get_local_rank() == 0:
                    log.warning(f"[encode_prompts] fallback: {e}")
        return torch.zeros(B, CR1_MAX_LENGTH, CR1_EMBEDDING_DIM,
                           device=self.device, dtype=torch.float32)

    def _process_batch_multiview(self, batch: dict, stage: str = "train") -> dict:
        """
        Process a batch of CT volumes to generate multiview training data with control signals.
        
        Renders frontal X-ray (reference view) and multiple target views from sampled
        camera parameters, ensuring all views share the same intrinsic parameters
        (FOV, depth range) but differ in extrinsic parameters (azimuth, elevation).
        Generates control signals (e.g., edge maps) for ControlNet conditioning.

        Args:
            batch: Dict with key "ct" containing CT volume tensor of shape (B, 1, D, H, W).
            stage: "train" or "val" - determines sampling and augmentation strategy.

        Returns:
            Dict containing:
                - video: Concatenated 93-frame videos across all views (B*V, 3, 93, H, W)
                - latent_control_input: Encoded control signals for ControlNet
                - text_embeddings: Encoded text prompts for all samples
                - view_indices: View index per sample
                - camera_params: Camera extrinsics/intrinsics for each sample
                - prompts: Raw prompt strings
                - frontal_xray: Reference X-ray images
                - target_xray: Target X-ray images for each view
                - control_images: Control signal images (e.g., edge maps)
        """
        ct_volume = batch["ct"].to(device=self.device, dtype=torch.float32)
        B = ct_volume.shape[0]

        # Sample FOV + distance ONCE for the batch (used by both frontal and target views)
        fov, distance = self._sample_shared_camera_params(B, stage=stage)

        # Generate frontal X-ray with frontal camera params (azimuth=0, elevation=0)
        # Share same FOV/distance as target views
        frontal_params = {
            "azimuth": torch.zeros(B, device=self.device),
            "elevation": torch.zeros(B, device=self.device),
            "distance": distance.clone(),
            "fov": fov.clone(),
        }
        frontal_xray = self._render_xray(ct_volume, frontal_params)

        # Target views use same FOV/distance as frontal
        all_view_params = self._sample_multiview_camera_params(B, stage=stage, fov=fov, distance=distance)
        all_target_xrays = self._render_multiview_xrays(ct_volume, all_view_params)
        all_prompts = self._generate_multiview_prompts(all_view_params)
        all_controls = self._generate_control_signals(all_target_xrays, ct_volume)

        videos, frontals, targets, prompts_flat = [], [], [], []
        latent_controls, view_indices = [], []
        control_images_flat = []
        cam_flat = {k: [] for k in ("azimuth", "elevation", "distance", "fov")}

        for v_idx, (vp, tgt, v_prompts, ctrl) in enumerate(
            zip(all_view_params, all_target_xrays, all_prompts, all_controls)
        ):
            video = self._construct_93_frame_tensor(frontal_xray, tgt)
            latent_ctrl = self._build_control_video(ctrl)

            videos.append(video)
            frontals.append(frontal_xray)
            targets.append(tgt)
            latent_controls.append(latent_ctrl)
            control_images_flat.append(ctrl)
            prompts_flat.extend(v_prompts)
            view_indices.append(torch.full((B,), v_idx, device=self.device, dtype=torch.long))
            for k in cam_flat:
                cam_flat[k].append(vp[k])

        return {
            "video": torch.cat(videos, dim=0),
            "latent_control_input": torch.cat(latent_controls, dim=0),
            "text_embeddings": self._encode_prompts(prompts_flat),
            "view_indices": torch.cat(view_indices, dim=0),
            "camera_params": {k: torch.cat(v, dim=0) for k, v in cam_flat.items()},
            "prompts": prompts_flat,
            "frontal_xray": torch.cat(frontals, dim=0),
            "target_xray": torch.cat(targets, dim=0),
            "control_images": torch.cat(control_images_flat, dim=0),
        }

    def _apply_cfg_dropout_per_sample(self, text_embeddings, dropout_rate=None):
        if dropout_rate is None:
            dropout_rate = self.hparams.cfg_dropout_rate
        B = text_embeddings.shape[0]
        device = text_embeddings.device
        if dropout_rate <= 0.0 or not self.training:
            return text_embeddings, torch.ones(B, dtype=torch.bool, device=device)
        keep = torch.bernoulli((1.0 - dropout_rate) * torch.ones(B, device=device))
        return text_embeddings * keep.view(B, 1, 1), keep.bool()

    def _get_condition(self, text_embeddings, latent,
                       num_conditional_frames=None, use_video_condition=True,
                       dtype=None, apply_cfg_dropout=False, **_kw):
        B, C, T, H, W = latent.shape
        device = latent.device
        if dtype is None:
            dtype = next(self.net.parameters()).dtype
        conditional_frames_probs = None
        if apply_cfg_dropout and self.training:
            text_embeddings, _ = self._apply_cfg_dropout_per_sample(text_embeddings)
            if use_video_condition:
                dr = self.hparams.cfg_dropout_rate
                mn, mx = self.hparams.min_num_conditional_frames, self.hparams.max_num_conditional_frames
                n = mx - mn + 1
                kp = (1.0 - dr) / n
                conditional_frames_probs = {0: dr}
                for nf in range(mn, mx + 1):
                    conditional_frames_probs[nf] = kp
        H_px, W_px = H * 8, W * 8
        padding_mask = torch.zeros(B, 1, H_px, W_px, device=device, dtype=dtype)
        fps = torch.full((B,), 24.0, device=device, dtype=dtype)
        base = Video2WorldCondition(
            crossattn_emb=text_embeddings.to(device=device, dtype=dtype),
            fps=fps, padding_mask=padding_mask,
            data_type=DataType.VIDEO, use_video_condition=use_video_condition,
        )
        return base.set_video_condition(
            gt_frames=latent.to(dtype=dtype),
            random_min_num_conditional_frames=self.hparams.min_num_conditional_frames,
            random_max_num_conditional_frames=self.hparams.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=conditional_frames_probs,
        )

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Video2WorldCondition,
        latent_control_input: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict velocity using MinimalV4LVGControlVaceDiT.

        The control signal is passed as latent_control_input directly to
        the network's forward() — the model internally routes it through
        control_embedder → control_blocks → hints → base blocks.
        """
        model_dtype = next(self.net.parameters()).dtype
        B, C, T, H, W = xt_B_C_T_H_W.shape
        condition_video_mask = None

        gt_frames = condition.gt_frames
        mask = condition.condition_video_input_mask_B_C_T_H_W

        if condition.is_video and gt_frames is not None and mask is not None:
            cs = gt_frames.type_as(xt_B_C_T_H_W)
            use_vc = condition.use_video_condition
            if isinstance(use_vc, torch.Tensor):
                assert bool((use_vc == use_vc[0]).all().item())
                use_vc = bool(use_vc[0].item())
            if not use_vc:
                cs = cs * 0
            condition_video_mask = mask.repeat(1, C, 1, 1, 1).type_as(xt_B_C_T_H_W)
            xt_B_C_T_H_W = cs * condition_video_mask + xt_B_C_T_H_W * (1 - condition_video_mask)
            cft = getattr(self.hparams, "conditional_frame_timestep", -1.0)
            if cft >= 0:
                cvm = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                tsc = torch.ones_like(cvm) * cft
                timesteps_B_T = tsc * cvm + timesteps_B_T.view(B, 1, 1, 1, 1) * (1 - cvm)
                timesteps_B_T = timesteps_B_T.squeeze()
                if timesteps_B_T.ndim == 1:
                    timesteps_B_T = timesteps_B_T.unsqueeze(0)

        # Build kwargs for MinimalV4LVGControlVaceDiT.forward()
        cond_dict = condition.to_dict()
        forward_kwargs = {
            "x_B_C_T_H_W": xt_B_C_T_H_W.to(device=self.device, dtype=model_dtype),
            "timesteps_B_T": timesteps_B_T.to(device=self.device, dtype=model_dtype),
            "crossattn_emb": cond_dict["crossattn_emb"],
            "latent_control_input": (
                latent_control_input.to(device=self.device, dtype=model_dtype)
                if latent_control_input is not None
                else torch.zeros(B, C, T, H, W, device=self.device, dtype=model_dtype)
            ),
            "control_context_scale": self.hparams.control_context_scale,
        }
        # Pass through optional condition fields
        for key in ("condition_video_input_mask_B_C_T_H_W", "fps", "padding_mask",
                     "data_type", "img_context_emb"):
            if key in cond_dict:
                forward_kwargs[key] = cond_dict[key]

        net_out = self.net(**forward_kwargs)
        if isinstance(net_out, (tuple, list)):
            net_out = net_out[0]
        net_out = net_out.float()

        # Replace velocity for conditioning frames with GT
        if condition.is_video and condition_video_mask is not None:
            gt_x0 = condition.gt_frames.type_as(net_out)
            gt_vel = noise.type_as(net_out) - gt_x0
            net_out = gt_vel * condition_video_mask + net_out * (1 - condition_video_mask)

        return net_out

    def ema_beta(self, iteration: int) -> float:
        iteration = iteration + self.hparams.ema_iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    @contextlib.contextmanager
    def ema_scope(self):
        # Only rank 0 has up-to-date EMA; other ranks use regular net for validation
        if self.hparams.enable_ema and self.net_ema is not None and get_rank() == 0:
            if self.global_step >= self.hparams.ema_iteration_shift:
                orig_net = self.net
                orig_dtype = next(self.net.parameters()).dtype
                self.net_ema.to(device=self.device, dtype=orig_dtype)
                self.net = self.net_ema
                try:
                    yield
                finally:
                    self.net = orig_net
                    if self.hparams.ema_offload_cpu:
                        self.net_ema.to(dtype=torch.float32, device="cpu")
            else:
                yield
        else:
            yield

    @contextlib.contextmanager
    def ema_scope_generation(self):
        use_ema = (
            self.hparams.enable_ema
            and self.net_ema is not None
            and self.global_step >= self.hparams.ema_iteration_shift
        )
        if use_ema:
            orig_net = self.net
            self.net_ema.to(device=self.device, dtype=torch.bfloat16)
            self.net = self.net_ema
            try:
                yield
            finally:
                self.net = orig_net
                if self.hparams.ema_offload_cpu:
                    self.net_ema.to(dtype=torch.float32, device="cpu")
        else:
            orig_dtype = next(self.net.parameters()).dtype
            self.net.to(dtype=torch.bfloat16)
            try:
                yield
            finally:
                self.net.to(dtype=orig_dtype)

    def on_before_zero_grad(self, optimizer):
        if self.hparams.enable_ema and self.net_ema is not None:
            # Only update EMA on rank 0: DDP guarantees identical weights.
            if get_rank() != 0:
                return
            beta = self.ema_beta(self.global_step)
            with torch.no_grad():
                if self.hparams.ema_offload_cpu:
                    # Update EMA directly on CPU — stream each tensor via .cpu()
                    # instead of moving the full 2.1B-param model to GPU.
                    for p_ema, p_net in zip(self.net_ema.parameters(), self.net.parameters()):
                        p_ema.data.mul_(beta).add_(p_net.data.to(p_ema.device), alpha=1.0 - beta)
                else:
                    # EMA lives on GPU — update in place
                    self.ema_updater.update_average(self.net, self.net_ema, beta=beta)

    def configure_optimizers(self):
        trainable_params = [p for p in self.net.parameters() if p.requires_grad]

        optimizer = get_base_optimizer(
            model=self.net,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            optim_type="fusedadam",
            betas=(0.9, 0.99), eps=1e-8,
            master_weights=True, capturable=True,
        )
        warmup_steps = min(self.hparams.warmup_steps, self.hparams.max_iters // 5)
        warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=0.5, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=self.hparams.max_iters - warmup_steps,
                                    eta_min=self.hparams.learning_rate * 0.2)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_train_start(self):
        self._ensure_tokenizer_device()
        if self.hparams.enable_ema and self.net_ema is not None:
            if self.hparams.ema_offload_cpu:
                self.net_ema.to(dtype=torch.float32)
            else:
                self.net_ema.to(device=self.device, dtype=torch.float32)
        if get_local_rank() == 0:
            log.info("[Transfer] Training started (step=0)")

    def on_save_checkpoint(self, checkpoint: dict):
        if self.hparams.enable_ema and self.net_ema is not None:
            checkpoint["net_ema"] = self.net_ema.state_dict()
            checkpoint["ema_exp_coefficient"] = self.ema_exp_coefficient

    def on_load_checkpoint(self, checkpoint: dict):
        if self.hparams.enable_ema and self.net_ema is not None:
            if "net_ema" in checkpoint:
                try:
                    self.net_ema.load_state_dict(checkpoint["net_ema"], strict=False)
                except Exception as e:
                    if get_local_rank() == 0:
                        log.warning(f"Could not load EMA weights: {e}")
            if "ema_exp_coefficient" in checkpoint:
                self.ema_exp_coefficient = checkpoint["ema_exp_coefficient"]

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        result = self._process_batch_multiview(batch, stage="train")
        video = result["video"]
        text_embeddings = result["text_embeddings"]
        latent_control = result["latent_control_input"]

        video_norm = video * 2.0 - 1.0
        x_1 = self.encode(video_norm)

        B_eff = x_1.shape[0]
        tk = {"device": x_1.device, "dtype": torch.float32}

        epsilon = torch.randn(x_1.size(), **tk)
        t_B = self.rectified_flow.sample_train_time(B_eff).to(**tk).view(B_eff, 1)
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, tk).view(B_eff, 1)
        sigmas = self.rectified_flow.get_sigmas(timesteps, tk).view(B_eff, 1)

        xt, vt = self.rectified_flow.get_interpolation(epsilon, x_1.float(), sigmas)

        condition = self._get_condition(text_embeddings, x_1, None, True, apply_cfg_dropout=True)

        vt_pred = self.denoise(
            noise=epsilon, xt_B_C_T_H_W=xt,
            timesteps_B_T=timesteps, condition=condition,
            latent_control_input=latent_control,
        )

        time_w = self.rectified_flow.train_time_weight(timesteps, tk)
        per_loss = torch.mean((vt_pred - vt) ** 2, dim=list(range(1, vt_pred.dim())))
        loss = torch.mean(time_w * per_loss) * self.hparams.loss_scale

        cam = result["camera_params"]
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=B_eff)
        self.log("train/mse", per_loss.mean(), on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=B_eff)

        output_batch = {
            "x_1": x_1.detach().cpu(),
            "loss": loss.detach().cpu(),
            "camera_params": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in cam.items()},
            "view_indices": result["view_indices"].detach().cpu(),
            "prompts": result["prompts"],
            "frontal_xray": result["frontal_xray"].detach().cpu(),
            "target_xray": result["target_xray"].detach().cpu(),
            "control_images": result["control_images"].detach().cpu(),
        }
        return {"loss": loss, "output_batch": output_batch}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        result = self._process_batch_multiview(batch, stage="val")
        video = result["video"]
        text_embeddings = result["text_embeddings"]
        latent_control = result["latent_control_input"]

        video_norm = video * 2.0 - 1.0
        x_1 = self.encode(video_norm)

        B_eff = x_1.shape[0]
        tk = {"device": x_1.device, "dtype": torch.float32}

        epsilon = torch.randn(x_1.size(), **tk)
        t_B = self.rectified_flow.sample_train_time(B_eff).to(**tk).view(B_eff, 1)
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, tk).view(B_eff, 1)
        sigmas = self.rectified_flow.get_sigmas(timesteps, tk).view(B_eff, 1)

        xt, vt = self.rectified_flow.get_interpolation(epsilon, x_1.float(), sigmas)
        condition = self._get_condition(text_embeddings, x_1, 1, True)

        with self.ema_scope():
            vt_pred = self.denoise(
                noise=epsilon, xt_B_C_T_H_W=xt,
                timesteps_B_T=timesteps, condition=condition,
                latent_control_input=latent_control,
            )

        time_w = self.rectified_flow.train_time_weight(timesteps, tk)
        per_loss = torch.mean((vt_pred - vt) ** 2, dim=list(range(1, vt_pred.dim())))
        loss = torch.mean(time_w * per_loss)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=B_eff)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=False,
                 sync_dist=True, batch_size=B_eff)

        output_batch = {
            "x_1": x_1.detach().cpu(),
            "loss": loss.detach().cpu(),
            "camera_params": {k: v.detach().cpu() if torch.is_tensor(v) else v
                              for k, v in result["camera_params"].items()},
            "view_indices": result["view_indices"].detach().cpu(),
            "prompts": result["prompts"],
            "frontal_xray": result["frontal_xray"].detach().cpu(),
            "target_xray": result["target_xray"].detach().cpu(),
            "control_images": result["control_images"].detach().cpu(),
        }
        return {"loss": loss, "output_batch": output_batch}

    @torch.inference_mode()
    def generate(
        self,
        image: Optional[Union[torch.Tensor]] = None,
        view_name: str = "xray_ap",
        control_image: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        num_steps: int = None,
        guidance_scale: float = None,
        control_context_scale: float = None,
        seed: int = None,
        shift: float = None,
        num_conditional_frames: int = 1,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Generate video for a view with control signal.

        Args:
            image: (B, C, H, W) frontal X-ray in [0,1].
            view_name: Target view name from XRAY_CAMERAS.
            control_image: (B, 1, H, W) control signal in [0,1].
            prompt: Text prompt (auto-generates if None).
            control_context_scale: Override control scale for this call.
        """
        if image is None:
            raise ValueError("image cannot be None")

        ext = XRAY_EXTRINSICS[view_name]
        fov_mid = sum(XRAY_FOV_RANGE) / 2.0
        dist_mid = sum(XRAY_DISTANCE_RANGE) / 2.0
        cam_params = {
            "azimuth": ext["azimuth"], "elevation": ext["elevation"],
            "distance": dist_mid, "fov": fov_mid,
        }

        if prompt is None:
            prompt = XRAY_PROMPT_TEMPLATE.format(
                prefix=XRAY_CAPTION_PREFIXES[view_name],
                azimuth=cam_params["azimuth"], elevation=cam_params["elevation"],
                distance=cam_params["distance"], fov=cam_params["fov"],
                znear=self.hparams.renderer_min_depth,
                zfar=self.hparams.renderer_max_depth,
            )

        text_embeddings = self._encode_prompts([prompt])

        num_steps = num_steps or self.hparams.num_inference_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.hparams.guidance_scale
        ccs = control_context_scale if control_context_scale is not None else self.hparams.control_context_scale
        shift = shift or self.hparams.rf_shift
        seed = seed if seed is not None else torch.randint(0, 2**32 - 1, (1,)).item()

        # Build latent control input
        latent_ctrl = None
        if control_image is not None:
            latent_ctrl = self._build_control_video(control_image)

        # Temporarily override control scale
        orig_scale = self.hparams.control_context_scale
        self.hparams.control_context_scale = ccs

        with self.ema_scope_generation():
            model = self.net
            was_training = model.training
            model.eval()
            gen_dtype = next(model.parameters()).dtype

            B, C_in, H_in, W_in = image.shape
            device = image.device
            input_norm = image.to(dtype=gen_dtype) * 2.0 - 1.0
            input_video = input_norm.unsqueeze(2)
            zeros_pad = torch.zeros(B, C_in, NUM_FRAMES - 1, H_in, W_in,
                                    device=device, dtype=gen_dtype)
            input_video = torch.cat([input_video, zeros_pad], dim=2)

            latent_cond = self.encode(input_video.to(device)).to(dtype=gen_dtype)
            _, C, T, H, W = latent_cond.shape

            condition = self._get_condition(
                text_embeddings, latent_cond, num_conditional_frames, True, dtype=gen_dtype,
            )
            uncondition = self._get_condition(
                torch.zeros_like(text_embeddings), latent_cond,
                num_conditional_frames, True, dtype=gen_dtype,
            )

            noise = arch_invariant_rand(
                shape=(B, C, T, H, W), dtype=torch.float32,
                device=device, seed=seed,
            ).to(dtype=gen_dtype)

            seed_gen = torch.Generator(device=device)
            seed_gen.manual_seed(seed)

            self.sample_scheduler.set_timesteps(
                num_inference_steps=num_steps, device=device,
                shift=shift, use_kerras_sigma=False,
            )

            initial_noise = noise
            latents = noise.clone()

            for i, t in enumerate(self.sample_scheduler.timesteps):
                ts = t.view(1, 1).expand(B, 1)
                v_cond = self.denoise(initial_noise, latents, ts, condition,
                                      latent_control_input=latent_ctrl)
                v_uncond = self.denoise(initial_noise, latents, ts, uncondition,
                                        latent_control_input=latent_ctrl)
                vel = v_uncond + guidance_scale * (v_cond - v_uncond)
                latents = self.sample_scheduler.step(
                    model_output=vel, timestep=t, sample=latents,
                    return_dict=False, generator=seed_gen,
                )[0]
                cond_mask = condition.condition_video_input_mask_B_C_T_H_W
                if cond_mask is not None:
                    latents = latents * (1 - cond_mask) + latent_cond * cond_mask

            self.hparams.control_context_scale = orig_scale

            video = self.decode(latents.float())
            video = (video / 2.0 + 0.5).clamp(0, 1)

            if was_training:
                model.train()

            return video
