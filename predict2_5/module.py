"""Cosmos-Predict 2.5 Multiview X-Ray Synthesis."""

from shared.utils import setup_early_logging
setup_early_logging()

import contextlib
from typing import Dict, List, Literal, Optional, Union

import numpy as np

import torch
torch.set_float32_matmul_precision("high")

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from lightning import LightningModule

from shared.utils import get_logger
log = get_logger(__name__)

from cosmos_predict2._src.imaginaire.utils.ema import FastEmaModelUpdater
from cosmos_predict2._src.imaginaire.utils.checkpointer import non_strict_load_model
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import download_checkpoint
from cosmos_predict2._src.predict2.utils.optim_instantiate import get_base_optimizer

from cosmos_oss.checkpoints_predict2 import register_checkpoints as _register_cosmos_checkpoints
_register_cosmos_checkpoints()

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_predict2._src.predict2.schedulers.rectified_flow import RectifiedFlow
from cosmos_predict2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler

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
    XRAY_VIEW_MAPPING,
    NUM_XRAY_VIEWS,
)

from renderers.diffdrr.renderer import DiffDRRVolumeRenderer

from shared.utils import get_local_rank, get_rank, get_world_size, sync_ema_ddp, arch_invariant_rand

from predict2_5.hf import hf_download as _hf_download, text_encoder_snapshot as _text_encoder_snapshot


# Global caches for resolved checkpoint and tokenizer paths to avoid redundant downloads across ranks.
_CHECKPOINT_RESOLVED = {}
_TOKENIZER_RESOLVED = {}


class CosmosXRay2XRayMultiview(LightningModule):
    """Cosmos-Predict 2.5 for multiview X-ray synthesis."""

    _setup_complete: bool = False
    _view_weights: Optional[torch.Tensor] = None

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
        # ── Classifier-Free Guidance ──
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
        # ── Distributed Training ──
        distributed_strategy: Literal["auto", "ddp", "fsdp"] = "auto",
        # ── X-Ray Rendering ──
        renderer_n_pts_per_ray: int = 1000,
        renderer_min_depth: float = 7.0,
        renderer_max_depth: float = 9.0,
        renderer_ndc_extent: float = 1.0,
        num_frontal_frames: int = 5,
        # ── Text Encoder ──
        text_encoder_device: Optional[str] = None,
        text_encoder_ckpt: Optional[str] = None,
        # ── Multiview-specific ──
        views_per_batch: int = 2,
        view_loss_weights: bool = False,
        prefer_difficult_views: bool = False,
    ):
        """
        Initialise the Cosmos-Predict 2.5 multiview X-ray synthesis module.

        Args:
            checkpoint_uuid: GCS/HF UUID for the pretrained DiT checkpoint.
            tokenizer_uuid: GCS/HF UUID for the Wan2pt1 VAE tokenizer.
            checkpoint_path: Local path override (skips HF download).
            tokenizer_path: Local path override (skips HF download).
            tokenizer_chunk_duration: VAE temporal input length in frames.
            tokenizer_temporal_window: VAE sliding-window size.
            model_size: Architecture size selector — ``"2B"``, ``"7B"``, or ``"14B"``.
            state_t: Number of latent time frames.
            state_ch: Latent channel count (must match VAE).
            learning_rate: Peak learning rate for AdamW.
            weight_decay: L2 regularisation coefficient.
            warmup_steps: Number of linear warmup steps.
            max_iters: Total training iterations (for cosine decay schedule).
            gradient_clip_val: Gradient norm clipping threshold.
            loss_scale: Loss multiplier applied before backward.
            num_inference_steps: Default denoising steps for ``generate()``.
            guidance_scale: Classifier-free guidance strength at inference.
            rf_shift: Rectified-flow logit-normal shift parameter.
            cfg_dropout_rate: Fraction of training samples with zeroed text embeddings.
            min_num_conditional_frames: Lower bound for randomly sampled condition frames.
            max_num_conditional_frames: Upper bound for randomly sampled condition frames.
            conditional_frame_timestep: Fixed sigma for condition frames (-1 = not overridden).
            enable_ema: Whether to maintain an EMA shadow model on rank 0.
            ema_rate: EMA decay "half-life" equivalent; drives exp_coefficient.
            ema_offload_cpu: Keep EMA weights on CPU to save GPU memory.
            ema_iteration_shift: Global step from which EMA updates begin.
            ema_sync_every_n_steps: How often to synchronise EMA across nodes.
            distributed_strategy: Lightning strategy hint (``"auto"``, ``"ddp"``, ``"fsdp"``).
            renderer_n_pts_per_ray: Ray-march sample count for the X-ray renderer.
            renderer_min_depth: Near clipping plane (metres) for renderer.
            renderer_max_depth: Far clipping plane (metres) for renderer.
            renderer_ndc_extent: NDC frustum half-extent.
            num_frontal_frames: How many times to tile the frontal conditioning frame.
            text_encoder_device: CUDA device string for the Cosmos-Reason1-7B encoder.
            text_encoder_ckpt: Path or GCS URI to text encoder weights.
            views_per_batch: Number of camera views to sample per training step.
                Recommended: 2-3 to avoid OOM. Use gradient_accumulate for effective 7-view training.
            view_loss_weights: When True, scales loss per-view by difficulty (cranial/oblique harder).
            prefer_difficult_views: When True, biases view sampling toward hard views during training.
        """
        super().__init__()

        # Save all hyperparameters to ``self.hparams`` for easy access and checkpointing.
        self.save_hyperparameters()

        # Initialize per-view difficulty weights for loss weighting
        self._init_view_weights()

        # Tensor kwargs for consistent device and data type when creating new tensors.
        self.tensor_kwargs = {"device": "cuda", "dtype": torch.float32}

        # Setup all the components
        self._setup_rectified_flow()
        self._setup_tokenizer()
        self._setup_network()
        self._setup_ema()
        self._setup_renderer()
        self._setup_text_encoder()

    def _setup_rectified_flow(self):
        """Initialise the RectifiedFlow velocity scheduler and UniPC sampler."""
        # Rectified Flow handler for training-time velocity scheduling
        self.rectified_flow = RectifiedFlow(
            velocity_field=lambda *a, **k: None,
            train_time_distribution="logitnormal",
            train_time_weight_method="uniform",
            use_dynamic_shift=False,
            shift=self.hparams.rf_shift,
            device=self.device if hasattr(self, "device") else torch.device("cpu"),
            dtype=torch.float32,
        )

        # Set number of training timesteps for the scheduler
        self._num_train_timesteps = self.rectified_flow.num_train_timesteps

        # The UniPC sampler used for inference
        self.sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, 
            shift=1, 
            use_dynamic_shifting=False,
        )

    def _setup_tokenizer(self):
        """Setup the Wan2pt1 VAE tokenizer interface."""
        global _TOKENIZER_RESOLVED

        if self.hparams.tokenizer_path:
            tokenizer_path = self.hparams.tokenizer_path
        else:
            uuid = self.hparams.tokenizer_uuid
            if uuid not in _TOKENIZER_RESOLVED:
                from cosmos_predict2._src.imaginaire.flags import INTERNAL
                
                # All ranks resolve the tokenizer path independently.
                if INTERNAL:
                    try:
                        tokenizer_path = download_checkpoint(uuid)
                    except Exception as e:
                        if get_local_rank() == 0:
                            log.info(
                                f"[Setup] Tokenizer checkpoint_db download failed ({type(e).__name__}); "
                                f"falling back to HuggingFace download …"
                            )
                        tokenizer_path = _hf_download(
                            repo_id="nvidia/Cosmos-Predict2.5-2B",
                            filename="tokenizer.pth",
                        )
                else:
                    if get_local_rank() == 0:
                        log.info("[Setup] External mode (INTERNAL=False): downloading tokenizer from HuggingFace …")
                    tokenizer_path = _hf_download(
                        repo_id="nvidia/Cosmos-Predict2.5-2B",
                        filename="tokenizer.pth",
                    )
                _TOKENIZER_RESOLVED[uuid] = tokenizer_path
            else:
                tokenizer_path = _TOKENIZER_RESOLVED[uuid]

        self.tokenizer = Wan2pt1VAEInterface(
            chunk_duration=self.hparams.tokenizer_chunk_duration,
            load_mean_std=False,
            vae_pth=tokenizer_path,
            temporal_window=self.hparams.tokenizer_temporal_window,
            keep_decoder_cache=False,
            keep_encoder_cache=False,
        )

        # Verify latent channel count matches the DiT's expected input dimension
        assert self.tokenizer.latent_ch == self.hparams.state_ch

        if get_local_rank() == 0:
            log.info(
                f"[Setup] VAE tokenizer loaded — path={tokenizer_path}, "
                f"latent_ch={self.tokenizer.latent_ch}, "
                f"chunk_dur={self.hparams.tokenizer_chunk_duration}, "
                f"temporal_window={self.hparams.tokenizer_temporal_window}"
            )

    def _get_model_config(self) -> dict:
        """
        Get the DiT architecture configuration corresponding to the selected model size.

        Returns:
            Dict with keys ``model_channels``, ``num_heads``, and ``num_blocks``.
        """
        configs = {
            "2B": {"model_channels": 2048, "num_heads": 16, "num_blocks": 28},
            "7B": {"model_channels": 4096, "num_heads": 32, "num_blocks": 28},
            "14B": {"model_channels": 5120, "num_heads": 40, "num_blocks": 36},
        }
        return configs[self.hparams.model_size]

    def _create_dit(self, device: str = "meta") -> MinimalV1LVGDiT:
        """
        Instantiate the DiT network

        Args:
            device: Torch device string. Use ``"meta"`` for lazy allocation
                    (memory-efficient; call ``.to_empty()`` before weight loading).

        Returns:
            Freshly constructed ``MinimalV1LVGDiT`` with all layers on *device*.
        """
        # Get the architecture config for the selected model size
        config = self._get_model_config()

        # Suppress per-block "Enable selective checkpoint for block N" INFO spam
        # (28 messages × number-of-ranks × number-of-calls).
        try:
            from cosmos_predict2._src.imaginaire.utils import log as _cl
            _cl.logger.disable("cosmos_predict2._src.predict2.networks.minimal_v4_dit")
        except Exception:
            _cl = None

        # Build the DiT on the specified device; "meta" defers memory allocation until weight loading
        with torch.device(device):
            net = MinimalV1LVGDiT(
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
            )

        return net

    def _fix_rope_buffers(self, module: torch.nn.Module, model_name: str = "net"):
        """
        Re-create ``VideoRopePosition3DEmb`` buffers after ``.to_empty()`` allocation.

        ``to_empty()`` moves the module to a device but does **not** re-initialise
        non-parameter buffers created in ``__init__`` with ``torch.arange``.  This
        method walks the module tree and regenerates those buffers in-place.

        Args:
            module: Module tree to recurse over (typically ``self.net`` or ``self.net_ema``).
            model_name: Label used in log messages to identify which model is being fixed.
        """
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

    def _setup_network(self):
        """
        Setup the DiT network.

        Steps:
          1. Create the DiT on the ``meta`` device (no memory allocation).
          2. Move to an empty CUDA tensor, initialise random weights, fix RoPE buffers.
          3. Resolve the checkpoint from ``hparams.checkpoint_path`` or HuggingFace.
          4. Load weights with ``non_strict_load_model`` and log any key mismatches.
        """
        # Create the DiT on the "meta" device to avoid unnecessary memory allocation during setup.
        self.net = self._create_dit(device="meta")

        # Move to an empty CUDA tensor
        self.net.to_empty(device="cuda")

        # Initialize random weights
        self.net.init_weights()

        # Fix the RoPE buffers that were not re-initialised by ``to_empty()``.
        self._fix_rope_buffers(self.net, "net")

        checkpoint_path = None

        # Resolve the checkpoint path
        if self.hparams.checkpoint_path:
            checkpoint_path = self.hparams.checkpoint_path
        elif self.hparams.checkpoint_uuid:
            uuid = self.hparams.checkpoint_uuid
            if uuid not in _CHECKPOINT_RESOLVED:
                # All ranks resolve the checkpoint path independently.
                # _hf_download uses a per-file filelock so concurrent calls are safe.
                from cosmos_predict2._src.imaginaire.flags import INTERNAL
                if INTERNAL:
                    try:
                        _CHECKPOINT_RESOLVED[uuid] = download_checkpoint(uuid)
                    except Exception as e:
                        if get_local_rank() == 0:
                            log.info(
                                f"[Setup] Checkpoint checkpoint_db download failed ({type(e).__name__}); "
                                f"falling back to HuggingFace download …"
                            )
                        _CHECKPOINT_RESOLVED[uuid] = _hf_download(
                            repo_id="nvidia/Cosmos-Predict2.5-2B",
                            filename="base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",
                        )
                else:
                    if get_local_rank() == 0:
                        log.info("[Setup] External mode (INTERNAL=False): downloading checkpoint from HuggingFace …")
                    _CHECKPOINT_RESOLVED[uuid] = _hf_download(
                        repo_id="nvidia/Cosmos-Predict2.5-2B",
                        filename="base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",
                    )
            checkpoint_path = _CHECKPOINT_RESOLVED[uuid]

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "net." in list(state_dict.keys())[0]:
                state_dict = {k.replace("net.", ""): v for k, v in state_dict.items() if k.startswith("net.")}
            result = non_strict_load_model(self.net, state_dict)
            if get_local_rank() == 0:
                log.info(
                    f"[Setup] DiT checkpoint loaded — source={checkpoint_path}, "
                    f"missing={len(result.missing_keys)}, "
                    f"unexpected={len(result.unexpected_keys)}"
                )
                if result.missing_keys:
                    log.warning(f"[Setup] DiT missing keys (first 5): {result.missing_keys[:5]}")
                if result.unexpected_keys:
                    log.warning(f"[Setup] DiT unexpected keys (first 5): {result.unexpected_keys[:5]}")
        else:
            if get_local_rank() == 0:
                log.warning("No checkpoint loaded! Model has random weights.")
        if get_local_rank() == 0:
            n_params = sum(p.numel() for p in self.net.parameters())
            n_train = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            log.info(
                f"[Setup] DiT ({self.hparams.model_size}) — "
                f"total={n_params/1e6:.1f}M, trainable={n_train/1e6:.1f}M"
            )
        self.net.train()
        self.net.requires_grad_(True)

    def _setup_ema(self):
        """
        Setup EMA training for the DiT.

        Only rank-0 maintains the EMA; DDP guarantees that all ranks have identical
        weights after each step, so the EMA update is deterministic and does not
        need to run on every GPU.

        Sets ``self.net_ema``, ``self.ema_updater``, and ``self.ema_exp_coefficient``.
        When EMA is disabled, all three are set to ``None``.
        """
        if self.hparams.enable_ema and get_rank() == 0:
            # Only rank 0 maintains EMA — DDP guarantees identical weights,
            # so EMA is deterministic and only needs to be computed once.
            self.net_ema = self._create_dit(device="meta")
            self.net_ema.to_empty(device="cpu")
            self.net_ema.init_weights()
            self._fix_rope_buffers(self.net_ema, "net_ema")
            self.net_ema.to(dtype=torch.float32)
            self.net_ema.eval()
            self.net_ema.requires_grad_(False)

            # Build EMA updater and compute exponential decay coefficient from ema_rate
            self.ema_updater = FastEmaModelUpdater()
            s = self.hparams.ema_rate
            self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

            # Initialise EMA weights as an exact copy of the net (initial drift = 0)
            with torch.no_grad():
                for p_ema, p_net in zip(self.net_ema.parameters(), self.net.parameters()):
                    p_ema.data.copy_(p_net.data.to("cpu"))

            if get_local_rank() == 0:
                n_ema = sum(p.numel() for p in self.net_ema.parameters())
                n_net = sum(p.numel() for p in self.net.parameters())
                # Verify param count matches and initial copy is exact
                max_diff = max(
                    (pe.data.float() - pn.data.float().cpu()).abs().max().item()
                    for pe, pn in zip(self.net_ema.parameters(), self.net.parameters())
                )
                log.info(
                    f"[Setup] EMA created — params={n_ema/1e6:.1f}M "
                    f"(net={n_net/1e6:.1f}M), device=cpu, "
                    f"ema_rate={self.hparams.ema_rate}, "
                    f"exp_coeff={self.ema_exp_coefficient:.4f}, "
                    f"init_max_diff={max_diff:.2e} (expect 0)"
                )
                if n_ema != n_net:
                    log.error(
                        f"[Setup] EMA/net param count MISMATCH: "
                        f"ema={n_ema} vs net={n_net}"
                    )
        else:
            self.net_ema = None
            self.ema_updater = None
            self.ema_exp_coefficient = None

    def _init_view_weights(self):
        """
        Pre-compute per-view difficulty weights for loss scaling based on camera extrinsics.
        """
        # Initialize with equal weight
        weights = torch.ones(len(XRAY_CAMERAS))

        for i, view_name in enumerate(XRAY_CAMERAS):
            ext = XRAY_EXTRINSICS[view_name]

            # Cranial has elevation != 0 → harder due to foreshortening
            if ext["elevation"] != 0:
                weights[i] = 1.5
                
            # LAO/RAO oblique views → harder due to complex projections
            elif view_name in ("xray_lao", "xray_rao"):
                weights[i] = 1.5

        # Normalize weights to have mean 1.0 so that the average loss scale is unchanged
        self._view_weights = weights / weights.mean()

    def _setup_renderer(self):
        """DiffDRR renderer setup (instantiated on demand per volume)."""
        pass

    def _setup_text_encoder(self):
        """Setup the Cosmos-Reason1-7B text encoder for view conditioning."""
        encoder = None
        try:
            from cosmos_predict2._src.predict2.text_encoders.text_encoder import (
                TextEncoder, TextEncoderConfig,
            )
            from cosmos_predict2._src.imaginaire.flags import INTERNAL
            ckpt = self.hparams.text_encoder_ckpt
            if ckpt is None:
                if INTERNAL:
                    # NVidia cluster: S3 path accessible directly.
                    ckpt = (
                        "s3://bucket/cosmos_reasoning1/sft_exp700/"
                        "sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/"
                        "checkpoints/iter_000016000/model/"
                    )
                else:
                    # External machine: download via curl (stall-resistant).
                    # nvidia/Cosmos-Reason1-7B is the public release of the same weights.
                    if get_local_rank() == 0:
                        log.info(
                            "[MultiviewXRay] Downloading nvidia/Cosmos-Reason1-7B text encoder "
                            "from HuggingFace (first run only, ~14 GB) …"
                        )
                    ckpt = _text_encoder_snapshot()
                    if get_local_rank() == 0:
                        log.info(f"[MultiviewXRay] Text encoder snapshot at {ckpt}")

            # Encoder device and config
            dev = self.hparams.text_encoder_device or "cuda:3"
            cfg = TextEncoderConfig(
                ckpt_path=ckpt, compute_online=True,
                embedding_concat_strategy="full_concat",
            )

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
                try:
                    n_te = sum(p.numel() for p in encoder.model.parameters())
                    log.info(
                        f"[Setup] Text encoder loaded — device={dev}, "
                        f"params={n_te/1e6:.0f}M, ckpt={ckpt}"
                    )
                except Exception:
                    log.info(f"[Setup] Text encoder loaded — device={dev}, ckpt={ckpt}")
        except Exception as e:
            if get_local_rank() == 0:
                log.warning(
                    f"[MultiviewXRay] Text encoder unavailable ({e}); using zero embeddings.\n"
                    "  *** View conditioning is DISABLED — all views will receive identical\n"
                    "  *** zero text embeddings, causing mode collapse at inference.\n"
                    "  *** Pass text_encoder_ckpt pointing to a Qwen-7B checkpoint to fix this."
                )
        object.__setattr__(self, "_text_encoder", encoder)

    def _ensure_tokenizer_device(self):
        """Move VAE parameters to the current training device if they have drifted."""
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

                    # Sync registered normalisation stat tensors (mean / std buffers)
                    for attr in ("mean", "std", "img_mean", "img_std", "video_mean", "video_std"):
                        if hasattr(self.tokenizer.model, attr):
                            val = getattr(self.tokenizer.model, attr)
                            if isinstance(val, torch.Tensor):
                                setattr(self.tokenizer.model, attr, val.to(target_device))

                    # Sync scale list (per-channel scale factors stored as a Python list of tensors)
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
        """
        Encode a video tensor to VAE latent space.

        Args:
            video: Float tensor of shape ``(B, 3, T, H, W)`` in ``[-1, 1]``.

        Returns:
            Float latent tensor of shape ``(B, 16, T//4, H//8, W//8)``.
        """
        self._ensure_tokenizer_device()
        return self.tokenizer.encode(video.float()).float()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode a VAE latent tensor back to video space.

        Args:
            latent: Float tensor produced by :meth:`encode`.

        Returns:
            Float video tensor of shape ``(B, 3, T, H, W)`` in ``[-1, 1]``.
        """
        self._ensure_tokenizer_device()
        video = self.tokenizer.decode(latent.float()).float()
        if not self.training and not torch.is_grad_enabled():
            v_min, v_max = video.min().item(), video.max().item()
            assert v_min >= -1.5 and v_max <= 1.5, (
                f"VAE decode range: [{v_min:.3f}, {v_max:.3f}]"
            )
        return video

    def _sample_shared_camera_params(
        self, batch_size: int, stage: str = "train"
    ) -> tuple:
        """
        Sample shared camera parameters once per batch.
        
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

        # Apply small random jitter during training for camera augmentation
        if stage == "train":
            fov += torch.randint_like(fov, low=-2, high=3, device=device)
            distance += torch.rand_like(distance, device=device) / 4.0 - 0.125 
    
        return fov, distance

    def _sample_multiview_camera_params(
        self, 
        batch_size: int, 
        stage: str = "train",
        fov: Optional[torch.Tensor] = None, 
        distance: Optional[torch.Tensor] = None,
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

        if fov is None or distance is None:
            fov, distance = self._sample_shared_camera_params(batch_size, stage)

        # Randomly select views for this batch.
        # Validation also uses random subsets to better cover view distribution.
        if n_views < NUM_XRAY_VIEWS and stage in ("train", "val"):
            if stage == "train" and self.hparams.prefer_difficult_views and n_views >= 2 and batch_size >= 2:
                # Define indices of more difficult views based on extrinsics (e.g., cranial and oblique views)
                difficult_indices = [2, 4, 6]

                # Mix: 50% difficult, 50% random
                n_difficult = max(1, n_views // 2)

                # Select difficult views randomly from the difficult set
                difficult_selected = torch.tensor(difficult_indices)[
                    torch.randperm(len(difficult_indices))[:n_difficult]
                ].tolist()

                # Remain views are selected randomly from the full set, excluding already selected difficult views
                remaining = n_views - len(difficult_selected)

                if remaining > 0:
                    other_indices = [i for i in range(NUM_XRAY_VIEWS) if i not in difficult_selected]
                    
                    # Randomly select the remaining views from the non-difficult set
                    other_selected = torch.tensor(other_indices)[
                        torch.randperm(len(other_indices))[:remaining]
                    ].tolist()
                    selected = sorted(difficult_selected + other_selected)
                else:
                    selected = sorted(difficult_selected[:n_views])
            else:
                # For small batches or validation, use random uniform sampling (no bias)
                selected = torch.randperm(NUM_XRAY_VIEWS)[:n_views].sort().values.tolist()
        else:
            selected = list(range(min(n_views, NUM_XRAY_VIEWS)))

        # Build one camera-param dict per selected view
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
    def _render_xray(
        self,
        ct_volume: torch.Tensor,
        camera_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Render a single-channel X-ray from a CT volume via differentiable ray marching.

        Args:
            ct_volume: Float tensor of shape ``(B, 1, D, H, W)`` in ``[0, 1]``.
            camera_params: Dict with keys ``azimuth``, ``elevation``, ``distance``,
                ``fov``, each a 1-D tensor of shape ``(B,)``.

        Returns:
            Float tensor of shape ``(B, 1, H, W)`` — the rendered X-ray images.
        """
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
    def _render_multiview_xrays(
        self,
        ct_volume: torch.Tensor,
        all_view_params: List[Dict[str, torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
        Render one X-ray per view.

        Args:
            ct_volume: ``(B, 1, D, H, W)`` CT volume tensor in ``[0, 1]``.
            all_view_params: List of camera param dicts (one per view) from
                :meth:`_sample_multiview_camera_params`.

        Returns:
            List of ``(B, 1, H, W)`` X-ray tensors, one per view.
        """
        return [self._render_xray(ct_volume, vp) for vp in all_view_params]

    def _construct_93_frame_tensor(
        self,
        frontal_xray: torch.Tensor,
        target_xray: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate frontal and target X-ray frames into a 93-frame video tensor.

        The frontal frame is tiled ``num_frontal_frames`` times as the conditioning
        prefix; the target view fills the remaining frames.

        Args:
            frontal_xray: Reference AP-view X-ray tensor of shape ``(B, 1, H, W)``.
            target_xray:  Target-view X-ray tensor of shape ``(B, 1, H, W)``.

        Returns:
            Float tensor of shape ``(B, 3, 93, H, W)`` ready for VAE encoding.
        """
        nf = self.hparams.num_frontal_frames
        nt = NUM_FRAMES - nf

        # Grayscale X-rays expanded to 3-channel so the video encoder accepts them
        frontal_3ch = frontal_xray.expand(-1, 3, -1, -1)
        target_3ch = target_xray.expand(-1, 3, -1, -1)

        # Tile frontal across nf frames, target across the remaining nt frames
        frontal_frames = frontal_3ch.unsqueeze(2).expand(-1, -1, nf, -1, -1)
        target_frames = target_3ch.unsqueeze(2).expand(-1, -1, nt, -1, -1)
        return torch.cat([frontal_frames, target_frames], dim=2)

    def _generate_multiview_prompts(
        self,
        all_view_params: List[Dict[str, torch.Tensor]],
    ) -> List[List[str]]:
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
    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """
        Encode a list of text prompts to fixed-length embedding tensors.

        Falls back to zero-filled tensors if the text encoder is unavailable.

        Args:
            prompts: List of B prompt strings describing the camera pose/view.

        Returns:
            Float tensor of shape ``(B, CR1_MAX_LENGTH, CR1_EMBEDDING_DIM)`` on
            ``self.device``.  All-zero when no encoder is loaded.
        """
        B = len(prompts)
        encoder = getattr(self, "_text_encoder", None)
        if encoder is not None:
            try:
                te_device = self.hparams.text_encoder_device or "cuda:3"
                with torch.cuda.device(te_device):
                    embeddings = encoder.compute_text_embeddings_online(
                        {"text": prompts}, input_caption_key="text",
                    ).detach().to(dtype=torch.float32, device=self.device).contiguous()
                seq_len = embeddings.shape[1]
                if seq_len < CR1_MAX_LENGTH:
                    pad = torch.zeros(B, CR1_MAX_LENGTH - seq_len, CR1_EMBEDDING_DIM,
                                      device=self.device, dtype=torch.float32)
                    embeddings = torch.cat([embeddings, pad], dim=1)
                elif seq_len > CR1_MAX_LENGTH:
                    embeddings = embeddings[:, :CR1_MAX_LENGTH, :]
                return embeddings
            except Exception as e:
                if get_local_rank() == 0:
                    log.warning(f"[encode_prompts] fallback: {e}")

        # No encoder available — return zero embeddings (mode collapse risk)
        if not getattr(self, "_zero_emb_warned", False):
            log.warning(
                "[TextEncoder] Returning ZERO embeddings — CFG will be non-functional. "
                "All views get identical conditioning → mode collapse risk!"
            )
            self._zero_emb_warned = True
        return torch.zeros(B, CR1_MAX_LENGTH, CR1_EMBEDDING_DIM,
                           device=self.device, dtype=torch.float32)

    def _process_batch_multiview(self, batch: dict, stage: str = "train") -> dict:
        """
        Process a batch of CT volumes to generate multiview training data.
        
        Renders frontal X-ray (reference view) and multiple target views from sampled
        camera parameters, ensuring all views share the same intrinsic parameters
        (FOV, depth range) but differ in extrinsic parameters (azimuth, elevation).

        Args:
            batch: Dict with key "ct" containing CT volume tensor of shape (B, 1, D, H, W).
            stage: "train" or "val" - determines sampling and augmentation strategy.

        Returns:
            Dict containing:
                - video: Concatenated 93-frame videos across all views (B*V, 3, 93, H, W)
                - text_embeddings: Encoded text prompts for all samples
                - view_indices: View index per sample
                - camera_params: Camera extrinsics/intrinsics for each sample
                - prompts: Raw prompt strings
                - frontal_xray: Reference X-ray images
                - target_xray: Target X-ray images for each view
        """
        ct_volume = batch["ct"].to(device=self.device, dtype=torch.float32)
        B = ct_volume.shape[0]

        # Sample FOV + distance ONCE — shared by frontal and all target views
        fov, distance = self._sample_shared_camera_params(B, stage=stage)

        # Generate frontal X-ray (azimuth=0, elevation=0, same intrinsics as target views)
        frontal_params = {
            "azimuth": torch.zeros(B, device=self.device),
            "elevation": torch.zeros(B, device=self.device),
            "distance": distance.clone(),
            "fov": fov.clone(),
        }
        frontal_xray = self._render_xray(ct_volume, frontal_params)

        all_view_params = self._sample_multiview_camera_params(B, stage=stage, fov=fov, distance=distance)
        all_target_xrays = self._render_multiview_xrays(ct_volume, all_view_params)
        all_prompts = self._generate_multiview_prompts(all_view_params)

        videos = []
        frontals_flat = []
        targets_flat = []
        prompts_flat = []
        view_indices = []
        cam_flat = {k: [] for k in ("azimuth", "elevation", "distance", "fov")}

        for v_idx, (vp, target_xr, v_prompts) in enumerate(
            zip(all_view_params, all_target_xrays, all_prompts)
        ):
            video = self._construct_93_frame_tensor(frontal_xray, target_xr)
            videos.append(video)
            frontals_flat.append(frontal_xray)
            targets_flat.append(target_xr)
            prompts_flat.extend(v_prompts)
            view_indices.append(torch.full((B,), vp["view_index"], device=self.device, dtype=torch.long))
            for k in cam_flat:
                cam_flat[k].append(vp[k])

        # Concatenate per-view lists into a single flat batch along the sample dimension
        video_all = torch.cat(videos, dim=0)
        frontal_all = torch.cat(frontals_flat, dim=0)
        target_all = torch.cat(targets_flat, dim=0)
        view_indices_all = torch.cat(view_indices, dim=0)
        cam_all = {k: torch.cat(v, dim=0) for k, v in cam_flat.items()}

        # Encode all concatenated prompts to conditioning embeddings in one batch call
        text_embeddings = self._encode_prompts(prompts_flat)

        return {
            "video": video_all,
            "text_embeddings": text_embeddings,
            "view_indices": view_indices_all,
            "camera_params": cam_all,
            "prompts": prompts_flat,
            "frontal_xray": frontal_all,
            "target_xray": target_all,
        }

    def _apply_cfg_dropout_per_sample(
        self, text_embeddings: torch.Tensor, dropout_rate: float = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Zero out text embeddings for randomly selected samples (CFG training).

        During training, a fraction of samples have their prompt embeddings replaced
        with zeros. This teaches the model to work both with and without conditioning,
        enabling classifier-free guidance at inference.

        Args:
            text_embeddings: Float tensor of shape ``(B, T, D)``.
            dropout_rate: Fraction of samples to drop; defaults to ``hparams.cfg_dropout_rate``.

        Returns:
            Tuple of:
                - ``masked_embeddings``: Tensor ``(B, T, D)`` with dropped rows zeroed.
                - ``keep_mask``: Boolean tensor ``(B,)`` — ``True`` where embeddings were kept.
        """
        if dropout_rate is None:
            dropout_rate = self.hparams.cfg_dropout_rate
        B = text_embeddings.shape[0]
        device = text_embeddings.device
        if dropout_rate <= 0.0 or not self.training:
            return text_embeddings, torch.ones(B, dtype=torch.bool, device=device)
        keep_mask_flat = torch.bernoulli(
            (1.0 - dropout_rate) * torch.ones(B, device=device)
        )
        text_out = text_embeddings * keep_mask_flat.view(B, 1, 1)
        return text_out, keep_mask_flat.bool()

    def _get_condition(
        self,
        text_embeddings: torch.Tensor,
        latent: torch.Tensor,
        num_conditional_frames: int = None,
        use_video_condition: bool = True,
        dtype: torch.dtype = None,
        apply_cfg_dropout: bool = False,
        input_height: int = None,
        input_width: int = None,
    ) -> Video2WorldCondition:
        """
        Build the condition object for a batch.

        Args:
            text_embeddings: ``(B, T, D)`` prompt embeddings from :meth:`_encode_prompts`.
            latent: ``(B, C, T, H, W)`` latent used to set conditional frame values.
            num_conditional_frames: Fixed number of frames to condition on.  When
                ``None``, sampled randomly from ``[min, max]_num_conditional_frames``.
            use_video_condition: Include video (latent) conditioning in output.
            dtype: Cast dtype; inferred from ``self.net`` if ``None``.
            apply_cfg_dropout: Apply per-sample CFG dropout (training only).
            input_height: Pixel height for padding mask; ``None`` disables masking.
            input_width: Pixel width for padding mask; ``None`` disables masking.

        Returns:
            :class:`Video2WorldCondition` ready for :meth:`denoise`.
        """
        B, C, T, H, W = latent.shape
        device = latent.device

        # Infer model dtype from the network parameters if not explicitly provided
        if dtype is None:
            dtype = next(self.net.parameters()).dtype

        # Apply CFG dropout: randomly zero text embeddings and build per-n_frames probs
        conditional_frames_probs = None
        if apply_cfg_dropout and self.training:
            text_embeddings, _ = self._apply_cfg_dropout_per_sample(text_embeddings)
            if use_video_condition:
                dr = self.hparams.cfg_dropout_rate
                min_cf = self.hparams.min_num_conditional_frames
                max_cf = self.hparams.max_num_conditional_frames
                n_opts = max_cf - min_cf + 1
                keep_per = (1.0 - dr) / n_opts
                conditional_frames_probs = {0: dr}
                for nf in range(min_cf, max_cf + 1):
                    conditional_frames_probs[nf] = keep_per

        # Build padding mask: 1 outside the valid image area, 0 inside
        H_px, W_px = H * 8, W * 8
        if input_height is not None and input_width is not None:
            padding_mask = torch.ones(B, 1, H_px, W_px, device=device, dtype=dtype)
            padding_mask[:, :, :input_height, :input_width] = 0.0
        else:
            padding_mask = torch.zeros(B, 1, H_px, W_px, device=device, dtype=dtype)

        fps = torch.full((B,), 24.0, device=device, dtype=dtype)

        # Assemble the condition and attach video (conditional frame) conditioning
        base_condition = Video2WorldCondition(
            crossattn_emb=text_embeddings.to(device=device, dtype=dtype),
            fps=fps,
            padding_mask=padding_mask,
            data_type=DataType.VIDEO,
            use_video_condition=use_video_condition,
        )
        return base_condition.set_video_condition(
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
    ) -> torch.Tensor:
        """
        Run one forward pass of the DiT denoising network.

        Handles conditional frame masking (the frontal frames are replaced with decoded
        GT latents), optional per-frame timestep overriding, and output blending so that
        the velocity prediction for conditioned frames equals the GT velocity.

        Args:
            noise: Pure Gaussian noise tensor ``(B, C, T, H, W)``; used to compute
                GT velocity for conditional frames.
            xt_B_C_T_H_W: Noisy latent at the current diffusion timestep ``(B, C, T, H, W)``.
            timesteps_B_T: Discrete timestep index tensor, shape ``(B, 1)``.
            condition: :class:`Video2WorldCondition` created by :meth:`_get_condition`.

        Returns:
            Predicted velocity field as a ``float32`` tensor ``(B, C, T, H, W)``.
        """
        model_dtype = next(self.net.parameters()).dtype
        B, C, T, H, W = xt_B_C_T_H_W.shape
        condition_video_mask = None

        gt_frames = condition.gt_frames
        mask = condition.condition_video_input_mask_B_C_T_H_W

        # Splice GT frames into the noisy latent for conditioned frame positions
        if condition.is_video and gt_frames is not None and mask is not None:
            condition_state_in = gt_frames.type_as(xt_B_C_T_H_W)
            use_vc = condition.use_video_condition
            if isinstance(use_vc, torch.Tensor):
                assert bool((use_vc == use_vc[0]).all().item())
                use_vc = bool(use_vc[0].item())
            if not use_vc:
                condition_state_in = condition_state_in * 0
            condition_video_mask = mask.repeat(1, C, 1, 1, 1).type_as(xt_B_C_T_H_W)
            xt_B_C_T_H_W = (
                condition_state_in * condition_video_mask
                + xt_B_C_T_H_W * (1 - condition_video_mask)
            )

            # Override timestep to a fixed low value for conditioned frames (optional)
            cft = getattr(self.hparams, "conditional_frame_timestep", -1.0)
            if cft >= 0:
                cvm = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                ts_cond = torch.ones_like(cvm) * cft
                timesteps_B_T = (
                    ts_cond * cvm + timesteps_B_T.view(B, 1, 1, 1, 1) * (1 - cvm)
                )
                timesteps_B_T = timesteps_B_T.squeeze()
                if timesteps_B_T.ndim == 1:
                    timesteps_B_T = timesteps_B_T.unsqueeze(0)

        # Forward pass through the DiT to get the predicted velocity field
        net_out = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(device=self.device, dtype=model_dtype),
            timesteps_B_T=timesteps_B_T.to(device=self.device, dtype=model_dtype),
            **condition.to_dict(),
        ).float()

        # Replace conditioned-frame predictions with GT velocity for exact reconstruction
        if condition.is_video and condition_video_mask is not None:
            gt_x0 = condition.gt_frames.type_as(net_out)
            gt_vel = noise.type_as(net_out) - gt_x0
            net_out = gt_vel * condition_video_mask + net_out * (1 - condition_video_mask)

        return net_out

    def ema_beta(self, iteration: int) -> float:
        """
        Compute the EMA decay coefficient for a given training step.

        Args:
            iteration: Current global step (before ``ema_iteration_shift`` is added).

        Returns:
            Float decay rate in ``[0, 1)``.  Returns ``0.0`` for the first step.
        """
        iteration = iteration + self.hparams.ema_iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    @contextlib.contextmanager
    def ema_scope(self):
        """Context manager for swapping in the EMA model during validation and sample generation."""
        # Only rank 0 has up-to-date EMA; other ranks use regular net for validation
        if self.hparams.enable_ema and self.net_ema is not None and get_rank() == 0:
            if self.global_step >= self.hparams.ema_iteration_shift:
                original_net = self.net
                original_dtype = next(self.net.parameters()).dtype

                # Move EMA to the active device+dtype and swap it in as the forward model
                self.net_ema.to(device=self.device, dtype=original_dtype)
                self.net = self.net_ema

                try:
                    yield
                finally:
                    # Restore original net; push EMA back to CPU if memory-offloaded
                    self.net = original_net
                    if self.hparams.ema_offload_cpu:
                        self.net_ema.to(dtype=torch.float32, device="cpu")
            else:
                yield
        else:
            yield

    @contextlib.contextmanager
    def ema_scope_generation(self):
        """Context manager for swapping in the EMA model during sample generation, with bfloat16 inference."""
        use_ema = (
            self.hparams.enable_ema
            and self.net_ema is not None
            and self.global_step >= self.hparams.ema_iteration_shift
        )
        if use_ema:
            original_net = self.net
            self.net_ema.to(device=self.device, dtype=torch.bfloat16)
            self.net = self.net_ema

            try:
                yield
            finally:
                # Restore net and optionally push EMA back to CPU
                self.net = original_net
                if self.hparams.ema_offload_cpu:
                    self.net_ema.to(dtype=torch.float32, device="cpu")
        else:
            # No EMA available — cast the main net to bfloat16 for inference speed
            original_dtype = next(self.net.parameters()).dtype
            self.net.to(dtype=torch.bfloat16)

            try:
                yield
            finally:
                self.net.to(dtype=original_dtype)

    def on_before_zero_grad(self, optimizer):
        """Update EMA shadow model on rank 0 after each parameter update."""
        if self.hparams.enable_ema and self.net_ema is not None:
            # Only update EMA on rank 0: DDP guarantees all ranks have identical
            # weights after allreduce, so EMA(model) is deterministic.
            if get_rank() != 0:
                return
            ema_beta = self.ema_beta(self.global_step)
            with torch.no_grad():
                if self.hparams.ema_offload_cpu:
                    # Update EMA directly on CPU — stream each tensor via .cpu()
                    # instead of moving the full 2.1B-param model to GPU.
                    for p_ema, p_net in zip(self.net_ema.parameters(), self.net.parameters()):
                        p_ema.data.mul_(ema_beta).add_(p_net.data.to(p_ema.device), alpha=1.0 - ema_beta)
                else:
                    # EMA lives on GPU — update in place
                    self.ema_updater.update_average(self.net, self.net_ema, beta=ema_beta)

    def configure_optimizers(self):
        """
        Build a FusedAdam optimiser with linear warmup + cosine LR schedule.

        Returns:
            Dict with ``'optimizer'`` and ``'lr_scheduler'`` keys for Lightning.
        """
        optimizer = get_base_optimizer(
            model=self.net,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            optim_type="fusedadam",
            betas=(0.9, 0.99), eps=1e-8,
            master_weights=True, capturable=True,
        )

        # Build LR schedule: short linear warmup followed by cosine decay to 20% of peak LR
        warmup_steps = min(self.hparams.warmup_steps, self.hparams.max_iters // 5)
        warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=0.5, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=self.hparams.max_iters - warmup_steps,
                                    eta_min=self.hparams.learning_rate * 0.2)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_train_start(self):
        """Ensure tokenizer is on the training device and log a training summary."""
        self._ensure_tokenizer_device()
        if self.hparams.enable_ema and self.net_ema is not None:
            if self.hparams.ema_offload_cpu:
                self.net_ema.to(dtype=torch.float32)
            else:
                self.net_ema.to(device=self.device, dtype=torch.float32)

        if get_local_rank() == 0:
            te_status = "LOADED" if getattr(self, "_text_encoder", None) is not None else "MISSING (zero embeddings → mode collapse risk!)"
            ema_status = f"ENABLED (rate={self.hparams.ema_rate}, offload_cpu={self.hparams.ema_offload_cpu})" if self.net_ema is not None else "DISABLED"
            log.info(
                f"[Train] Started — step={self.global_step}, "
                f"text_encoder={te_status}, ema={ema_status}, "
                f"guidance={self.hparams.guidance_scale}, "
                f"cfg_dropout={self.hparams.cfg_dropout_rate}, "
                f"cond_frames=[{self.hparams.min_num_conditional_frames}, "
                f"{self.hparams.max_num_conditional_frames}], "
                f"frontal_frames={self.hparams.num_frontal_frames}"
            )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Persist EMA weights and exp_coefficient alongside the main model checkpoint."""
        if self.hparams.enable_ema and self.net_ema is not None:
            checkpoint["net_ema"] = self.net_ema.state_dict()
            checkpoint["ema_exp_coefficient"] = self.ema_exp_coefficient

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Restore EMA weights from a saved checkpoint."""
        if self.hparams.enable_ema and self.net_ema is not None:
            if "net_ema" in checkpoint and isinstance(checkpoint["net_ema"], dict):
                try:
                    result = self.net_ema.load_state_dict(checkpoint["net_ema"], strict=False)

                    if "ema_exp_coefficient" in checkpoint:
                        self.ema_exp_coefficient = checkpoint["ema_exp_coefficient"]

                    if get_local_rank() == 0:
                        n_keys = len(checkpoint["net_ema"])
                        log.info(
                            f"[EMA] Loaded from checkpoint — keys={n_keys}, "
                            f"missing={len(result.missing_keys)}, "
                            f"unexpected={len(result.unexpected_keys)}, "
                            f"exp_coeff={self.ema_exp_coefficient:.4f}"
                        )

                        # Verify EMA differs from net (should diverge after training)
                        diffs = [
                            (pe.data.float().cpu() - pn.data.float().cpu()).abs().mean().item()
                            for pe, pn in zip(
                                list(self.net_ema.parameters())[:10],
                                list(self.net.parameters())[:10],
                            )
                        ]
                        mean_diff = sum(diffs) / len(diffs) if diffs else 0

                        if mean_diff < 1e-8:
                            log.warning(
                                "[EMA] EMA weights are nearly identical to net — "
                                "EMA may not have been updated during training!"
                            )
                        else:
                            log.info(f"[EMA] Verified: EMA ≠ net (mean_abs_diff={mean_diff:.6f})")
                except Exception as e:
                    if get_local_rank() == 0:
                        log.warning(f"Could not load EMA weights: {e}")
            else:
                if get_local_rank() == 0:
                    log.warning("[EMA] No 'net_ema' key in checkpoint — EMA keeps init weights!")

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Compute rectified flow velocity loss for one multiview batch.

        Args:
            batch: Dict with key ``"ct"`` containing a CT volume tensor ``(B, 1, D, H, W)``.
            batch_idx: Batch index (unused).

        Returns:
            Dict with:
                - ``"loss"``: Scalar loss tensor for Lightning backprop.
                - ``"output_batch"``: Images and metadata consumed by callbacks.
        """
        result = self._process_batch_multiview(batch, stage="train")
        video = result["video"]
        text_embeddings = result["text_embeddings"]

        # Encode video to latent space
        video_norm = video * 2.0 - 1.0
        x_1 = self.encode(video_norm)

        # Sample noise and diffusion timesteps
        B_eff = x_1.shape[0]
        tk = {"device": x_1.device, "dtype": torch.float32}

        epsilon = torch.randn(x_1.size(), **tk)
        t_B = self.rectified_flow.sample_train_time(B_eff).to(**tk).view(B_eff, 1)
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, tk).view(B_eff, 1)
        sigmas = self.rectified_flow.get_sigmas(timesteps, tk).view(B_eff, 1)

        xt, vt = self.rectified_flow.get_interpolation(epsilon, x_1.float(), sigmas)

        # Build conditioning and run denoising forward pass
        condition = self._get_condition(text_embeddings, x_1, None, True, apply_cfg_dropout=True)
        cond_mask = condition.condition_video_input_mask_B_C_T_H_W
        mean_cond = cond_mask[:, 0, :, 0, 0].sum(dim=1).mean().item()

        vt_pred = self.denoise(noise=epsilon, xt_B_C_T_H_W=xt,
                               timesteps_B_T=timesteps, condition=condition)

        # Compute time-weighted MSE loss
        time_w = self.rectified_flow.train_time_weight(timesteps, tk)
        per_loss = torch.mean((vt_pred - vt) ** 2, dim=list(range(1, vt_pred.dim())))
        
        # Apply per-view difficulty weighting if enabled
        if self.hparams.view_loss_weights and self._view_weights is not None:
            view_indices = result["view_indices"]  # (B,) with values in [0, NUM_XRAY_VIEWS)
            view_indices = view_indices.to(device=per_loss.device, dtype=torch.long)
            view_w = self._view_weights.to(device=per_loss.device)[view_indices]  # (B,)
            weighted_loss = time_w.squeeze(1) * per_loss * view_w
            loss = torch.mean(weighted_loss) * self.hparams.loss_scale
        else:
            loss = torch.mean(time_w * per_loss) * self.hparams.loss_scale

        # Log scalars
        cam = result["camera_params"]
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=B_eff)
        self.log("train/mse", per_loss.mean(), on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=B_eff)
        self.log("train/mean_cond_frames", mean_cond, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=B_eff)

        # Assemble output dict for callbacks
        output_batch = {
            "xt": xt.detach().cpu(),
            "v_pred": vt_pred.detach().cpu(),
            "sigma": sigmas.detach().cpu(),
            "loss": loss.detach().cpu(),
            "camera_params": {k: v.detach().cpu() if torch.is_tensor(v) else v
                              for k, v in cam.items()},
            "view_indices": result["view_indices"].detach().cpu(),
            "prompts": result["prompts"],
            "frontal_xray": result["frontal_xray"].detach().cpu(),
            "target_xray": result["target_xray"].detach().cpu(),
        }
        return {"loss": loss, "output_batch": output_batch}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Evaluate rectified flow velocity loss on a validation batch using the EMA model.

        Args:
            batch: Dict with key ``"ct"`` containing a CT volume tensor ``(B, 1, D, H, W)``.
            batch_idx: Batch index (unused).

        Returns:
            Dict with ``"loss"`` and ``"output_batch"`` (same structure as :meth:`training_step`).
        """
        result = self._process_batch_multiview(batch, stage="val")
        video = result["video"]
        text_embeddings = result["text_embeddings"]

        # Encode video to latent space
        video_norm = video * 2.0 - 1.0
        x_1 = self.encode(video_norm)

        # Sample noise and diffusion timesteps
        B_eff = x_1.shape[0]
        tk = {"device": x_1.device, "dtype": torch.float32}

        epsilon = torch.randn(x_1.size(), **tk)
        t_B = self.rectified_flow.sample_train_time(B_eff).to(**tk).view(B_eff, 1)
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, tk).view(B_eff, 1)
        sigmas = self.rectified_flow.get_sigmas(timesteps, tk).view(B_eff, 1)

        xt, vt = self.rectified_flow.get_interpolation(epsilon, x_1.float(), sigmas)

        # Build conditioning (fixed 1 conditional frame for consistent val metric)
        condition = self._get_condition(text_embeddings, x_1, 1, True)

        # Forward pass under EMA weights (rank 0) for a cleaner loss signal
        with self.ema_scope():
            vt_pred = self.denoise(noise=epsilon, xt_B_C_T_H_W=xt,
                                   timesteps_B_T=timesteps, condition=condition)

        # Compute time-weighted MSE loss
        time_w = self.rectified_flow.train_time_weight(timesteps, tk)
        per_loss = torch.mean((vt_pred - vt) ** 2, dim=list(range(1, vt_pred.dim())))
        loss = torch.mean(time_w * per_loss)

        # Log scalars
        cam = result["camera_params"]
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=B_eff)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=False,
                 sync_dist=True, batch_size=B_eff)
        self.log("val/mse", per_loss.mean(), on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=B_eff)

        # Assemble output dict for callbacks
        output_batch = {
            "xt": xt.detach().cpu(),
            "v_pred": vt_pred.detach().cpu(),
            "sigma": sigmas.detach().cpu(),
            "loss": loss.detach().cpu(),
            "camera_params": {k: v.detach().cpu() if torch.is_tensor(v) else v
                              for k, v in cam.items()},
            "view_indices": result["view_indices"].detach().cpu(),
            "prompts": result["prompts"],
            "frontal_xray": result["frontal_xray"].detach().cpu(),
            "target_xray": result["target_xray"].detach().cpu(),
        }
        return {"loss": loss, "output_batch": output_batch}

    @torch.inference_mode()
    def generate(
        self,
        image: Optional[Union[torch.Tensor, "PILImage"]] = None,
        view_name: str = "xray_ap",
        camera_params: Optional[Dict[str, float]] = None,
        prompt: Optional[str] = None,
        num_steps: int = None,
        guidance_scale: float = None,
        seed: int = None,
        shift: float = None,
        num_conditional_frames: int = 1,
        verbose: bool = False,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Dict]:
        """
        Generate a novel-view X-ray video conditioned on a frontal reference image.

        Args:
            image: Conditioning frontal X-ray as a PIL Image or a
                ``(1, C, H, W)`` float tensor in ``[0, 1]``.
            view_name: Target camera name from ``XRAY_CAMERAS``.
            camera_params: Override the default camera pose dict with keys
                ``azimuth``, ``elevation``, ``distance``, ``fov``.
            prompt: Override the auto-generated view description.
            num_steps: Diffusion denoising steps; defaults to ``hparams.num_inference_steps``.
            guidance_scale: CFG strength; defaults to ``hparams.guidance_scale``.
            seed: RNG seed for reproducibility.
            shift: Flow-matching shift parameter; defaults to ``hparams.rf_shift``.
            num_conditional_frames: Number of frontal frames to freeze as conditioning.
            verbose: Log generation info at INFO level.
            return_intermediates: If ``True``, return a dict with keys ``"video"``,
                ``"noise"``, and ``"denoised_latent"`` instead of just the video.

        Returns:
            Float tensor ``(1, 3, T, H, W)`` in ``[0, 1]``, or a dict when
            ``return_intermediates=True``.
        """
        try:
            from PIL.Image import Image as PILImage
        except ImportError:
            PILImage = None

        # Preprocess PIL Image to normalised float tensor on the model device
        if image is not None and PILImage is not None and isinstance(image, PILImage):
            import numpy as _np
            img_array = _np.array(image).astype(_np.float32) / 255.0
            if len(img_array.shape) == 2:
                img_array = _np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            image = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if image is None:
            raise ValueError("image cannot be None")

        # Fill default camera pose from the standard extrinsics table
        if camera_params is None:
            ext = XRAY_EXTRINSICS[view_name]
            fov_mid = sum(XRAY_FOV_RANGE) / 2.0
            dist_mid = sum(XRAY_DISTANCE_RANGE) / 2.0
            camera_params = {
                "azimuth": ext["azimuth"],
                "elevation": ext["elevation"],
                "distance": dist_mid,
                "fov": fov_mid,
            }

        # Build text description for the target view
        if prompt is None:
            prefix = XRAY_CAPTION_PREFIXES[view_name]
            prompt = XRAY_PROMPT_TEMPLATE.format(
                prefix=prefix,
                azimuth=camera_params["azimuth"],
                elevation=camera_params["elevation"],
                distance=camera_params["distance"],
                fov=camera_params["fov"],
                znear=self.hparams.renderer_min_depth,
                zfar=self.hparams.renderer_max_depth,
            )

        text_embeddings = self._encode_prompts([prompt])

        # Resolve generation hyperparameters, falling back to hparams defaults
        num_steps = num_steps or self.hparams.num_inference_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.hparams.guidance_scale
        shift = shift or self.hparams.rf_shift
        seed = seed if seed is not None else torch.randint(0, 2**32 - 1, (1,)).item()

        with self.ema_scope_generation():
            model = self.net
            was_training = model.training
            model.eval()
            gen_dtype = next(model.parameters()).dtype

            # Build input video tensor from the conditioning image (tile to NUM_FRAMES)
            if image.dim() == 4:
                B, C_in, H_in, W_in = image.shape
                device = image.device
                input_norm = image.to(dtype=gen_dtype) * 2.0 - 1.0
                input_video = input_norm.unsqueeze(2)
                # Repeat the frontal frame for all remaining frames so the temporal VAE
                # receives a uniform input matching its training distribution (which always
                # had meaningful content in every frame, never all-black padding).
                frame_pad = input_norm.unsqueeze(2).expand(
                    -1, -1, NUM_FRAMES - 1, -1, -1
                ).clone()
                input_video = torch.cat([input_video, frame_pad], dim=2)
                if not getattr(self, "_gen_frame_logged", False):
                    nf = self.hparams.num_frontal_frames
                    log.info(
                        f"[Generate] input_video={tuple(input_video.shape)} "
                        f"range=[{input_video.min():.2f},{input_video.max():.2f}], "
                        f"all {NUM_FRAMES} frames=frontal "
                        f"(train: frontal×{nf}+target×{NUM_FRAMES-nf}), "
                        f"cond_frames={num_conditional_frames}"
                    )
                    self._gen_frame_logged = True
            else:
                B = image.shape[0]
                device = image.device
                input_video = image.to(dtype=gen_dtype) * 2.0 - 1.0
                if input_video.shape[2] < NUM_FRAMES:
                    last = input_video[:, :, -1:, :, :]
                    pad = last.repeat(1, 1, NUM_FRAMES - input_video.shape[2], 1, 1)
                    input_video = torch.cat([input_video, pad], dim=2)

            # Encode to latent space and derive the spatial state shape
            latent_cond = self.encode(input_video.to(device)).to(dtype=gen_dtype)
            _, C, T, H, W = latent_cond.shape
            state_shape = (C, T, H, W)

            # Build conditional and unconditional guidance objects
            condition = self._get_condition(
                text_embeddings, latent_cond, num_conditional_frames, True, dtype=gen_dtype,
            )
            uncondition = self._get_condition(
                torch.zeros_like(text_embeddings), latent_cond,
                num_conditional_frames, True, dtype=gen_dtype,
            )

            # Sample reproducible initial noise from a seeded generator
            noise = arch_invariant_rand(
                shape=(B,) + state_shape, dtype=torch.float32,
                device=device, seed=seed,
            ).to(dtype=gen_dtype)

            seed_gen = torch.Generator(device=device)
            seed_gen.manual_seed(seed)

            # Configure the UniPC sampler and run the denoising loop
            self.sample_scheduler.set_timesteps(
                num_inference_steps=num_steps, device=device,
                shift=shift, use_kerras_sigma=False,
            )
            timesteps = self.sample_scheduler.timesteps

            initial_noise = noise
            latents = noise.clone()

            for i, t in enumerate(timesteps):
                ts = t.view(1, 1).expand(B, 1)

                # Conditional and unconditional velocity predictions for CFG
                v_cond = self.denoise(initial_noise, latents, ts, condition)
                v_uncond = self.denoise(initial_noise, latents, ts, uncondition)
                vel = v_uncond + guidance_scale * (v_cond - v_uncond)

                latents = self.sample_scheduler.step(
                    model_output=vel, timestep=t, sample=latents,
                    return_dict=False, generator=seed_gen,
                )[0]

                # Re-inject conditioning frames after each step
                cond_mask = condition.condition_video_input_mask_B_C_T_H_W
                if cond_mask is not None:
                    latents = latents * (1 - cond_mask) + latent_cond * cond_mask

            # Decode latent to video and rescale from [-1, 1] to [0, 1]
            denoised_latent = latents.detach().clone() if return_intermediates else None
            video = self.decode(latents.float())
            video = (video / 2.0 + 0.5).clamp(0, 1)

            if was_training:
                model.train()

            if return_intermediates:
                return {"video": video, "noise": initial_noise.cpu(), "denoised_latent": denoised_latent.cpu()}
            return video

    @torch.inference_mode()
    def generate_all_views(
        self,
        image: Optional[Union[torch.Tensor, "PILImage"]] = None,
        num_steps: int = None,
        guidance_scale: float = None,
        seed: int = None,
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate novel-view X-ray videos for every camera in ``XRAY_CAMERAS``.

        Args:
            image: Conditioning frontal X-ray (PIL Image or ``(1, C, H, W)`` tensor).
            num_steps: Diffusion steps per view; defaults to ``hparams.num_inference_steps``.
            guidance_scale: CFG strength; defaults to ``hparams.guidance_scale``.
            seed: Shared RNG seed (same seed for every view).
            verbose: Log progress for each view.

        Returns:
            Dict mapping view name → float tensor ``(1, 3, T, H, W)`` in ``[0, 1]``.
        """
        results = {}
        for view_name in XRAY_CAMERAS:
            if verbose:
                log.info(f"Generating view: {view_name}")
            results[view_name] = self.generate(
                image=image,
                view_name=view_name,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                verbose=verbose,
            )
        return results
