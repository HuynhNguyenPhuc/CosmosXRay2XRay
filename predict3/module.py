"""Cosmos 3 multiview X-ray synthesis (predict3) — SCAFFOLDING, see docs/cosmos-predict3/PLAN.md §7-9.

Builds on the real `Cosmos3OmniTransformer` / `AutoencoderKLWan` / `Cosmos3OmniPipeline`
classes from `diffusers` (installed from GitHub `main` — the released 0.39.0 wheel silently
drops the `nvidia/Cosmos3-Edge` `use_und_k_norm_for_gen` config field; `main` has the fix,
see PLAN.md §3/§6.2). This backbone needs a SEPARATE virtualenv from `predict2_5`/
`transfer2_5`: `diffusers` `main` needs `huggingface-hub>=1.23,<2.0` + `transformers>=5.15`,
while `cosmos_predict2`/`cosmos_transfer2` need `huggingface-hub<1.0`. Do not try to install
this alongside `predict2_5`/`transfer2_5` in the same venv (PLAN.md §7).

Conditioning uses all three of Cosmos 3's ports, where `predict2_5` had only one
(cross-attention text):

* **Vision (native I2V)** — the anchor X-ray view is encoded as latent frame 0 and excluded
  from `vision_noisy_frame_indexes` (`condition_frame_indexes=[0]`), so joint self-attention
  conditions on it directly instead of splicing at every denoising step.
* **Action (`camera_pose`)** — camera geometry enters as continuous 9D pose vectors through
  `action_proj_in`, NOT as text. See `predict3/camera.py` for why this beats formatting
  ``azimuth {:.1f} deg`` into a prompt. All action tokens are pure conditioning (never
  noised, excluded from the loss).
* **Text** — left to carry anatomy/appearance, with camera numerics removed.

SCOPE (per PLAN.md §7 — "scaffolding + meta-device smoke test"): real `nvidia/Cosmos3-Edge`
checkpoint weights are NOT downloaded/loaded here — `_create_transformer`/`_create_vae`
build the real architecture from the real published config so shapes and the joint-sequence
packing can be verified end to end, but the weights themselves are randomly initialized.
Loading real pretrained weights and wiring a real X-ray datamodule are follow-ups tracked in
PLAN.md §5 (dataset conversion to `cosmos-framework`'s MP4 + captions.jsonl format comes
first, since that also determines the text-prompt/caption format used here).
"""

from __future__ import annotations

from typing import Any

import torch
from lightning import LightningModule

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline
from transformers import AutoTokenizer

from predict3.camera import (
    CAMERA_POSE_DOMAIN_ID,
    PoseConvention,
    orbit_view_matrices,
    pad_actions_to_model_dim,
    poses_to_camera_pose_actions,
)
from predict3.constants import COSMOS3_EDGE_REPO_ID, COSMOS3_LATENT_CHANNELS
from predict3.hf import load_scheduler_config, load_transformer_config, load_vae_config
from shared.constants import NUM_FRAMES, VOL_SIZE
from shared.utils import get_logger

log = get_logger(__name__)


class CosmosXRay2XRayPredict3Multiview(LightningModule):
    """Cosmos 3 (`Cosmos3OmniTransformer`) fine-tuning for 7-view X-ray synthesis.

    See module docstring for scope — this constructs the real architecture but does not
    yet load real pretrained weights or a real datamodule.
    """

    def __init__(
        self,
        model_repo_id: str = COSMOS3_EDGE_REPO_ID,
        state_ch: int = COSMOS3_LATENT_CHANNELS,
        num_frames: int = NUM_FRAMES,
        learning_rate: float = 2 ** (-14.5),
        weight_decay: float = 0.001,
        warmup_steps: int = 2000,
        max_iters: int = 100000,
        default_prompt: str = "A 360-degree rotational view of a chest CT scan.",
        use_camera_action: bool = True,
        camera_pose_convention: PoseConvention = "backward_anchored",
        num_hidden_layers_override: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._pipeline: Cosmos3OmniPipeline | None = None
        # `Cosmos3OmniPipeline` is a `DiffusionPipeline`, not an `nn.Module` — storing the
        # backbone only inside `self._pipeline` would make `self.parameters()` (hence
        # `configure_optimizers`'s AdamW) and the Lightning `Trainer`'s automatic device
        # placement / DDP wrapping silently see zero parameters. Register both as direct
        # submodule attributes (same objects as `self._pipeline.transformer`/`.vae`, so
        # weights/gradients are shared, not copied) so `nn.Module.__setattr__` picks them up
        # and `Trainer`/`.to(device)` move them correctly — matching predict2_5's convention
        # of registering even its frozen VAE (`self.tokenizer`) as a real submodule. The VAE
        # is frozen (encode-only; `configure_optimizers` below only includes
        # `self.transformer.parameters()`) but still needs to move with the module.
        self.transformer: Cosmos3OmniTransformer | None = None
        self.vae: AutoencoderKLWan | None = None

    # ------------------------------------------------------------------
    # Construction — mirrors predict2_5._create_dit / transfer2_5._create_control_vace_dit
    # ------------------------------------------------------------------

    def _create_transformer(self, device: str = "meta") -> Cosmos3OmniTransformer:
        """Build the real `Cosmos3OmniTransformer` from the published `nvidia/Cosmos3-Edge`
        config. ``device="meta"`` defers memory allocation (no weights loaded)."""
        config = load_transformer_config(self.hparams.model_repo_id)

        # Smoke-test-only escape hatch. The full 28-layer model is 3.37B params, which needs
        # ~27 GB just for fp32 weights + gradients — more than is reliably free on a
        # workstation also running other jobs. Shrinking the layer count keeps every other
        # dimension (hidden size, latent channels, patch size, mRoPE axes) at its real
        # published value, so the joint-sequence packing and action/vision wiring under test
        # are bit-identical; only depth changes. Never set this for real training: it would
        # silently make pretrained weights unloadable.
        override = self.hparams.num_hidden_layers_override
        if override is not None:
            log.warning(
                f"[predict3] num_hidden_layers_override={override} (real config: "
                f"{config['num_hidden_layers']}) — smoke-test scaffolding only, not for training."
            )
            config = {**config, "num_hidden_layers": override}

        with torch.device(device):
            net = Cosmos3OmniTransformer.from_config(config)
        return net

    def _create_vae(self, device: str = "meta") -> AutoencoderKLWan:
        config = load_vae_config(self.hparams.model_repo_id)
        with torch.device(device):
            vae = AutoencoderKLWan.from_config(config)
        return vae

    def _create_scheduler(self) -> UniPCMultistepScheduler:
        config = load_scheduler_config(self.hparams.model_repo_id)
        return UniPCMultistepScheduler.from_config(config)

    def _create_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.hparams.model_repo_id, subfolder="text_tokenizer")

    def _build_pipeline(self, device: str = "meta") -> Cosmos3OmniPipeline:
        """Wrap the real transformer/vae/tokenizer/scheduler in a real `Cosmos3OmniPipeline`
        instance, purely to reuse its own (private) joint-sequence packing helpers
        (``_prepare_text_segment``, ``_prepare_vision_segment``, ``_prepare_action_segment``,
        ``_encode_video``) instead of re-implementing Cosmos 3's mRoPE/token-packing scheme by
        hand — that packing is intricate (see
        `diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py`) and re-deriving it
        independently risks a silently-wrong training signal that still "runs". Disabling the
        safety checker: it is a separate, large, irrelevant-to-X-ray model, not needed for
        training and not something we want implicitly downloaded here."""
        return Cosmos3OmniPipeline(
            transformer=self._create_transformer(device=device),
            text_tokenizer=self._create_tokenizer(),
            vae=self._create_vae(device=device),
            scheduler=self._create_scheduler(),
            enable_safety_checker=False,
        )

    def setup(self, stage: str | None = None) -> None:
        if self._pipeline is None:
            # Real architecture, randomly initialized — see module docstring SCOPE note.
            self._pipeline = self._build_pipeline(device=str(self.device))
            self.transformer = self._pipeline.transformer
            self.vae = self._pipeline.vae
            self.vae.requires_grad_(False)

    # ------------------------------------------------------------------
    # Conditioning helpers
    # ------------------------------------------------------------------

    def _encode_anchor_video(self, video: torch.Tensor) -> torch.Tensor:
        """``video``: ``(B, 3, T, H, W)`` in ``[-1, 1]``, anchor view as frame 0. Returns
        normalized latents ``(B, C, T_lat, H_lat, W_lat)`` via the pipeline's own
        ``_encode_video`` (bit-exact with how the pretrained checkpoint's own inference path
        encodes — see its docstring)."""
        return self._pipeline._encode_video(video)

    def _resolve_camera_actions(self, batch: dict[str, Any], num_pixel_frames: int) -> torch.Tensor:
        """Resolve ``(T_pixel - 1, action_dim)`` camera-pose conditioning for this batch.

        Prefers explicit per-sample geometry from the batch, falling back to a synthetic
        orbit so the wiring is exercisable before the real datamodule exists:

        1. ``batch["camera_actions"]`` — pre-encoded ``(T-1, 9)`` or ``(T-1, action_dim)``.
        2. ``batch["camera_poses"]`` — ``(T, 4, 4)`` camera-to-world, encoded here.
        3. Fallback: a 360-degree orbit matching ``render_orbit_video``'s sweep.

        Actions describe *transitions*, so ``T`` frames yield ``T - 1`` action tokens — this
        is Cosmos 3's ``chunk_size`` / ``chunk_size + 1`` frames contract.
        """
        action_dim = self.transformer.action_dim

        actions = batch.get("camera_actions")
        if actions is None:
            poses = batch.get("camera_poses")
            if poses is None:
                poses = orbit_view_matrices(num_pixel_frames)
            actions = poses_to_camera_pose_actions(
                poses, pose_convention=self.hparams.camera_pose_convention
            )

        actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.ndim != 2:
            raise ValueError(f"camera actions must have shape (T-1, D), got {tuple(actions.shape)}.")

        expected_len = num_pixel_frames - 1
        if actions.shape[0] != expected_len:
            raise ValueError(
                f"camera actions length {actions.shape[0]} does not match the expected "
                f"{expected_len} transitions for {num_pixel_frames} video frames."
            )

        return pad_actions_to_model_dim(actions, action_dim).to(self.device)

    # ------------------------------------------------------------------
    # Flow-matching training step
    # ------------------------------------------------------------------

    @staticmethod
    def _flow_matching_interpolate(x_1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rectified-flow interpolation, same convention as `cosmos_predict2`'s
        `RectifiedFlow.get_interpolation` (``x_0`` = noise, ``x_1`` = clean data):
        ``x_t = eps*t + x_1*(1-t)``, ``v_target = eps - x_1``.

        Not importing `cosmos_predict2`'s `RectifiedFlow` directly: this backbone's
        `diffusers`-main + `transformers>=5.15` environment cannot coexist with
        `cosmos_predict2`/`cosmos_transfer2`'s `huggingface-hub<1.0` pin in the same venv
        (see module docstring), so the few lines of interpolation math are duplicated here
        rather than shared across environments that can't both be installed at once.
        """
        epsilon = torch.randn_like(x_1)
        t_view = t.view(t.shape[0], *([1] * (x_1.dim() - 1)))
        x_t = epsilon * t_view + x_1 * (1 - t_view)
        v_target = epsilon - x_1
        return x_t, v_target

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, torch.Tensor]:
        """One flow-matching training step with native I2V + `camera_pose` action conditioning.

        Args:
            batch: dict with ``"video"``: ``(1, 3, T, H, W)`` in ``[0, 1]``, anchor view as
                   frame 0, and ``"prompt"``: ``str`` or length-1 ``list[str]``. Optionally
                   ``"camera_actions"`` / ``"camera_poses"`` (see
                   :meth:`_resolve_camera_actions`). Batch size is fixed at 1:
                   `Cosmos3OmniTransformer` packs one joint sequence per call (see its
                   ``vision_tokens: list[torch.Tensor]`` signature) — use gradient
                   accumulation for an effective batch size > 1, matching how the pretrained
                   checkpoint's own pipeline only ever runs batch=1 (see
                   `Cosmos3OmniPipeline._mask_velocity_predictions`'s docstring).
        """
        if self._pipeline is None:
            self.setup()

        video = batch["video"] * 2.0 - 1.0  # [0,1] -> [-1,1]
        prompt = batch["prompt"]
        prompt = prompt[0] if isinstance(prompt, (list, tuple)) else prompt
        device = video.device
        num_pixel_frames = video.shape[2]

        latents = self._encode_anchor_video(video)  # (1, C, T_lat, H_lat, W_lat)
        if latents.shape[0] != 1:
            raise ValueError("batch size must be 1 — see training_step docstring")

        cond_input_ids, _ = self._pipeline.tokenize_prompt(
            prompt, num_frames=self.hparams.num_frames, height=VOL_SIZE, width=VOL_SIZE,
        )
        text_segment = self._pipeline._prepare_text_segment(cond_input_ids, device)

        vision_segment = self._pipeline._prepare_vision_segment(
            input_vision_tokens=latents,
            has_image_condition=True,
            mrope_offset=text_segment["vision_start_temporal_offset"],
            vision_fps=None,
            curr=text_segment["und_len"],
            device=device,
            condition_frame_indexes=[0],  # native I2V: latent frame 0 = anchor view, never noised
        )

        mrope_segments = [text_segment["text_mrope_ids"], vision_segment["vision_mrope_ids"]]
        sequence_length = text_segment["und_len"] + vision_segment["num_vision_tokens"]

        action_kwargs: dict[str, Any] = {}
        if self.hparams.use_camera_action:
            camera_actions = self._resolve_camera_actions(batch, num_pixel_frames)
            action_segment = self._pipeline._prepare_action_segment(
                input_action_tokens=camera_actions,
                # Every action token is conditioning: the camera trajectory is *given*, not
                # generated. This empties `action_mse_loss_indexes`, which the transformer
                # explicitly guards on (`if action_mse_loss_indexes.numel() > 0`) to skip
                # timestep embedding, so the poses enter as clean projected conditioning.
                condition_frame_indexes=list(range(camera_actions.shape[0])),
                mrope_offset=text_segment["vision_start_temporal_offset"],
                action_fps=None,
                curr=sequence_length,
                device=device,
            )
            mrope_segments.append(action_segment["action_mrope_ids"])
            sequence_length += action_segment["action_len"]
            action_kwargs = {
                "action_tokens": [camera_actions],
                "action_token_shapes": action_segment["action_token_shapes"],
                "action_sequence_indexes": action_segment["action_sequence_indexes"],
                "action_mse_loss_indexes": action_segment["action_mse_loss_indexes"],
                "action_timesteps": None,  # unused: no noisy action tokens
                "action_noisy_frame_indexes": action_segment["action_noisy_frame_indexes"],
                "action_domain_ids": [torch.tensor([CAMERA_POSE_DOMAIN_ID], device=device)],
            }

        t = torch.rand(1, device=device, dtype=torch.float32)
        noisy_latents, v_target = self._flow_matching_interpolate(latents, t)
        noisy_latents = noisy_latents.clone()
        noisy_latents[:, :, 0] = latents[:, :, 0]  # keep the anchor frame clean

        dtype = self.transformer.dtype
        num_noisy = vision_segment["num_noisy_vision_tokens"]
        preds_vision, _preds_sound, _preds_action = self.transformer(
            input_ids=text_segment["input_ids"],
            text_indexes=text_segment["text_indexes"],
            position_ids=torch.cat(mrope_segments, dim=1),
            und_len=text_segment["und_len"],
            sequence_length=sequence_length,
            vision_tokens=[noisy_latents.to(dtype=dtype)],
            vision_token_shapes=vision_segment["vision_token_shapes"],
            vision_sequence_indexes=vision_segment["vision_sequence_indexes"],
            vision_mse_loss_indexes=vision_segment["vision_mse_loss_indexes"],
            vision_timesteps=torch.full((num_noisy,), t.item(), device=device),
            vision_noisy_frame_indexes=vision_segment["vision_noisy_frame_indexes"],
            return_dict=False,
            **action_kwargs,
        )

        # preds_vision[0] is (1, C, T_lat, H_lat, W_lat), matching `latents` — the transformer
        # itself zero-fills the conditioning frame (index 0) in its unpatchify step (see
        # Cosmos3OmniTransformer._unpatchify_and_unpack_latents), so slicing [:, :, 1:] on both
        # sides below is what excludes the anchor frame from the loss.
        v_pred = preds_vision[0].float()
        loss = torch.mean((v_pred[:, :, 1:] - v_target[:, :, 1:].float()) ** 2)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
