"""Inference pipeline for Cosmos Predict 2.5."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
import torch

from cosmos_predict2._src.imaginaire.utils.checkpointer import non_strict_load_model
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from predict2_5.constants import COSMOS_TOKENIZER_UUID, IMG_HEIGHT, IMG_WIDTH, NUM_FRAMES, PROMPTS
from predict2_5.hf import hf_download, resolve_hf_uri
from predict2_5.text_encoder import CR1TextEncoder
from shared.utils import (
    arch_invariant_rand,
    fix_rope_buffers,
    get_logger,
    is_uuid_format,
    move_tokenizer_to_device,
    safe_torch_load,
)

log = get_logger(__name__)

DEFAULT_PROMPT = PROMPTS[0] if PROMPTS else None


class Inferencer:
    """Inference pipeline for post-trained Cosmos Predict 2.5."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        model_size: str = "2B",
        tokenizer_path: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        device: str = "cuda",
        cpu_offload: bool = False,
    ) -> None:
        """
        Initialize inferencer by loading DiT, tokenizer, and text encoder.

        Args:
            checkpoint_path: Path, UUID, or hf:// URI for DiT checkpoint.
            config_path: Path or URI for DiT config JSON.
            model_size: Model size label (e.g. "2B").
            tokenizer_path: Path, UUID, or URI for VAE tokenizer.
            text_encoder_path: Path, UUID, or URI for text encoder.
            device: Device for inference.
            cpu_offload: Whether to offload weights to CPU when idle.
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.runtime_dtype = torch.bfloat16
        self.cpu_offload = bool(cpu_offload)
        self.model_size = model_size

        resolved_checkpoint_path = self._resolve_artifact(
            spec=checkpoint_path,
            artifact_name="checkpoint",
        )
        resolved_config_path = self._resolve_config_path(
            config_path=config_path,
            checkpoint_path=resolved_checkpoint_path,
        )
        resolved_tokenizer_path = self._resolve_artifact(
            spec=tokenizer_path or COSMOS_TOKENIZER_UUID,
            artifact_name="tokenizer",
        )
        resolved_text_encoder_path = (
            self._resolve_artifact(
                spec=text_encoder_path,
                artifact_name="text_encoder",
            )
            if text_encoder_path
            else None
        )

        self.checkpoint_path = resolved_checkpoint_path
        self.config_path = resolved_config_path

        self.sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )

        self.tokenizer = self._build_tokenizer(resolved_tokenizer_path)
        move_tokenizer_to_device(self.tokenizer, self.device)

        self.dit = self._build_dit(
            config_path=resolved_config_path,
            checkpoint_path=resolved_checkpoint_path,
        )

        self.text_encoder = CR1TextEncoder(
            text_encoder_ckpt_path=resolved_text_encoder_path,
            device=self.device,
            cpu_offload=self.cpu_offload,
        )

    def _resolve_artifact(self, spec: Optional[str], artifact_name: str) -> Optional[str]:
        """Resolve artifact specification to a local file path."""
        if not spec:
            return spec

        path_candidate = Path(spec)
        if path_candidate.exists():
            resolved = str(path_candidate)
            log.info("Resolved %s from local path: %s", artifact_name, resolved)
            return resolved

        if spec.startswith("hf://"):
            repo_id, filename, revision = resolve_hf_uri(spec)
            resolved = hf_download(repo_id=repo_id, filename=filename, revision=revision)
            log.info("Resolved %s from HuggingFace: %s", artifact_name, resolved)
            return resolved

        if is_uuid_format(spec):
            os.environ.setdefault("COSMOS_EXPERIMENTAL_CHECKPOINTS", "1")
            from cosmos_oss.checkpoints_predict2 import register_checkpoints
            from cosmos_predict2._src.imaginaire.utils.checkpoint_db import download_checkpoint

            register_checkpoints()
            resolved = download_checkpoint(spec)
            log.info("Resolved %s from checkpoint UUID: %s", artifact_name, resolved)
            return resolved

        log.info("Using %s spec as-is: %s", artifact_name, spec)
        return spec

    def _resolve_config_path(self, config_path: Optional[str], checkpoint_path: str) -> str:
        """Resolve config path with sensible defaults."""
        if config_path is not None:
            resolved_config = self._resolve_artifact(config_path, artifact_name="config")
        else:
            local_default = Path(checkpoint_path).parent / "config.json"
            resolved_config = str(local_default)

        if not Path(resolved_config).exists():
            raise FileNotFoundError(f"Config file not found at resolved path: {resolved_config}")
        
        return resolved_config

    def _build_tokenizer(self, tokenizer_path: str) -> Wan2pt1VAEInterface:
        """Create VAE tokenizer from resolved path."""
        log.info("Loading tokenizer: %s", tokenizer_path)
        return Wan2pt1VAEInterface(
            chunk_duration=93,
            load_mean_std=False,
            vae_pth=tokenizer_path,
            temporal_window=16,
            keep_decoder_cache=False,
            keep_encoder_cache=False,
        )

    def _build_dit(self, config_path: str, checkpoint_path: str) -> MinimalV1LVGDiT:
        """Create DiT model from config and load checkpoint weights."""
        with open(config_path, "r", encoding="utf-8") as f:
            dit_config = json.load(f)

        if isinstance(dit_config.get("sac_config"), dict):
            dit_config["sac_config"] = SACConfig(**dit_config["sac_config"])

        log.info("Initializing DiT from config: %s", config_path)
        with torch.device("meta"):
            dit = MinimalV1LVGDiT(**dit_config)

        dit.to_empty(device=self.device)
        dit.init_weights()
        fix_rope_buffers(dit)

        log.info("Loading checkpoint: %s", checkpoint_path)
        ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "net" in ckpt:
            state_dict = ckpt["net"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = {
                k.removeprefix("net."): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("net.")
            }
        else:
            state_dict = ckpt

        load_result = non_strict_load_model(dit, state_dict)
        if isinstance(load_result, tuple) and len(load_result) >= 2:
            missing, unexpected = load_result[0], load_result[1]
            if missing:
                log.warning("Missing keys while loading DiT: %s", missing[:5])
            if unexpected:
                log.warning("Unexpected keys while loading DiT: %s", unexpected[:5])

        dit = dit.to(self.device)
        dit.eval()
        return dit

    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompt into embeddings."""
        return self.text_encoder.encode_prompt(prompt)

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video tensor to latent space."""
        latent = self.tokenizer.encode(video.float())
        return latent.float()

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor to video tensor."""
        video = self.tokenizer.decode(latent.float())
        return video.float()

    def _get_condition(
        self,
        text_embeddings: torch.Tensor,
        latent: torch.Tensor,
        num_conditional_frames: int = 1,
        dtype: torch.dtype | None = None,
    ) -> Video2WorldCondition:
        """Build Video2World condition object for inference."""
        if dtype is None:
            dtype = next(self.dit.parameters()).dtype

        bsz, _, _, height, width = latent.shape
        device = latent.device

        padding_mask = torch.zeros(
            bsz,
            1,
            height * 8,
            width * 8,
            device=device,
            dtype=dtype,
        )

        fps = torch.full((bsz,), 24.0, device=device, dtype=dtype)

        base_condition = Video2WorldCondition(
            crossattn_emb=text_embeddings.to(device=device, dtype=dtype),
            fps=fps,
            padding_mask=padding_mask,
            data_type=DataType.VIDEO,
            use_video_condition=True,
        )

        return base_condition.set_video_condition(
            gt_frames=latent.to(dtype=dtype),
            random_min_num_conditional_frames=num_conditional_frames,
            random_max_num_conditional_frames=num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=None,
        )

    def denoise(
        self,
        noise: torch.Tensor,
        xt_b_c_t_h_w: torch.Tensor,
        timesteps_b_t: torch.Tensor,
        condition: Video2WorldCondition,
    ) -> torch.Tensor:
        """Predict velocity at a single denoising step using DiT."""
        model_dtype = next(self.dit.parameters()).dtype
        batch_size, channels, _, _, _ = xt_b_c_t_h_w.shape

        condition_video_mask = None
        gt_frames = condition.gt_frames
        mask = condition.condition_video_input_mask_B_C_T_H_W

        if condition.is_video and gt_frames is not None and mask is not None:
            condition_state_in = gt_frames.type_as(xt_b_c_t_h_w)
            condition_video_mask = mask.repeat(1, channels, 1, 1, 1).type_as(xt_b_c_t_h_w)

            xt_b_c_t_h_w = (
                condition_state_in * condition_video_mask
                + xt_b_c_t_h_w * (1 - condition_video_mask)
            )

        autocast_device = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.autocast(device_type=autocast_device, dtype=self.runtime_dtype): 
            net_output = self.dit(
                x_B_C_T_H_W=xt_b_c_t_h_w.to(device=self.device, dtype=model_dtype),
                timesteps_B_T=timesteps_b_t.to(device=self.device, dtype=model_dtype),
                **condition.to_dict()
            ).float()

        if condition.is_video and condition_video_mask is not None:
            gt_frames_x0 = condition.gt_frames.type_as(net_output)
            gt_velocity = noise.type_as(net_output) - gt_frames_x0
            net_output = (
                gt_velocity * condition_video_mask 
                + net_output * (1 - condition_video_mask)
            )

        return net_output

    @staticmethod
    def _to_pil_rgb(image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Convert input image to RGB Pillow image format."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        return Image.fromarray(image, mode="RGB")

    @staticmethod
    def _scale_image_intensity(image: np.ndarray) -> np.ndarray:
        """Scale image intensity to [0, 1] range using fixed /255 scaling."""
        return np.asarray(image, dtype=np.float32).copy() / 255.0

    @torch.inference_mode()
    def predict(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: Optional[str] = DEFAULT_PROMPT,
        cfg_scale: float = 1.5,
        num_steps: int = 35,
        seed: Optional[int] = None,
    ) -> list[np.ndarray]:
        """Generate 93 360-degree rotational video frames from one input image."""
        prompt = prompt if prompt is not None else DEFAULT_PROMPT
        pil_img = self._to_pil_rgb(image)

        resized_image = pil_img.resize((IMG_WIDTH, IMG_HEIGHT), resample=Image.BICUBIC)
        np_image = self._scale_image_intensity(np.asarray(resized_image, dtype=np.float32))

        anchor_frame = (
            torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)
        ).to(self.device)
        anchor_frame = anchor_frame * 2.0 - 1.0
        anchor_frame = anchor_frame.unsqueeze(2)

        batch_size = anchor_frame.shape[0]
        input_video = anchor_frame.repeat(1, 1, NUM_FRAMES, 1, 1)

        text_embeddings = self.encode_text(prompt).to(self.device)
        latent_cond = self.encode_video(input_video)
        gen_dtype = next(self.dit.parameters()).dtype
        latent_cond = latent_cond.to(dtype=gen_dtype)
        _, channels, timesteps, height, width = latent_cond.shape
        state_shape = (channels, timesteps, height, width)

        run_seed = seed if seed is not None else int(torch.randint(0, 2**32 - 1, (1,)).item())
        
        condition = self._get_condition(
            text_embeddings,
            latent_cond,
            num_conditional_frames=1,
            dtype=gen_dtype,
        )
        uncondition = self._get_condition(
            torch.zeros_like(text_embeddings),
            latent_cond,
            num_conditional_frames=1,
            dtype=gen_dtype,
        )

        noise = arch_invariant_rand(
            shape=(batch_size,) + state_shape,
            dtype=torch.float32,
            device=self.device,
            seed=run_seed,
        ).to(dtype=gen_dtype)

        seed_generator = torch.Generator(device=self.device)
        seed_generator.manual_seed(run_seed)

        self.sample_scheduler.set_timesteps(
            num_inference_steps=num_steps,
            device=self.device,
            shift=5.0,
            use_kerras_sigma=False,
        )

        latents = noise.clone()

        for timestep in self.sample_scheduler.timesteps:
            timestep_b_t = timestep.view(1, 1).expand(batch_size, 1)

            v_cond = self.denoise(noise, latents, timestep_b_t, condition)
            v_uncond = self.denoise(noise, latents, timestep_b_t, uncondition)
            velocity_pred = v_uncond + cfg_scale * (v_cond - v_uncond)

            latents = self.sample_scheduler.step(
                model_output=velocity_pred,
                timestep=timestep,
                sample=latents,
                return_dict=False,
                generator=seed_generator,
            )[0]

            cond_mask = condition.condition_video_input_mask_B_C_T_H_W
            if cond_mask is not None:
                latents = latents * (1 - cond_mask) + latent_cond * cond_mask

        video = self.decode_latent(latents.float())
        video = (video / 2.0 + 0.5).clamp(0, 1)

        video_np = (
            (video[0].permute(1, 2, 3, 0).detach().cpu().numpy() * 255.0)
            .clip(0, 255)
            .astype(np.uint8)
        )

        return [video_np[i] for i in range(video_np.shape[0])]
