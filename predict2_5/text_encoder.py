"""Cosmos-Reason 1.0 Text Encoder Wrapper."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from predict2_5.constants import CR1_EMBEDDING_DIM, CR1_MAX_LENGTH, DEFAULT_EMBEDDING_DIR
from shared.utils import get_logger

log = get_logger(__name__)


class CR1TextEncoder:
    """Text encoder wrapper for Cosmos-Reason 1.0."""

    def __init__(
        self,
        text_encoder_ckpt_path: Optional[str] = None,
        embedding_dir: str = DEFAULT_EMBEDDING_DIR,
        device: str = "cuda",
        cpu_offload: bool = True,
    ) -> None:
        """
        Initializes the Cosmos-Reason 1.0 Text Encoder Wrapper.

        Args:
            text_encoder_ckpt_path: Checkpoint path for the Cosmos-Reason 1.0 text encoder.
            embedding_dir: Directory where precomputed .pkl embeddings are stored.
            device: Target device for output tensors.
            cpu_offload: Whether to offload text encoder to CPU when not in use.
        """
        self.text_encoder_ckpt_path = text_encoder_ckpt_path
        self.embedding_dir = Path(embedding_dir)
        self.device = device
        self.cpu_offload = cpu_offload

        self._text_encoder = None
        self._cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def prompt_to_filename(prompt: str) -> str:
        """Convert prompt text to deterministic embedding filename stem."""
        return (
            (prompt.strip().replace(" ", "_").replace("'", "").replace(",", "")[:100])
            or "null"
        )

    def _load_embedding_from_disk(self, prompt: str) -> Optional[torch.Tensor]:
        """Load precomputed embedding from disk if available."""
        emb_path = self.embedding_dir / f"{self.prompt_to_filename(prompt)}.pkl"
        
        if not emb_path.exists():
            return None

        try:
            with open(emb_path, "rb") as f:
                data = pickle.load(f)

            raw = data[0] if isinstance(data, list) and len(data) > 0 else data
            emb = torch.as_tensor(raw, dtype=torch.float32)

            if emb.ndim != 2:
                log.warning("Invalid embedding ndim=%s at %s; expected 2D", emb.ndim, emb_path)
                return None

            padded = torch.zeros((1, CR1_MAX_LENGTH, CR1_EMBEDDING_DIM), dtype=torch.float32)
            use_len = min(emb.shape[0], CR1_MAX_LENGTH)
            use_dim = min(emb.shape[1], CR1_EMBEDDING_DIM)
            padded[0, :use_len, :use_dim] = emb[:use_len, :use_dim]

            return padded
        
        except Exception as exc:
            log.warning("Could not load %s: %s", emb_path, exc)
            return None

    def _ensure_online_text_encoder(self) -> None:
        """Ensure the Cosmos-Reason 1.0 Text Encoder is initialized."""
        if self._text_encoder is not None:
            return

        if not self.text_encoder_ckpt_path:
            raise ValueError("Text encoder checkpoint path is required for online fallback when .pkl is missing")

        from cosmos_predict2._src.predict2.text_encoders.text_encoder import (
            TextEncoder,
            TextEncoderConfig,
        )

        cfg = TextEncoderConfig(
            embedding_concat_strategy="full_concat",
            ckpt_path=self.text_encoder_ckpt_path,
        )

        online_device = "cpu" if self.cpu_offload else self.device
        self._text_encoder = TextEncoder(cfg, device=online_device)

        log.info("Text encoder initialized on %s", online_device)

    def _compute_text_embedding_online(self, prompt: str) -> torch.Tensor:
        """Compute text embedding on-the-fly using Cosmos-Reason 1.0 Text Encoder."""
        self._ensure_online_text_encoder()

        if self.cpu_offload and self.device.startswith("cuda"):
            self._text_encoder.model = self._text_encoder.model.to(self.device)

        text_embeddings = self._text_encoder.compute_text_embeddings_online(
            {"text": [prompt]}, "text"
        )

        if self.cpu_offload and self.device.startswith("cuda"):
            self._text_encoder.model = self._text_encoder.model.to("cpu")
            torch.cuda.empty_cache()

        if isinstance(text_embeddings, np.ndarray):
            text_embeddings = torch.from_numpy(text_embeddings)

        text_embeddings = text_embeddings.to(dtype=torch.float32)

        padded = torch.zeros((1, CR1_MAX_LENGTH, CR1_EMBEDDING_DIM), dtype=torch.float32)
        use_len = min(text_embeddings.shape[1], CR1_MAX_LENGTH)
        use_dim = min(text_embeddings.shape[2], CR1_EMBEDDING_DIM)
        padded[0, :use_len, :use_dim] = text_embeddings[0, :use_len, :use_dim]

        return padded

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a single prompt."""
        key = prompt or ""
        if key in self._cache:
            return self._cache[key]

        embedding = self._load_embedding_from_disk(key)
        if embedding is None:
            if self.text_encoder_ckpt_path:
                embedding = self._compute_text_embedding_online(key)
            else:
                log.warning("Missing embedding for prompt '%s' and no checkpoint provided; returning zeros", key)
                embedding = torch.zeros((1, CR1_MAX_LENGTH, CR1_EMBEDDING_DIM), dtype=torch.float32)

        self._cache[key] = embedding
        return embedding

    def encode_prompts(self, prompts: list[str], device: torch.device | str) -> torch.Tensor:
        """Encode multiple prompts into a batch tensor (N, L, D)."""
        tensors = [self.encode_prompt(p).squeeze(0) for p in prompts]
        return torch.stack(tensors, dim=0).to(device=device, dtype=torch.float32)
