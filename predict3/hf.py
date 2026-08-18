"""HuggingFace config resolution for Cosmos 3 (`nvidia/Cosmos3-Edge`).

Scope: config/metadata files only (each a few KB) — enough to construct the real
Cosmos3OmniTransformer/AutoencoderKLWan/UniPCMultistepScheduler architecture for wiring
verification (docs/cosmos-predict3/PLAN.md §7). Downloading the real multi-GB safetensors
weight shards is out of scope here; see PLAN.md §5 for that follow-up (analogous to
predict2_5/hf.py's `text_encoder_snapshot()` curl-based large-shard downloader).
"""

from __future__ import annotations

import json
from typing import Any, Dict

from huggingface_hub import hf_hub_download

from shared.utils import get_logger

log = get_logger(__name__)


def _load_json_config(repo_id: str, filename: str) -> Dict[str, Any]:
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(path) as f:
        return json.load(f)


def load_transformer_config(repo_id: str) -> Dict[str, Any]:
    """Real `Cosmos3OmniTransformer` config (``transformer/config.json``)."""
    return _load_json_config(repo_id, "transformer/config.json")


def load_vae_config(repo_id: str) -> Dict[str, Any]:
    """Real `AutoencoderKLWan` config (``vae/config.json``)."""
    return _load_json_config(repo_id, "vae/config.json")


def load_scheduler_config(repo_id: str) -> Dict[str, Any]:
    """Real `UniPCMultistepScheduler` config (``scheduler/scheduler_config.json``)."""
    return _load_json_config(repo_id, "scheduler/scheduler_config.json")
