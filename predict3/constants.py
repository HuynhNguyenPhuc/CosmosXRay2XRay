"""Cosmos 3 (`nvidia/Cosmos3-Edge`) architecture constants.

Values below were read directly from the real published config files on
`https://huggingface.co/nvidia/Cosmos3-Edge` (`transformer/config.json`, `vae/config.json`,
`scheduler/scheduler_config.json`) on 2026-08-18 — see docs/cosmos-predict3/PLAN.md §6.2 for
the verification trail. Do not hand-edit these to "round" values; re-derive from the real
config if the checkpoint changes.
"""

from __future__ import annotations

from typing import Final

# ── Model identity ──
COSMOS3_EDGE_REPO_ID: Final[str] = "nvidia/Cosmos3-Edge"

# ── Transformer (transformer/config.json) ──
COSMOS3_HIDDEN_SIZE: Final[int] = 2048
COSMOS3_NUM_HIDDEN_LAYERS: Final[int] = 28
COSMOS3_NUM_ATTENTION_HEADS: Final[int] = 16
COSMOS3_NUM_KEY_VALUE_HEADS: Final[int] = 8
COSMOS3_HEAD_DIM: Final[int] = 128
COSMOS3_LATENT_PATCH_SIZE: Final[int] = 2
COSMOS3_PATCH_LATENT_DIM: Final[int] = 192  # latent_channel * latent_patch_size**2 = 48 * 4

# ── VAE (vae/config.json) — Wan2.2-TI2V-5B AutoencoderKLWan ──
# 16x spatial + 2x2 patch merge (see COSMOS3_LATENT_PATCH_SIZE above) = 32x effective
# spatial token compression, vs. predict2_5's Wan2.1 VAE 8x spatial (no patch merge).
COSMOS3_LATENT_CHANNELS: Final[int] = 48  # vae "z_dim" == transformer "latent_channel"
COSMOS3_VAE_SCALE_SPATIAL: Final[int] = 16
COSMOS3_VAE_SCALE_TEMPORAL: Final[int] = 4  # same as predict2_5's Wan2.1 VAE -> NUM_LATENT_FRAMES formula is reusable

# ── Scheduler (scheduler/scheduler_config.json) ──
COSMOS3_SCHEDULER_PREDICTION_TYPE: Final[str] = "flow_prediction"
COSMOS3_SCHEDULER_NUM_TRAIN_TIMESTEPS: Final[int] = 1000
