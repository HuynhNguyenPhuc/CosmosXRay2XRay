"""Mixture-of-Transformers tower split for `Cosmos3OmniTransformer`.

Cosmos 3 carries **two parameter sets per layer** that share only the joint self-attention
operator (PLAN.md §2):

* the **AR Reasoner tower** ("understanding" / ``und``) — causal, consumes language + ViT
  vision tokens;
* the **DM Generator tower** ("generation" / ``gen``) — bidirectional, consumes VAE
  video/audio/action tokens.

Post-training here trains the generator only and leaves the reasoner frozen. That is safe
precisely because the towers are asymmetric: in ``Cosmos3AttnProcessor.__call__`` the causal
pathway attends over ``q_und/k_und/v_und`` alone, so **und activations never depend on gen** —
freezing the reasoner cannot starve the generator of signal, while fine-tuning it on a narrow
X-ray corpus would degrade the language/reasoning prior we want to keep.

Naming convention, read off the real module (not guessed): the generator duplicates a layer's
parameters under an ``_moe_gen`` suffix (``mlp_moe_gen``, ``input_layernorm_moe_gen``,
``post_attention_layernorm_moe_gen``) and, inside attention, under diffusers' "added-stream"
``add_``/``_added_`` prefixes (``add_q_proj``, ``add_k_proj``, ``add_v_proj``, ``to_add_out``,
``norm_added_q``, ``norm_added_k``). The unsuffixed twins (``mlp``, ``input_layernorm``,
``to_q``/``to_k``/``to_v``/``to_out``) are the reasoner's.

Two names need reading the attention source rather than pattern-matching:

* ``k_norm_und_for_gen`` — despite sitting on the *und* key path, its output ``k_und_for_gen``
  is consumed **only** by the generation pathway (``all_k = cat([k_und_for_gen, k_gen])``);
  the causal pathway uses the unnormalized ``k_und``. It therefore affects the generator's
  reading of und keys and nothing about the reasoner's own output → **generator side**.
* ``time_embedder`` — diffusion-timestep embedding. The causal AR tower has no timestep, so
  this is generator-only despite having no ``_moe_gen`` suffix.

NOTE — this corrects PLAN.md §5, which (secondhand from `CosmosXRay360`) named the trainable
keys ``moe_gen``, ``time_embedder``, ``vae2llm``, ``llm2vae``. Only ``time_embedder`` exists
verbatim in the `diffusers` implementation; ``moe_gen`` is a *suffix* not a module, and the
latent<->hidden projections are called ``proj_in``/``proj_out``. Those names may still be
correct for `cosmos-framework`'s native checkpoint layout, but they do not match what this
code trains against.
"""

from __future__ import annotations

import torch.nn as nn

from shared.utils import get_logger

log = get_logger(__name__)


# Top-level modules belonging to the AR Reasoner tower: text token embedding, the final
# understanding-stream norm, and the language-modelling head. `lm_head` matters only for text
# generation, which this pipeline never does.
REASONER_TOP_LEVEL: tuple[str, ...] = ("embed_tokens", "lm_head", "norm")

# Top-level modules belonging to the DM Generator tower: VAE-latent patch projections in/out,
# the generation-stream final norm, the diffusion timestep embedding, and the action-modality
# parameters (`camera_pose` conditioning).
GENERATOR_TOP_LEVEL: tuple[str, ...] = (
    "proj_in",
    "proj_out",
    "norm_moe_gen",
    "time_embedder",
    "action_proj_in",
    "action_proj_out",
    "action_modality_embed",
)

# Per-layer substrings marking a parameter as generator-side (see module docstring).
GENERATOR_LAYER_MARKERS: tuple[str, ...] = (
    "_moe_gen",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "norm_added_q",
    "norm_added_k",
    "k_norm_und_for_gen",
)


def is_generator_param(name: str) -> bool:
    """Whether ``name`` (a ``Cosmos3OmniTransformer.named_parameters()`` key) is generator-side.

    Everything not classified as generator-side is reasoner-side; there is no third category,
    so :func:`freeze_reasoner_tower` cannot silently skip a parameter.
    """
    top = name.split(".")[0]
    if top in GENERATOR_TOP_LEVEL:
        return True
    if top in REASONER_TOP_LEVEL:
        return False
    return any(marker in name for marker in GENERATOR_LAYER_MARKERS)


def freeze_reasoner_tower(transformer: nn.Module) -> dict[str, int]:
    """Freeze the AR Reasoner tower in-place; leave the DM Generator tower trainable.

    Returns a summary dict with ``trainable``/``frozen`` parameter-element counts, which the
    caller logs so a misclassification shows up as an obviously wrong split rather than
    silently training (or freezing) the wrong half.
    """
    trainable = frozen = 0
    for name, param in transformer.named_parameters():
        if is_generator_param(name):
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)
            frozen += param.numel()

    if trainable == 0:
        raise RuntimeError(
            "freeze_reasoner_tower() left zero trainable parameters — the generator-side naming "
            "convention likely changed upstream; re-check predict3/tower.py against "
            "Cosmos3OmniTransformer.named_parameters()."
        )

    total = trainable + frozen
    log.info(
        f"[predict3] Reasoner tower frozen — trainable {trainable:,} ({trainable / total:.1%}), "
        f"frozen {frozen:,} ({frozen / total:.1%})"
    )
    return {"trainable": trainable, "frozen": frozen}
