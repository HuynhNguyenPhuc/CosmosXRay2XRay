# Cosmos 3 Backbone Migration Plan (`PLAN.md`)

**Status (2026-08-18):** `predict3/` now exists — scaffolding + meta-device/synthetic smoke test only (§7,
§8). Real pretrained-weight loading and a real X-ray datamodule are NOT wired up — see §8's "not done" list
and §5 for what's still needed before this is a real training pipeline.
**Scope:** Whether/how to add a third pipeline to `CosmosXRay2XRay` that post-trains NVIDIA **Cosmos 3**
for 7-view X-ray synthesis, alongside the existing `predict2_5/` (base DiT) and `transfer2_5/` (ControlNet)
pipelines.

---

## 1. Context

`predict2_5/` fine-tunes Cosmos-Predict2.5 (`MinimalV1LVGDiT`, 2B) and `transfer2_5/` adds a ControlNet/VACE
branch (`MinimalV4LVGControlVaceDiT`) on top of it. Both are complete and are the production pipelines today.

NVIDIA's **Cosmos 3** (tech report [arXiv:2606.02800](https://arxiv.org/abs/2606.02800),
[NVIDIA/cosmos](https://github.com/NVIDIA/cosmos),
[NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework)) unifies Predict, Transfer, and Reason
into one Mixture-of-Transformers (MoT) omni-model. There is no repo named `cosmos-predict3` — that name is
used here only as a working label for "the migration effort," matching the sibling `CosmosXRay360` repo's
convention.

**This plan exists only as a design/status doc.** No `predict3/` code should be written against it until the
blocker in §3 is cleared, per the outcome of a real attempt already made in `CosmosXRay360` (see §5).

---

## 2. What Cosmos 3 actually is (architecture deltas vs. `predict2_5`/`transfer2_5`)

| Aspect | `predict2_5` / `transfer2_5` (this repo, today) | Cosmos 3 |
| --- | --- | --- |
| Architecture | Standalone video DiT (`MinimalV1LVGDiT`), ControlNet-style parallel `control_blocks` for `transfer2_5` | Mixture-of-Transformers: causal **AR Reasoner tower** (language + vision tokens) + bidirectional **DM Generator tower** (VAE video tokens), sharing one joint self-attention op |
| Variants | `predict2_5-2B` | `nvidia/Cosmos3-Edge` (4B), `nvidia/Cosmos3-Nano` (16B), `nvidia/Cosmos3-Super` (64B) |
| VAE tokenizer | Wan 2.1 VAE (8× spatial downsample) | Wan 2.2 VAE (16× spatial + 2×2 patch merge = 32× effective spatial compression) |
| I2V / view conditioning | `transfer2_5` splices the control latent in via `control_embedder → control_blocks → hints → base blocks`; `predict2_5` uses per-step frame-replacement splicing for the anchor view | Native token concatenation — a clean conditioning latent (e.g. 0° PA view) is concatenated as frame 0 ahead of the noisy target latents; no custom splicing loop needed |
| Text conditioning | `predict2_5.text_encoder` (CR1 encoder), concatenated T5/Reason1 embeddings | Native AR Reasoner tower, structured JSON captions |
| Training surface | PyTorch Lightning + `torchrun`/FSDP (`predict2_5/trainer.py`, `transfer2_5/trainer.py`) | `cosmos-framework`'s `cosmos_framework.scripts.train`, pydantic/Hydra TOML recipes |
| Inference surface | Custom UniPC sampling loop in-repo | `diffusers.Cosmos3OmniPipeline`, or TRT-LLM/NIM |

This table is derived from the equivalent audited comparison in `CosmosXRay360/docs/cosmos-predict3/PLAN.md`
(dated 2026-08-17), not independently re-verified against Cosmos 3 source in this repo — flag as
"external, secondhand" if it needs re-checking before anything is built on it.

---

## 3. Current upstream blocker

A real GCP A100 test in the sibling `CosmosXRay360` repo (documented in its `CHAT.md`, 2026-08-17) attempted
`diffusers.Cosmos3OmniPipeline` zero-shot inference against `nvidia/Cosmos3-Edge` and hit a hard, unresolved
blocker:

1. **`diffusers==0.39.0`** (current PyPI release at the time) does not recognize several config fields on the
   published `nvidia/Cosmos3-Edge` checkpoint (e.g. `use_und_k_norm_for_gen`) — pipeline construction silently
   random-initializes ~120 attention/MLP weights instead of loading them from the checkpoint.
2. The fix exists on `diffusers`'s unreleased GitHub `main`, but installing it pulls in `huggingface-hub>=1.0`,
   which conflicts with `transformers`'s `<1.0` pin — no combination of currently-installable packages
   resolves both at once.
3. Outcome: zero-shot Cosmos 3 inference is **not currently runnable** with released packages. This is an
   upstream ecosystem gap, not something fixable from application code in either repo.

This has not been independently re-checked in `CosmosXRay2XRay`'s own environment — see §6, Next Action.

---

## 4. Prior art: don't repeat the `CosmosXRay360` scaffolding mistake

`CosmosXRay360`'s own `predict3/` package was audited (same `PLAN.md`) and found to be a ~95% copy of
`predict2_5/` with cosmetic `V3` renames — still importing `MinimalV1LVGDiT` and the Wan 2.1 VAE, with
invented model ids (`cosmos3-nano-16b-mot-v3`) that don't correspond to real artifacts. It did not actually
implement Cosmos 3.

**Implication for this repo:** do not create a `predict3/` directory that structurally mirrors `predict2_5/`
as a starting point "to fill in later." If/when this migration proceeds, it should be built directly against
verified real Cosmos 3 APIs (`diffusers.Cosmos3OmniPipeline`, `cosmos_framework.scripts.train`), once §3 is
cleared, not as a renamed copy of the existing DiT pipeline.

**Followed as of §7/§8:** `predict3/module.py` imports the real `Cosmos3OmniTransformer` / `AutoencoderKLWan`
/ `Cosmos3OmniPipeline` classes from `diffusers` directly (not `cosmos_predict2`'s `MinimalV1LVGDiT`), and
constructs them from the real published `nvidia/Cosmos3-Edge` config (not invented ids). It does not use
`cosmos_framework.scripts.train` yet — that's still open, see §5/§8.

---

## 5. Path forward

- `predict2_5/` and `transfer2_5/` remain the only production pipelines in `CosmosXRay2XRay` until §3 clears.
- Once `diffusers`/`transformers`/`huggingface-hub` versions align (tracked upstream, not in this repo):
  1. Convert the 7-view DRR dataset to the format `cosmos-framework`'s `SFTDataset` expects (lossless MP4 +
     structured-JSON `captions.jsonl`), analogous to the conversion already scoped in `CosmosXRay360`.
  2. Write a TOML SFT recipe selecting only the generator-side keys for training
     (`moe_gen`, `time_embedder`, `vae2llm`, `llm2vae`) so the causal AR Reasoner tower stays frozen.
     > **Correction (2026-08-18):** those four names are secondhand from `CosmosXRay360` and do **not**
     > match the `diffusers` implementation this repo actually trains against. Only `time_embedder`
     > exists verbatim; `moe_gen` is a *suffix* (`mlp_moe_gen`, `norm_moe_gen`, …) not a module, and the
     > latent↔hidden projections are `proj_in`/`proj_out`. The verified split is implemented in
     > `predict3/tower.py` and documented in §10 — use that, not this list. The names above may still
     > be correct for `cosmos-framework`'s own checkpoint layout, which is a separate naming scheme.
  3. Re-derive the physical regularizers (view-angle consistency, attenuation mass conservation) against
     Cosmos 3's velocity-prediction formulation and the Wan 2.2 VAE's 16×16 latent grid (for a 256×256 frame).
  4. Only then create `predict3/` (or fold it into `predict2_5/` as a backend switch) as real code.

## 6. Next action

Before any of §5 happens: re-check whether `diffusers`, `transformers`, and `huggingface-hub` have released
versions that resolve the §3 conflict since 2026-08-17 — this plan's blocker is only a few days old relative
to today. A quick `pip index versions diffusers/transformers/huggingface-hub` check (or checking release
notes) is enough to know if it's worth revisiting; don't assume it's still broken indefinitely.

### 6.1 Recheck result (2026-08-18, via PyPI/GitHub metadata only — not an install test)

- **`diffusers`**: still `0.39.0` on PyPI (published 2026-07-03, unchanged) — the stable release still lacks
  the `nvidia/Cosmos3-Edge` config fields (e.g. `use_und_k_norm_for_gen`). The fix is still only on GitHub
  `main`, unreleased.
- **`transformers`**: now `5.15.0` on PyPI — a major-version jump since the 2026-08-17 test, and its
  `huggingface-hub` requirement is now `<2.0,>=1.5.0` (previously pinned `<1.0`, which is what caused the
  original conflict).
- **`huggingface-hub`**: `1.27.0` on PyPI; `diffusers`'s GitHub `main` branch now requires
  `huggingface-hub>=1.23.0,<2.0`.
- **Conclusion**: `[1.23.0, 2.0)` (diffusers main) and `[1.5.0, 2.0)` (transformers 5.15.0) overlap at
  `[1.5.0, 2.0)`, so the specific `huggingface-hub` version conflict that blocked `CosmosXRay360`'s test
  looks **resolved** — installing `diffusers` from GitHub `main` alongside current `transformers` should no
  longer collide. This is a metadata-only check (`pip show`/PyPI JSON/GitHub `setup.py`), not a real install
  + `Cosmos3OmniPipeline` load test, so treat it as "worth re-attempting," not "confirmed fixed."

### 6.2 Real install + load test (2026-08-18, confirmed on real hardware)

Ran the actual install in an isolated venv on this machine (which has a real NVIDIA GPU, driver
580.173.02 / CUDA 13.0, previously unused because the checked-in `torch` was a CPU-only build):

- `pip install git+https://github.com/huggingface/diffusers.git@main "transformers>=5.15.0" "huggingface-hub>=1.23.0,<2.0" accelerate`
  resolved and installed cleanly — `diffusers==0.40.0.dev0`, `transformers==5.15.0`,
  `huggingface-hub==1.27.0`, **zero dependency conflicts**.
- `from diffusers import Cosmos3OmniPipeline` — imports.
- Pulled the real `nvidia/Cosmos3-Edge` config from HF Hub (`transformer/config.json`): its
  `_diffusers_version` is `0.40.0.dev0` (exact match to the installed dev build) and it sets
  `use_und_k_norm_for_gen: true` — the exact field §3 says the stable `diffusers==0.39.0` release
  drops silently.
- `Cosmos3OmniTransformer.from_config(cfg)` on `meta` device (no weight download): **constructed
  successfully**, `model.config.use_und_k_norm_for_gen == True` (correctly honored, not dropped),
  3.37B params — consistent with the "4B" `Cosmos3-Edge` variant. Two harmless config keys
  (`backbone_type`, `temporal_compression_factor`) were ignored with a warning, not an error.
- **Conclusion**: the §3 blocker is confirmed resolved as of today, using `diffusers` `main` +
  current `transformers`/`huggingface-hub`. This did not download real weights or run inference
  end-to-end (`from_pretrained` + a real forward pass), so full runtime correctness is still
  unverified — but the specific failure mode in §3 (config fields silently dropped, dependency
  conflict blocking install) no longer reproduces.

## 7. Backbone verification (2026-08-18)

All three backbones this repo cares about were checked on this machine (real GPU, driver 580.173.02
/ CUDA 13.0):

| Backbone | Result |
| --- | --- |
| `predict2_5` (`MinimalV1LVGDiT`) | Constructed for real on `meta` device via `cosmos-predict2.5[cu130]`: 2.06B params, matches the documented "2B" config. `predict2_5.CosmosXRay2XRayMultiview` imports cleanly. |
| `transfer2_5` (`MinimalV4LVGControlVaceDiT`) | Constructed for real on `meta` device: 3.09B params, 14 `control_blocks` (consistent with `vace_block_every_n=2` over 28 base blocks) + `control_embedder` both present. Required a code fix — see below. |
| Cosmos 3 (`Cosmos3OmniTransformer`, no `predict3/` package yet) | Constructed for real on `meta` device from the actual published `nvidia/Cosmos3-Edge` config via `diffusers` `main` — see §6.2. Not integrated into this repo (by design, §4). |

**Real bug found and fixed**: `transfer2_5/module.py` could not be imported at all in a correctly
installed environment (with or without `predict2_5` imported first). Root cause: `cosmos_predict2`
and `cosmos_transfer2` each vendor their own copy of `imaginaire/lazy_config/lazy.py`, and both
unconditionally call `OmegaConf.register_new_resolver("add", ...)` at import time. Since
`transfer2_5/module.py` necessarily imports from both (`cosmos_predict2._src.predict2.*` for the base
DiT config classes it reuses, `cosmos_transfer2._src.transfer2.*` for the control DiT), the second
registration always raised `ValueError: resolver 'add' is already registered`, independent of import
order. Fixed in `transfer2_5/module.py` by temporarily making `OmegaConf.register_new_resolver`
idempotent for the duration of the `cosmos_transfer2` import only (not a submodule patch — the pinned
vendored code is untouched). Verified: both real backbones now construct, and the full `tests/` suite
(7 tests) passes.

Also pinned `huggingface-hub<1.0,>=0.30.0` in `requirements.txt`: following `CLAUDE.md`'s documented
setup exactly (`uv pip install -e ".[cu130]"` for both submodules, then `uv pip install -r
requirements.txt`) resolves `huggingface-hub` to `1.27.0` (pulled in by `diffusers`/`gradio`/
`accelerate`/`peft`/`timm`, all present via the submodules' own dependency trees), which conflicts
with the pinned `transformers==4.51.3`'s runtime-enforced `huggingface-hub<1.0` check and made
`import predict2_5` / `import transfer2_5` fail outright. Confirmed the pin fixes it end to end.

## 8. `predict3/` scaffolding build (2026-08-18)

Built `predict3/` (`__init__.py`, `constants.py`, `hf.py`, `module.py`, `trainer.py`) and
`tests/test_predict3.py`, per the "scaffolding + meta-device smoke test" scope agreed for this
first pass (real weight download / real training deferred — see "Not done" below).

### 8.1 Design decisions

- **No separate text encoder.** Unlike `predict2_5`'s `CR1TextEncoder`, Cosmos 3 has no external
  text encoder — `Cosmos3OmniPipeline.tokenize_prompt`'s own docstring: "This pipeline does not run a
  separate text encoder: the joint Cosmos3 transformer consumes raw token IDs alongside vision
  tokens." `predict3/module.py` reuses `AutoTokenizer` + the pipeline's own `tokenize_prompt`.
- **Native I2V via `condition_frame_indexes=[0]`.** `Cosmos3OmniPipeline._prepare_vision_segment`
  takes a list of latent-frame indices to exclude from `vision_noisy_frame_indexes` (never noised,
  never included in the loss — the transformer's own `_unpatchify_and_unpack_latents` zero-fills
  predictions at those positions). Passing `condition_frame_indexes=[0]` marks the anchor X-ray
  view (encoded as latent frame 0) as the I2V conditioning frame — this *is* the "native I2V frame-0
  token concatenation" described in §2, confirmed by reading the actual `diffusers` source rather
  than inferred from documentation.
- **Reuses the pipeline's private packing helpers, not reimplemented.** Cosmos 3's joint-sequence
  packing (mRoPE position ids per modality, `text_indexes`/`vision_sequence_indexes`/
  `vision_mse_loss_indexes`, per-frame noisy/clean splitting) is intricate — see
  `diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py`'s `_prepare_text_segment`/
  `_prepare_vision_segment`. `predict3/module.py` constructs a real `Cosmos3OmniPipeline` instance
  purely to call these (private, underscore-prefixed) methods directly, rather than re-deriving the
  packing scheme independently — reimplementing it was judged too easy to get silently wrong (right
  shapes, wrong RoPE positions or loss mask) in a way that "runs" without erroring. This is the same
  reuse-over-reimplementation principle `predict2_5`/`transfer2_5` already follow for
  `cosmos_predict2`/`cosmos_transfer2`'s DiT internals.
- **Rectified-flow math reimplemented locally, not imported from `cosmos_predict2`.** `predict2_5`
  uses `cosmos_predict2._src.predict2.schedulers.rectified_flow.RectifiedFlow` for its
  interpolation/timestep-sampling math. `predict3` can't import that: `cosmos_predict2` needs
  `huggingface-hub<1.0`, `diffusers` `main` needs `huggingface-hub>=1.23,<2.0` — the two can't be
  installed in one venv (§7 first surfaced this; §8.3 confirms it directly). The formula itself
  (`x_t = eps*t + x_1*(1-t)`, `v_target = eps - x_1`) is copied verbatim from
  `RectifiedFlow.get_interpolation`'s docstring/implementation for convention consistency, as a
  few self-contained lines in `predict3/module.py` rather than a cross-venv import.

### 8.2 Real bug caught by the smoke test, fixed

First smoke-test run completed (`loss=1.515843`) but reported **`params_with_grad=0/0`** —
`Cosmos3OmniPipeline` is a `DiffusionPipeline`, not an `nn.Module`; storing the transformer only
inside `self._pipeline` meant `LightningModule.parameters()` (and therefore
`configure_optimizers`'s `AdamW`, and `Trainer`'s automatic device placement/DDP wrapping) walked
right past it and saw nothing. Training would have silently done nothing — loss computed and
backpropagated correctly, but no optimizer step would ever change any weight.

Fixed by registering `self.transformer = self._pipeline.transformer` and
`self.vae = self._pipeline.vae` as direct submodule attributes in `__init__`/`setup()` (same
objects, not copies, so gradients still flow through the shared parameters) — matching
`predict2_5`'s existing convention of registering even its frozen VAE (`self.tokenizer`) as a real
submodule so it follows the module across devices. `configure_optimizers` now builds `AdamW` from
`self.transformer.parameters()` only (VAE frozen via `requires_grad_(False)`).
`tests/test_predict3.py::test_predict3_transformer_registered_as_submodule` regression-tests this.

### 8.3 Verification

Ran in the isolated `diffusers`-main venv (§6.2's), plus a second confirmation of §3's bug found by
constructing directly from the real config in the *main* repo venv (`diffusers==0.39.0`,
`transformers==4.51.3` — the pins `predict2_5`/`transfer2_5` need):

- **Main venv, real `Cosmos3OmniTransformer.from_config(real_config)`**: logged
  `The config attributes {..., 'use_und_k_norm_for_gen': True} were ... ignored` — independently
  reproduces §3's bug via direct construction, not just metadata inspection. Then failed with
  `ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.` —
  `transformers==4.51.3` doesn't know the tokenizer backend class `nvidia/Cosmos3-Edge`'s
  `text_tokenizer/tokenizer_config.json` specifies (needs `transformers>=5`). Both failures
  confirm, independently of §6.2, that `predict3` cannot run in the same venv as `predict2_5`/
  `transfer2_5` — not just via `huggingface-hub` version metadata, but via real construction
  failures.
- **`diffusers`-main venv (§6.2's), CPU**: `predict3.trainer --smoke-test --device cpu` — real
  forward + backward pass through the real 3.37B-param `Cosmos3OmniTransformer` with native I2V
  packing, finite loss, `params_with_grad=543/745` (543 real transformer tensors gradient-bearing,
  202 frozen VAE tensors correctly excluded).
- **Same venv, GPU, bf16**: peak 16.8 GB VRAM, same 543/745 gradient split, finite loss — fits
  comfortably on the 23.46 GB GPU (fp32 alone OOM'd at `backward()`, as expected for a 3.37B-param
  model with no memory optimization — not a wiring bug, just realistic fp32 memory pressure).
  Surfaced one more real detail for the future real-training implementation: casting the whole
  transformer to bf16 via a blanket `.to(torch.bfloat16)` triggers `Cosmos3OmniTransformer`'s own
  warning that `time_embedder` should stay float32 — use Lightning's `Trainer(precision="bf16-mixed")`
  (autocast, per-op) rather than a blanket dtype cast when real training is wired up.
- **`tests/test_predict3.py`** (3 tests): pass in the `diffusers`-main venv; skip cleanly (not
  fail, not hang) in the main repo venv via a guard on `transformers`'s major version — a bare
  `hasattr(diffusers, "Cosmos3OmniPipeline")` check is insufficient, since `diffusers==0.39.0`
  already exports that class by name, it just silently mishandles the real config (§3/§6.2's bug).
- **Existing `tests/` suite**: still 7 passed (+ predict3's 1 skipped) in the main venv, confirming
  the new package didn't regress `predict2_5`/`transfer2_5`.

### Not done (tracked, not implemented)

- Real `nvidia/Cosmos3-Edge` weight download + `non_strict_load`-style loading into the constructed
  architecture (currently randomly initialized).
- A real datamodule feeding `training_step` — needs the DRR dataset → captions conversion from §5
  step 1 first, since that also determines the real prompt/caption format used by
  `tokenize_prompt`.
- `cosmos-framework`'s SFT TOML recipe path (§5 steps 2–3) — `predict3` currently trains via a
  hand-written Lightning `training_step`, not `cosmos_framework.scripts.train`.
- Physical regularizers (view-angle consistency, attenuation mass conservation) re-derived for
  Cosmos 3's velocity formulation (§5 step 3).

## 9. Camera conditioning via the `camera_pose` action port (2026-08-18)

### 9.1 Why not the text prompt

`predict2_5` formats camera geometry into prose (`XRAY_PROMPT_TEMPLATE`, `shared/constants.py`):
`azimuth {azimuth:.1f} deg, elevation {elevation:.1f} deg`. That was the only option available —
`MinimalV1LVGDiT` has exactly one conditioning port (cross-attention text embeddings). It has three
structural weaknesses:

- **Discretization** — BPE turns `"315.0"` and `"314.0"` into unrelated token sequences; the model
  must recover continuous geometry from discrete token identity.
- **No cyclic prior** — 0° and 359° are adjacent geometrically but maximally distant in token space.
- **Signal dilution** — the camera numbers are ~6 words inside ~150 words of DRR-physics boilerplate
  that is *identical across all 7 views*, so the only view-discriminative content is a tiny fraction
  of the prompt.

### 9.2 What Cosmos 3 offers instead

Cosmos 3 has a third conditioning port — **action** — with a registered `camera_pose` embodiment
domain. Verified in both `diffusers` and the `cosmos-framework` submodule (they agree exactly):

| Fact | Value | Source |
| --- | --- | --- |
| Domain id | `2` | `EMBODIMENT_TO_DOMAIN_ID["camera_pose"]` |
| Raw action width | `9` = translation(3) + rot6d(6) | `EMBODIMENT_TO_RAW_ACTION_DIM["camera_pose"]` |
| Projection | `DomainAwareLinear(64, 2048, 32)` — continuous, per-domain weights | `Cosmos3OmniTransformer.action_proj_in` |
| Model action width | `64` (raw 9 zero-padded) | `transformer/config.json` `action_dim` |

Because `action_proj_in` is a plain linear map, nearby viewing angles produce nearby vectors — the
metric and continuity properties the text path destroys.

### 9.3 Implementation

`predict3/camera.py` composes two existing implementations rather than re-deriving either, since a
silent mismatch would corrupt the training signal without raising:

- **Geometry** — `renderers.diffdrr.renderer.look_at_view_matrices`, the same look-at math the DRR
  renderer uses to actually render the views, so conditioning poses cannot drift from rendered
  geometry. This required a small behavior-preserving refactor: `look_at_view_poses` previously
  built the `(N,4,4)` matrix and wrapped it in a `diffdrr` `RigidTransform` in one function, so the
  matrix was unreachable without `diffdrr` installed. The matrix construction is now
  `look_at_view_matrices` and `look_at_view_poses` calls it — one implementation, and `predict3`
  (whose venv has no `diffdrr`) can reach it. Verified `look_at_view_poses(...).matrix` is
  bit-identical to `look_at_view_matrices(...)`.
- **Pose → action encoding** — `cosmos-framework`'s `pose_abs_to_rel`, NVIDIA's own implementation
  of the layout the pretrained head expects: `(T,4,4)` camera-to-world → `(T-1, 9)`.

**Conventions resolved from the source** (these were open questions before the `cosmos-framework`
submodule was added):

- Actions are **relative**, not absolute. Default here is `backward_anchored`
  (`T_0^{-1} @ T_{i+1}`), i.e. every view encoded relative to the anchor view — matching this
  task's framing ("given the anchor X-ray, produce the view at this relative offset") and keeping
  each action independent of its neighbours. `backward_framewise` (consecutive deltas) suits a
  continuous orbit sweep and is selectable via `camera_pose_convention`.
- Actions describe **transitions**: `T` frames yield `T-1` action tokens. This is Cosmos 3's
  `chunk_size` / `chunk_size + 1` frames contract, not an off-by-one.
- Camera poses are **pure conditioning** — never noised, excluded from the loss. Passing every
  action index as a condition frame empties `action_mse_loss_indexes`, which the transformer
  explicitly guards on (`if action_mse_loss_indexes.numel() > 0`) to skip timestep embedding, so
  the poses enter as clean projected conditioning. The library supports this directly.

### 9.4 Verification

- **Geometric self-consistency** (`test_camera_actions_are_geometrically_symmetric`): under
  `backward_anchored`, RAO (315°) and LAO (45°) produce equal-norm action vectors (6.284), as do
  lateral-right (270°) and lateral-left (90°) (11.402), and PA (180°, the AP antipode) is the
  largest (16.062). Shape checks alone would not catch an axis/sign error; this does.
- **The action port actually receives gradients** (`test_camera_action_conditioning_reaches_action_pathway`):
  a silently-ignored action tensor would still yield a finite loss, so shape checks prove nothing.
  Comparing gradient-bearing parameters with and without conditioning gives exactly three extra:
  `action_proj_in.fc.weight`, `action_proj_in.bias.weight`, `action_modality_embed` — and
  `action_proj_out` correctly absent, confirming actions are conditioning-only.
- **Real full-depth run**: the real 28-layer, 3.37B-param `Cosmos3OmniTransformer` on GPU in bf16
  with camera conditioning — finite loss, 546/549 parameters gradient-bearing, action parameters
  among them, 16.84 GB peak VRAM.
- **`tests/test_predict3.py`**: 7 tests pass in the `diffusers`-main venv, skip cleanly in the main
  venv. Existing suite still 7 passed + 1 skipped, so the renderer refactor did not regress
  `predict2_5`/`transfer2_5`.

### 9.5 Open risk, not resolved

The 9D representation transfers exactly, but the **pretrained `camera_pose` weights were trained on
natural-scene camera motion**. X-ray is a *transmission* modality: views at azimuth θ and θ+180° are
near-mirror images (the same ray paths traversed in reverse), whereas orbiting an opaque scene by
180° reveals entirely different, mutually-occluded content. The pretrained prior therefore carries an
inductive bias that is actively **wrong** for X-ray in that specific respect, and fine-tuning must
overwrite it. Worth an explicit ablation (`--no-camera-action`, and text-encoded camera as a third
arm) rather than assuming the pretrained action prior helps.

### 9.6 Smoke-test scale-down

The full 28-layer model needs ~27 GB for fp32 weights + gradients — more than is reliably free on a
workstation running other jobs (this surfaced as a SIGKILL/OOM, exit 137, once gradients actually
began allocating after §8.2's fix). `num_hidden_layers_override` (default 2 in `Predict3Config`, and
`--layers 0` for real depth) shrinks **only** depth; hidden size, latent channels, patch size and
mRoPE axes stay at their real published values, so the packing and wiring under test are unchanged.
It is logged as a warning and must never be set for real training — pretrained weights would not load.

## 10. Reasoner freeze — generator-only post-training (2026-08-18)

`predict3/` now post-trains the **DM Generator tower** only and freezes the **AR Reasoner tower**
(`freeze_reasoner=True`, the default on `CosmosXRay2XRayPredict3Multiview`). Implemented in
`predict3/tower.py`; the cross-backbone conditioning context is in `docs/CONDITIONING.md` §4.

### 10.1 Correction to §5's key list

§5 named the trainable keys `moe_gen`, `time_embedder`, `vae2llm`, `llm2vae` — secondhand from
`CosmosXRay360`, never checked against code. Enumerating the real
`Cosmos3OmniTransformer.named_parameters()` shows only `time_embedder` exists verbatim:

- `moe_gen` is a **suffix**, not a module — `mlp_moe_gen`, `norm_moe_gen`,
  `input_layernorm_moe_gen`, `post_attention_layernorm_moe_gen`.
- The latent↔hidden projections are `proj_in` / `proj_out`, not `vae2llm` / `llm2vae`.

Had the §5 list been used literally as a parameter-name filter, it would have matched almost
nothing and the run would have trained a near-empty parameter set. `freeze_reasoner_tower()` therefore
raises if the split leaves zero trainable parameters, rather than proceeding quietly.

### 10.2 The verified split

| Tower | Top-level | Per-layer |
| --- | --- | --- |
| **Reasoner** (frozen) | `embed_tokens`, `lm_head`, `norm` | `self_attn.to_{q,k,v,out}`, `mlp.*`, `input_layernorm`, `post_attention_layernorm` |
| **Generator** (trained) | `proj_in`, `proj_out`, `norm_moe_gen`, `time_embedder`, `action_proj_in`, `action_proj_out`, `action_modality_embed` | `*_moe_gen`, `self_attn.{add_q_proj,add_k_proj,add_v_proj,to_add_out}`, `norm_added_{q,k}`, `k_norm_und_for_gen` |

Two names could not be settled by pattern-matching and required reading
`Cosmos3AttnProcessor.__call__`:

- **`k_norm_und_for_gen`** sits on the *und* key path but its output is consumed **only** by the
  generation pathway (`all_k = cat([k_und_for_gen, k_gen])`); the causal pathway uses the
  unnormalized `k_und`. It affects the generator's reading of und keys and nothing about the
  reasoner's own output → **generator side, trainable**.
- **`time_embedder`** carries no `_moe_gen` suffix but is the diffusion-timestep embedding, and
  the causal AR tower has no timestep → **generator side, trainable**.

### 10.3 Why freezing is safe here

The towers are asymmetric: the causal pathway attends over `q_und`/`k_und`/`v_und` alone, so und
activations never depend on gen activations. Freezing the reasoner cannot starve the generator of
signal. The converse motivation also holds — fine-tuning a language/reasoning prior on a narrow
DRR corpus would erode it for no benefit, since this pipeline never generates text.

### 10.4 Measured split and verification

On the real full-depth `nvidia/Cosmos3-Edge` config (28 layers, 3.37B):
**1.42B trainable (42.2%) / 1.95B frozen (57.8%)**. The frozen share is large partly because the
131k-vocab `embed_tokens` and `lm_head` (~268M each) are reasoner-side.

Three tests in `tests/test_predict3.py` (10 passing total):

- `test_reasoner_frozen_generator_trainable` — asserts the classification against real parameter
  names and that the partition is **total and disjoint**, so an upstream rename surfaces as a
  failure instead of silently-untrained weights.
- `test_frozen_reasoner_receives_no_gradients` — `requires_grad=False` is weak evidence on its
  own; this runs a real backward and asserts every parameter that accumulated a gradient is
  generator-side, and that `configure_optimizers` excludes frozen params (which would otherwise
  allocate AdamW state for weights that never update).
- `test_freeze_can_be_disabled` — full fine-tuning remains available via `freeze_reasoner=False`.

### 10.5 Open optimization

Because the reasoner is fully frozen *and* its activations never depend on the generator, the
understanding pathway could run once under `torch.no_grad()` rather than building an autograd
graph that accumulates no gradients — a material activation-memory saving. It needs a
restructured forward, so it is left open rather than done speculatively.
