# Conditioning Ports by Backbone — Which Signal Goes Where

This repository trains three different backbones, and they do **not** share a conditioning
surface. Sending a signal through the wrong port is the single most common way to lose accuracy
without producing an error — the run still trains, the loss still falls, the conditioning is just
weak. This document classifies the ports each backbone exposes and states which signal belongs in
which.

**The governing rule: geometry belongs in a geometric port, never in the text port.** Camera
angles formatted into prose are tokenized (`"315.0"` and `"314.0"` become unrelated token
sequences, `0°` and `359°` land maximally far apart despite being adjacent), then diluted inside
boilerplate that is identical across views. Every backbone here has somewhere better to put it.

---

## 1. Ports by backbone

### `predict2_5/` — `MinimalV1LVGDiT` / `CameraMiniTrainDITwithConditionalMask` (Cosmos Predict 2.5)

| Port | Mechanism | Currently used for |
| --- | --- | --- |
| Cross-attention text | CR1 embeddings, `crossattn_emb` | anatomy prose; camera numerics too on `camera_cond="text"` (see §2) |
| Latent frame replacement | `condition_video_input_mask_B_C_T_H_W` | anchor/source view |
| **Plücker ray maps** | `cosmos_predict2._src.predict2.camera` | camera geometry, on `camera_cond="plucker"` (see §2) |

### `transfer2_5/` — `MinimalV4LVGControlVaceDiT` (Cosmos Transfer 2.5)

Everything Predict 2.5 has, plus:

| Port | Mechanism | Currently used for |
| --- | --- | --- |
| ControlNet/VACE branch | `latent_control_input` → `control_embedder` → `control_blocks` → hints | Sobel edge / depth / segmentation |

### `predict3/` — `Cosmos3OmniTransformer` (Cosmos 3)

| Port | Mechanism | Currently used for |
| --- | --- | --- |
| Text tokens | raw token IDs into the AR Reasoner — **no separate text encoder** | anatomy prose |
| Vision tokens | per-frame clean/noisy split via `condition_frame_indexes` | anchor view as native I2V frame 0 |
| **Action tokens** | `camera_pose` domain (id 2), 9D pose → `action_proj_in` | camera geometry |
| Sound tokens | `sound_*` | unused |

No ControlNet equivalent exists in Cosmos 3.

---

## 2. The two camera ports are not equivalent

Both non-text options beat prose, but they differ in granularity, and the right choice differs
per backbone:

| | Predict 2.5 — Plücker rays | Cosmos 3 — `camera_pose` action |
| --- | --- | --- |
| Representation | 6D ray (direction + moment) **per patch** | 9D pose (3D translation + 6D rotation) **per frame** |
| Granularity | dense, spatially aligned to the latent grid | one global vector per transition |
| Entry point | `convert_camera_to_plucker_rays(extrinsics, intrinsics)` → `CameraToPluckerRays` conditioner | `action_proj_in` (`DomainAwareLinear`, per-domain weights) |
| Status in this repo | wired and tested (`predict2_5/camera_plucker.py`, `camera_cond="plucker"`) — real forward+backward confirmed on A100, `cam_encoder` loads real pretrained weights (not random init), see §2.1 | wired and tested (`predict3/camera.py`) |

Plücker rays are the stronger signal — per-patch geometry tells the model where *each region* of
the image is looking from, not merely where the camera is. Cosmos 3's action port is coarser, but
it is the native port on a much stronger backbone.

### 2.1 Pretrained camera-conditioned weights — VERIFIED AVAILABLE AND WIRED IN (2026-08-19)

Yes, they are publicly downloadable, and `predict2_5/module.py`'s checkpoint resolution now loads them
by default whenever `camera_cond="plucker"` (`COSMOS_2B_CAMERA_PRETRAINED_UUID`, `shared/constants.py`) —
you do **not** have to train the camera branch from scratch, and no longer have to remember to ask for
this checkpoint by hand. Downloaded and loaded for real (not just checked via HF metadata):
`non_strict_load_model` reports **`missing=0, unexpected=0`** against a freshly constructed
`CameraMiniTrainDITwithConditionalMask` — every parameter the network has, including all 28 blocks'
`cam_encoder.weight`, is present in the checkpoint and gets real pretrained values. Loaded param count
(`2147.3M`) matches the arithmetic below (`2059M` base + `~88M` for 28× `cam_encoder`) exactly.

| | |
| --- | --- |
| Repository | `nvidia/Cosmos-Predict2.5-2B` (public, `gated: auto` — accept the license, access is instant) |
| Revision | `fbe72c18d152053029a19db3b211cf78671ad422` |
| File | `robot/multiview-agibot/f740321e-2cd6-4370-bbfe-545f4eca2065_ema_bf16.pt` (4.29 GB) |
| Registered as | `nvidia/Cosmos-Predict2.5-2B/robot/multiview-agibot`, uuid `f740321e-2cd6-4370-bbfe-545f4eca2065` |
| Experiment | `multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot_frameinit` |

That experiment resolves through `camera_conditioned_frameinit_video_conditioner` →
`CameraToPluckerRays(patch_spatial=16, camera_patch_average=False)`, i.e. genuine Plücker
conditioning, not merely multi-camera training.

**Confirmed by three independent checks**, without downloading the full 4.29 GB (ranged HTTP
fetch of the checkpoint head, whose pickle lists tensor names):

1. The checkpoint contains **28 `net.blocks.{N}.cam_encoder.weight` tensors** — one per DiT block.
2. The two `auto/multiview` checkpoints contain **zero** such tensors, so the difference is real
   and specific to this file.
3. The arithmetic closes: `cam_encoder` is `nn.Linear(cam_dim=1536, 2048, bias=False)` and
   1536 = 6 × 16 × 16 (6D Plücker ray × 16×16 patch, unaveraged). 1536 × 2048 × 28 blocks in bf16
   = **0.17616 GB**, against an observed size delta of **0.17605 GB** vs. the non-camera
   checkpoints — a 0.06% match.

**Do not grab the wrong file:** `auto/multiview/{524af350…,6b9d7548…}_ema_bf16.pt` are *not*
camera-conditioned. Their only "camera" keys are `pos_embedder_options.n_cameras_*`, which are
multi-view positional-embedding options, not ray conditioning.

**Domain caveat.** These weights are trained on AgiBot **robot** multi-camera footage at
720p/16fps — natural-scene robotics video. The *geometric* prior (how a Plücker ray field maps to
viewpoint change) is what transfers; the *appearance* prior does not, and X-ray is a transmission
modality where a view at azimuth θ and one at θ+180° are near-mirror images rather than mutually
occluded. Expect to fine-tune the appearance behaviour, and treat this as an ablation arm rather
than an assumed win — the same caveat that applies to Cosmos 3's `camera_pose` domain.

**Useful for fine-tuning:** the net ships a `freeze_parameters()` method that unfreezes only
`cam_encoder` and `self_attn` and freezes everything else — the Predict 2.5 analogue of
`predict3/tower.py`'s reasoner freeze. Note it is **commented out at the call site**
(`# self.freeze_parameters()` in `CameraMiniTrainDITwithConditionalMask.__init__`), so it is
opt-in, not the default.

---

## 3. Recommended assignment

| Signal | `predict2_5` | `transfer2_5` | `predict3` |
| --- | --- | --- | --- |
| Source / anchor view | latent frame replacement | latent frame replacement | native I2V frame 0 |
| **Camera geometry** | **Plücker rays** (not text) | **Plücker rays** (not text) | **`camera_pose` action tokens** |
| Anatomy, appearance, pathology | CR1 cross-attention | CR1 cross-attention | native text tokens (structured JSON) |
| Structural edges / depth / masks | — | ControlNet branch | — (no equivalent port) |

When camera geometry moves out of the prompt, drop the numeric fields from the caption too
(`XRAY_PROMPT_TEMPLATE` → a camera-free variant) so the geometry is not being supplied twice in
two different encodings. Keep the per-view anatomical prefix: that is legitimate view
description, not camera numerics.

---

## 4. Cosmos 3 only: which half of the model trains

Cosmos 3 is a Mixture-of-Transformers with two parameter sets per layer sharing one joint
self-attention op. `predict3/` post-trains the **DM Generator tower** and freezes the **AR
Reasoner tower** (`freeze_reasoner=True`, the default).

This is safe because the towers are asymmetric: in `Cosmos3AttnProcessor.__call__` the causal
pathway attends over `q_und`/`k_und`/`v_und` alone, so understanding activations never depend on
generation activations. Freezing the reasoner cannot starve the generator of signal, while
fine-tuning it on a narrow X-ray corpus would erode the language prior we want to keep.

On the real `nvidia/Cosmos3-Edge` (3.37B): **1.42B trainable (42.2%) / 1.95B frozen (57.8%)**.
The frozen share is large partly because the 131k-vocab `embed_tokens` and `lm_head` (~268M each)
sit on the reasoner side.

The verified parameter split lives in `predict3/tower.py` — see its docstring for the naming
convention and the two names (`k_norm_und_for_gen`, `time_embedder`) that required reading the
attention source rather than pattern-matching. `docs/cosmos-predict3/PLAN.md` §5 carries an
earlier, incorrect key list; §10 there records the correction.

**Possible optimization, not implemented:** because the reasoner is entirely frozen *and* its
activations never depend on the generator, the understanding pathway could be run once under
`torch.no_grad()` instead of building an autograd graph that accumulates no gradients. That would
cut activation memory materially. It needs a restructured forward, so it is deliberately left
open.
