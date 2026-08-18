from __future__ import annotations

import pytest
import torch

# predict3 needs diffusers `main` + transformers>=5.15 + huggingface-hub>=1.23,<2.0 — a
# SEPARATE venv from predict2_5/transfer2_5 (which need huggingface-hub<1.0, and whose
# transformers==4.51.3 doesn't know the `TokenizersBackend` class nvidia/Cosmos3-Edge's
# tokenizer needs). Skip cleanly here instead of failing when run in the main repo venv —
# see docs/cosmos-predict3/PLAN.md §7 for the environment split and how to set up the
# predict3-specific venv.
#
# A bare `hasattr(diffusers, "Cosmos3OmniPipeline")` is NOT enough to detect this: the
# released diffusers==0.39.0 wheel already exports that class by name, it just silently
# drops `use_und_k_norm_for_gen` from the real config (PLAN.md §3) — so check the actual
# transformers major version instead, which is what the tokenizer failure above depends on.
diffusers = pytest.importorskip("diffusers", reason="predict3 needs a separate diffusers-main venv (see PLAN.md §7)")
transformers = pytest.importorskip("transformers")
if not hasattr(diffusers, "Cosmos3OmniPipeline") or int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip(
        "installed diffusers/transformers are too old for real Cosmos3-Edge support "
        "(needs diffusers `main` + transformers>=5) — see PLAN.md §7",
        allow_module_level=True,
    )

from predict3.camera import (
    CAMERA_POSE_DOMAIN_ID,
    CAMERA_POSE_RAW_DIM,
    orbit_view_matrices,
    xray_view_actions,
    xray_view_matrices,
)
from predict3.constants import COSMOS3_EDGE_REPO_ID
from predict3.module import CosmosXRay2XRayPredict3Multiview
from predict3.trainer import Predict3Config, smoke_test
from shared.constants import XRAY_CAMERAS

# Depth-only scale-down: the real 28-layer model needs ~27 GB for fp32 weights + gradients.
# Every other dimension stays at its real published value, so packing/wiring is unchanged.
TEST_LAYERS = 2


def test_predict3_configuration():
    assert COSMOS3_EDGE_REPO_ID == "nvidia/Cosmos3-Edge"
    # Must match cosmos-framework's own domain table, which the pretrained action head keys off.
    assert CAMERA_POSE_DOMAIN_ID == 2
    assert CAMERA_POSE_RAW_DIM == 9


def test_camera_poses_and_action_shapes():
    """7 views -> 7 poses -> 6 transitions of [translation(3), rot6d(6)]."""
    poses = xray_view_matrices()
    assert poses.shape == (len(XRAY_CAMERAS), 4, 4)
    # Bottom row of a homogeneous camera-to-world transform.
    assert torch.allclose(poses[:, 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(len(XRAY_CAMERAS), 4))

    actions = xray_view_actions()
    assert actions.shape == (len(XRAY_CAMERAS) - 1, CAMERA_POSE_RAW_DIM)

    padded = xray_view_actions(action_dim=64)
    assert padded.shape == (len(XRAY_CAMERAS) - 1, 64)
    assert torch.all(padded[:, CAMERA_POSE_RAW_DIM:] == 0)
    assert torch.allclose(padded[:, :CAMERA_POSE_RAW_DIM], actions)


def test_camera_actions_are_geometrically_symmetric():
    """Views mirrored about AP must yield equal-magnitude relative poses.

    RAO (315 deg) / LAO (45 deg) and lateral-right (270 deg) / lateral-left (90 deg) are each
    symmetric about the AP anchor, so under `backward_anchored` their action vectors must have
    equal norms. Guards against an axis/sign error in the pose encoding that shape checks miss.
    """
    actions = xray_view_actions()
    norms = {name: float(actions[i - 1].norm()) for i, name in enumerate(XRAY_CAMERAS) if i > 0}

    assert norms["xray_rao"] == pytest.approx(norms["xray_lao"], rel=1e-5)
    assert norms["xray_lateral_right"] == pytest.approx(norms["xray_lateral_left"], rel=1e-5)
    # PA is the antipode of the AP anchor, so it must be the largest displacement.
    assert norms["xray_pa"] == max(norms.values())


def test_orbit_actions_match_frame_count():
    """Actions describe transitions: T frames -> T-1 action tokens (Cosmos 3's chunk contract)."""
    poses = orbit_view_matrices(num_frames=9)
    assert poses.shape == (9, 4, 4)
    with pytest.raises(ValueError):
        orbit_view_matrices(num_frames=1)


def test_predict3_smoke_test_forward_backward():
    """Real Cosmos3OmniTransformer/AutoencoderKLWan architecture, one real forward + backward
    with native I2V (frame-0) and `camera_pose` action conditioning — see PLAN.md §9."""
    cfg = Predict3Config(
        device="cpu", smoke_test_num_frames=5, smoke_test_height=32, smoke_test_width=32,
        num_hidden_layers_override=TEST_LAYERS,
    )
    loss = smoke_test(cfg)
    assert torch.isfinite(torch.tensor(loss))


def test_predict3_transformer_registered_as_submodule():
    """Regression test: the transformer/vae must be registered as real nn.Module submodules
    (not just held inside the non-Module Cosmos3OmniPipeline wrapper), otherwise
    `configure_optimizers` silently trains on zero parameters — see PLAN.md §7."""
    model = CosmosXRay2XRayPredict3Multiview(num_hidden_layers_override=TEST_LAYERS)
    model.setup()
    assert any(p.requires_grad for p in model.transformer.parameters())
    assert not any(p.requires_grad for p in model.vae.parameters())
    optimizer = model.configure_optimizers()
    assert len(optimizer.param_groups[0]["params"]) > 0


def test_camera_action_conditioning_reaches_action_pathway():
    """The camera_pose action port must actually receive gradients.

    Shapes alone can't prove the conditioning is used — a silently-ignored action tensor would
    still produce a finite loss. Asserting that `action_proj_in` / `action_modality_embed` get
    gradients only when conditioning is enabled proves the poses reach the model.
    """

    def action_grad_params(use_camera_action: bool) -> set[str]:
        model = CosmosXRay2XRayPredict3Multiview(
            use_camera_action=use_camera_action, num_hidden_layers_override=TEST_LAYERS
        )
        model.setup()
        video = torch.rand(1, 3, 5, 32, 32)
        model.training_step({"video": video, "prompt": "test"}, batch_idx=0)["loss"].backward()
        return {n for n, p in model.transformer.named_parameters() if p.grad is not None and "action" in n}

    with_action = action_grad_params(True)
    without_action = action_grad_params(False)

    assert "action_proj_in.fc.weight" in with_action
    assert "action_modality_embed" in with_action
    assert without_action == set()
    # All action tokens are conditioning (never denoised), so the output head must stay unused.
    assert not any("action_proj_out" in n for n in with_action)
