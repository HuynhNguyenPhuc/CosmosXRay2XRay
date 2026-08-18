# Quality Assurance & Peer Review Criteria — CosmosXRay2XRay

This document defines the quality assurance criteria, validation gates, and peer review standards for **CosmosXRay2XRay**.

---

## 1. Quality Assurance Checklists

### Code Quality & Formatting Gate
- [x] Python 3.10+ compatibility with `from __future__ import annotations`.
- [x] All module imports grouped cleanly (stdlib $\to$ third-party $\to$ project local).
- [x] Strict typing hints across functions and class methods.
- [x] No raw `print()` statements in library modules; `logging` utilized exclusively.

### Physical & Mathematical Accuracy Gate
- [x] CT volumes windowed to $[-1000, +1000]$ HU and scaled to $[0, 1]$.
- [x] Beer-Lambert exponential attenuation ($I = I_0 e^{-\int \mu dx}$) correctly modeled.
- [x] 7 camera views (AP, PA, LAT-L, LAT-R, LAO, RAO, Cranial) strictly mapped with documented extrinsics.
- [x] VAE temporal frame constraint ($93 \to 24$ latents) preserved.

### Test Automation Gate
- [x] All unit test modules executed in isolated subprocesses via `run_all_tests_isolate.py`.
- [x] DDP multi-GPU training confirmed stable under mixed precision (`bf16-mixed`).

---

## 2. Peer Review Submission Criteria

1. **Reproducibility**: Shell scripts (`launch/train_predict25.sh`, `launch/train_transfer25.sh`) and CLI flags documented.
2. **Benchmark Reporting**: PSNR, SSIM, LPIPS metrics reported for both Predict 2.5 and Transfer 2.5.
3. **Interactive Demo**: Gradio `app.py` validated for web deployment.
