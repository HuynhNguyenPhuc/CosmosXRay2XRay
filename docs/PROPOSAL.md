# Project Proposal & Roadmap — CosmosXRay2XRay

This document outlines the strategic proposal, core objectives, target deliverables, and milestone roadmap for **CosmosXRay2XRay**.

---

## 1. Executive Summary

**CosmosXRay2XRay** addresses the fundamental challenge of 3D anatomical perception from 2D chest radiographs. By leveraging NVIDIA's **Cosmos 2.5** video world foundation models, the project enables high-fidelity, spatio-temporally consistent 7-view X-ray synthesis (AP, PA, LAT-L, LAT-R, LAO, RAO, Cranial) from a single 2D radiograph or CT volume prompt.

---

## 2. Core Objectives

1. **Predict 2.5 Pipeline**: Fine-tune `MinimalV1LVGDiT` on continuous 7-view X-ray video latents to capture $360^\circ$ anatomical rotation.
2. **Transfer 2.5 ControlNet Pipeline**: Integrate Sobel edge map and depth map conditioning via `MinimalV4LVGControlVaceDiT` to enforce anatomical edge fidelity.
3. **Clinical Validation**: Evaluate zero-shot transfer performance on real-world clinical X-rays (`VinDr-CXR`).

---

## 3. Milestone Roadmap

```mermaid
gantt
    title CosmosXRay2XRay Development Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1: Data & Infrastructure
    CT Cache Builder & Normalization  :done, p1_1, 2026-08-01, 2026-08-05
    7-View Camera Setup & Renderer     :done, p1_2, 2026-08-05, 2026-08-10
    section Phase 2: Core Pipelines
    Predict 2.5 Fine-Tuning Pipeline   :active, p2_1, 2026-08-10, 2026-08-25
    Transfer 2.5 ControlNet Pipeline   :active, p2_2, 2026-08-15, 2026-08-30
    section Phase 3: Benchmarking & App
    Quantitative Benchmark (PSNR/SSIM) : p3_1, 2026-09-01, 2026-09-10
    Gradio Web App Deployment          : p3_2, 2026-09-05, 2026-09-15
```
