# Quantitative Benchmarking & Protocol — CosmosXRay2XRay

This document outlines the quantitative evaluation protocol, metrics, baseline comparisons, and dataset partitions for benchmarking **CosmosXRay2XRay**.

---

## 1. Evaluation Protocol & Metrics

Models are evaluated on synthesizing 7 2D X-ray projections from a single frontal input X-ray or CT prompt. Synthesized outputs $\hat{\mathbf{I}}$ are compared against ground-truth DRR renders $\mathbf{I}$ across three standard image quality metrics:

### 1. Peak Signal-to-Noise Ratio (PSNR)
$$\text{PSNR}(\mathbf{I}, \hat{\mathbf{I}}) = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}(\mathbf{I}, \hat{\mathbf{I}})} \right)$$

### 2. Structural Similarity Index Measure (SSIM)
$$\text{SSIM}(\mathbf{I}, \hat{\mathbf{I}}) = \frac{(2\mu_I \mu_{\hat{I}} + c_1)(2\sigma_{I\hat{I}} + c_2)}{(\mu_I^2 + \mu_{\hat{I}}^2 + c_1)(\sigma_I^2 + \sigma_{\hat{I}}^2 + c_2)}$$

### 3. Learned Perceptual Image Patch Similarity (LPIPS)
$$\text{LPIPS}(\mathbf{I}, \hat{\mathbf{I}}) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \left\| w_l \odot \left( \hat{y}_{hw}^l - y_{hw}^l \right) \right\|_2^2$$

---

## 2. Model Pipeline Comparison Matrix

| Model Pipeline | Backbone | Conditioning Strategy | Primary Strength |
| :--- | :--- | :--- | :--- |
| **Predict 2.5** | `MinimalV1LVGDiT` (2B) | Flow-matching fine-tuning over 7-view latents | High spatio-temporal continuity across full 360° orbit. |
| **Transfer 2.5 (Edge)** | `MinimalV4LVGControlVaceDiT` | Sobel edge map ControlNet branch | Superior anatomical boundary definition (ribs, spine). |
| **Transfer 2.5 (Depth)** | `MinimalV4LVGControlVaceDiT` | Depth map ControlNet branch | Accurate volumetric depth perception and organ layering. |

---

## 3. Evaluation Split Protocol

- **Train Set**: `NSCLC` + `TCIA` + `MELA2022` (paired CT volume and rendered 7-view X-rays).
- **Validation Set**: Held-out patient scans from `MELA2022` and `MOSMED`.
- **Out-of-Distribution (OOD) Test Set**: `VinDr-CXR` real clinical 2D radiographs to test zero-shot transfer performance.
