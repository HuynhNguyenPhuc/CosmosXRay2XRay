# Dataset Specifications & Preprocessing — CosmosXRay2XRay

This document details the CT and X-ray datasets used by **CosmosXRay2XRay**, along with Hounsfield Unit (HU) windowing, spatial normalization, rendering transformations, and cache directory structure.

---

## 1. Supported CT Datasets

| Dataset | Modality | Primary Application | Partition / Usage |
| :--- | :--- | :--- | :--- |
| **NSCLC** | Thoracic CT | Non-Small Cell Lung Cancer scans | Training / Evaluation |
| **MELA2022** | Thoracic CT | Mediastinal Lesion Analysis | Training / Validation |
| **TCIA** | Thoracic CT | The Cancer Imaging Archive public scans | Pretraining / Fine-tuning |
| **MosMed** | Thoracic CT | COVID-19 & Pulmonary infection CTs | Domain Adaptation |
| **VinDr-CXR** | 2D Chest X-ray | Real-world clinical 2D X-rays | Clinical Inference / Validation |

---

## 2. Hounsfield Unit (HU) Windowing & Normalization

Medical CT volumes store physical density measurements in Hounsfield Units (HU):
- Air: $-1000$ HU
- Lung tissue: $-700$ to $-500$ HU
- Soft tissue (heart, muscle): $+40$ to $+80$ HU
- Bone (ribs, vertebrae): $+300$ to $+1500$ HU

### Thoracic Windowing Policy
For X-ray raymarching and model inputs, raw HU values are clamped and linearly normalized to $[0, 1]$:

$$\text{HU}_{\text{clamped}} = \text{clip}(\text{HU}, -1000, +1000)$$

$$\mu_{\text{normalized}} = \frac{\text{HU}_{\text{clamped}} + 1000}{2000}$$

This mapping ensures:
- Background air maps to $0.0$.
- Dense bone structures map to $\approx 0.65 - 1.0$.
- Soft tissue renders with smooth intermediate grays.

---

## 3. Spatial Transformations & Resampling

All CT volumes undergo MONAI spatial processing:
1. **Orientation**: Reoriented to standard `RAS` (Right, Anterior, Superior) coordinate frame.
2. **Isotropic Voxel Resampling**: Resampled to uniform $1.0\,\text{mm} \times 1.0\,\text{mm} \times 1.0\,\text{mm}$ voxel resolution.
3. **Volume Resizing**: Resized/cropped to uniform cube tensor of shape $[1, 256, 256, 256]$ (`VOL_SIZE = 256`).

---

## 4. Cache Directory Layout

The `CacheBuilder` (`shared/cache_builder.py`) processes raw NIfTI files into fast-loading PyTorch `.npy` arrays:

```
cache/
├── train/
│   ├── ct/    # [1, 256, 256, 256] float32 CT volumes (.npy)
│   └── xr/    # [1, 256, 256] float32 rendered frontal X-rays (.npy)
├── val/
│   ├── ct/
│   └── xr/
└── test/
    ├── ct/
    └── xr/
```

Caching reduces IO overhead by pre-rendering frontal reference X-rays and converting NIfTI headers to raw binary numpy arrays, enabling maximum GPU utilization during multi-GPU DDP training.
