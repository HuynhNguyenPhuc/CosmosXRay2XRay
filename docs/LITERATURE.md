# Academic Literature & Theoretical Foundations — CosmosXRay2XRay

This document reviews the foundational computer vision, generative AI, and medical imaging literature underpinning **CosmosXRay2XRay**.

---

## 1. Video Diffusion Foundation Models (Cosmos 2.5)

NVIDIA's **Cosmos 2.5** family introduces large-scale video diffusion transformers (DiT) trained on massive video corpora. 
- **Predict 2.5 (`MinimalV1LVGDiT`)**: A 2B-parameter DiT operating in the 3D latent space of a Wan2.1 3D VAE tokenizer. It uses continuous flow-matching formulation to generate temporally coherent video frames.
- **Transfer 2.5 (`MinimalV4LVGControlVaceDiT`)**: Extends the DiT architecture with a ControlNet-style parallel VACE control branch. This enables structure-guided generation without altering the generative priors of the base model.

---

## 2. 2D-to-3D Multiview X-Ray Synthesis & DRR Physics

Digitally Reconstructed Radiographs (DRRs) bridge 3D CT volume representations and 2D projection imaging:
- **Physics**: Governed by the Beer-Lambert attenuation law ($I = I_0 e^{-\int \mu(x) dx}$).
- **Challenge**: Traditional single-view X-ray novel view synthesis suffers from depth ambiguity and occlusion overlap.
- **Solution**: Leveraging pretrained video diffusion world models treats 7-view orbital camera trajectories as a 93-frame contiguous video sequence, enforcing global anatomical consistency across the full $360^\circ$ span.

---

## 3. Key References

1. **NVIDIA Cosmos Team** (2025). *Cosmos 2.5: Physical World Foundation Models*. NVIDIA Technical Report.
2. **Zhang & Agrawala** (2023). *Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)*. ICCV.
3. **Siddon, R. L.** (1985). *Fast calculation of the exact radiological path for a three-dimensional CT array*. Medical Physics.
4. **Gopalakrishnan et al.** (2024). *DiffDRR: Differentiable Digitally Reconstructed Radiographs in PyTorch*. Journal of Open Source Software.
