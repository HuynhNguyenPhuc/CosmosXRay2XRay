# Domain, Physical & Technical Gotchas — CosmosXRay2XRay

This document catalogs technical, physical, and implementation gotchas specific to **CosmosXRay2XRay**.

---

## 1. Frame & VAE Latent Dimensions

- **Frame Count Constraint**: `NUM_FRAMES = 93` is hardcoded in `shared/constants.py`.
- **Latent Temporal Compression**: The Wan2.1 3D VAE uses a temporal stride of 4.
  $$\text{NUM\_LATENT\_FRAMES} = 1 + \frac{93 - 1}{4} = 24$$
- **Gotcha**: Passing video sequences whose frame counts are not of the form $4k + 1$ will cause shape mismatches inside the VAE encoder/decoder blocks.

---

## 2. Submodule CUDA Build Requirements

- `cosmos-predict2.5` and `cosmos-transfer2.5` require CUDA extension compilation.
- **Python Version Lock**: 
  - `cu128` extra requires **Python 3.10** (flash-attn wheels are built for `cp310`).
  - `cu130` extra supports **Python 3.11**.
- **Gotcha**: Attempting to install `cu128` under Python 3.11 will fail with `no wheels with a matching Python ABI tag`. Re-create `.venv` with `--python 3.10`.

---

## 3. PyTorch3D Installation & Build Isolation

- Building `pytorch3d` from source requires PyTorch headers to be present in the build environment.
- **Gotcha**: Running standard `pip install git+...` without `--no-build-isolation` fails because isolated build environments lack pre-installed PyTorch.
- **Fix**: Always use `uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"`.

---

## 4. DDP Process Isolation in Testing

- Executing PyTorch3D raymarchers and multiple DiT model instances within a single pytest session can pollute global CUDA contexts and JIT extension registries.
- **Gotcha**: Direct `pytest` on all test files simultaneously may cause random CUDA initialization hangs or OOMs.
- **Fix**: Always use `python run_all_tests_isolate.py` to run each test module in a separate subprocess.
