# Volume Rendering & 7-View Camera Setup — CosmosXRay2XRay

This document describes the physics-based X-ray volume rendering engine, 7-view camera geometry, and Beer-Lambert attenuation model implemented in **CosmosXRay2XRay**.

---

## 1. Physical Attenuation Model (Beer-Lambert Law)

Chest radiography measures the attenuation of X-ray photons passing through heterogeneous tissue. The transmitted intensity $I$ along a ray path $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ is governed by the Beer-Lambert law:

$$I = I_0 \exp\left( -\int_{t_{\text{near}}}^{t_{\text{far}}} \mu(\mathbf{r}(t)) \, dt \right)$$

where $\mu(\mathbf{r}(t))$ is the linear attenuation coefficient at spatial coordinate $\mathbf{r}(t)$.

In `shared/renderer.py` and `shared/raymarcher.py`, PyTorch3D absorption-emission raymarching approximates this integral over $N$ discrete steps:

$$\text{RayIntegral}(u, v) = \sum_{i=1}^{N} \mu(x_i, y_i, z_i) \cdot \Delta t$$

$$\text{Intensity}(u, v) = 1.0 - \exp\left( -\text{RayIntegral}(u, v) \right)$$

---

## 2. 7-View Multiview Camera Configuration

**CosmosXRay2XRay** defines a fixed 7-view clinical radiograph camera orbital setup in `shared/constants.py`:

| Camera Identifier | Projection Name | Azimuth ($\theta$) | Elevation ($\phi$) | Clinical Rationale & Features |
| :--- | :--- | :---: | :---: | :--- |
| `xray_ap` | Anteroposterior (AP) | $0.0^\circ$ | $0.0^\circ$ | Standard bedside view; cardiac silhouette enlarged due to geometric magnification. |
| `xray_lateral_right` | Right Lateral | $270.0^\circ$ | $0.0^\circ$ | Right-to-left beam; retrosternal and retrocardiac spaces clearly visible. |
| `xray_rao` | Right Anterior Oblique | $315.0^\circ$ | $0.0^\circ$ | $45^\circ$ oblique; separates right cardiac border and atrium from spine. |
| `xray_pa` | Posteroanterior (PA) | $180.0^\circ$ | $0.0^\circ$ | Standard upright view; minimizes cardiac magnification; compact heart shadow. |
| `xray_lao` | Left Anterior Oblique | $45.0^\circ$ | $0.0^\circ$ | $45^\circ$ oblique; opens aortic arch and displays left atrium. |
| `xray_lateral_left` | Left Lateral | $90.0^\circ$ | $0.0^\circ$ | Left-to-right beam; left hemidiaphragm and posterior cardiac shadow. |
| `xray_cranial` | Cranial AP Tilt | $0.0^\circ$ | $+30.0^\circ$ | $30^\circ$ superior tilt; projects clavicles over lung apices; highlights aortic knob. |

---

## 3. Camera Extrinsics & Intrinsics

- **Isocenter**: The 3D CT volume center is anchored at the origin $(0, 0, 0)$.
- **Source-to-Isocenter Distance**: Fixed at $R = 8.0$ scene units (`XRAY_DISTANCE_DEFAULT`).
- **Field of View (FOV)**: Cone-beam FOV of $12.0^\circ$ (`XRAY_FOV_DEFAULT`).
- **Ray Range**: Raymarching brackets the volume between $z_{\text{near}} = 6.0$ and $z_{\text{far}} = 10.0$ scene units.
