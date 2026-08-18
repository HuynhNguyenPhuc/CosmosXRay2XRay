"""DiffDRR-backed X-ray volume renderer."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchio as tio
    from diffdrr.drr import DRR
    from diffdrr.pose import RigidTransform
    HAS_DIFFDRR = True
except ImportError:
    tio = None
    DRR = None
    RigidTransform = None
    HAS_DIFFDRR = False

from shared.transforms import minimized, normalized, standardized


Number = Union[int, float]
AngleLike = Union[Number, Sequence[Number], torch.Tensor]

FIXED_LINE_INTEGRAL_MAX = 1.0


def _look_at_camera_positions(
    dist: torch.Tensor,
    elev: torch.Tensor,
    azim: torch.Tensor,
    degrees: bool = True,
) -> torch.Tensor:
    """Calculates camera positions in world space."""
    if degrees:
        elev = elev * (math.pi / 180.0)
        azim = azim * (math.pi / 180.0)

    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)

    return torch.stack([x, y, z], dim=-1)


def _look_at_rotation_matrices(
    camera_position: torch.Tensor,
    at: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Calculates look-at rotation matrices."""
    device, dtype = camera_position.device, camera_position.dtype
    at_t = torch.as_tensor(at, device=device, dtype=dtype).expand_as(camera_position)
    up_t = torch.as_tensor(up, device=device, dtype=dtype).expand_as(camera_position)

    z_axis = F.normalize(at_t - camera_position, dim=-1, eps=1e-5)
    x_axis = F.normalize(torch.cross(up_t, z_axis, dim=-1), dim=-1, eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-5)

    is_close = torch.isclose(x_axis, torch.zeros_like(x_axis), atol=5e-3).all(dim=-1, keepdim=True)

    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=-1), dim=-1, eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)

    return torch.stack([x_axis, y_axis, z_axis], dim=-1)


def look_at_view_matrices(
    dist: AngleLike,
    elev: AngleLike,
    azim: AngleLike,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Builds camera-to-world view matrices as a plain ``(N, 4, 4)`` tensor.

    Split out of :func:`look_at_view_poses` so callers that only need the raw matrices —
    notably ``predict3.camera``, which converts them into Cosmos 3 ``camera_pose`` action
    vectors — can reuse the exact same camera geometry as the DRR renderer without
    requiring ``diffdrr``/``torchio`` to be installed. Keeping one implementation here
    matters: if the conditioning poses ever drifted from the geometry actually rendered,
    the mismatch would silently corrupt the training signal rather than raise.

    Returns:
        ``(N, 4, 4)`` camera-to-world transforms: ``[:3, :3]`` is the look-at rotation and
        ``[:3, 3]`` the camera position in world space.
    """
    dist_t = torch.as_tensor(dist, device=device, dtype=dtype)
    elev_t = torch.as_tensor(elev, device=device, dtype=dtype)
    azim_t = torch.as_tensor(azim, device=device, dtype=dtype)
    dist_t, elev_t, azim_t = torch.broadcast_tensors(dist_t, elev_t, azim_t)

    C = _look_at_camera_positions(dist_t, elev_t, azim_t)
    R = _look_at_rotation_matrices(C)

    batch_shape = C.shape[:-1]
    matrix = torch.zeros(*batch_shape, 4, 4, device=C.device, dtype=C.dtype)
    matrix[..., :3, :3] = R
    matrix[..., :3, 3] = C
    matrix[..., 3, 3] = 1.0

    return matrix.reshape(-1, 4, 4)


def look_at_view_poses(
    dist: AngleLike,
    elev: AngleLike,
    azim: AngleLike,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> RigidTransform:
    """Builds camera view pose RigidTransform."""
    if not HAS_DIFFDRR:
        raise ImportError("diffdrr and torchio are required for look_at_view_poses.")

    return RigidTransform(look_at_view_matrices(dist, elev, azim, device=device, dtype=dtype))


def _fov_to_pixel_spacing(
    fov: float,
    sdd: float,
    image_width: int,
    image_height: int,
    ndc_extent: float,
    degrees: bool = True,
) -> Tuple[float, float]:
    """Converts field of view to detector pixel spacing."""
    fov_rad = math.radians(fov) if degrees else fov
    f_ndc = 1.0 / math.tan(fov_rad / 2.0)
    delx = sdd * 2.0 * ndc_extent / (image_width * f_ndc)
    dely = sdd * 2.0 * ndc_extent / (image_height * f_ndc)

    return delx, dely


class DiffDRRVolumeRenderer(nn.Module):
    """DiffDRR-backed X-ray volume renderer."""

    def __init__(
        self,
        image_width: int = 256,
        image_height: int = 256,
        n_pts_per_ray: int = 320,
        ndc_extent: float = 1.0,
        renderer: str = "trilinear",
        flip_horizontal: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()

        if renderer != "trilinear":
            raise ValueError("DiffDRRVolumeRenderer requires renderer='trilinear'.")

        self.image_width = image_width
        self.image_height = image_height
        self.n_pts_per_ray = n_pts_per_ray
        self.ndc_extent = ndc_extent
        self.renderer_backend = renderer
        self.flip_horizontal = flip_horizontal
        self.device = torch.device(device)

        self._drr: Optional[DRR] = None

    def _build_subject(self, volume: torch.Tensor) -> Tuple[tio.Subject, float]:
        """Wraps a CT tensor into a torchio Subject."""
        vol = volume.detach()

        if vol.ndim == 5:
            if vol.shape[0] != 1:
                raise ValueError("Expected single CT volume.")
            vol = vol[0]

        if vol.ndim == 4:
            vol = vol[0]

        if vol.ndim != 3:
            raise ValueError(f"Expected 3D CT volume, got shape {tuple(volume.shape)}.")

        D, H, W = vol.shape
        shape = max(D, H)
        voxel_size = 2.0 * float(self.ndc_extent) / float(shape)

        tio_tensor = vol.permute(2, 1, 0).unsqueeze(0).contiguous().to(torch.float32).cpu()

        offset_w = -0.5 * (W - 1) * voxel_size
        offset_h = -0.5 * (H - 1) * voxel_size
        offset_d = -0.5 * (D - 1) * voxel_size

        affine = np.array(
            [
                [voxel_size, 0.0, 0.0, offset_w],
                [0.0, voxel_size, 0.0, offset_h],
                [0.0, 0.0, voxel_size, offset_d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        image = tio.ScalarImage(tensor=tio_tensor, affine=affine)
        subject = tio.Subject(
            volume=image,
            density=image,
            mask=None,
            reorient=torch.eye(4, dtype=torch.float32),
        )

        return subject, voxel_size

    def set_volume(self, volume: torch.Tensor) -> "DiffDRRVolumeRenderer":
        """Binds a CT volume to the renderer."""
        subject, _ = self._build_subject(volume)

        self._drr = DRR(
            subject,
            sdd=1.0,
            height=self.image_height,
            width=self.image_width,
            delx=1.0,
            dely=1.0,
            renderer=self.renderer_backend,
        ).to(self.device)

        return self

    @torch.no_grad()
    def render(
        self,
        azimuth: AngleLike,
        volume: Optional[torch.Tensor] = None,
        elev: AngleLike = 0.0,
        dist: AngleLike = 8.0,
        fov: float = 12.0,
        min_depth: float = 7.0,
        max_depth: float = 9.0,
        norm_type: str = "standardized",
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Renders X-ray projections of the CT volume."""
        if volume is not None:
            self.set_volume(volume)

        if self._drr is None:
            raise RuntimeError("Call set_volume(...) or pass volume=... before rendering.")

        azim_t = torch.as_tensor(azimuth, device=self.device, dtype=torch.float32).reshape(-1)
        n_frames = azim_t.shape[0]

        elev_t = torch.as_tensor(elev, device=self.device, dtype=torch.float32).expand_as(azim_t)
        dist_t = torch.as_tensor(dist, device=self.device, dtype=torch.float32).expand_as(azim_t)

        poses = look_at_view_poses(dist_t, elev_t, azim_t, device=self.device, dtype=torch.float32)

        sdd = float(max_depth)
        delx, dely = _fov_to_pixel_spacing(
            fov=float(fov),
            sdd=sdd,
            image_width=self.image_width,
            image_height=self.image_height,
            ndc_extent=float(self.ndc_extent),
        )

        alphamin = float(min_depth) / sdd
        alphamax = float(max_depth) / sdd

        self._drr.set_intrinsics_(sdd=sdd, delx=delx, dely=dely, height=self.image_height, width=self.image_width)

        frames = []
        for start in range(0, n_frames, batch_size):
            chunk = RigidTransform(poses.matrix[start:start + batch_size].to(self.device))
            img = self._drr(
                chunk,
                n_points=self.n_pts_per_ray,
                alphamin=alphamin,
                alphamax=alphamax,
            )
            frames.append(img)

        img = torch.cat(frames, dim=0)

        if self.flip_horizontal:
            img = torch.flip(img, dims=[-1])

        if norm_type == "minimized":
            img = minimized(img)
        elif norm_type == "normalized":
            img = normalized(img)
        elif norm_type == "standardized":
            img = normalized(standardized(img))

        return img


def create_diffdrr_renderer(
    img_shape: int = 256,
    n_pts_per_ray: int = 1000,
    ndc_extent: float = 1.0,
    flip_horizontal: bool = False,
    device: Union[str, torch.device] = "cuda",
) -> DiffDRRVolumeRenderer:
    """Creates a DiffDRRVolumeRenderer instance."""
    return DiffDRRVolumeRenderer(
        image_width=img_shape,
        image_height=img_shape,
        n_pts_per_ray=n_pts_per_ray,
        ndc_extent=ndc_extent,
        flip_horizontal=flip_horizontal,
        device=device,
    )


def render_diffdrr_multiview_frames(
    vol: torch.Tensor,
    num_frames: int = 93,
    img_shape: int = 256,
    device: Union[str, torch.device] = "cuda",
    dist: float = 8.0,
    elev: float = 0.0,
    fov: float = 12.0,
    min_depth: float = 7.0,
    max_depth: float = 9.0,
    renderer: Optional[DiffDRRVolumeRenderer] = None,
    batch_size: int = 1,
) -> torch.Tensor:
    """Renders a 360-degree azimuth sweep video."""
    if renderer is None:
        renderer = create_diffdrr_renderer(img_shape=img_shape, device=device)

    renderer.set_volume(vol)
    azimuths = torch.linspace(0.0, 360.0, num_frames, device=device)

    return renderer.render(
        azimuth=azimuths,
        elev=elev,
        dist=dist,
        fov=fov,
        min_depth=min_depth,
        max_depth=max_depth,
        norm_type="standardized",
        batch_size=batch_size,
    )
