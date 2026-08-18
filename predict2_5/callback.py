"""WanDB and TensorBoard callbacks for multiview X-ray synthesis logging."""

import torch
import numpy as np

from shared.utils import get_local_rank, get_logger
from shared.constants import XRAY_CAMERAS, NUM_XRAY_VIEWS

log = get_logger(__name__)

try:
    import wandb
except ImportError:
    wandb = None

from lightning import Trainer
from lightning.pytorch.callbacks import Callback


class MultiviewWanDBCallback(Callback):
    """Log multiview X-ray training/validation image batches and generated samples to WandB."""

    def __init__(
        self,
        max_samples_to_log: int = 2,
        log_every_n_epochs: int = 1,
    ):
        """
        Args:
            max_samples_to_log: Maximum number of samples to upload per view per epoch.
            log_every_n_epochs: How often to run epoch-end generation logging.
        """
        super().__init__()
        self.max_samples_to_log = max_samples_to_log
        self.log_every_n_epochs = log_every_n_epochs
        self._last_train_ob = None

    def _should_log_epoch(self, trainer: Trainer) -> bool:
        """Return True when rank-0, WandB is active, and the epoch interval is reached."""
        return (
            get_local_rank() == 0
            and wandb is not None
            and wandb.run is not None
            and trainer.current_epoch % self.log_every_n_epochs == 0
        )

    def _tensor_to_image(self, t: torch.Tensor) -> np.ndarray:
        """
        Convert a CHW or HWC tensor to a uint8 HWC numpy array suitable for wandb.Image.

        Args:
            t: Tensor of shape (C, H, W), (H, W, C), (H, W), or (1, C, H, W).

        Returns:
            uint8 numpy array of shape (H, W, 3) with values in [0, 255].
        """
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)

        arr = t.detach().cpu().float().numpy()

        # Determine value range and normalize to [0, 255]
        if arr.max() <= 1.0 and arr.min() >= -0.01:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        elif arr.max() > 1.0:
            arr = arr.clip(0, 255).astype(np.uint8)
        else:
            # Assume [-1, 1] range (diffusion model output)
            arr = ((arr + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

        # Keep 1-channel grayscale (H, W, 1)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]  # (H, W) -> (H, W, 1)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)[:, :, np.newaxis]  # (1, H, W) -> (H, W, 1)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            pass  # Already (H, W, 1)

        return arr

    def _to_grayscale(self, t: torch.Tensor) -> torch.Tensor:
        """Convert image tensor to grayscale (1 channel)."""
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.dim() == 3 and t.shape[0] == 1:
            return t
        if t.dim() == 3 and t.shape[0] > 1:
            return t[:1].float().mean(dim=0, keepdim=True) if t.shape[0] == 1 else t.float().mean(dim=0, keepdim=True)
        return t

    def _log_decoded_images(self, trainer, pl_module, ob, prefix):
        """
        Use target_xray as GT and decode the single-step predicted latent for Pred.
        Logs GT_target / Pred_target / Error_target for one randomly selected view per epoch.

        Prediction is estimated as: x_1_pred = xt + (1 - sigma) * v_pred
        """
        if ob is None or not (wandb is not None and wandb.run is not None):
            return
        try:
            view_indices = ob.get("view_indices")
            target = ob.get("target_xray")  # (B, C, H, W) pixel-space GT
            xt     = ob.get("xt")
            sigma  = ob.get("sigma")
            v_pred = ob.get("v_pred")
            if any(v is None for v in [view_indices, target, xt, sigma, v_pred]):
                return

            sigma_bc = sigma.float().view(sigma.shape[0], *([1] * (xt.dim() - 1)))
            x_1_pred = xt.float() + (1.0 - sigma_bc) * v_pred.float()

            view_to_idxs: dict = {}
            for i, vi in enumerate(view_indices.tolist()):
                view_to_idxs.setdefault(vi, []).append(i)

            # Randomly select one view from the batch
            if not view_to_idxs:
                return
            selected_v_idx = list(view_to_idxs.keys())[
                torch.randint(len(view_to_idxs), (1,)).item()
            ]
            sample_idxs = view_to_idxs[selected_v_idx]
            view_name = XRAY_CAMERAS[selected_v_idx] if selected_v_idx < len(XRAY_CAMERAS) else f"view_{selected_v_idx}"

            device = pl_module.device
            global_idx = sample_idxs[0]
            pred_lat = x_1_pred[global_idx : global_idx + 1].to(device)
            with torch.no_grad():
                pred_vid = pl_module.decode(pred_lat)  # (1, 3, T, H, W) in [-1, 1]
            T = pred_vid.shape[2]
            # GT directly from rendered pixel image (CPU); pred decoded on GPU → move to CPU
            gt_frame   = target[global_idx].float().clamp(0, 1)                          # (C, H, W) CPU
            pred_frame = ((pred_vid[0, :, T - 1] + 1) / 2).clamp(0, 1).cpu()            # (3, H, W) CPU
            gt_frame = self._to_grayscale(gt_frame)
            pred_frame = self._to_grayscale(pred_frame)
            err_frame  = (gt_frame - pred_frame).abs()
            
            section = f"{prefix}_results"
            images_to_log = {
                f"{section}/GT_target":    wandb.Image(self._tensor_to_image(gt_frame)),
                f"{section}/Pred_target":  wandb.Image(self._tensor_to_image(pred_frame)),
                f"{section}/Error_target": wandb.Image(self._tensor_to_image(err_frame)),
            }
            if images_to_log:
                wandb.log(images_to_log, step=trainer.global_step)
        except Exception as e:
            log.warning(f"[MultiviewWanDBCallback] failed to log decoded images: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Cache the last training batch output for epoch-end logging."""
        if isinstance(outputs, dict):
            self._last_train_ob = outputs.get("output_batch")

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._should_log_epoch(trainer) or self._last_train_ob is None:
            return
        ob, self._last_train_ob = self._last_train_ob, None
        self._log_decoded_images(trainer, pl_module, ob, prefix="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log val GT batch images then render GT / Pred / Error for one rotating view."""
        if not self._should_log_epoch(trainer):
            return

        try:
            val_dl = trainer.val_dataloaders
            if val_dl is None:
                return

            batch = next(iter(val_dl))
            ct = batch.get("ct")
            if ct is None:
                return
            ct = ct[:1].to(pl_module.device, dtype=torch.float32)
            device = pl_module.device

            # Rotate through views each epoch
            view_name = XRAY_CAMERAS[trainer.current_epoch % len(XRAY_CAMERAS)]

            from shared.constants import XRAY_EXTRINSICS, XRAY_FOV_DEFAULT, XRAY_DISTANCE_DEFAULT

            frontal_params = {
                "azimuth":   torch.zeros(1, device=device),
                "elevation": torch.zeros(1, device=device),
                "distance":  torch.full((1,), pl_module.hparams.get("renderer_max_depth", 8.0) - 0.1, device=device),
                "fov":       torch.full((1,), 12.5, device=device),
            }
            with torch.no_grad():
                frontal = pl_module._render_xray(ct, frontal_params)
            frontal_3ch = frontal.expand(-1, 3, -1, -1)

            ext = XRAY_EXTRINSICS[view_name]
            gt_params = {
                "azimuth":   torch.tensor([ext["azimuth"]],        device=device),
                "elevation": torch.tensor([ext["elevation"]],      device=device),
                "distance":  torch.tensor([XRAY_DISTANCE_DEFAULT], device=device),
                "fov":       torch.tensor([XRAY_FOV_DEFAULT],      device=device),
            }
            with torch.no_grad():
                gt_img = pl_module._render_xray(ct, gt_params)

            with torch.no_grad():
                pred_video = pl_module.generate(
                    image=frontal_3ch,
                    view_name=view_name,
                    num_steps=pl_module.hparams.num_inference_steps,
                    guidance_scale=pl_module.hparams.guidance_scale,
                    seed=42,
                )
            pred_frame = self._to_grayscale(pred_video[0, :, -1].clamp(0, 1))
            gt_frame   = self._to_grayscale(gt_img[0].clamp(0, 1))
            err_frame  = (gt_frame - pred_frame).abs()

            wandb.log({
                f"val_results/GT_target":    wandb.Image(self._tensor_to_image(gt_frame),   caption=f"GT {view_name}"),
                f"val_results/Pred_target":  wandb.Image(self._tensor_to_image(pred_frame), caption=f"Pred {view_name}"),
                f"val_results/Error_target": wandb.Image(self._tensor_to_image(err_frame),  caption=f"Error {view_name}"),
            }, step=trainer.global_step)
            log.info(f"[MultiviewWanDBCallback] logged GT/Pred/Error for view={view_name} at epoch={trainer.current_epoch}")
        except Exception as e:
            log.warning(f"[MultiviewWanDBCallback] generation logging failed: {e}")


class MultiviewTensorBoardCallback(Callback):
    """Log multiview X-ray training/validation image batches and GT vs Pred comparisons to TensorBoard."""

    def __init__(
        self,
        max_samples_to_log: int = 2,
        log_every_n_epochs: int = 1,
    ):
        """
        Args:
            max_samples_to_log: Maximum number of samples to write per view per epoch.
            log_every_n_epochs: How often to run epoch-end GT/Pred/Error generation.
        """
        super().__init__()
        self.max_samples_to_log = max_samples_to_log
        self.log_every_n_epochs = log_every_n_epochs
        self._last_train_ob = None

    def _get_writer(self, trainer: Trainer):
        """Return the SummaryWriter if a TensorBoardLogger is active, else None."""
        from lightning.pytorch.loggers import TensorBoardLogger
        if isinstance(trainer.logger, TensorBoardLogger):
            return trainer.logger.experiment
        return None

    def _should_log_epoch(self, trainer: Trainer) -> bool:
        """Return True when rank-0, a TensorBoard writer exists, and the epoch interval is reached."""
        return (
            get_local_rank() == 0
            and self._get_writer(trainer) is not None
            and trainer.current_epoch % self.log_every_n_epochs == 0
        )
    
    def _tensor_to_tb_image(self, t: torch.Tensor) -> np.ndarray:
        """
        Convert a tensor to a CHW float32 array in [0, 1] for TensorBoard add_image.
        Expands single-channel to 3-channel replica for proper grayscale display.

        Args:
            t: Tensor of shape (C, H, W), (H, W), or (1, C, H, W).

        Returns:
            float32 numpy array of shape (3, H, W) with values in [0, 1].
        """
        if t.dim() == 4:
            t = t[0]

        arr = t.detach().cpu().float().numpy()

        # Ensure CHW format (TensorBoard expects C first)
        if arr.ndim == 3 and arr.shape[0] not in (1, 3):
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 2:
            arr = arr[np.newaxis]  # Add channel dim: (H, W) → (1, H, W)

        # Normalize to [0, 1]
        if arr.max() > 1.5:
            arr = arr / 255.0
        elif arr.min() < -0.5:
            arr = (arr + 1.0) / 2.0  # Assume [-1, 1] diffusion output

        arr = arr.clip(0.0, 1.0).astype(np.float32)
        
        # Expand single-channel to 3-channel replica for TensorBoard grayscale display
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)  # (1, H, W) -> (3, H, W) with replicated values

        return arr

    def _to_grayscale(self, t: torch.Tensor) -> torch.Tensor:
        """Convert image tensor to grayscale (1 channel)."""
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.dim() == 3 and t.shape[0] == 1:
            return t
        if t.dim() == 3 and t.shape[0] > 1:
            return t[:1].float().mean(dim=0, keepdim=True) if t.shape[0] == 1 else t.float().mean(dim=0, keepdim=True)
        return t

    def _log_decoded_images(self, trainer, pl_module, ob, prefix):
        """
        Use target_xray as GT and decode the single-step predicted latent for Pred.
        Logs GT_target / Pred_target / Error_target for one randomly selected view per epoch.

        Prediction is estimated as: x_1_pred = xt + (1 - sigma) * v_pred
        """
        writer = self._get_writer(trainer)
        if ob is None or writer is None:
            return
        try:
            view_indices = ob.get("view_indices")
            target = ob.get("target_xray")  # (B, C, H, W) pixel-space GT
            xt     = ob.get("xt")
            sigma  = ob.get("sigma")
            v_pred = ob.get("v_pred")
            if any(v is None for v in [view_indices, target, xt, sigma, v_pred]):
                return

            sigma_bc = sigma.float().view(sigma.shape[0], *([1] * (xt.dim() - 1)))
            x_1_pred = xt.float() + (1.0 - sigma_bc) * v_pred.float()

            view_to_idxs: dict = {}
            for i, vi in enumerate(view_indices.tolist()):
                view_to_idxs.setdefault(vi, []).append(i)

            # Randomly select one view from the batch
            if not view_to_idxs:
                return
            selected_v_idx = list(view_to_idxs.keys())[
                torch.randint(len(view_to_idxs), (1,)).item()
            ]
            sample_idxs = view_to_idxs[selected_v_idx]
            view_name = XRAY_CAMERAS[selected_v_idx] if selected_v_idx < len(XRAY_CAMERAS) else f"view_{selected_v_idx}"

            device = pl_module.device
            global_idx = sample_idxs[0]
            pred_lat = x_1_pred[global_idx : global_idx + 1].to(device)
            with torch.no_grad():
                pred_vid = pl_module.decode(pred_lat)  # (1, 3, T, H, W) in [-1, 1]
            T = pred_vid.shape[2]
            
            # GT: Render directly from camera params (not from cached target_xray)
            # to ensure consistency with Val rendering
            cam = ob.get("camera_params")
            if cam is not None:
                # Render GT fresh from current batch CT volume
                # Extract CT from first sample (all samples share frontal view)
                ct_volume = ob.get("ct_volume")
                if ct_volume is None:
                    # Fallback: use cached target_xray
                    gt_frame = target[global_idx].float().clamp(0, 1)
                else:
                    ct_batch = ct_volume[:1].to(device, dtype=torch.float32)
                    # Get camera params for this sample's view
                    azim = cam["azimuth"][global_idx:global_idx+1].to(device)
                    elev = cam["elevation"][global_idx:global_idx+1].to(device)
                    dist = cam["distance"][global_idx:global_idx+1].to(device)
                    fov = cam["fov"][global_idx:global_idx+1].to(device)
                    
                    gt_params = {
                        "azimuth": azim, "elevation": elev,
                        "distance": dist, "fov": fov,
                    }
                    with torch.no_grad():
                        gt_frame = pl_module._render_xray(ct_batch, gt_params)[0]  # (1, H, W)
            else:
                gt_frame = target[global_idx].float().clamp(0, 1)
            
            pred_frame = ((pred_vid[0, :, T - 1] + 1) / 2).clamp(0, 1).cpu()            # (3, H, W) CPU
            gt_frame = self._to_grayscale(gt_frame)
            pred_frame = self._to_grayscale(pred_frame)
            err_frame  = (gt_frame - pred_frame).abs()
            
            section = f"{prefix}_results"
            writer.add_image(f"{section}/GT_target",    self._tensor_to_tb_image(gt_frame),   trainer.global_step)
            writer.add_image(f"{section}/Pred_target",  self._tensor_to_tb_image(pred_frame), trainer.global_step)
            writer.add_image(f"{section}/Error_target", self._tensor_to_tb_image(err_frame),  trainer.global_step)
        except Exception as e:
            log.warning(f"[MultiviewTensorBoardCallback] failed to log decoded images: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Cache the last training batch output for epoch-end logging."""
        if isinstance(outputs, dict):
            self._last_train_ob = outputs.get("output_batch")

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._should_log_epoch(trainer) or self._last_train_ob is None:
            return
        ob, self._last_train_ob = self._last_train_ob, None
        self._log_decoded_images(trainer, pl_module, ob, prefix="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Log val GT batch images, then render GT / Pred / Error for one rotating view.
        Runs every log_every_n_epochs epochs on rank-0.
        """
        if not self._should_log_epoch(trainer):
            return
        try:
            val_dl = trainer.val_dataloaders
            if val_dl is None:
                return
            batch = next(iter(val_dl))
            ct = batch.get("ct")
            if ct is None:
                return
            ct = ct[:1].to(pl_module.device, dtype=torch.float32)
            device = pl_module.device
            writer = self._get_writer(trainer)
            step   = trainer.global_step

            # Build the frontal conditioning image (AP view at azimuth=0, elevation=0)
            frontal_params = {
                "azimuth":   torch.zeros(1, device=device),
                "elevation": torch.zeros(1, device=device),
                "distance":  torch.full((1,), pl_module.hparams.get("renderer_max_depth", 8.0) - 0.1, device=device),
                "fov":       torch.full((1,), 12.5, device=device),
            }
            with torch.no_grad():
                frontal = pl_module._render_xray(ct, frontal_params)   # (1, 1, H, W)
            frontal_3ch = frontal.expand(-1, 3, -1, -1)                # (1, 3, H, W)

            # Render GT and model prediction for one view (rotates each epoch)
            from shared.constants import XRAY_EXTRINSICS, XRAY_FOV_DEFAULT, XRAY_DISTANCE_DEFAULT
            view_name = XRAY_CAMERAS[trainer.current_epoch % len(XRAY_CAMERAS)]

            # Render ground-truth X-ray for this view
            ext = XRAY_EXTRINSICS[view_name]
            gt_params = {
                "azimuth":   torch.tensor([ext["azimuth"]],          device=device),
                "elevation": torch.tensor([ext["elevation"]],        device=device),
                "distance":  torch.tensor([XRAY_DISTANCE_DEFAULT],   device=device),
                "fov":       torch.tensor([XRAY_FOV_DEFAULT],        device=device),
            }
            with torch.no_grad():
                gt_img = pl_module._render_xray(ct, gt_params)      # (1, 1, H, W)

            # Run 10-step diffusion for fast qualitative feedback during training
            with torch.no_grad():
                pred_video = pl_module.generate(
                    image=frontal_3ch,
                    view_name=view_name,
                    num_steps=pl_module.hparams.num_inference_steps,
                    guidance_scale=pl_module.hparams.guidance_scale,
                    seed=42,
                )   # (1, 3, T, H, W), values in [0, 1]

            # Extract last frame as the "predicted" still image
            pred_frame = self._to_grayscale(pred_video[0, :, -1].clamp(0, 1))
            gt_frame   = self._to_grayscale(gt_img[0].clamp(0, 1))
            err_frame  = (gt_frame - pred_frame).abs()              # (1, H, W)

            # Upload GT, prediction, and absolute error (same key each epoch, view rotates)
            writer.add_image(
                f"val_results/GT_target",
                self._tensor_to_tb_image(gt_frame),
                step,
            )
            writer.add_image(
                f"val_results/Pred_target",
                self._tensor_to_tb_image(pred_frame),
                step,
            )
            writer.add_image(
                f"val_results/Error_target",
                self._tensor_to_tb_image(err_frame),
                step,
            )

            log.info(f"[MultiviewTensorBoardCallback] logged GT vs Pred for view={view_name} at step {step}")
        except Exception as e:
            log.warning(f"[MultiviewTensorBoardCallback] generation logging failed: {e}")
