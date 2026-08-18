"""WanDB and TensorBoard callbacks for Transfer2.5 multiview X-ray synthesis logging."""

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


class TransferMultiviewWanDBCallback(Callback):
    """Log Transfer2.5 multiview training/validation results to WanDB."""

    def __init__(
        self,
        log_every_n_steps: int = 500,
        max_samples_to_log: int = 2,
        log_generation_every_n_steps: int = 2000,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.max_samples_to_log = max_samples_to_log
        self.log_generation_every_n_steps = log_generation_every_n_steps

    def _should_log(self, trainer: Trainer) -> bool:
        return (
            get_local_rank() == 0
            and wandb is not None
            and wandb.run is not None
            and trainer.global_step % self.log_every_n_steps == 0
            and trainer.global_step > 0
        )

    def _should_generate(self, trainer: Trainer) -> bool:
        return (
            get_local_rank() == 0
            and wandb is not None
            and wandb.run is not None
            and trainer.global_step % self.log_generation_every_n_steps == 0
            and trainer.global_step > 0
        )

    def _tensor_to_image(self, t: torch.Tensor) -> np.ndarray:
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3:
            if t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)
        arr = t.detach().cpu().float().numpy()
        if arr.max() <= 1.0 and arr.min() >= -0.01:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        elif arr.max() > 1.0:
            arr = arr.clip(0, 255).astype(np.uint8)
        else:
            arr = ((arr + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 1:
            arr = np.concatenate([arr] * 3, axis=-1)
        return arr

    def _log_batch_images(self, trainer, pl_module, outputs, prefix="train"):
        if not self._should_log(trainer):
            return

        ob = outputs.get("output_batch") if isinstance(outputs, dict) else None
        if ob is None:
            return

        try:
            view_indices = ob.get("view_indices")
            frontal = ob.get("frontal_xray")
            target = ob.get("target_xray")
            control = ob.get("control_images")

            if view_indices is None or frontal is None or target is None:
                return

            n_views = NUM_XRAY_VIEWS
            B_per_view = frontal.shape[0] // n_views
            n_show = min(self.max_samples_to_log, B_per_view)

            images_to_log = {}
            for v_idx in range(n_views):
                view_name = XRAY_CAMERAS[v_idx]
                start = v_idx * B_per_view
                for s_idx in range(n_show):
                    idx = start + s_idx
                    fr_img = self._tensor_to_image(frontal[idx])
                    tg_img = self._tensor_to_image(target[idx])
                    images_to_log[f"{prefix}/frontal_{view_name}_s{s_idx}"] = wandb.Image(fr_img)
                    images_to_log[f"{prefix}/target_{view_name}_s{s_idx}"] = wandb.Image(tg_img)
                    if control is not None and idx < control.shape[0]:
                        ct_img = self._tensor_to_image(control[idx])
                        images_to_log[f"{prefix}/control_{view_name}_s{s_idx}"] = wandb.Image(ct_img)

            if images_to_log:
                wandb.log(images_to_log, step=trainer.global_step)

        except Exception as e:
            log.warning(f"[TransferWandBCallback] failed to log images: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log_batch_images(trainer, pl_module, outputs, prefix="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._log_batch_images(trainer, pl_module, outputs, prefix="val")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._should_generate(trainer):
            return
        try:
            from transfer2_5.control_signal import generate_control_signal

            val_dl = trainer.val_dataloaders
            if val_dl is None:
                return
            batch = next(iter(val_dl))
            xr_input = batch["xr"][:1].to(pl_module.device)

            gen_images = {}
            for view_name in XRAY_CAMERAS:
                # Generate control signal from frontal for demonstration
                ctrl = generate_control_signal(
                    drr_image=xr_input,
                    control_type=pl_module.hparams.control_type,
                )
                video = pl_module.generate(
                    image=xr_input, view_name=view_name,
                    control_image=ctrl,
                    num_steps=20, guidance_scale=1.5, seed=42,
                )
                mid_frame = video.shape[2] // 2
                frame = video[0, :, mid_frame]
                gen_images[f"gen/{view_name}"] = wandb.Image(
                    self._tensor_to_image(frame), caption=view_name,
                )
            if gen_images:
                wandb.log(gen_images, step=trainer.global_step)
        except Exception as e:
            log.warning(f"[TransferMultiviewWanDBCallback] generation logging failed: {e}")


class TransferMultiviewTensorBoardCallback(Callback):
    """Log Transfer2.5 multiview training/validation results to TensorBoard."""

    def __init__(
        self,
        log_every_n_steps: int = 500,
        max_samples_to_log: int = 2,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.max_samples_to_log = max_samples_to_log

    def _get_writer(self, trainer: Trainer):
        from lightning.pytorch.loggers import TensorBoardLogger
        if isinstance(trainer.logger, TensorBoardLogger):
            return trainer.logger.experiment
        return None

    def _should_log(self, trainer: Trainer) -> bool:
        return (
            get_local_rank() == 0
            and self._get_writer(trainer) is not None
            and trainer.global_step % self.log_every_n_steps == 0
            and trainer.global_step > 0
        )

    def _tensor_to_tb_image(self, t: torch.Tensor) -> np.ndarray:
        """Return CHW float32 array in [0, 1] for TensorBoard add_image."""
        if t.dim() == 4:
            t = t[0]
        arr = t.detach().cpu().float().numpy()
        # Ensure CHW format
        if arr.ndim == 3 and arr.shape[0] not in (1, 3):
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 2:
            arr = arr[np.newaxis]
        # Normalize to [0, 1]
        if arr.max() > 1.5:
            arr = arr / 255.0
        elif arr.min() < -0.5:
            arr = (arr + 1.0) / 2.0
        return arr.clip(0.0, 1.0).astype(np.float32)

    def _log_batch_images(self, trainer: Trainer, outputs, prefix: str = "train"):
        if not self._should_log(trainer):
            return
        writer = self._get_writer(trainer)

        ob = outputs.get("output_batch") if isinstance(outputs, dict) else None
        if ob is None:
            return

        try:
            view_indices = ob.get("view_indices")
            frontal = ob.get("frontal_xray")
            target = ob.get("target_xray")
            control = ob.get("control_images")

            if view_indices is None or frontal is None or target is None:
                return

            n_views = NUM_XRAY_VIEWS
            B_per_view = frontal.shape[0] // n_views
            n_show = min(self.max_samples_to_log, B_per_view)

            for v_idx in range(n_views):
                view_name = XRAY_CAMERAS[v_idx]
                start = v_idx * B_per_view
                for s_idx in range(n_show):
                    idx = start + s_idx
                    fr_img = self._tensor_to_tb_image(frontal[idx])
                    tg_img = self._tensor_to_tb_image(target[idx])
                    writer.add_image(f"{prefix}/frontal_{view_name}_s{s_idx}", fr_img, trainer.global_step)
                    writer.add_image(f"{prefix}/target_{view_name}_s{s_idx}", tg_img, trainer.global_step)
                    if control is not None and idx < control.shape[0]:
                        ct_img = self._tensor_to_tb_image(control[idx])
                        writer.add_image(f"{prefix}/control_{view_name}_s{s_idx}", ct_img, trainer.global_step)

        except Exception as e:
            log.warning(f"[TransferMultiviewTensorBoardCallback] failed to log images: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log_batch_images(trainer, outputs, prefix="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._log_batch_images(trainer, outputs, prefix="val")
