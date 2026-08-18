"""Shared callbacks for training (Predict2.5 & Transfer2.5)."""

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

import lightning as L
from lightning.pytorch.callbacks import Callback

from shared.utils import get_logger, get_local_rank

log = get_logger(__name__)


class EMAMonitorCallback(Callback):
    """Monitor EMA dtype and update health."""

    def __init__(self, log_every_n_steps: int = 500, warmup_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.warmup_steps = warmup_steps
        self.last_ema_params = None
        self.zero_diff_count = 0
        self.dtype_checked = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not hasattr(pl_module, "net_ema") or pl_module.net_ema is None:
            return

        if trainer.global_step == 1 and not self.dtype_checked:
            ema_dtype = next(pl_module.net_ema.parameters()).dtype
            if ema_dtype != torch.float32:
                raise RuntimeError(
                    f"EMA dtype is {ema_dtype}, MUST be float32 for stable accumulation"
                )
            self.dtype_checked = True

        if trainer.global_step < self.warmup_steps:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        try:
            if hasattr(pl_module, "_get_ema_beta"):
                shift = pl_module.hparams.ema_iteration_shift
                effective_step = trainer.global_step - shift + 1
                if effective_step >= 1:
                    beta = pl_module._get_ema_beta(effective_step)
                    pl_module.log("ema/beta", beta, prog_bar=False)

            current_params = next(pl_module.net_ema.parameters()).detach().cpu().clone()

            if self.last_ema_params is not None:
                diff = (current_params - self.last_ema_params).abs().mean().item()
                pl_module.log("ema/param_diff", diff, prog_bar=False)
                if diff == 0.0:
                    self.zero_diff_count += 1
                    if self.zero_diff_count >= 5:
                        pl_module.log("ema/frozen_warning", 1.0, prog_bar=False)
                else:
                    self.zero_diff_count = 0

            model_params = next(pl_module.net.parameters()).detach().cpu().float()
            divergence = (model_params - current_params.float()).abs().mean().item()
            pl_module.log("ema/model_divergence", divergence, prog_bar=False)

            self.last_ema_params = current_params

        except Exception:
            pass


class GradClipCallback(Callback):
    """Gradient clipping with NaN/Inf sanitization."""

    def __init__(self, clip_norm: float = 1.0, force_finite: bool = True, log_every_n_steps: int = 50):
        super().__init__()
        self.clip_norm = clip_norm
        self.force_finite = force_finite
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.force_finite:
            for param in pl_module.net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)

        total_norm = clip_grad_norm_(
            pl_module.net.parameters(),
            max_norm=self.clip_norm,
            norm_type=2.0,
            error_if_nonfinite=False,
        )

        if trainer.global_step % self.log_every_n_steps == 0:
            pl_module.log("grad/norm", total_norm, prog_bar=False)


class DDPSamplerEpochCallback(Callback):
    """Update DistributedSampler epoch at the start of each training epoch."""

    def on_train_epoch_start(self, trainer, pl_module):
        loader = trainer.train_dataloader
        if loader is None:
            return

        if hasattr(loader, "loaders"):
            for _loader in loader.loaders.values():
                self._set_sampler_epoch(_loader, trainer.current_epoch, trainer.global_rank)
        else:
            self._set_sampler_epoch(loader, trainer.current_epoch, trainer.global_rank)

    def _set_sampler_epoch(self, loader, epoch: int, rank: int):
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
            if rank == 0:
                log.debug(f"Updated DistributedSampler epoch to {epoch}")
        elif hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
            if rank == 0:
                log.debug(f"Updated BatchSampler epoch to {epoch}")


class GradientMonitorCallback(Callback):
    """Log gradient norms including cross-attention health."""

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        try:
            total_norm_sq = 0.0
            crossattn_norm_sq = 0.0

            for name, p in pl_module.net.named_parameters():
                if p.grad is not None:
                    grad_norm_sq = p.grad.data.norm(2).item() ** 2
                    total_norm_sq += grad_norm_sq
                    if "crossattn" in name.lower() or "cross_attn" in name.lower():
                        crossattn_norm_sq += grad_norm_sq

            pl_module.log("grad/norm", total_norm_sq ** 0.5, prog_bar=False)
            pl_module.log("grad/crossattn_norm", crossattn_norm_sq ** 0.5, prog_bar=False)

        except Exception:
            pass
