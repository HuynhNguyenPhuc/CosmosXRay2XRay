import torch
from monai.transforms import MapTransform


class ClipMinIntensityDict(MapTransform):
    """Clip intensity values to minimum threshold."""

    def __init__(self, keys, min_val: float = -512):
        super().__init__(keys)
        self.min_val = min_val

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = torch.clamp(d[key], min=self.min_val)
        return d

def correct_window(T_old, a_min=-1024, a_max=3071, b_min=-512, b_max=3071):
    """Correct CT window level to new range."""
    range_old = a_max - a_min
    range_new = b_max - b_min
    T_raw = (T_old * range_old) + a_min
    T_new = (T_raw - b_min) / range_new
    return T_new.clamp(0, 1)


def rescaled(x, val=64, eps=1e-8):
    """Rescale by fixed divisor."""
    return (x + eps) / (val + eps)


def minimized(x, eps=1e-8):
    """Normalize by max value."""
    return (x + eps) / (x.max() + eps)


def normalized(x, eps=1e-8):
    """Min-max normalization."""
    return (x - x.min() + eps) / (x.max() - x.min() + eps)


def standardized(x, eps=1e-8):
    """Standardize to zero mean, unit variance."""
    return (x - x.mean()) / (x.std() + eps)
