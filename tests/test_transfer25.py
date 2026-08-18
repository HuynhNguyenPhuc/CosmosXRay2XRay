from __future__ import annotations

import pytest
import torch

from transfer2_5.control_signal import generate_control_signal


def test_control_signal_generation():
    # Test control signal generation on a dummy image tensor
    dummy_image = torch.zeros(1, 1, 256, 256, dtype=torch.float32)
    dummy_image[:, :, 50:150, 50:150] = 1.0
    
    edge_signal = generate_control_signal(dummy_image, control_type="edge_map")
    assert edge_signal is not None
    assert isinstance(edge_signal, torch.Tensor)
