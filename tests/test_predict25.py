from __future__ import annotations

import pytest
from shared.constants import COSMOS_2B_PRETRAINED_UUID, NUM_LATENT_FRAMES
from predict2_5.text_encoder import CR1TextEncoder


def test_predict25_configuration():
    assert COSMOS_2B_PRETRAINED_UUID == "d20b7120-df3e-4911-919d-db6e08bad31c"
    assert NUM_LATENT_FRAMES == 24


def test_prompt_to_filename():
    prompt = "A 360-degree rotational view, of a chest CT scan."
    filename = CR1TextEncoder.prompt_to_filename(prompt)
    assert "," not in filename
    assert " " not in filename
    assert filename.startswith("A_360-degree")

