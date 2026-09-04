from __future__ import annotations

import random

from PIL import Image


def test_window_augmentation_samples_bounded_parameters_and_is_deterministic():
    from idv_agent.training.image_augmentation import (
        apply_window_augmentation,
        sample_window_augmentation,
    )

    params = sample_window_augmentation(random.Random(7))
    assert -0.05 <= params.hue <= 0.05
    assert 0.8 <= params.saturation <= 1.2
    assert 0.8 <= params.brightness <= 1.2
    assert 0.8 <= params.contrast <= 1.2
    assert -8 <= params.translate_x <= 8
    assert -8 <= params.translate_y <= 8

    image = Image.new("RGB", (16, 16), (40, 90, 140))
    assert list(apply_window_augmentation(image, params).get_flattened_data()) == (
        list(apply_window_augmentation(image, params).get_flattened_data())
    )
