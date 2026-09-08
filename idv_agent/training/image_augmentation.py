"""Label-preserving image augmentation for VLA training windows."""

from __future__ import annotations

from dataclasses import dataclass
import random

from PIL import Image, ImageEnhance


@dataclass(frozen=True)
class WindowAugmentation:
    """One transform shared by every real frame in a temporal window."""

    hue: float
    saturation: float
    brightness: float
    contrast: float
    translate_x: int
    translate_y: int


def sample_window_augmentation(rng: random.Random | None = None) -> WindowAugmentation:
    """Sample bounded photometric and translation parameters."""
    rng = rng or random
    return WindowAugmentation(
        hue=rng.uniform(-0.05, 0.05),
        saturation=rng.uniform(0.8, 1.2),
        brightness=rng.uniform(0.8, 1.2),
        contrast=rng.uniform(0.8, 1.2),
        translate_x=rng.randint(-8, 8),
        translate_y=rng.randint(-8, 8),
    )


def apply_window_augmentation(image: Image.Image, params: WindowAugmentation) -> Image.Image:
    """Apply one label-preserving transform without mutating ``image``."""
    if not isinstance(params, WindowAugmentation):
        raise TypeError("params 必须是 WindowAugmentation")
    hsv = image.convert("HSV")
    h, s, v = hsv.split()
    h = h.point(lambda value: int((value + params.hue * 255.0) % 256))
    result = Image.merge("HSV", (h, s, v)).convert("RGB")
    result = ImageEnhance.Color(result).enhance(params.saturation)
    result = ImageEnhance.Brightness(result).enhance(params.brightness)
    result = ImageEnhance.Contrast(result).enhance(params.contrast)
    if params.translate_x or params.translate_y:
        result = result.transform(
            result.size,
            Image.Transform.AFFINE,
            (1, 0, -params.translate_x, 0, 1, -params.translate_y),
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
    return result
