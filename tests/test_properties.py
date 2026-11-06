import numpy as np
from PIL import Image

from neurotfinder.properties import GREY_TOLERANCE, channel_spread, image_properties


def grey_pixels(height: int = 64, width: int = 96) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(height, width), dtype=np.uint8)


def test_identical_channels_have_zero_spread():
    rgb = np.stack([grey_pixels()] * 3, axis=-1)

    assert channel_spread(rgb) == 0.0


def test_coloured_image_has_large_spread():
    g = grey_pixels()
    rgb = np.stack([g, g // 2, 255 - g], axis=-1)

    assert channel_spread(rgb) > GREY_TOLERANCE


def test_greyscale_jpeg_saved_as_rgb_counts_as_grey(tmp_path):
    path = tmp_path / "grey.jpg"
    Image.fromarray(np.stack([grey_pixels()] * 3, axis=-1)).save(path, quality=75)

    assert image_properties(path)["channel_spread"] <= GREY_TOLERANCE


def test_size_mode_and_aspect_ratio(tmp_path):
    path = tmp_path / "slice.png"
    Image.fromarray(grey_pixels(height=50, width=100)).save(path)

    props = image_properties(path)

    assert (props["width"], props["height"], props["mode"]) == (100, 50, "L")
    assert props["aspect_ratio"] == 2.0
    assert props["file_bytes"] == path.stat().st_size
