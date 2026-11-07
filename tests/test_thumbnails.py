import numpy as np
from PIL import Image

from neurotfinder.thumbnails import build_thumbnails, thumbnail


def save_image(path, height, width, mode="L", value=128):
    channels = {"L": (height, width), "RGB": (height, width, 3)}[mode]
    Image.fromarray(np.full(channels, value, dtype=np.uint8), mode).save(path)


def test_any_size_and_mode_becomes_square_greyscale(tmp_path):
    save_image(tmp_path / "wide.png", 100, 300, mode="RGB")

    thumb = thumbnail(tmp_path / "wide.png", size=16)

    assert thumb.shape == (16, 16)
    assert thumb.dtype == np.uint8


def test_uniform_image_keeps_its_grey_level(tmp_path):
    save_image(tmp_path / "flat.png", 50, 50, value=200)

    assert np.all(thumbnail(tmp_path / "flat.png", size=8) == 200)


def test_thumbnails_are_stacked_in_path_order(tmp_path):
    save_image(tmp_path / "dark.png", 20, 20, value=10)
    save_image(tmp_path / "light.png", 20, 20, value=240)

    thumbs = build_thumbnails(tmp_path, ["light.png", "dark.png"], size=4)

    assert thumbs.shape == (2, 4, 4)
    assert thumbs[0, 0, 0] == 240 and thumbs[1, 0, 0] == 10
