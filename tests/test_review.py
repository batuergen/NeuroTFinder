import numpy as np

from neurotfinder.review import GAP, cluster_thumbnails, contact_sheet


def test_two_obviously_different_groups_get_separate_clusters():
    rng = np.random.default_rng(0)
    dark = rng.integers(0, 40, size=(20, 16, 16), dtype=np.uint8)
    light = rng.integers(200, 255, size=(20, 16, 16), dtype=np.uint8)

    clusters = cluster_thumbnails(np.concatenate([dark, light]), n_clusters=2, n_components=5)

    assert len(set(clusters[:20])) == 1
    assert len(set(clusters[20:])) == 1
    assert clusters[0] != clusters[20]


def test_contact_sheet_grid_size():
    images = [np.zeros((8, 8), dtype=np.uint8)] * 7

    sheet = contact_sheet(images, columns=3)

    assert sheet.size == (3 * (8 + GAP), 3 * (8 + GAP))  # 7 images -> 3 rows of 3


def test_contact_sheet_accepts_colour_images():
    images = [np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8)]

    sheet = contact_sheet(images, columns=1)

    assert sheet.getpixel((0, 0)) == (255, 0, 0)


def test_frames_mark_each_image_in_its_colour():
    images = [np.zeros((8, 8), dtype=np.uint8)] * 2

    sheet = contact_sheet(images, columns=2, frames=[(0, 255, 0), (255, 0, 0)])

    assert sheet.getpixel((0, 0)) == (0, 255, 0)
    assert sheet.getpixel((8 + GAP, 0)) == (255, 0, 0)
    assert sheet.getpixel((4, 4)) == (0, 0, 0)  # inside the frame the image is untouched
