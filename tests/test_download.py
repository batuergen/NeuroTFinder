from pathlib import Path

from neurotfinder.download import count_images


def make_images(folder: Path, names: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).touch()


def test_counts_per_split_and_class(tmp_path):
    make_images(tmp_path / "Training" / "glioma", ["a.jpg", "b.jpg"])
    make_images(tmp_path / "Testing" / "notumor", ["c.jpg"])

    assert count_images(tmp_path) == {
        "Testing": {"notumor": 1},
        "Training": {"glioma": 2},
    }


def test_all_image_suffixes_count_regardless_of_case(tmp_path):
    # The notebook's `cp *jpg *JPG` skipped .jpeg files; these must all count.
    make_images(tmp_path / "Training" / "glioma", ["a.jpg", "b.JPG", "c.jpeg", "d.png"])

    assert count_images(tmp_path)["Training"]["glioma"] == 4


def test_non_image_files_are_ignored(tmp_path):
    make_images(tmp_path / "Training" / "glioma", ["a.jpg", "notes.txt", ".DS_Store"])

    assert count_images(tmp_path)["Training"]["glioma"] == 1