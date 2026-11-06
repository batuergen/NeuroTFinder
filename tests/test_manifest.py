from pathlib import Path

import pytest

from neurotfinder.manifest import CLASSES, binary_label, build_manifest


def make_dataset(root: Path, layout: dict[str, dict[str, list[str]]]) -> None:
    for split, classes in layout.items():
        (root / split).mkdir(parents=True, exist_ok=True)
        for class_name, files in classes.items():
            folder = root / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for name in files:
                (folder / name).touch()


def test_only_notumor_is_negative():
    assert [binary_label(c) for c in CLASSES] == [1, 1, 0, 1]


def test_rows_carry_split_and_both_labels(tmp_path):
    make_dataset(
        tmp_path,
        {"Training": {"glioma": ["a.jpg"]}, "Testing": {"notumor": ["b.jpg"]}},
    )

    rows = build_manifest(tmp_path, version="1")

    assert rows == [
        {
            "path": "Training/glioma/a.jpg",
            "version": "1",
            "split": "train",
            "class_name": "glioma",
            "multiclass_label": 0,
            "binary_label": 1,
        },
        {
            "path": "Testing/notumor/b.jpg",
            "version": "1",
            "split": "test",
            "class_name": "notumor",
            "multiclass_label": 2,
            "binary_label": 0,
        },
    ]


def test_label_does_not_depend_on_which_folders_exist(tmp_path):
    # With only two class folders present, pituitary must still be 3, not 1.
    make_dataset(
        tmp_path,
        {"Training": {"notumor": ["a.jpg"], "pituitary": ["b.jpg"]}, "Testing": {}},
    )

    labels = {r["class_name"]: r["multiclass_label"] for r in build_manifest(tmp_path, "1")}

    assert labels == {"notumor": 2, "pituitary": 3}


def test_unknown_class_folder_is_rejected(tmp_path):
    make_dataset(tmp_path, {"Training": {"Glioma": ["a.jpg"]}, "Testing": {}})

    with pytest.raises(ValueError, match="unknown class folder"):
        build_manifest(tmp_path, "1")


def test_non_image_files_are_skipped(tmp_path):
    make_dataset(
        tmp_path,
        {"Training": {"glioma": ["a.jpg", ".DS_Store", "notes.txt"]}, "Testing": {}},
    )

    assert len(build_manifest(tmp_path, "1")) == 1
