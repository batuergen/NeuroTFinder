"""Build a manifest: one row per image with its split, class and labels.

The notebook loaded images with Keras' `flow_from_directory` and then
collapsed four classes into two by overwriting `train.classes` in place. That
only worked while the folders happened to sort alphabetically, and the
iterator still reported four classes. Here every label comes from one explicit
mapping, and the result is a plain CSV that every later step reads.

Usage:
    uv run python -m neurotfinder.manifest --record data/raw/download_v1.json
"""

import argparse
import csv
import json
from pathlib import Path

from neurotfinder.download import IMAGE_SUFFIXES

# Fixed order, so a class always gets the same index regardless of folder order.
CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
SPLITS = {"Training": "train", "Testing": "test"}  # Kaggle folder -> our name
FIELDS = ["path", "version", "split", "class_name", "multiclass_label", "binary_label"]


def binary_label(class_name: str) -> int:
    """1 = tumour (glioma, meningioma, pituitary), 0 = no tumour."""
    return 0 if class_name == "notumor" else 1


def build_manifest(root: Path, version: str) -> list[dict]:
    """List every image under root/<split>/<class>/ with its labels."""
    rows = []
    for folder, split in SPLITS.items():
        for class_dir in sorted((root / folder).iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in CLASSES:
                # Fail loudly: a renamed or extra folder must never get a silent label.
                raise ValueError(f"unknown class folder {class_dir}, expected one of {CLASSES}")
            for image in sorted(class_dir.iterdir()):
                if image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                rows.append(
                    {
                        "path": image.relative_to(root).as_posix(),
                        "version": version,
                        "split": split,
                        "class_name": class_name,
                        "multiclass_label": CLASSES.index(class_name),
                        "binary_label": binary_label(class_name),
                    }
                )
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, required=True, help="download record JSON")
    parser.add_argument("--out", type=Path, help="default: data/manifests/v<version>.csv")
    args = parser.parse_args()

    record = json.loads(args.record.read_text())
    rows = build_manifest(Path(record["path"]), record["version"])
    out = args.out or Path(f"data/manifests/v{record['version']}.csv")
    write_manifest(rows, out)

    n_tumour = sum(r["binary_label"] for r in rows)
    print(f"{len(rows)} images ({n_tumour} tumour, {len(rows) - n_tumour} no tumour) -> {out}")


if __name__ == "__main__":
    main()
