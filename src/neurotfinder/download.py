"""downloads the Brain Tumor MRI dataset from Kaaggle and records what was downloaded.

uv run python -m neurotfinder.download      #latest version
uv run python -m neurotfinder.download --version 1

"""
import argparse
import json
from pathlib import Path

import kagglehub

DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}  # compared lower-case, so .JPG counts too


def count_images(root: Path) -> dict[str, dict[str, int]]:
    """Count images per split folder (Training/Testing) and class folder."""
    counts: dict[str, dict[str, int]] = {}
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        counts[split_dir.name] = {
            class_dir.name: sum(
                1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES
            )
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir())
        }
    return counts


def download(version: int | None = None) -> Path:
    """Download (or reuse the cached copy of) the dataset and return its folder."""
    handle = DATASET if version is None else f"{DATASET}/versions/{version}"
    return Path(kagglehub.dataset_download(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", type=int, help="Kaggle dataset version (default: latest)")
    parser.add_argument(
        "--record",
        type=Path,
        default=Path("data/raw/download.json"),
        help="where to write the download record",
    )
    args = parser.parse_args()

    path = download(args.version)
    record = {
        "dataset": DATASET,
        "version": path.name,  # kagglehub stores each version in .../versions/<n>
        "path": str(path),
        "counts": count_images(path),
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()