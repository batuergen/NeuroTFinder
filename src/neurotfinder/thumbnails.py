"""Make a small greyscale thumbnail of every image in a manifest.

Thumbnails are what the data review clusters and what near-duplicate detection
compares, so they are computed once and stored as one array in manifest order.

Usage:
    uv run python -m neurotfinder.thumbnails --record data/raw/download_v1.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 64


def thumbnail(path: Path, size: int = SIZE) -> np.ndarray:
    """Greyscale, resized to size x size, as uint8 of shape (size, size).

    Stretched rather than padded: two copies of the same slice have the same
    aspect ratio, so stretching keeps them identical, and padding would make
    small images mostly border.
    """
    with Image.open(path) as img:
        grey = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(grey, dtype=np.uint8)


def build_thumbnails(root: Path, paths: list[str], size: int = SIZE) -> np.ndarray:
    """Stack thumbnails of root/path for every path, shape (N, size, size)."""
    thumbs = np.empty((len(paths), size, size), dtype=np.uint8)
    for i, rel in enumerate(paths):
        thumbs[i] = thumbnail(root / rel, size)
    return thumbs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, required=True, help="download record JSON")
    parser.add_argument("--manifest", type=Path, help="default: data/manifests/v<version>.csv")
    parser.add_argument("--out", type=Path, help="default: data/review/v<version>_thumbs.npz")
    args = parser.parse_args()

    record = json.loads(args.record.read_text())
    root, version = Path(record["path"]), record["version"]
    manifest = args.manifest or Path(f"data/manifests/v{version}.csv")
    out = args.out or Path(f"data/review/v{version}_thumbs.npz")

    with manifest.open() as fh:
        paths = [row["path"] for row in csv.DictReader(fh)]

    thumbs = build_thumbnails(root, paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Paths are stored alongside, so the array can never be matched to the wrong rows.
    np.savez_compressed(out, thumbs=thumbs, paths=np.array(paths))
    print(f"{len(paths)} thumbnails of {SIZE}x{SIZE} -> {out}")


if __name__ == "__main__":
    main()
