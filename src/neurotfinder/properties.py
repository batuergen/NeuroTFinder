"""Measure file-level properties of every image in a manifest.

Two uses. First, finding images that are not greyscale MRI: an MRI or CT slice
has identical colour channels, so an image whose channels differ is a PET scan,
a fused overlay or an annotated screenshot. Second, measuring how much the
image source shows through properties that have nothing to do with anatomy
(size, aspect ratio, file size), since sources and labels are entangled here.

Usage:
    uv run python -m neurotfinder.properties --record data/raw/download_v1.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

# JPEG compression adds small differences between the channels of a greyscale
# image, so "greyscale" means the channels agree to within this many grey levels.
GREY_TOLERANCE = 10
FIELDS = ["path", "width", "height", "aspect_ratio", "mode", "file_bytes", "channel_spread"]


def channel_spread(rgb: np.ndarray) -> float:
    """99th percentile over pixels of the largest difference between two channels.

    A percentile rather than the maximum, so a handful of compression artefacts
    does not make a greyscale image look coloured.
    """
    rgb = rgb.astype(np.int16)  # (H, W, 3); int16 so differences can go negative
    per_pixel = rgb.max(axis=2) - rgb.min(axis=2)
    return float(np.percentile(per_pixel, 99))


def image_properties(path: Path) -> dict:
    with Image.open(path) as img:
        width, height = img.size
        mode = img.mode
        rgb = np.asarray(img.convert("RGB"))
    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 4),
        "mode": mode,
        "file_bytes": path.stat().st_size,
        "channel_spread": channel_spread(rgb),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, required=True, help="download record JSON")
    parser.add_argument("--manifest", type=Path, help="default: data/manifests/v<version>.csv")
    parser.add_argument("--out", type=Path, help="default: data/review/v<version>_properties.csv")
    args = parser.parse_args()

    record = json.loads(args.record.read_text())
    root, version = Path(record["path"]), record["version"]
    manifest = args.manifest or Path(f"data/manifests/v{version}.csv")
    out = args.out or Path(f"data/review/v{version}_properties.csv")

    with manifest.open() as fh:
        paths = [row["path"] for row in csv.DictReader(fh)]

    out.parent.mkdir(parents=True, exist_ok=True)
    n_colour = 0
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for rel in paths:
            props = image_properties(root / rel)
            n_colour += props["channel_spread"] > GREY_TOLERANCE
            writer.writerow({"path": rel, **props})

    print(f"{len(paths)} images, {n_colour} not greyscale -> {out}")


if __name__ == "__main__":
    main()
