"""Group visually similar images and render contact sheets for manual review.

Looking at 7,000 images one by one is not practical, but looking at a few
dozen groups of similar images is. Images are compressed to their main
directions of variation, clustered, and each cluster is drawn as one sheet of
thumbnails, together with a sheet of every image flagged as not greyscale.
Clusters that hold a single class, or an unexpected kind of image, show where
the data is not what the labels claim.

Usage:
    uv run python -m neurotfinder.review --record data/raw/download_v1.json
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from neurotfinder.manifest import CLASSES
from neurotfinder.properties import GREY_TOLERANCE

N_COMPONENTS = 50
N_CLUSTERS = 16
SHEET_COLUMNS = 10
SHEET_SIZE = 60  # thumbnails per cluster sheet
GAP = 2  # pixels between thumbnails
SEED = 0
# Frame colour per class on the cluster sheets, so labels can be checked by eye.
CLASS_COLOURS = {
    "glioma": (230, 60, 60),
    "meningioma": (240, 170, 0),
    "notumor": (40, 180, 80),
    "pituitary": (60, 120, 240),
}


def cluster_thumbnails(
    thumbs: np.ndarray, n_clusters: int = N_CLUSTERS, n_components: int = N_COMPONENTS
) -> np.ndarray:
    """Cluster id per thumbnail, shape (N,).

    64x64 pixels are 4,096 dimensions in which distances say little; the first
    principal components keep the coarse structure (head shape, slice plane,
    contrast) that separates kinds of image.
    """
    flat = thumbs.reshape(len(thumbs), -1).astype(np.float32) / 255.0  # (N, 4096)
    features = PCA(n_components=n_components, random_state=SEED).fit_transform(flat)
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED).fit_predict(features)


def contact_sheet(
    images: list[np.ndarray],
    columns: int = SHEET_COLUMNS,
    frames: list[tuple[int, int, int]] | None = None,
) -> Image.Image:
    """Arrange equally sized greyscale (H, W) or colour (H, W, 3) images in a grid.

    If frames is given, each image gets a 2-pixel frame in its colour.
    """
    height, width = images[0].shape[:2]
    rows = -(-len(images) // columns)  # ceiling division
    sheet = Image.new("RGB", (columns * (width + GAP), rows * (height + GAP)), "white")
    for i, image in enumerate(images):
        row, col = divmod(i, columns)
        x, y = col * (width + GAP), row * (height + GAP)
        sheet.paste(Image.fromarray(image).convert("RGB"), (x, y))
        if frames is not None:
            ImageDraw.Draw(sheet).rectangle(
                [x, y, x + width - 1, y + height - 1], outline=frames[i], width=2
            )
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, required=True, help="download record JSON")
    args = parser.parse_args()

    record = json.loads(args.record.read_text())
    root, version = Path(record["path"]), record["version"]
    review = Path("data/review")
    sheets = review / f"v{version}_sheets"
    sheets.mkdir(parents=True, exist_ok=True)

    with Path(f"data/manifests/v{version}.csv").open() as fh:
        manifest = list(csv.DictReader(fh))
    with (review / f"v{version}_properties.csv").open() as fh:
        spread = [float(row["channel_spread"]) for row in csv.DictReader(fh)]
    data = np.load(review / f"v{version}_thumbs.npz")
    thumbs, paths = data["thumbs"], list(data["paths"])
    if paths != [row["path"] for row in manifest]:
        raise ValueError("thumbnails are out of date with the manifest; rebuild them")

    clusters = cluster_thumbnails(thumbs)
    with (review / f"v{version}_clusters.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "cluster"])
        writer.writerows(zip(paths, clusters.tolist(), strict=True))

    rng = np.random.default_rng(SEED)
    print(f"{'cluster':>7} {'size':>5}  " + "  ".join(f"{c:>10}" for c in CLASSES))
    for k in range(clusters.max() + 1):
        members = np.flatnonzero(clusters == k)
        shown = rng.choice(members, size=min(SHEET_SIZE, len(members)), replace=False)
        frames = [CLASS_COLOURS[manifest[i]["class_name"]] for i in shown]
        contact_sheet([thumbs[i] for i in shown], frames=frames).save(
            sheets / f"cluster_{k:02d}.png"
        )
        counts = Counter(manifest[i]["class_name"] for i in members)
        print(f"{k:>7} {len(members):>5}  " + "  ".join(f"{counts[c]:>10}" for c in CLASSES))

    # Colour images are drawn from the originals, since thumbnails are greyscale.
    colour = [i for i, s in enumerate(spread) if s > GREY_TOLERANCE]
    if colour:
        originals = []
        for i in colour:
            with Image.open(root / paths[i]) as img:
                originals.append(np.asarray(img.convert("RGB").resize((96, 96))))
        contact_sheet(originals).save(sheets / "not_greyscale.png")
    print(f"{len(colour)} not-greyscale images, sheets -> {sheets}")
    print("frames: " + ", ".join(f"{c} {CLASS_COLOURS[c]}" for c in CLASSES))


if __name__ == "__main__":
    main()
