"""Pack a sample of images into one self-contained HTML page for labelling.

Sequence (T1, T2, FLAIR, ...) and slice plane are not recorded anywhere in the
dataset, but both differ between classes, so they have to be labelled by eye.
The page needs nothing but a browser: the rater opens it, labels each image
and downloads a CSV. Progress is kept in the browser between sittings.

The page is blind. Images are shuffled and renamed img_001, img_002, ..., so
neither file names nor class folders reveal the class. The key that maps ids
back to paths is written next to the page and is never sent to the rater.

Usage:
    uv run python -m neurotfinder.labelling --record data/raw/download_v1.json \\
        --sample data/labelling/sample.csv --rater batu
"""

import argparse
import base64
import csv
import hashlib
import json
from importlib.resources import files
from pathlib import Path

import numpy as np

# (key in the CSV, text on the button). Order sets the keyboard shortcuts 1-7.
SEQUENCES = (
    ("t1", "T1"),
    ("t1c", "T1 with contrast"),
    ("t1_unclear", "T1, contrast unclear"),
    ("t2", "T2"),
    ("flair", "FLAIR"),
    ("ct", "CT"),
    ("other", "Other"),
)
# Shortcut is the first letter: a, c, s.
PLANES = (("axial", "Axial"), ("coronal", "Coronal"), ("sagittal", "Sagittal"))
BATCH = 50  # images between suggested breaks
SEED = 0
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
PLACEHOLDER = "/*DATA*/null"


def assign_ids(paths: list[str], seed: int = SEED) -> list[tuple[str, str]]:
    """Shuffle paths and give each a neutral id, as (id, path) pairs in page order.

    Without shuffling, images from one class folder would arrive in a run and
    the order alone would give the class away.
    """
    order = np.random.default_rng(seed).permutation(len(paths))
    width = max(3, len(str(len(paths))))
    return [(f"img_{i + 1:0{width}d}", paths[j]) for i, j in enumerate(order)]


def data_url(path: Path) -> str:
    """The file's original bytes as a data URL: full resolution, no re-encoding."""
    mime = MIME[path.suffix.lower()]
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def render_page(items: list[dict], rater: str) -> str:
    """Fill the page template with the images and label options.

    items holds {"id", "src"} dicts in page order. The session string keys the
    saved progress in the browser, so two different packs never overwrite
    each other's answers.
    """
    session = hashlib.sha1(
        (rater + "|" + "|".join(item["id"] + item["src"][-64:] for item in items)).encode()
    ).hexdigest()[:12]
    data = {
        "rater": rater,
        "session": session,
        "batch": BATCH,
        "sequences": SEQUENCES,
        "planes": PLANES,
        "items": items,
    }
    # "</" inside a script block would end it early, so escape it.
    payload = json.dumps(data).replace("</", "<\\/")
    template = files("neurotfinder").joinpath("labeller.html").read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError("labeller.html has no data placeholder")
    return template.replace(PLACEHOLDER, payload)


def build_labeller(
    root: Path, paths: list[str], rater: str, out_dir: Path, seed: int = SEED
) -> tuple[Path, Path]:
    """Write the page and its key for one rater; returns (page, key) paths."""
    pairs = assign_ids(paths, seed)
    items = [{"id": image_id, "src": data_url(root / path)} for image_id, path in pairs]
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / f"labeller_{rater}.html"
    key = out_dir / f"key_{rater}.csv"
    page.write_text(render_page(items, rater), encoding="utf-8")
    with key.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "path"])
        writer.writerows(pairs)
    return page, key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, required=True, help="download record JSON")
    parser.add_argument("--sample", type=Path, required=True, help="CSV with a 'path' column")
    parser.add_argument("--rater", required=True, help="short name, used in file names")
    parser.add_argument("--out", type=Path, default=Path("data/labelling"))
    args = parser.parse_args()

    record = json.loads(args.record.read_text())
    with args.sample.open() as fh:
        paths = [row["path"] for row in csv.DictReader(fh)]
    if len(set(paths)) != len(paths):
        raise ValueError("the sample lists the same image more than once")

    page, key = build_labeller(Path(record["path"]), paths, args.rater, args.out)
    size_mb = page.stat().st_size / 1e6
    print(f"{len(paths)} images -> {page} ({size_mb:.1f} MB)")
    print(f"key -> {key} (keep this; do not send it to the rater)")


if __name__ == "__main__":
    main()
