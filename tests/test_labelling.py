import base64
import csv
import json
import re

from PIL import Image

from neurotfinder.labelling import assign_ids, build_labeller, data_url, render_page


def _payload(page: str) -> dict:
    match = re.search(r"const DATA = (.*);\n", page)
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_ids_are_unique_padded_and_cover_every_path():
    paths = [f"Training/glioma/{i}.jpg" for i in range(12)]

    pairs = assign_ids(paths)

    ids = [image_id for image_id, _ in pairs]
    assert ids == [f"img_{i:03d}" for i in range(1, 13)]
    assert sorted(path for _, path in pairs) == sorted(paths)


def test_order_is_shuffled_but_reproducible():
    paths = [f"p{i}.jpg" for i in range(50)]

    first, again = assign_ids(paths, seed=1), assign_ids(paths, seed=1)

    assert first == again
    assert [path for _, path in first] != paths


def test_data_url_keeps_the_original_bytes(tmp_path):
    image = tmp_path / "scan.png"
    Image.new("L", (512, 512), 90).save(image)

    url = data_url(image)

    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == image.read_bytes()


def test_page_is_blind_to_paths_and_classes(tmp_path):
    root = tmp_path / "data"
    paths = []
    for class_name in ("glioma", "notumor"):
        (root / "Training" / class_name).mkdir(parents=True)
        for i in range(3):
            path = f"Training/{class_name}/Tr-{class_name}_{i}.jpg"
            Image.new("L", (64, 64), 30 * i).save(root / path)
            paths.append(path)

    page, key = build_labeller(root, paths, "batu", tmp_path / "out")

    html = page.read_text()
    for word in ("glioma", "notumor", "Training", "Tr-"):
        assert word not in html
    with key.open() as fh:
        rows = list(csv.DictReader(fh))
    assert [row["id"] for row in rows] == [item["id"] for item in _payload(html)["items"]]
    assert sorted(row["path"] for row in rows) == sorted(paths)


def test_text_cannot_break_out_of_the_script_block():
    page = render_page([{"id": "img_001", "src": "data:image/png;base64,AA=="}], "a</script>b")

    assert page.count("</script>") == 1
    assert _payload(page)["rater"] == "a</script>b"
