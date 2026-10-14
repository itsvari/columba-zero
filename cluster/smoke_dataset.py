"""Build small CC0 synthetic C1 archives for infrastructure checks only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from cluster.staging import sha256_file

IMAGE_SIZE_PX = 640
IMAGES_PER_SPLIT = 4


def build_dataset(root: Path, name: str, splits: tuple[str, ...]) -> Path:
    import zstandard

    dataset = root / name
    records: list[dict[str, str | int | list[str] | None]] = []
    for split in splits:
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index in range(IMAGES_PER_SPLIT):
            identifier = f"synthetic-{split}-{index}"
            image = Image.new("RGB", (IMAGE_SIZE_PX, IMAGE_SIZE_PX), (30 + index * 20, 70, 100))
            drawing = ImageDraw.Draw(image)
            drawing.ellipse((160, 160, 480, 480), fill=(180, 180, 190))
            image_path = dataset / "images" / split / f"{identifier}.jpg"
            image.save(image_path)
            (dataset / "labels" / split / f"{identifier}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
            records.append(
                {
                    "id": identifier,
                    "split": split,
                    "file": f"images/{split}/{identifier}.jpg",
                    "source": "synthetic-ws3",
                    "source_id": identifier,
                    "group": identifier,
                    "license": "CC0-1.0",
                    "attribution": "WS3 generated infrastructure fixture",
                    "sha256": sha256_file(image_path),
                    "phash": "0000000000000000",
                    "width": IMAGE_SIZE_PX,
                    "height": IMAGE_SIZE_PX,
                    "style": "original",
                    "derived_from": None,
                    "label_origin": "source",
                    "classes_present": ["pigeon"],
                }
            )
    configuration = {
        "path": ".",
        "names": ["pigeon"],
        **{split: f"images/{split}" for split in splits},
    }
    if "train" in splits:
        configuration["test"] = "../smoke-coco8-test/images/test"
    (dataset / "data.yaml").write_text(yaml.safe_dump(configuration))
    (dataset / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    (dataset / "DATASET.json").write_text(
        json.dumps(
            {
                "name": name,
                "taxonomy": "single",
                "taxonomy_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "builder_commit": "synthetic",
                "sources": ["synthetic-ws3"],
                "counts": {
                    split: {
                        "images": IMAGES_PER_SPLIT,
                        "boxes": {"pigeon": IMAGES_PER_SPLIT},
                        "backgrounds": 0,
                    }
                    for split in splits
                },
                "dedupe": {"phash_max_distance": 0, "test_dataset": "smoke-coco8-test"},
                "archive_sha256": None,
                "notes": "synthetic circles, never an accuracy dataset",
            },
            indent=2,
        )
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.add(dataset, arcname=name)
    destination = root / f"{name}.tar.zst"
    destination.write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue()))
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_name(destination.name + ".sha256").write_text(
        f"{checksum}  {destination.name}\n"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    for name, splits in (("smoke-coco8", ("train", "val")), ("smoke-coco8-test", ("test",))):
        print(build_dataset(arguments.out, name, splits))


if __name__ == "__main__":
    main()
