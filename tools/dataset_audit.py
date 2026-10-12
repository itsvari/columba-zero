#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from app.dataset import load_dataset
from app.json_data import JsonValue

DATASET_SPLITS = ("train", "val", "test")
SPLIT_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))
IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")


def label_directory(image_directory: Path) -> Path:
    parts = list(image_directory.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts)
    return image_directory.parent / "labels"


def resolve_image_directory(
    configuration_path: Path,
    dataset_root: JsonValue,
    split_directory: str,
) -> Path:
    image_directory = Path(split_directory)
    if image_directory.is_absolute():
        return image_directory
    if dataset_root is not None and not isinstance(dataset_root, str):
        raise SystemExit("dataset path must be a directory path")
    root = Path(dataset_root) if isinstance(dataset_root, str) else configuration_path.parent
    if not root.is_absolute():
        root = configuration_path.parent / root
    return root / image_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a deployment dataset before training")
    parser.add_argument("data_yaml")
    arguments = parser.parse_args()
    configuration_path = Path(arguments.data_yaml).resolve()
    try:
        data = load_dataset(configuration_path)
    except ValueError as error:
        raise SystemExit(f"dataset audit failed: {error}") from error
    problems: list[str] = []
    roots: dict[str, set[str]] = {}
    for split in DATASET_SPLITS:
        value = data.get(split)
        if not value:
            problems.append(f"missing {split} split")
            continue
        if not isinstance(value, str):
            problems.append(f"{split} split must be an image directory path")
            continue
        image_directory = resolve_image_directory(configuration_path, data.get("path"), value)
        images = {path.stem for pattern in IMAGE_PATTERNS for path in image_directory.glob(pattern)}
        roots[split] = images
        labels_directory = label_directory(image_directory)
        backgrounds = 0
        for stem in images:
            label_path = labels_directory / f"{stem}.txt"
            if not label_path.exists() or not label_path.read_text(encoding="utf-8").strip():
                backgrounds += 1
        print(f"{split}: {len(images)} images, {backgrounds} background-only images")
        if not images:
            problems.append(f"{split} split is empty")
    for left, right in SPLIT_PAIRS:
        overlap = roots.get(left, set()) & roots.get(right, set())
        if overlap:
            problems.append(f"{left}/{right} share {len(overlap)} filename stems")
    if problems:
        raise SystemExit("dataset audit failed:\n- " + "\n- ".join(problems))
    print(
        "Dataset structure audit passed. Session/location split provenance still requires review."
    )


if __name__ == "__main__":
    main()
