#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
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


def read_c1_data_yaml(path: Path) -> dict[str, JsonValue]:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid dataset {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"invalid dataset {path}: expected a mapping")
    return {str(key): value for key, value in data.items()}


def c1_audit(dataset: Path, data: dict[str, JsonValue], configuration_path: Path) -> list[str]:
    """CONTRACTS.md C1 checks for datasets built by tools/build_dataset.py (DATASET.json present):
    manifest fields and licences, labels against the taxonomy, groups within one split, and for
    training datasets no exact or near duplicate of a test image."""
    if __package__ in (None, ""):  # run as `python tools/dataset_audit.py`
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.build_dataset import DatasetError, load_dataset_manifest, validate_dataset
    from tools.dedupe_images import nearest_distances

    problems: list[str] = []
    try:
        for note in validate_dataset(dataset):
            print(note)
    except DatasetError as error:
        problems.append(str(error))
    records = load_dataset_manifest(dataset)
    licences: dict[str, int] = {}
    for record in records:
        licences[str(record.get("license"))] = licences.get(str(record.get("license")), 0) + 1
    print("licences: " + ", ".join(f"{name} {count}" for name, count in sorted(licences.items())))
    test_split = data.get("test")
    if "train" not in {record["split"] for record in records} or not isinstance(test_split, str):
        return problems
    test_images = resolve_image_directory(configuration_path, data.get("path"), test_split)
    test_dataset = test_images.parent.parent
    if not (test_dataset / "manifest.jsonl").exists():
        problems.append(f"test dataset {test_dataset} has no manifest.jsonl")
        return problems
    summary = json.loads((dataset / "DATASET.json").read_text(encoding="utf-8"))
    limit = int(summary["dedupe"]["phash_max_distance"])
    test_records = load_dataset_manifest(test_dataset)
    shared = {record["sha256"] for record in records} & {r["sha256"] for r in test_records}
    if shared:
        problems.append(f"{len(shared)} images are byte-identical to test images")
    nearest = nearest_distances(
        {record["id"]: [record["phash"]] for record in records},
        {record["id"]: [record["phash"]] for record in test_records},
    )
    closest = min(nearest.values(), default=(65, None))
    print(f"closest train/val image to {test_dataset.name}: phash distance {closest[0]}")
    if closest[0] <= limit:
        problems.append(f"train/val image within phash distance {limit} of a test image")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a deployment dataset before training")
    parser.add_argument("data_yaml")
    arguments = parser.parse_args()
    configuration_path = Path(arguments.data_yaml).resolve()
    try:
        if (configuration_path.parent / "DATASET.json").exists():
            # C1 datasets may use the multi-class `species` taxonomy; c1_audit checks the
            # names against training/classes.yaml instead of the single-class runtime rule.
            data = read_c1_data_yaml(configuration_path)
        else:
            data = load_dataset(configuration_path)
    except ValueError as error:
        raise SystemExit(f"dataset audit failed: {error}") from error
    problems: list[str] = []
    roots: dict[str, set[str]] = {}
    dataset = configuration_path.parent
    required_splits: tuple[str, ...] = DATASET_SPLITS
    if (dataset / "DATASET.json").exists():
        summary = json.loads((dataset / "DATASET.json").read_text(encoding="utf-8"))
        if set(summary.get("counts", {})) == {"test"}:
            required_splits = ("test",)  # C1 test datasets hold only a test split
        problems.extend(c1_audit(dataset, data, configuration_path))
    for split in required_splits:
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
