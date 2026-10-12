#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REQUIRED_CLASSES = {"pigeon"}


def label_directory(image_directory: Path) -> Path:
    parts = list(image_directory.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts)
    return image_directory.parent / "labels"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a deployment dataset before training")
    parser.add_argument("data_yaml")
    args = parser.parse_args()
    yaml_path = Path(args.data_yaml).resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names_value = data.get("names", [])
    names = list(names_value.values()) if isinstance(names_value, dict) else list(names_value)
    normalized = {str(name).lower() for name in names}
    missing = normalized != REQUIRED_CLASSES
    problems: list[str] = []
    if missing:
        problems.append("expected the single class pigeon")
    roots: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        value = data.get(split)
        if not value:
            problems.append(f"missing {split} split")
            continue
        image_dir = Path(value)
        if not image_dir.is_absolute():
            root = Path(data.get("path", yaml_path.parent))
            if not root.is_absolute():
                root = yaml_path.parent / root
            image_dir = root / image_dir
        images = {
            path.stem
            for pattern in ("*.jpg", "*.jpeg", "*.png")
            for path in image_dir.glob(pattern)
        }
        roots[split] = images
        label_dir = label_directory(image_dir)
        backgrounds = 0
        for stem in images:
            label_path = label_dir / f"{stem}.txt"
            if not label_path.exists() or not label_path.read_text(encoding="utf-8").strip():
                backgrounds += 1
        print(f"{split}: {len(images)} images, {backgrounds} background-only images")
        if not images:
            problems.append(f"{split} split is empty")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
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
