#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REQUIRED_CLASSES = {"pigeon"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the pigeon detector and evaluate its test split"
    )
    parser.add_argument("data_yaml")
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="pigeon_detector")
    parser.add_argument("--name", default="pigeon_experiment")
    args = parser.parse_args()

    data = yaml.safe_load(Path(args.data_yaml).read_text(encoding="utf-8"))
    names_value = data.get("names", [])
    names = list(names_value.values()) if isinstance(names_value, dict) else list(names_value)
    normalized = {str(name).lower() for name in names}
    if normalized != REQUIRED_CLASSES:
        raise SystemExit("refusing to train: expected the single class pigeon")
    if not data.get("test"):
        raise SystemExit("refusing to train: data.yaml has no held-out test split")

    from ultralytics import YOLO

    model = YOLO(args.base_model)
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        pretrained=True,
        patience=10,
        plots=True,
    )
    best_model = Path(args.project) / args.name / "weights" / "best.pt"
    YOLO(str(best_model)).val(data=args.data_yaml, split="test", imgsz=args.imgsz, plots=True)
    print(best_model)


if __name__ == "__main__":
    main()
