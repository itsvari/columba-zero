#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from app.dataset import load_dataset

EARLY_STOPPING_PATIENCE_EPOCHS = 10


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
    arguments = parser.parse_args()

    try:
        data = load_dataset(Path(arguments.data_yaml))
    except ValueError as error:
        raise SystemExit(f"refusing to train: {error}") from error
    if not data.get("test"):
        raise SystemExit("refusing to train: data.yaml has no held-out test split")

    from ultralytics import YOLO

    model = YOLO(arguments.base_model, task="detect")
    model.train(
        data=arguments.data_yaml,
        epochs=arguments.epochs,
        imgsz=arguments.imgsz,
        batch=arguments.batch,
        project=arguments.project,
        name=arguments.name,
        pretrained=True,
        patience=EARLY_STOPPING_PATIENCE_EPOCHS,
        plots=True,
    )
    best_model = Path(arguments.project) / arguments.name / "weights" / "best.pt"
    YOLO(str(best_model), task="detect").val(
        data=arguments.data_yaml, split="test", imgsz=arguments.imgsz, plots=True
    )
    print(best_model)


if __name__ == "__main__":
    main()
