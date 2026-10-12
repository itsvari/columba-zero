#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from app.inference import NMS_INTERSECTION_OVER_UNION
from app.json_data import JsonValue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the held-out test split"
    )
    parser.add_argument("model")
    parser.add_argument("data_yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--output", default="test_metrics.json")
    arguments = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(arguments.model, task="detect")
    metrics = model.val(
        data=arguments.data_yaml,
        split="test",
        imgsz=arguments.imgsz,
        conf=arguments.confidence,
        plots=True,
        device="cpu",
        batch=1,
        workers=0,
        rect=False,
        iou=NMS_INTERSECTION_OVER_UNION,
        project=str(Path("outputs/evaluation").resolve()),
        name=Path(arguments.model).stem,
    )
    result: dict[str, JsonValue] = {
        "platform": platform.platform(),
        "split": "test",
        "device": "cpu",
        "rect": False,
        "iou": NMS_INTERSECTION_OVER_UNION,
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "confidence": arguments.confidence,
        "image_size": arguments.imgsz,
    }
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
