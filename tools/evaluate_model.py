#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the held-out test split"
    )
    parser.add_argument("model")
    parser.add_argument("data_yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--output", default="test_metrics.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data_yaml,
        split="test",
        imgsz=args.imgsz,
        conf=args.confidence,
        plots=True, device="cpu", batch=1, workers=0, rect=False, iou=0.7,
        project=str(Path("outputs/evaluation").resolve()), name=Path(args.model).stem,
    )
    result = {
        "platform": platform.platform(),
        "split": "test", "device": "cpu", "rect": False, "iou": 0.7,
        "speed_ms_per_image": metrics.speed,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "confidence": args.confidence,
        "image_size": args.imgsz,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
