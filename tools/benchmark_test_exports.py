"""Compare held-out accuracy and CPU validation timings at matched input settings."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from app.inference import NMS_INTERSECTION_OVER_UNION
from app.json_data import JsonValue

IMAGE_SIZE_PX = 640
VALIDATION_CONFIDENCE_FLOOR = 0.001
MILLISECONDS_PER_SECOND = 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml")
    parser.add_argument("--models", default="models/pigeon_baseline_v1-2")
    parser.add_argument("--output", default="outputs/pc_benchmark")
    arguments = parser.parse_args()
    from ultralytics import YOLO

    model_directory = Path(arguments.models)
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, JsonValue] = {}
    report: dict[str, JsonValue] = {
        "platform": platform.platform(),
        "imgsz": IMAGE_SIZE_PX,
        "device": "cpu",
        "batch": 1,
        "conf": VALIDATION_CONFIDENCE_FLOOR,
        "iou": NMS_INTERSECTION_OVER_UNION,
        "rect": False,
        "results": results,
    }
    for label, model_path in [
        ("pytorch", model_directory / "best.pt"),
        ("ncnn", model_directory / "best_ncnn_model"),
    ]:
        metrics = YOLO(str(model_path), task="detect").val(
            data=arguments.data_yaml,
            split="test",
            imgsz=IMAGE_SIZE_PX,
            device="cpu",
            batch=1,
            workers=0,
            conf=VALIDATION_CONFIDENCE_FLOOR,
            iou=NMS_INTERSECTION_OVER_UNION,
            rect=False,
            plots=True,
            project=str(output),
            name=label,
            verbose=False,
        )
        speed: dict[str, float] = {key: float(value) for key, value in metrics.speed.items()}
        results[label] = {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "speed_ms_per_image": {key: value for key, value in speed.items()},
            "inference_fps": MILLISECONDS_PER_SECOND / speed["inference"],
            "pipeline_fps": MILLISECONDS_PER_SECOND
            / sum(speed[key] for key in ("preprocess", "inference", "postprocess")),
        }
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results[label], indent=2))


if __name__ == "__main__":
    main()
