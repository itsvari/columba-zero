#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def predictions(
    model: object, image: Path, size: int, confidence: float
) -> list[dict[str, object]]:
    result = model.predict(str(image), imgsz=size, conf=confidence, verbose=False)[0]
    output = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        output.append(
            {
                "label": str(result.names[class_id]),
                "confidence": float(box.conf.item()),
                "box": [float(value) for value in box.xyxy[0].tolist()],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PT and NCNN detections on fixed images")
    parser.add_argument("pt_model")
    parser.add_argument("ncnn_model")
    parser.add_argument("images", help="image directory")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--min-iou", type=float, default=0.8)
    parser.add_argument("--output", default="export_comparison.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    pt = YOLO(args.pt_model, task="detect")
    ncnn = YOLO(args.ncnn_model, task="detect")
    image_paths = sorted(
        path
        for path in Path(args.images).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    records = []
    failures = 0
    for image in image_paths:
        expected = predictions(pt, image, args.imgsz, args.confidence)
        actual = predictions(ncnn, image, args.imgsz, args.confidence)
        unmatched = list(range(len(actual)))
        matches = []
        for source in expected:
            candidates = [
                (iou(source["box"], actual[index]["box"]), index)
                for index in unmatched
                if source["label"] == actual[index]["label"]
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_index >= 0:
                unmatched.remove(best_index)
            matches.append(best_iou)
        passed = len(expected) == len(actual) and all(value >= args.min_iou for value in matches)
        failures += not passed
        records.append(
            {
                "image": str(image),
                "pt_count": len(expected),
                "ncnn_count": len(actual),
                "matched_ious": matches,
                "passed": passed,
            }
        )
    summary = {"images": len(records), "failures": failures, "records": records}
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(records), "failures": failures}, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
