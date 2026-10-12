"""Compare CPU PT and NCNN predictions on identical square-preprocessed images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iou(left: list[float], right: list[float]) -> float:
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1]))
    union = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1]) - intersection)
    return intersection / union if union else 0.0


def predictions(model, image: Path, size: int, confidence: float) -> list[dict]:
    result = model.predict(str(image), imgsz=size, conf=confidence, device="cpu",
                           rect=False, iou=0.7, verbose=False)[0]
    return [{"label": result.names[int(box.cls.item())],
             "confidence": float(box.conf.item()), "box": box.xyxy[0].tolist()}
            for box in result.boxes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt_model")
    parser.add_argument("ncnn_model")
    parser.add_argument("images")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--min-iou", type=float, default=0.8)
    parser.add_argument("--max-confidence-delta", type=float, default=0.05)
    parser.add_argument("--output", default="export_comparison.json")
    args = parser.parse_args()
    from ultralytics import YOLO
    pt, ncnn = YOLO(args.pt_model, task="detect"), YOLO(args.ncnn_model, task="detect")
    images = sorted(p for p in Path(args.images).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit("no images to compare")
    records = []
    for image in images:
        expected = predictions(pt, image, args.imgsz, args.confidence)
        actual = predictions(ncnn, image, args.imgsz, args.confidence)
        unmatched = set(range(len(actual)))
        matches = []
        for index, source in enumerate(expected):
            candidates = [(iou(source["box"], actual[j]["box"]), j) for j in unmatched
                          if source["label"] == actual[j]["label"]]
            overlap, other = max(candidates, default=(0.0, -1))
            if other < 0 or overlap == 0:
                continue
            unmatched.remove(other)
            matches.append({"pt_index": index, "ncnn_index": other, "iou": overlap,
                            "confidence_delta": abs(source["confidence"] - actual[other]["confidence"]),
                            "box_delta_px": [b - a for a, b in zip(source["box"], actual[other]["box"])]})
        passed = (len(matches) == len(expected) == len(actual)
                  and all(m["iou"] >= args.min_iou
                          and m["confidence_delta"] <= args.max_confidence_delta for m in matches))
        records.append({"image": str(image), "pt_count": len(expected), "ncnn_count": len(actual),
                        "pt": expected, "ncnn": actual, "matches": matches, "passed": passed})
    failures = sum(not r["passed"] for r in records)
    report = {"images": len(images), "failures": failures, "image_size": args.imgsz,
              "confidence": args.confidence, "device": "cpu", "rect": False, "iou": 0.7,
              "min_iou": args.min_iou, "max_confidence_delta": args.max_confidence_delta,
              "records": records}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"images": len(images), "failures": failures}))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
