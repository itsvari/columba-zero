#!/usr/bin/env python
"""Pseudo-label bird boxes on unlabelled source images and gate them on a recorded review.

    python -m tools.auto_label detect ~/columba-data/raw/inat-pigeon/provenance.jsonl \
        --weights /abs/path/yolo11x.pt --device mps
    python -m tools.auto_label sample ~/columba-data/raw/inat-pigeon/provenance.jsonl \
        --count 200 --output-dir ~/columba-data/contact_sheets/review/auto-inat-pigeon
    python -m tools.auto_label summarize data_sources/review/inat-pigeon.review.jsonl

`detect` runs a COCO-pretrained detector (YOLO11x by default) and keeps its `bird` boxes.
Each box gets the species of the source taxon (`extra.taxon` in the provenance). It writes
`raw/<source>/auto_labels.jsonl`, one line per image:

    {"id": ..., "status": "pseudo_unreviewed", "decision": "keep" | "drop",
     "flags": [...], "boxes": [{"class": "pigeon", "xyxy": [...], "score": 0.91}],
     "uncertain": [...], "model": "yolo11x.pt", "conf": 0.35, "low_conf": 0.1}

Decisions (`label_policy`): drop an image when no box reaches `conf` (the target bird was
missed), when a detection between `low_conf` and `conf` does not overlap a kept box (a
bird may be unlabelled), or, for non-pigeon taxa, when there is more than one bird (the
others may be a different species; pigeon flocks are kept because they carry the small
pigeons we need).

`sample` renders a seeded random sample of kept images on numbered contact sheets for
review. The reviewer writes `data_sources/review/<source>.review.jsonl` with one line per
reviewed image: `{"id", "boxes": n, "wrong_boxes": n, "missed_birds": n, "drop": bool,
"note"}`. `summarize` computes the box error rate (wrong + missed) / (boxes + missed) and
writes `<source>.summary.json`; the builder accepts a source's pseudo labels
(`pseudo_reviewed`) only if that summary exists, covers at least 200 images and the error
rate is at most 5%. Images marked `drop` in the review are always left out.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from tools.contact_sheet import SheetEntry, render_sheets
from tools.fetch_sources import read_provenance, write_jsonl

COCO_BIRD_CLASS = 14
DEFAULT_CONF = 0.35
DEFAULT_LOW_CONF = 0.10
DEFAULT_IMGSZ = 1024
OVERLAP_IOU = 0.3
MIN_REVIEWED_IMAGES = 200
MAX_ERROR_RATE = 0.05
TRIGGER_TAXON = "pigeon"
BATCH_SIZE = 8

Record = dict[str, Any]
Detection = tuple[float, list[float]]  # (score, xyxy) of one `bird` detection
Detector = Callable[[Sequence[Path]], list[list[Detection]]]


def intersection_over_union(left: Sequence[float], right: Sequence[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    overlap = width * height
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - overlap
    )
    return overlap / union if union > 0 else 0.0


def label_policy(
    detections: Sequence[Detection], taxon: str, conf: float, low_conf: float
) -> Record:
    """Turn one image's bird detections into boxes for `taxon`, flags and a keep/drop decision."""
    kept = [(score, box) for score, box in detections if score >= conf]
    uncertain = [
        (score, box)
        for score, box in detections
        if low_conf <= score < conf
        and all(intersection_over_union(box, other) < OVERLAP_IOU for _, other in kept)
    ]
    flags = []
    if not kept:
        flags.append("no_bird")
    if uncertain:
        flags.append("uncertain_box")
    if len(kept) > 1:
        flags.append("multi_bird")
    drop = "no_bird" in flags or "uncertain_box" in flags
    if taxon != TRIGGER_TAXON and len(kept) > 1:
        drop = True
    return {
        "decision": "drop" if drop else "keep",
        "flags": flags,
        "boxes": [
            {"class": taxon, "xyxy": [round(v, 2) for v in box], "score": round(score, 4)}
            for score, box in sorted(kept, key=lambda item: -item[0])
        ],
        "uncertain": [
            {"xyxy": [round(v, 2) for v in box], "score": round(score, 4)}
            for score, box in uncertain
        ],
    }


def ultralytics_detector(weights: Path, device: str, imgsz: int, low_conf: float) -> Detector:
    if not weights.is_absolute():
        raise SystemExit("pass --weights as an absolute path (bare names download into cwd)")
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def detect(paths: Sequence[Path]) -> list[list[Detection]]:
        results = model.predict(
            [str(path) for path in paths],
            classes=[COCO_BIRD_CLASS],
            conf=low_conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        output = []
        for result in results:
            boxes = result.boxes
            output.append(
                [
                    (float(score), [float(value) for value in box])
                    for score, box in zip(boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True)
                ]
            )
        return output

    return detect


def auto_label(
    provenance_path: Path,
    detector: Detector,
    model_name: str,
    conf: float = DEFAULT_CONF,
    low_conf: float = DEFAULT_LOW_CONF,
    batch_size: int = BATCH_SIZE,
) -> Path:
    """Write `auto_labels.jsonl` next to the provenance; images already labelled are kept."""
    root = provenance_path.parent
    output = root / "auto_labels.jsonl"
    done: dict[str, Record] = {}
    if output.exists():
        done = {record["id"]: record for record in read_provenance(output)}
    todo = [record for record in read_provenance(provenance_path) if record["id"] not in done]
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        detections = detector([root / record["file"] for record in batch])
        for record, found in zip(batch, detections, strict=True):
            taxon = (record.get("extra") or {}).get("taxon") or record["classes_present"][0]
            done[record["id"]] = {
                "id": record["id"],
                "status": "pseudo_unreviewed",
                "model": model_name,
                "conf": conf,
                "low_conf": low_conf,
                **label_policy(found, taxon, conf, low_conf),
            }
        if (start // batch_size) % 50 == 0:
            print(f"{root.name}: {min(start + batch_size, len(todo))}/{len(todo)}")
            write_jsonl(output, sorted(done.values(), key=lambda record: record["id"]))
    write_jsonl(output, sorted(done.values(), key=lambda record: record["id"]))
    decisions = Counter(record["decision"] for record in done.values())
    flags = Counter(flag for record in done.values() for flag in record["flags"])
    print(f"{root.name}: {dict(decisions)}; flags {dict(flags)}")
    return output


def review_sample(
    provenance_path: Path, count: int, seed: int, output_dir: Path, flag: str | None = None
) -> list[Path]:
    """Render a seeded random sample of kept auto-labelled images; returns the sheet paths."""
    root = provenance_path.parent
    labels = {record["id"]: record for record in read_provenance(root / "auto_labels.jsonl")}
    records = {record["id"]: record for record in read_provenance(provenance_path)}
    kept = sorted(
        item
        for item, label in labels.items()
        if label["decision"] == "keep" and (flag is None or flag in label["flags"])
    )
    random.Random(seed).shuffle(kept)
    entries = [
        SheetEntry(
            root / records[item]["file"],
            item,
            [(box["class"], box["xyxy"]) for box in labels[item]["boxes"]],
        )
        for item in kept[:count]
    ]
    return render_sheets(entries, output_dir, root.name, per_sheet=24, columns=6, tile_px=320)


def summarize_review(review_path: Path) -> Record:
    reviews = read_provenance(review_path)
    boxes = sum(int(review["boxes"]) for review in reviews)
    wrong = sum(int(review.get("wrong_boxes", 0)) for review in reviews)
    missed = sum(int(review.get("missed_birds", 0)) for review in reviews)
    denominator = boxes + missed
    error_rate = (wrong + missed) / denominator if denominator else 0.0
    summary = {
        "source": review_path.name.removesuffix(".review.jsonl"),
        "reviewed_images": len(reviews),
        "boxes": boxes,
        "wrong_boxes": wrong,
        "missed_birds": missed,
        "images_with_errors": sum(
            1 for review in reviews if review.get("wrong_boxes") or review.get("missed_birds")
        ),
        "dropped_images": sum(1 for review in reviews if review.get("drop")),
        "box_error_rate": round(error_rate, 4),
        "passed": len(reviews) >= MIN_REVIEWED_IMAGES and error_rate <= MAX_ERROR_RATE,
        "rule": f">= {MIN_REVIEWED_IMAGES} images and (wrong + missed) / (boxes + missed) "
        f"<= {MAX_ERROR_RATE}",
    }
    review_path.with_name(f"{summary['source']}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def reviewed_labels(provenance_path: Path, review_dir: Path) -> dict[str, Record] | None:
    """Kept auto labels of a source whose review passed, without images the review dropped.
    Returns None if the source has not passed review."""
    source = provenance_path.parent.name
    summary_path = review_dir / f"{source}.summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("passed"):
        return None
    dropped = {
        review["id"]
        for review in read_provenance(review_dir / f"{source}.review.jsonl")
        if review.get("drop")
    }
    labels = read_provenance(provenance_path.parent / "auto_labels.jsonl")
    return {
        label["id"]: {**label, "status": "pseudo_reviewed"}
        for label in labels
        if label["decision"] == "keep" and label["id"] not in dropped
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    detect = commands.add_parser("detect", help="write auto_labels.jsonl for a source")
    detect.add_argument("provenance", type=Path)
    detect.add_argument("--weights", type=Path, required=True, help="absolute path")
    detect.add_argument("--device", default="mps")
    detect.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    detect.add_argument("--conf", type=float, default=DEFAULT_CONF)
    detect.add_argument("--low-conf", type=float, default=DEFAULT_LOW_CONF)
    sample = commands.add_parser("sample", help="render a review sample")
    sample.add_argument("provenance", type=Path)
    sample.add_argument("--count", type=int, default=MIN_REVIEWED_IMAGES)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--flag", help="only images with this flag, e.g. multi_bird")
    sample.add_argument("--output-dir", type=Path, required=True)
    summarize = commands.add_parser("summarize", help="error rate of a review file")
    summarize.add_argument("review", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "detect":
        detector = ultralytics_detector(
            arguments.weights, arguments.device, arguments.imgsz, arguments.low_conf
        )
        auto_label(
            arguments.provenance,
            detector,
            arguments.weights.name,
            arguments.conf,
            arguments.low_conf,
        )
    elif arguments.command == "sample":
        sheets = review_sample(
            arguments.provenance,
            arguments.count,
            arguments.seed,
            arguments.output_dir,
            arguments.flag,
        )
        print("\n".join(str(sheet) for sheet in sheets))
    else:
        print(json.dumps(summarize_review(arguments.review), indent=2))


if __name__ == "__main__":
    main()
