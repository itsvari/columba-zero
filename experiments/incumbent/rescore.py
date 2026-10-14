"""Reproduce incumbent scoring and conservative historical split-exposure analysis.

Run with the shared environment:
    python -m experiments.incumbent.rescore --data-root ~/columba-data

Published datasets and source exports are read only. C3 predictions are scratch;
small metrics and per-image exposure evidence are written beside this script.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from app.json_data import JsonValue
from cluster.artifacts import atomic_json, leaderboard_row, update_leaderboard
from tools.build_dataset import verify_freeze
from tools.dedupe_images import assign_groups
from tools.evaluate_predictions import (
    InputShape,
    evaluate,
    mapping,
    read_json,
    sidecar_path,
    string,
    string_sequence,
)
from tools.predict_to_coco import predict_to_coco

Exposure = Literal["seen_train", "seen_val", "unseen"]
REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path.home() / "columba-data"
DEFAULT_MODEL_ROOT = Path("/Users/vari/Developer/columba-zero")
COCO_SEED = 42
COCO_TRAIN_FRACTION = 0.8
COCO_TRAIN_VAL_FRACTION = 0.9
PHASH_MAX_DISTANCE = 10
INCUMBENT_EPOCHS_DONE = 60
INCUMBENT_TRAIN_SIZE_PX = 640
SOURCE_NAMES = ("roboflow-nea", "roboflow-yolo1199", "coco-birds")
EXPOSURE_ORDER: tuple[Exposure, ...] = ("seen_train", "seen_val", "unseen")
REVIEW_FILE = Path(__file__).with_name("exposure_review.json")


@dataclass(frozen=True)
class HistoricalImage:
    image_id: str
    source: str
    source_split: str
    exposure: Exposure
    original_stem: str
    scene_id: str
    sha256: str
    phash: str
    phash_mirror: str

    def grouping_record(self) -> dict[str, JsonValue]:
        return {
            "id": self.image_id,
            "scene_id": self.scene_id,
            "sha256": self.sha256,
            "phash": self.phash,
            "phash_mirror": self.phash_mirror,
        }


@dataclass(frozen=True)
class Configuration:
    name: str
    model: Path
    input_hw: InputShape


def records(path: Path) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        mapping(json.loads(line), str(path))
        for line in path.read_text().splitlines()
        if line.strip()
    )


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def coco_splits(annotation_path: Path) -> tuple[dict[int, str], tuple[int, ...]]:
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(annotation_path))
    image_ids = cast(list[int], coco.getImgIds(catIds=coco.getCatIds(catNms=["bird"])))
    original_order = tuple(image_ids)
    random.Random(COCO_SEED).shuffle(image_ids)
    training_end = int(len(image_ids) * COCO_TRAIN_FRACTION)
    validation_end = int(len(image_ids) * COCO_TRAIN_VAL_FRACTION)
    splits = {
        image_id: "train" if index < training_end else "valid" if index < validation_end else "test"
        for index, image_id in enumerate(image_ids)
    }
    return splits, original_order


def exposure(source_split: str) -> Exposure:
    return (
        "seen_train"
        if source_split == "train"
        else "seen_val"
        if source_split == "valid"
        else "unseen"
    )


def historical_images(data_root: Path, coco: dict[int, str]) -> tuple[HistoricalImage, ...]:
    images: list[HistoricalImage] = []
    for source in SOURCE_NAMES:
        source_root = data_root / "raw" / source
        hashes = {
            string(record["id"], "hash id"): record
            for record in records(source_root / "hashes.jsonl")
        }
        for record in records(source_root / "provenance.jsonl"):
            boxes = record.get("boxes")
            if not isinstance(boxes, list):
                continue
            # the notebook retained every yolo1199 image; one current export label is empty
            if source == "roboflow-nea" and not any(
                mapping(box, "raw box").get("class") == "pigeon" for box in boxes
            ):
                continue
            image_id = string(record["id"], "raw id")
            hashed = hashes[image_id]
            source_split = (
                coco[int(string(record["source_id"], "COCO id"))]
                if source == "coco-birds"
                else string(record["split_in_source"], "source split")
            )
            if source.startswith("roboflow"):
                image_path = source_root / string(record["file"], "raw file")
                if not image_path.is_file() or image_path.parent.parent.name != source_split:
                    raise ValueError(f"raw export split/path mismatch: {image_path}")
            extra = mapping(record.get("extra", {}), "raw extra")
            images.append(
                HistoricalImage(
                    image_id,
                    source,
                    source_split,
                    exposure(source_split),
                    str(extra.get("original_stem", "")),
                    str(record.get("scene_id", image_id)),
                    string(hashed["sha256"], "sha256"),
                    string(hashed["phash"], "phash"),
                    string(hashed["phash_mirror"], "mirror phash"),
                )
            )
    return tuple(images)


def primary_exposure(images: Sequence[HistoricalImage]) -> Exposure:
    present = {image.exposure for image in images}
    return next(status for status in EXPOSURE_ORDER if status in present) if present else "unseen"


def analyse_exposure(data_root: Path, output: Path) -> dict[str, JsonValue]:
    annotation_path = data_root / "raw/_coco-annotations/annotations/instances_val2017.json"
    coco, original_order = coco_splits(annotation_path)
    history = historical_images(data_root, coco)
    historical_by_id = {image.image_id: image for image in history}
    pool = {image.image_id: image.grouping_record() for image in history}
    manifests = {
        "test": tuple(
            record
            for record in records(data_root / "columba-test-v2/manifest.jsonl")
            if record["split"] == "test"
        ),
        "val": tuple(
            record
            for record in records(data_root / "columba-train-v1clean/manifest.jsonl")
            if record["split"] == "val"
        ),
    }
    query_sources = {
        string(record["source"], "source")
        for manifest in manifests.values()
        for record in manifest
        if record["id"] not in pool
    }
    query_mirrors = {
        string(record["id"], "hash id"): string(record["phash_mirror"], "mirror hash")
        for source in query_sources
        for record in records(data_root / "raw" / source / "hashes.jsonl")
    }
    for manifest in manifests.values():
        for record in manifest:
            image_id = string(record["id"], "manifest id")
            if image_id not in pool:
                pool[image_id] = {
                    "id": image_id,
                    "sha256": record["sha256"],
                    "phash": record["phash"],
                    "phash_mirror": query_mirrors[image_id],
                    "scene_id": image_id,
                }
    groups, _ = assign_groups(list(pool.values()), PHASH_MAX_DISTANCE)
    review = mapping(read_json(REVIEW_FILE), "exposure review")
    rejected_groups = review["rejected_groups"]
    if not isinstance(rejected_groups, list):
        raise ValueError("rejected_groups must be a list")
    rejected = {
        string(mapping(item, "review")["id"], "review id"): mapping(item, "review")
        for item in rejected_groups
    }
    history_by_group: dict[str, list[HistoricalImage]] = defaultdict(list)
    for image in history:
        history_by_group[groups[image.image_id]].append(image)
    summaries: dict[str, JsonValue] = {}
    output.mkdir(parents=True, exist_ok=True)
    for split, manifest in manifests.items():
        evidence: list[dict[str, JsonValue]] = []
        for record in manifest:
            image_id = string(record["id"], "manifest id")
            direct = historical_by_id.get(image_id)
            linked = history_by_group[groups[image_id]]
            status = primary_exposure(linked)
            reviewed = rejected.get(image_id)
            if reviewed is not None:
                reviewed_ids = reviewed["matched_ids"]
                if set(string_sequence(reviewed_ids, "reviewed IDs")) != {
                    image.image_id for image in linked
                }:
                    raise ValueError(
                        f"review no longer covers the entire candidate group: {image_id}"
                    )
                status = "unseen"
            direct_status = direct.exposure if direct is not None else "unseen"
            present = cast(list[str], record["classes_present"])
            evidence.append(
                {
                    "id": image_id,
                    "source": record["source"],
                    "group": record["group"],
                    "analysis_group": groups[image_id],
                    "exposure": status,
                    "direct_exposure": direct_status,
                    "source_split": direct.source_split if direct is not None else None,
                    "original_stem": direct.original_stem if direct is not None else None,
                    "pigeon": "pigeon" in present,
                    "review_rejection": reviewed["reason"] if reviewed is not None else None,
                    "matched_train_ids": cast(
                        list[JsonValue],
                        sorted(
                            image.image_id for image in linked if image.exposure == "seen_train"
                        ),
                    ),
                    "matched_val_ids": cast(
                        list[JsonValue],
                        sorted(image.image_id for image in linked if image.exposure == "seen_val"),
                    ),
                }
            )
        (output / f"exposure_{split}.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in evidence) + "\n"
        )
        unseen = sorted(string(row["id"], "id") for row in evidence if row["exposure"] == "unseen")
        (output / f"unseen_{split}.txt").write_text("\n".join(unseen) + ("\n" if unseen else ""))
        counts = Counter(str(row["exposure"]) for row in evidence)
        source_counts = {
            source: {
                status: sum(
                    row["source"] == source and row["exposure"] == status for row in evidence
                )
                for status in EXPOSURE_ORDER
            }
            for source in sorted({str(row["source"]) for row in evidence})
        }
        summaries[split] = {
            "images": len(evidence),
            "counts": dict(counts),
            "direct_counts": dict(Counter(str(row["direct_exposure"]) for row in evidence)),
            "pigeon_images": {
                status: sum(row["exposure"] == status and row["pigeon"] is True for row in evidence)
                for status in EXPOSURE_ORDER
            },
            "by_source": cast(dict[str, JsonValue], source_counts),
        }
    summary: dict[str, JsonValue] = {
        "label": "measured",
        "host": platform.node(),
        "method": (
            "reconstructed notebook membership; conservative WS2 stem/scene/SHA256/"
            "mirror-phash connected components with recorded visual false-match rejection; "
            "train precedes val; old test is unseen"
        ),
        "phash_max_distance": PHASH_MAX_DISTANCE,
        "coco_seed": COCO_SEED,
        "coco_annotation_sha256": file_hash(annotation_path),
        "coco_getImgIds_order": list(original_order),
        "coco_split_counts": dict(Counter(coco.values())),
        "historical_images_with_local_hashes": len(history),
        "review_file": str(REVIEW_FILE.relative_to(REPOSITORY)),
        "splits": summaries,
    }
    atomic_json(output / "contamination.json", summary)
    atomic_json(
        output / "coco_original_splits.json",
        {
            "seed": COCO_SEED,
            "splits": {str(image_id): split for image_id, split in sorted(coco.items())},
        },
    )
    return summary


def configurations(model_root: Path) -> tuple[Configuration, ...]:
    checkpoint = model_root / "runs/detect/pigeon_detector/pigeon_baseline_v1-2/weights/best.pt"
    return (
        Configuration("pt-640x640", checkpoint, (640, 640)),
        Configuration("ncnn-640x640", model_root / "models/pigeon_baseline_v1-2", (640, 640)),
        Configuration("pt-384x640", checkpoint, (384, 640)),
    )


def prediction_fingerprint(
    configuration: Configuration, dataset: Path, split: str
) -> dict[str, JsonValue]:
    model = configuration.model
    if model.is_dir():
        model = model / "best_ncnn_model" if (model / "best_ncnn_model").is_dir() else model
        model_files = tuple(
            model / filename
            for filename in (
                "model.ncnn.param",
                "model.ncnn.bin",
                "metadata.yaml",
            )
        )
    else:
        model_files = (model,)
    return {
        "model_sha256": {path.name: file_hash(path) for path in model_files},
        "dataset_sha256": {
            name: file_hash(dataset / name)
            for name in (
                "data.yaml",
                "manifest.jsonl",
                "DATASET.json",
            )
        },
        "split": split,
        "input_hw": list(configuration.input_hw),
        "conf_floor": 0.001,
        "nms_iou": 0.7,
        "max_det": 300,
        "device": "cpu",
        "ncnn_compute": "runtime_default",
    }


def score_configuration(
    configuration: Configuration, data_root: Path, scratch: Path, output: Path, git_commit: str
) -> list[dict[str, str]]:
    predictions_directory = scratch / configuration.name
    metrics_directory = output / configuration.name
    metrics_directory.mkdir(parents=True, exist_ok=True)
    for split, dataset_name in (("val", "columba-train-v1clean"), ("test", "columba-test-v2")):
        dataset = data_root / dataset_name
        predictions = predictions_directory / f"predictions_{split}.json"
        fingerprint = prediction_fingerprint(configuration, dataset, split)
        fingerprint_path = predictions_directory / f"predictions_{split}.fingerprint.json"
        if (
            not predictions.exists()
            or not fingerprint_path.exists()
            or read_json(fingerprint_path) != fingerprint
        ):
            print(f"predicting {configuration.name} {split}", flush=True)
            predict_to_coco(
                configuration.model,
                dataset,
                split,
                predictions,
                imgsz=configuration.input_hw,
                model_id=f"pigeon_baseline_v1-2/{configuration.name}",
                host=platform.node(),
            )
            atomic_json(fingerprint_path, fingerprint)
        metrics = evaluate(
            dataset,
            split,
            predictions,
            select_threshold=split == "val",
            threshold_from=metrics_directory / "metrics_val.json" if split == "test" else None,
        )
        atomic_json(metrics_directory / f"metrics_{split}.json", metrics)
        shutil.copy2(
            sidecar_path(predictions), metrics_directory / f"predictions_{split}.meta.json"
        )
        atomic_json(metrics_directory / f"predictions_{split}.fingerprint.json", fingerprint)
        subset = output / f"unseen_{split}.txt"
        if subset.read_text().strip():
            unseen = evaluate(
                dataset,
                split,
                predictions,
                subset=subset,
                threshold_from=metrics_directory / "metrics_val.json",
            )
            atomic_json(metrics_directory / f"metrics_{split}_unseen.json", unseen)
        else:
            atomic_json(
                metrics_directory / f"metrics_{split}_unseen.json",
                {
                    "dataset": dataset_name,
                    "split": split,
                    "subset": str(subset),
                    "status": "not_evaluable",
                    "reason": "no unseen images in this split",
                    "counts": {"images": 0},
                    "pigeon": None,
                    "label": "measured",
                    "host": platform.node(),
                },
            )
    report: dict[str, JsonValue] = {
        "recipe_id": f"pigeon_baseline_v1-2/{configuration.name}",
        "seed": 0,
        "status": "incumbent-contaminated",
        "dataset": "columba-test-v2",
        "taxonomy": "single",
        "model": "pigeon_baseline_v1-2",
        "train_imgsz": INCUMBENT_TRAIN_SIZE_PX,
        "host": platform.node(),
        "gpu": "cpu",
        "epochs_done": INCUMBENT_EPOCHS_DONE,
        "git_commit": git_commit,
        "input_hw": list(configuration.input_hw),
        "label": "measured",
        "model_path": str(configuration.model),
        "predictions_directory": str(predictions_directory),
        "contamination_report": str(output / "contamination.json"),
    }
    atomic_json(metrics_directory / "run.json", report)
    rows = [leaderboard_row(metrics_directory)]
    if metrics_directory.is_relative_to(REPOSITORY):
        rows[0]["run_path"] = str(metrics_directory.relative_to(REPOSITORY))
    if (metrics_directory / "metrics_test_unseen.json").exists():
        unseen_row = dict(rows[0])
        unseen_row["recipe_id"] += "/unseen"
        unseen_row["status"] = "incumbent-unseen-subset"
        unseen_row["dataset"] = "columba-test-v2/unseen"
        unseen_metrics = mapping(
            read_json(metrics_directory / "metrics_test_unseen.json"), "metrics"
        )
        pigeon = mapping(unseen_metrics["pigeon"], "pigeon")
        for column, metric in (
            ("test_recall_at_p90", "recall_at_p90"),
            ("test_small_recall_at_p90", "small_recall_at_p90"),
            ("test_ap50", "ap50"),
            ("test_ap50_95", "ap50_95"),
        ):
            unseen_row[column] = "" if pigeon[metric] is None else str(pigeon[metric])
        for column, metric in (
            ("test_empty_far", "empty_false_alarm_rate"),
            ("test_confuser_far", "confuser_false_alarm_rate"),
        ):
            unseen_row[column] = (
                "" if unseen_metrics[metric] is None else str(unseen_metrics[metric])
            )
        rows.append(unseen_row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--scratch", type=Path, default=REPOSITORY / "outputs/ws1/incumbent")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--analyse-only", action="store_true")
    arguments = parser.parse_args()
    verify_freeze(
        arguments.data_root / "columba-test-v2", REPOSITORY / "data_sources/columba-test-v2.sha256"
    )
    summary = analyse_exposure(arguments.data_root, arguments.out)
    print(json.dumps(summary["splits"], indent=2), flush=True)
    if arguments.analyse_only:
        return
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    rows = [
        row
        for configuration in configurations(arguments.model_root)
        for row in score_configuration(
            configuration, arguments.data_root, arguments.scratch, arguments.out, commit
        )
    ]
    update_leaderboard(REPOSITORY / "experiments/leaderboard.csv", rows)


if __name__ == "__main__":
    main()
