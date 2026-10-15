#!/usr/bin/env python
"""Export, evaluate and select C8 candidates using C4 accuracy and C7 latency.

Run export on the Mac, evaluate in an allocated xcng job, then select locally.
Each evaluation directory is self-contained and can be supplied to select later.
Selection ranks three-seed arms by validation recall; test is a promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.json_data import JsonValue
from app.models import Frame
from native.verify_native import NativeModel
from tools.bench_ncnn import apply_calibration, calibration_factor, read_report, summary
from tools.compare_exports import ExportPrediction, match_predictions
from tools.evaluate_predictions import (
    Dataset,
    evaluate,
    load_dataset,
    mapping,
    metric_group,
    number,
    read_predictions,
    select_operating_threshold,
    sidecar_path,
)
from tools.predict_to_coco import model_settings
from tools.reaction_budget import ReactionSettings, processing_samples, simulate_reaction

CONFIDENCE_FLOOR = 0.001
PARITY_CONFIDENCE = 0.25
MAXIMUM_ACCURACY_DROP = 0.01
INT8_MINIMUM_IOU = 0.7
INT8_MAXIMUM_SCORE_DELTA = 0.10
EMPTY_FAR_LIMIT = 0.02
CONFUSER_FAR_LIMIT = 0.05
FRAME_P95_LIMIT_MS = 100.0
REACTION_P95_LIMIT_MS = 500.0
THREAD_COUNT = 4
TIMED_ITERATIONS = 200
WARMUP_ITERATIONS = 20
REQUIRED_SEEDS = frozenset({0, 1, 2})
RSYNC_REMOTE_SHELL = "ssh -o BatchMode=yes -o ConnectTimeout=20"
CAMERA_INPUT_HW_PX = (480, 640)
BENCHMARK_FRAME_COUNT = 20
VALIDATION_TIE_RECALL_FRACTION = 0.01
LATENCY_TIE_RATIO = 1.05
MAXIMUM_SPECIES_RECALL_DROP = 0.01
SPECIES_LATENCY_RATIO = 1.05
SQUARE_LANDSCAPE_BOUNDARY_RATIO = (4 / 3) ** 0.5
WIDE_LANDSCAPE_BOUNDARY_RATIO = ((4 / 3) * (16 / 9)) ** 0.5
ASPECT_BUCKETS = ("approx-16:9", "4:3", "square", "portrait")


def affinity_cores() -> list[int]:
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []


@contextmanager
def taxonomy_view(dataset: Path, taxonomy: str) -> Iterator[Path]:
    from cluster.staging import align_test_taxonomy

    if taxonomy == "single":
        yield dataset
        return
    job_id = os.environ.get("SLURM_JOB_ID")
    scratch = Path(f"/tmp/cz-{job_id}") if job_id else Path("outputs/ws7/tmp").resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cz-taxonomy-", dir=scratch) as temporary:
        view = Path(temporary) / dataset.name
        view.mkdir()
        for child in dataset.iterdir():
            destination = view / child.name
            if child.name == "data.yaml":
                shutil.copy2(child, destination)
            else:
                destination.symlink_to(child.resolve(), target_is_directory=child.is_dir())
        align_test_taxonomy(view, taxonomy, Path("training/classes.yaml"))
        yield view


def complete_sidecars(directory: Path) -> None:
    for path in directory.glob("predictions_*.meta.json"):
        metadata = read_report(path)
        if "created_at" in metadata:
            continue
        metadata["created_at"] = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
        metadata["created_at_source"] = "original prediction sidecar modification time"
        write_json(path, metadata)


def write_json(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temporary:
        temporary.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def command(*arguments: str) -> None:
    subprocess.run([sys.executable, "-m", *arguments], check=True)


def export_candidate(arguments: argparse.Namespace) -> None:
    for decode in arguments.decodes:
        for precision in ("fp32", "fp16"):
            destination = arguments.output / arguments.recipe / f"seed{arguments.seed}" / decode
            package = destination / f"{precision}_ncnn_model"
            if package.exists():
                settings = model_settings(package, arguments.input_hw, decode)
                exported = read_report(package / "export.json")
                source_sha256 = hashlib.sha256(arguments.checkpoint.read_bytes()).hexdigest()
                if (
                    exported.get("source_sha256") != source_sha256
                    or settings.format != f"ncnn-{precision}"
                ):
                    raise ValueError(f"existing export has different source/precision: {package}")
                continue
            command(
                "tools.export_ncnn",
                str(arguments.checkpoint.resolve()),
                "--imgsz",
                *map(str, arguments.input_hw),
                "--decode",
                decode,
                "--taxonomy",
                arguments.taxonomy,
                "--quantization",
                precision,
                "--model-id",
                f"{arguments.recipe}/seed{arguments.seed}",
                "--output",
                str(package),
            )


def native_predictions(arguments: argparse.Namespace) -> None:
    import columba_native
    import cv2

    cv2.setNumThreads(1)
    settings = model_settings(arguments.model, None, None)
    dataset = load_dataset(arguments.dataset, arguments.split)
    detector = cast(
        NativeModel, columba_native.Detector(str(arguments.model.resolve()), THREAD_COUNT)
    )
    if tuple(detector.names[index] for index in range(len(detector.names))) != dataset.names:
        raise ValueError("native model names differ from dataset")
    predictions: list[JsonValue] = []
    for image in dataset.images:
        frame = cast(Frame | None, cv2.imread(str(image.path)))
        if frame is None:
            raise ValueError(f"cannot read {image.path}")
        for row in detector.detect(frame, CONFIDENCE_FLOOR):
            left_px, top_px, right_px, bottom_px, score, category = map(float, row)
            if right_px <= left_px or bottom_px <= top_px:
                continue
            predictions.append(
                {
                    "image_id": image.image_id,
                    "category_id": int(category),
                    "bbox": [left_px, top_px, right_px - left_px, bottom_px - top_px],
                    "score": score,
                }
            )
    write_json(arguments.out, predictions)
    write_json(
        sidecar_path(arguments.out),
        {
            "model_id": arguments.model_id,
            "framework": "columba_native",
            "created_at": datetime.now(UTC).isoformat(),
            "format": settings.format,
            "input_hw": list(settings.input_hw),
            "decode": settings.decode,
            "conf_floor": CONFIDENCE_FLOOR,
            "nms_iou": 0.7 if settings.decode == "yolo11" else None,
            "agnostic_nms": False,
            "max_det": 300,
            "host": platform.node(),
            "dataset": dataset.name,
            "split": arguments.split,
            "label": "measured",
            "ncnn_compute": "runtime_default",
            "effective_compute": detector.compute,
            "compute_flags": {key: bool(value) for key, value in detector.compute_flags.items()},
            "threads": THREAD_COUNT,
            "pinned_cores": [int(core) for core in affinity_cores()],
        },
    )


def score(directory: Path, validation: Path, frozen_test: Path) -> None:
    for split, dataset in (("val", validation), ("test", frozen_test)):
        report = evaluate(
            dataset,
            split,
            directory / f"predictions_{split}.json",
            select_threshold=split == "val",
            threshold_from=directory / "metrics_val.json" if split == "test" else None,
        )
        write_json(directory / f"metrics_{split}.json", report)


def grouped_predictions(path: Path) -> dict[str, list[ExportPrediction]]:
    records = json.loads(path.read_text())
    grouped: dict[str, list[ExportPrediction]] = defaultdict(list)
    for record in records:
        if record["score"] < PARITY_CONFIDENCE:
            continue
        left_px, top_px, width_px, height_px = record["bbox"]
        grouped[record["image_id"]].append(
            {
                "label": str(record["category_id"]),
                "confidence": record["score"],
                "box": [left_px, top_px, left_px + width_px, top_px + height_px],
            }
        )
    return grouped


def parity(
    reference: Path, actual: Path, precision: str, image_ids: list[str]
) -> dict[str, JsonValue]:
    expected = grouped_predictions(reference)
    observed = grouped_predictions(actual)
    minimum_iou = INT8_MINIMUM_IOU if precision == "int8" else 0.8
    maximum_score_delta = INT8_MAXIMUM_SCORE_DELTA if precision == "int8" else 0.05
    failures = count_mismatches = 0
    maximum_box_delta_px = maximum_confidence_delta = 0.0
    for image_id in image_ids:
        left, right = expected[image_id], observed[image_id]
        matches = match_predictions(
            left,
            right,
            minimum_intersection_over_union=minimum_iou,
            maximum_confidence_delta=maximum_score_delta,
        )
        failures += len(matches) != len(left) or len(left) != len(right)
        count_mismatches += len(left) != len(right)
        maximum_box_delta_px = max(
            maximum_box_delta_px,
            max((abs(delta) for match in matches for delta in match["box_delta_px"]), default=0.0),
        )
        maximum_confidence_delta = max(
            maximum_confidence_delta,
            max((match["confidence_delta"] for match in matches), default=0.0),
        )
    return {
        "label": "measured",
        "host": platform.node(),
        "images": len(image_ids),
        "confidence": PARITY_CONFIDENCE,
        "failures": failures,
        "count_mismatches": count_mismatches,
        "passed": failures == 0,
        "minimum_iou": minimum_iou,
        "maximum_confidence_delta": maximum_score_delta,
        "maximum_matched_box_delta_px": maximum_box_delta_px,
        "maximum_matched_confidence_delta": maximum_confidence_delta,
        "comparator": "tools.compare_exports.match_predictions",
        "reference": str(reference),
        "actual": str(actual),
        "zero_area_boxes": "excluded from both C3 files, matching WS1 positive-area contract",
    }


def accuracy_gate(reference: Path, actual: Path) -> dict[str, JsonValue]:
    deltas: dict[str, JsonValue] = {}
    for split in ("val", "test"):
        expected = mapping(read_report(reference / f"metrics_{split}.json")["pigeon"], "pigeon")
        observed = mapping(read_report(actual / f"metrics_{split}.json")["pigeon"], "pigeon")
        for metric in ("recall_at_p90", "ap50", "recall_at_threshold"):
            deltas[f"{split}_{metric}"] = number(observed[metric], metric) - number(
                expected[metric], metric
            )
    return {
        "passed": all(
            number(value, key) >= -MAXIMUM_ACCURACY_DROP for key, value in deltas.items()
        ),
        "maximum_drop": MAXIMUM_ACCURACY_DROP,
        "deltas": deltas,
        "rule": "each val/test R@P90, AP50 and val-threshold recall drops by at most 0.01",
    }


def benchmark_native(arguments: argparse.Namespace) -> None:
    if platform.machine() != "aarch64" or platform.node().split(".")[0] not in ("xcng0", "xcng1"):
        raise ValueError("native C7 gate requires an allocated xcng host")
    import columba_native
    import cv2

    from tools.export_ncnn import fingerprint

    cv2.setNumThreads(1)
    detector = cast(
        NativeModel, columba_native.Detector(str(arguments.model.resolve()), THREAD_COUNT)
    )
    confidence = min(arguments.confidence, 1.0)
    dataset = load_dataset(arguments.dataset, "val")
    decoded_frames = [
        cast(Frame | None, cv2.imread(str(image.path)))
        for image in dataset.images[:BENCHMARK_FRAME_COUNT]
    ]
    if any(frame is None for frame in decoded_frames):
        raise ValueError("unreadable benchmark frame")
    frames = [
        cast(Frame, cv2.resize(frame, (CAMERA_INPUT_HW_PX[1], CAMERA_INPUT_HW_PX[0])))
        for frame in decoded_frames
        if frame is not None
    ]
    stages: dict[str, list[float]] = {
        stage: [] for stage in ("preprocess", "extract", "postprocess", "total")
    }
    wall_ms: list[float] = []
    for index in range(WARMUP_ITERATIONS + TIMED_ITERATIONS):
        started_s = time.perf_counter()
        rows, timings = detector.detect_with_timings(frames[index % len(frames)], confidence)
        bool(len(rows))
        elapsed_ms = (time.perf_counter() - started_s) * 1000
        if index < WARMUP_ITERATIONS:
            continue
        wall_ms.append(elapsed_ms)
        for stage in stages:
            stages[stage].append(float(timings[stage]))
    settings = model_settings(arguments.model, None, None)
    report: dict[str, JsonValue] = {
        "model_id": arguments.model_id,
        "format": settings.format,
        "input_hw": list(settings.input_hw),
        "host": platform.node(),
        "host_kind": "xcng-n1",
        "threads": THREAD_COUNT,
        "pinned_cores": ",".join(map(str, affinity_cores())),
        "ncnn_version": "1.0.20260526",
        "ncnn_build_flags": (arguments.prefix / "build-evidence.txt").read_text(),
        "warmup": WARMUP_ITERATIONS,
        "iterations": TIMED_ITERATIONS,
        "inference_ms": summary(stages["extract"]),
        "preprocess_ms": summary(stages["preprocess"]),
        "postprocess_ms": summary(stages["postprocess"]),
        "end_to_end_ms": summary(wall_ms),
        "samples_ms": [float(value) for value in wall_ms],
        "label": "measured",
        "backend": "columba_native",
        "compute": detector.compute,
        "compute_flags": {key: bool(value) for key, value in detector.compute_flags.items()},
        "camera_hw": list(CAMERA_INPUT_HW_PX),
        "confidence": confidence,
        "requested_operating_threshold": arguments.confidence,
        "weights_sha256": fingerprint(arguments.model / "model.ncnn.bin"),
        "parameters_sha256": fingerprint(arguments.model / "model.ncnn.param"),
        "int8_calibration": read_report(arguments.model / "calibration.json")
        if (arguments.model / "calibration.json").exists()
        else None,
        "end_to_end_boundary": (
            "native letterbox, inference, decode/NMS, Python return and bool decision"
        ),
        "frame_source": f"first {BENCHMARK_FRAME_COUNT} val photographs at camera shape, cycling",
    }
    wait_policy = os.environ.get("OMP_WAIT_POLICY", "").upper()
    if wait_policy in ("ACTIVE", "PASSIVE"):
        report["omp_wait_policy"] = wait_policy
    write_json(arguments.out, apply_calibration(report, read_report(arguments.calibration)))


def reaction_for_recall(samples_ms: list[float], recall: float) -> dict[str, JsonValue]:
    if recall <= 0:
        return {
            "label": "simulated",
            "reaction_ms": {"p50": None, "p95": None},
            "status": "zero observed recall cannot confirm",
            "recall": recall,
        }
    try:
        return simulate_reaction(samples_ms, ReactionSettings(recall=recall))
    except ValueError as error:
        if "confirmation did not finish" not in str(error):
            raise
        return {
            "label": "simulated",
            "reaction_ms": {"p50": None, "p95": None},
            "status": str(error),
            "recall": recall,
        }


def evaluate_candidate(arguments: argparse.Namespace) -> None:
    with taxonomy_view(arguments.test_dataset, arguments.taxonomy) as aligned:
        evaluate_aligned_candidate(
            argparse.Namespace(**{**vars(arguments), "test_dataset": aligned})
        )


def evaluate_aligned_candidate(arguments: argparse.Namespace) -> None:
    if platform.machine() != "aarch64" or platform.node().split(".")[0] not in ("xcng0", "xcng1"):
        raise ValueError(
            "evaluate must run on allocated xcng with the A76-compatible native extension"
        )
    affinity = affinity_cores()
    if len(affinity) != THREAD_COUNT:
        raise ValueError("pin evaluate to exactly four allocated cores with taskset")
    source = arguments.exports / arguments.recipe / f"seed{arguments.seed}"
    destination = arguments.output / arguments.recipe / f"seed{arguments.seed}"
    for decode in arguments.decodes:
        reference = destination / decode / "pt"
        reference.mkdir(parents=True, exist_ok=True)
        if arguments.references:
            cached = (
                arguments.references / arguments.recipe / f"seed{arguments.seed}" / decode / "pt"
            )
            for name in ("predictions_val.json", "predictions_test.json"):
                shutil.copy2(cached / name, reference / name)
                shutil.copy2(sidecar_path(cached / name), sidecar_path(reference / name))
            for name in (
                "predictions_styles.json",
                "predictions_styles.meta.json",
                "metrics_styles.json",
            ):
                if (cached / name).is_file():
                    shutil.copy2(cached / name, reference / name)
        for split, dataset in (("val", arguments.dataset), ("test", arguments.test_dataset)):
            output = reference / f"predictions_{split}.json"
            if output.exists():
                continue
            command(
                "tools.predict_to_coco",
                "--model",
                str(arguments.checkpoint),
                "--dataset",
                str(dataset),
                "--split",
                split,
                "--out",
                str(output),
                "--imgsz",
                *map(str, arguments.input_hw),
                "--decode",
                decode,
            )
        score(reference, arguments.dataset, arguments.test_dataset)
        if (reference / "metrics_styles.json").is_file():
            styles_threshold = mapping(
                read_report(reference / "metrics_styles.json")["operating_threshold"], "threshold"
            )
            reference_threshold = mapping(
                read_report(reference / "metrics_val.json")["operating_threshold"], "threshold"
            )
            if styles_threshold["value"] != reference_threshold["value"]:
                raise ValueError("cached PyTorch styles threshold differs from its val threshold")
        for precision in ("fp32", "fp16", "int8"):
            package = source / decode / f"{precision}_ncnn_model"
            if precision == "int8" and not package.exists():
                command(
                    "tools.quantize_ncnn",
                    str(source / decode / "fp32_ncnn_model"),
                    "--output",
                    str(package),
                    "--dataset",
                    str(arguments.dataset),
                    "--test-dataset",
                    str(arguments.test_dataset),
                    "--prefix",
                    str(arguments.prefix),
                )
            directory = destination / decode / precision
            if (directory / "candidate.json").exists():
                continue
            directory.mkdir(parents=True, exist_ok=True)
            model_id = f"{arguments.recipe}/seed{arguments.seed}/{decode}/{precision}"
            for split, dataset in (("val", arguments.dataset), ("test", arguments.test_dataset)):
                command(
                    "tools.select_model",
                    "predict-native",
                    "--model",
                    str(package),
                    "--dataset",
                    str(dataset),
                    "--split",
                    split,
                    "--model-id",
                    model_id,
                    "--out",
                    str(directory / f"predictions_{split}.json"),
                )
            score(directory, arguments.dataset, arguments.test_dataset)
            comparisons = parity(
                reference / "predictions_test.json",
                directory / "predictions_test.json",
                precision,
                [image.image_id for image in load_dataset(arguments.test_dataset, "test").images],
            )
            write_json(directory / "parity.json", comparisons)
            accuracy = accuracy_gate(reference, directory)
            write_json(directory / "accuracy_gate.json", accuracy)
            threshold = mapping(
                read_report(directory / "metrics_val.json")["operating_threshold"], "threshold"
            )
            command(
                "tools.select_model",
                "benchmark-native",
                "--model",
                str(package),
                "--dataset",
                str(arguments.dataset),
                "--prefix",
                str(arguments.prefix),
                "--model-id",
                model_id,
                "--confidence",
                str(threshold["value"]),
                "--calibration",
                str(arguments.calibration),
                "--out",
                str(directory / "latency.json"),
            )
            latency = read_report(directory / "latency.json")
            test = mapping(read_report(directory / "metrics_test.json")["pigeon"], "pigeon")
            samples, distribution = processing_samples(
                latency, calibration_factor(read_report(arguments.calibration), "xcng-n1")
            )
            reaction = reaction_for_recall(samples, number(test["recall_at_threshold"], "recall"))
            reaction.update(
                {
                    "processing_host": "estimated Pi5",
                    "distribution": distribution,
                    "recall_source": (
                        "test recall at independently val-selected threshold; per-frame proxy"
                    ),
                }
            )
            write_json(directory / "reaction.json", reaction)
            write_json(
                directory / "candidate.json",
                {
                    "recipe": arguments.recipe,
                    "seed": arguments.seed,
                    "decode": decode,
                    "precision": precision,
                    "input_hw": list(arguments.input_hw),
                    "model_id": model_id,
                    "package": str(package),
                    "reference": str(reference),
                    "taxonomy": arguments.taxonomy,
                    "single_recipe": arguments.single_recipe,
                    "provisional": True,
                    "accuracy_gate_passed": accuracy["passed"],
                    "parity_passed": comparisons["passed"],
                },
            )
            print(f"completed {model_id}", flush=True)


def candidate_record(directory: Path) -> dict[str, JsonValue]:
    descriptor = read_report(directory / "candidate.json")
    validation = read_report(directory / "metrics_val.json")
    test = read_report(directory / "metrics_test.json")
    latency = read_report(directory / "latency.json")
    reaction = read_report(directory / "reaction.json")
    threshold = mapping(validation["operating_threshold"], "threshold")
    test_threshold = mapping(test["operating_threshold"], "threshold")
    if validation.get("split") != "val" or threshold.get("source") != "selected":
        raise ValueError(f"{directory}: operating threshold was not selected on validation")
    if threshold.get("value") != test_threshold.get("value"):
        raise ValueError(f"{directory}: test threshold differs from validation")
    native = read_report(sidecar_path(directory / "predictions_test.json"))
    reference_meta = read_report(directory.parent / "pt/predictions_test.meta.json")
    if native.get("framework") != "columba_native" or not str(native.get("host", "")).startswith(
        "xcng"
    ):
        raise ValueError(f"{directory}: deployed native xcng accuracy is required")
    if latency.get("backend") != "columba_native":
        raise ValueError(f"{directory}: native end-to-end C7 is required")
    if test.get("split") != "test":
        raise ValueError(f"{directory}: requires test C4 metrics")
    for report in (validation, test, latency, native):
        if report.get("input_hw") != descriptor.get("input_hw"):
            raise ValueError(f"{directory}: mismatched candidate input dimensions")
    for report in (latency, native):
        if report.get("model_id") != descriptor.get("model_id"):
            raise ValueError(f"{directory}: mismatched candidate model identity")
        if report.get("format") != f"ncnn-{descriptor['precision']}":
            raise ValueError(f"{directory}: mismatched candidate precision")
    if native.get("compute_flags") != latency.get("compute_flags"):
        raise ValueError(f"{directory}: accuracy/latency compute flags differ")
    if native.get("ncnn_compute") != "runtime_default":
        raise ValueError(f"{directory}: accuracy must use deployed package-default compute")
    flags = mapping(native["compute_flags"], "compute flags")
    expected_fp16 = descriptor["precision"] in ("fp16", "int8")
    for flag in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic"):
        if flags.get(flag) is not expected_fp16:
            raise ValueError(f"{directory}: compute mode does not match deployed package")
    if latency.get("omp_wait_policy") != "ACTIVE":
        raise ValueError(f"{directory}: remeasure latency with OMP_WAIT_POLICY=ACTIVE")
    if test.get("dataset") != "columba-test-v2":
        raise ValueError(f"{directory}: primary gates require frozen columba-test-v2")
    fp32_parity_passed = False
    if descriptor["precision"] == "fp16":
        fp32_directory = directory.parent / "fp32"
        if (fp32_directory / "candidate.json").exists():
            fp32 = read_report(fp32_directory / "candidate.json")
            graph_fields = ("recipe", "seed", "decode", "input_hw", "taxonomy")
            if any(fp32.get(field) != descriptor.get(field) for field in graph_fields):
                raise ValueError(f"{directory}: FP32 parity reference is a different graph")
            fp32_parity_passed = read_report(fp32_directory / "parity.json")["passed"] is True
    styles = None
    if (directory / "metrics_styles.json").exists():
        styles = read_report(directory / "metrics_styles.json")
        styles_meta = read_report(directory / "predictions_styles.meta.json")
        if mapping(styles["operating_threshold"], "threshold")["value"] != threshold["value"]:
            raise ValueError(f"{directory}: styles threshold differs from validation")
        if styles.get("dataset") != "columba-test-v2-styles" or styles.get("split") != "test":
            raise ValueError(f"{directory}: secondary metrics require columba-test-v2-styles")
        for report in (styles, styles_meta):
            if report.get("input_hw") != descriptor["input_hw"]:
                raise ValueError(f"{directory}: styles input dimensions differ from candidate")
        for field in ("framework", "format", "model_id", "compute_flags", "ncnn_compute"):
            if styles_meta.get(field) != native.get(field):
                raise ValueError(f"{directory}: styles {field} differs from native candidate")
        if not str(styles_meta.get("host", "")).startswith("xcng"):
            raise ValueError(f"{directory}: styles accuracy requires native xcng inference")
    return {
        **descriptor,
        "directory": os.path.relpath(directory, Path.cwd()),
        "val": validation["pigeon"],
        "test": test["pigeon"],
        "empty_far": test["empty_false_alarm_rate"],
        "confuser_far": test["confuser_false_alarm_rate"],
        "val_empty_far": validation["empty_false_alarm_rate"],
        "val_confuser_far": validation["confuser_false_alarm_rate"],
        "pi5_p95_ms": mapping(latency["estimated_pi5"], "estimated Pi")["end_to_end_p95_ms"],
        "reaction_p95_ms": mapping(reaction["reaction_ms"], "reaction")["p95"],
        "threshold": threshold["value"],
        "host": native["host"],
        "reference_host": reference_meta["host"],
        "reference_device": reference_meta["device"],
        "reference_styles": read_report(directory.parent / "pt/metrics_styles.json")["pigeon"]
        if (directory.parent / "pt/metrics_styles.json").exists()
        else None,
        "parity_passed": read_report(directory / "parity.json")["passed"],
        "fp32_parity_passed": fp32_parity_passed,
        "accuracy_gate_passed": read_report(directory / "accuracy_gate.json")["passed"],
        "accuracy_deltas": read_report(directory / "accuracy_gate.json")["deltas"],
        "parity_failures": read_report(directory / "parity.json")["failures"],
        "styles": styles["pigeon"] if styles is not None else None,
    }


def baseline_scores(directory: Path) -> tuple[float, ...]:
    scores: list[float] = []
    for seed in sorted(REQUIRED_SEEDS):
        report = read_report(directory / f"seed{seed}/metrics_test.json")
        if report.get("dataset") != "columba-test-v2" or report.get("input_hw") != [384, 640]:
            raise ValueError("baseline requires frozen test C4 at rectangular 384x640")
        scores.append(number(mapping(report["pigeon"], "pigeon")["recall_at_p90"], "recall"))
    return tuple(scores)


def passes_validation_false_alarms(seed: dict[str, JsonValue]) -> bool:
    return all(
        seed[field] is not None and number(seed[field], field) <= limit
        for field, limit in (
            ("val_empty_far", EMPTY_FAR_LIMIT),
            ("val_confuser_far", CONFUSER_FAR_LIMIT),
        )
    )


def deployment_parity(seed: dict[str, JsonValue]) -> bool:
    if seed["precision"] == "fp16":
        return seed.get("fp32_parity_passed") is True and seed["accuracy_gate_passed"] is True
    return seed["parity_passed"] is True


def metric(record: dict[str, JsonValue], split: str, name: str) -> float:
    return number(mapping(record[split], split)[name], name)


def enforce_species_gate(arms: list[dict[str, JsonValue]]) -> None:
    for arm in arms:
        seeds = [mapping(seed, "seed") for seed in cast(list[JsonValue], arm["seeds"])]
        if seeds[0].get("taxonomy", "single") != "species":
            continue
        matched_recipe = seeds[0].get("single_recipe")
        reference = next(
            (
                candidate
                for candidate in arms
                if candidate["recipe"] == matched_recipe
                and candidate["taxonomy"] == "single"
                and candidate["decode"] == arm["decode"]
                and candidate["precision"] == arm["precision"]
                and candidate["input_hw"] == arm["input_hw"]
            ),
            None,
        )
        reasons = cast(list[JsonValue], arm["failures"])
        if reference is None:
            reasons.append("species gate needs --single-recipe and its evaluated three-seed arm")
        elif (
            number(arm["test_recall_mean"], "recall")
            < number(reference["test_recall_mean"], "recall") - MAXIMUM_SPECIES_RECALL_DROP
        ):
            reasons.append("species pigeon recall is more than 0.01 below matched single")
        if reference is not None:
            reference_seed = next(
                (
                    mapping(seed, "seed")
                    for seed in cast(list[JsonValue], reference["seeds"])
                    if mapping(seed, "seed")["seed"] == reference["shipped_seed"]
                ),
                None,
            )
            shipped_seed = next(
                (seed for seed in seeds if seed["seed"] == arm["shipped_seed"]), None
            )
            if reference_seed is None or shipped_seed is None:
                reasons.append("species comparison requires val-eligible shipped seeds")
            else:
                for field in ("empty_far", "confuser_far"):
                    if number(shipped_seed[field], field) > number(reference_seed[field], field):
                        reasons.append(f"species shipped {field} is worse than matched single")
            if (
                number(arm["pi5_p95_ms"], "latency")
                > number(reference["pi5_p95_ms"], "latency") * SPECIES_LATENCY_RATIO
            ):
                reasons.append("species latency is over 5% slower than matched single")
        arm["failures"] = reasons
        arm["passes"] = not reasons


def styles_tie_break(eligible: list[dict[str, JsonValue]]) -> dict[str, JsonValue] | None:
    selected = max(
        eligible,
        key=lambda arm: (
            number(arm["val_recall_mean"], "recall"),
            -number(arm["pi5_p95_ms"], "latency"),
        ),
        default=None,
    )
    if selected is None:
        return None
    recipe = str(selected["recipe"]).replace("-v2s-", "-v2-")
    peers = [
        arm
        for arm in eligible
        if str(arm["recipe"]).replace("-v2s-", "-v2-") == recipe
        and arm["decode"] == selected["decode"]
        and arm["precision"] == selected["precision"]
        and arm["input_hw"] == selected["input_hw"]
        and abs(
            number(arm["val_recall_mean"], "recall") - number(selected["val_recall_mean"], "recall")
        )
        <= VALIDATION_TIE_RECALL_FRACTION
        and max(number(arm["pi5_p95_ms"], "latency"), number(selected["pi5_p95_ms"], "latency"))
        <= min(number(arm["pi5_p95_ms"], "latency"), number(selected["pi5_p95_ms"], "latency"))
        * LATENCY_TIE_RATIO
    ]
    if len(peers) < 2:
        return selected
    if any(arm["styles_recall_mean"] is None for arm in peers):
        return None
    return max(
        peers,
        key=lambda arm: (
            number(arm["styles_recall_mean"], "styles recall"),
            number(arm["val_recall_mean"], "recall"),
            -number(arm["pi5_p95_ms"], "latency"),
        ),
    )


def pareto_front(arms: list[dict[str, JsonValue]]) -> list[JsonValue]:
    return [
        arm["id"]
        for arm in arms
        if not any(
            number(other["val_recall_mean"], "recall") >= number(arm["val_recall_mean"], "recall")
            and number(other["pi5_p95_ms"], "latency") <= number(arm["pi5_p95_ms"], "latency")
            and (
                other["val_recall_mean"] != arm["val_recall_mean"]
                or other["pi5_p95_ms"] != arm["pi5_p95_ms"]
            )
            for other in arms
        )
    ]


def arm_gates(arm: dict[str, JsonValue], baseline: tuple[float, ...]) -> list[JsonValue]:
    improvement = number(arm["test_recall_mean"], "recall") - statistics.mean(baseline)
    uncertainty = max(statistics.stdev(baseline), number(arm["test_recall_std"], "deviation"))
    gates: list[JsonValue] = [
        {
            "gate": "Three-seed test improvement",
            "value": improvement,
            "requirement": f"> {uncertainty:.6f}, larger seed SD",
            "passed": improvement > uncertainty,
        }
    ]
    seed = next(
        (
            mapping(value, "seed")
            for value in cast(list[JsonValue], arm["seeds"])
            if mapping(value, "seed")["seed"] == arm["shipped_seed"]
        ),
        None,
    )
    if seed is None:
        return gates + [
            {
                "gate": "Val-eligible shipped seed",
                "value": None,
                "requirement": "one seed passes val FAR",
                "passed": False,
            }
        ]
    for field, label, limit in (
        ("val_empty_far", "Val empty FAR", EMPTY_FAR_LIMIT),
        ("val_confuser_far", "Val confuser FAR", CONFUSER_FAR_LIMIT),
        ("empty_far", "Test empty FAR", EMPTY_FAR_LIMIT),
        ("confuser_far", "Test confuser FAR", CONFUSER_FAR_LIMIT),
        ("pi5_p95_ms", "Estimated Pi frame p95 ms", FRAME_P95_LIMIT_MS),
        ("reaction_p95_ms", "Simulated reaction p95 ms", REACTION_P95_LIMIT_MS),
    ):
        value = seed[field]
        gates.append(
            {
                "gate": label,
                "value": value,
                "requirement": f"<= {limit}",
                "passed": value is not None and number(value, field) <= limit,
            }
        )
    deltas = mapping(seed.get("accuracy_deltas", {}), "accuracy deltas")
    maximum_drop = max(
        0.0, -min((number(value, "delta") for value in deltas.values()), default=0.0)
    )
    gates.extend(
        [
            {
                "gate": "Precision accuracy maximum drop",
                "value": maximum_drop,
                "requirement": "<= 0.01 in each val/test R@P90, AP50, operating recall",
                "passed": seed["accuracy_gate_passed"],
            },
            {
                "gate": "Export parity",
                "value": seed["deployment_parity_passed"],
                "requirement": "strict same-graph fp32 + accuracy for fp16",
                "passed": seed["deployment_parity_passed"],
            },
            {
                "gate": "Deployment policy",
                "value": f"{arm['decode']} / {arm['precision']}",
                "requirement": "standard head; int8 excluded",
                "passed": arm["decode"] == "yolo11" and arm["precision"] != "int8",
            },
        ]
    )
    return gates


def select_candidates(directories: list[Path], baseline: tuple[float, ...]) -> dict[str, JsonValue]:
    candidates = [candidate_record(directory) for directory in directories]
    for candidate in candidates:
        candidate["deployment_parity_passed"] = deployment_parity(candidate)
    grouped: dict[tuple[str, str, str, str], list[dict[str, JsonValue]]] = defaultdict(list)
    for candidate in candidates:
        key = tuple(
            str(candidate[field]) for field in ("recipe", "decode", "precision", "input_hw")
        )
        grouped[cast(tuple[str, str, str, str], key)].append(candidate)
    arms: list[dict[str, JsonValue]] = []
    baseline_mean, baseline_std = statistics.mean(baseline), statistics.stdev(baseline)
    for key, seeds in sorted(grouped.items()):
        recalls = [metric(seed, "test", "recall_at_p90") for seed in seeds]
        mean_recall = statistics.mean(recalls)
        standard_deviation = statistics.stdev(recalls) if len(recalls) > 1 else 0.0
        reasons: list[str] = []
        if {seed["seed"] for seed in seeds} != REQUIRED_SEEDS or len(seeds) != len(REQUIRED_SEEDS):
            reasons.append("requires exactly seeds 0, 1, 2")
        if mean_recall - baseline_mean <= max(baseline_std, standard_deviation):
            reasons.append("three-seed test accuracy does not beat baseline by seed uncertainty")
        representative = max(
            (seed for seed in seeds if passes_validation_false_alarms(seed)),
            key=lambda seed: (metric(seed, "val", "recall_at_p90"), -cast(int, seed["seed"])),
            default=None,
        )
        if representative is None:
            reasons.append("no seed passes validation false-alarm gates")
        if key[2] == "int8":
            reasons.append("int8 excluded by coordinator decision")
        if key[1] == "yolo26":
            reasons.append("one-to-one head excluded by coordinator decision")
        if representative is not None:
            for field, limit in (
                ("empty_far", EMPTY_FAR_LIMIT),
                ("confuser_far", CONFUSER_FAR_LIMIT),
                ("pi5_p95_ms", FRAME_P95_LIMIT_MS),
                ("reaction_p95_ms", REACTION_P95_LIMIT_MS),
            ):
                if representative[field] is None or number(representative[field], field) > limit:
                    reasons.append(f"shipped seed{representative['seed']} fails {field}")
            if representative["deployment_parity_passed"] is not True:
                reasons.append(f"shipped seed{representative['seed']} fails export parity")
            if representative["accuracy_gate_passed"] is not True:
                reasons.append(
                    f"shipped seed{representative['seed']} fails deployed precision accuracy"
                )
        timing_seeds = [representative] if representative is not None else seeds
        arms.append(
            {
                "id": "/".join(key[:3]) + "/" + key[3],
                "recipe": key[0],
                "input_hw": seeds[0]["input_hw"],
                "taxonomy": seeds[0].get("taxonomy", "single"),
                "decode": key[1],
                "precision": key[2],
                "val_recall_mean": statistics.mean(
                    metric(seed, "val", "recall_at_p90") for seed in seeds
                ),
                "test_recall_mean": mean_recall,
                "test_recall_std": standard_deviation,
                "test_ap50_mean": statistics.mean(metric(seed, "test", "ap50") for seed in seeds),
                "reference_styles_recall_mean": statistics.mean(
                    metric(seed, "reference_styles", "recall_at_p90") for seed in seeds
                )
                if all(seed.get("reference_styles") is not None for seed in seeds)
                else None,
                "styles_recall_mean": statistics.mean(
                    metric(seed, "styles", "recall_at_p90") for seed in seeds
                )
                if all(seed.get("styles") is not None for seed in seeds)
                else None,
                "test_operating_recall_mean": statistics.mean(
                    metric(seed, "test", "recall_at_threshold") for seed in seeds
                ),
                "pi5_p95_ms": max(number(seed["pi5_p95_ms"], "latency") for seed in timing_seeds),
                "reaction_p95_ms": max(
                    number(seed["reaction_p95_ms"], "reaction") for seed in timing_seeds
                )
                if all(seed["reaction_p95_ms"] is not None for seed in timing_seeds)
                else None,
                "passes": not reasons,
                "failures": [str(reason) for reason in reasons],
                "representative": representative["directory"] if representative else None,
                "shipped_seed": representative["seed"] if representative else None,
                "seeds": [dict(seed) for seed in seeds],
            }
        )
    enforce_species_gate(arms)
    for arm in arms:
        arm["gates"] = arm_gates(arm, baseline)
    eligible = [arm for arm in arms if arm["passes"]]
    front = pareto_front(eligible)
    selected = styles_tie_break(eligible)
    comparable = [
        arm
        for arm in arms
        if len(cast(list[JsonValue], arm["seeds"])) == len(REQUIRED_SEEDS)
        and arm["precision"] != "int8"
        and arm["decode"] == "yolo11"
        and any(
            mapping(seed, "seed")["seed"] == arm["shipped_seed"]
            and mapping(seed, "seed")["deployment_parity_passed"] is True
            and mapping(seed, "seed")["accuracy_gate_passed"] is True
            for seed in cast(list[JsonValue], arm["seeds"])
        )
    ]
    diagnostic = styles_tie_break(comparable)
    return {
        "provisional": True,
        "baseline": {
            "recipe": "baseline-repro-p50",
            "test_mean": baseline_mean,
            "test_std": baseline_std,
            "input_hw": [384, 640],
            "format": "ncnn-fp16",
            "source": "rectangular native baseline C4, three seeds",
        },
        "ranking": "three-seed validation R@P90, latency tie-break; test used for promotion only",
        "false_alarm_policy": "choose seed on val FAR and recall; gate shipped export alone",
        "seed_selection": "highest val R@P90 among val FAR-passing seeds; no test-driven fallback",
        "fp16_parity_policy": (
            "same-graph strict fp32 parity and precision accuracy; strict fp16 diagnostic"
        ),
        "excluded": ["int8", "decode:yolo26"],
        "precision_accuracy_gate": (
            "each split R@P90/AP50/operating recall drop <=0.01 vs same-head rectangular PyTorch"
        ),
        "arms": [dict(arm) for arm in arms],
        "pareto_front": pareto_front(arms),
        "admissible_front": front,
        "styles_tie_break": (
            "paired v2/v2s, val recall within 0.01, same decode/precision, latency within 5%"
        ),
        "selected": selected["id"] if selected else None,
        "package_candidate": selected["representative"] if selected else None,
        "diagnostic_candidate": diagnostic["representative"] if diagnostic else None,
        "diagnostic_failures": diagnostic["failures"] if diagnostic else [],
        "awaiting": ["round2-distill", "round3-species"],
    }


def remeasure_latency(arguments: argparse.Namespace) -> None:
    if os.environ.get("OMP_WAIT_POLICY", "").upper() != "ACTIVE":
        raise ValueError("WS7 latency comparison requires the OMP_WAIT_POLICY=ACTIVE profile")
    for descriptor_path in sorted(arguments.candidates.rglob("candidate.json")):
        directory = descriptor_path.parent
        descriptor = read_report(descriptor_path)
        package = (
            arguments.exports
            / str(descriptor["recipe"])
            / f"seed{descriptor['seed']}"
            / str(descriptor["decode"])
            / f"{descriptor['precision']}_ncnn_model"
        )
        if not (package / "model.ncnn.bin").is_file():
            continue
        complete_sidecars(directory)
        current = read_report(directory / "latency.json")
        if current.get("omp_wait_policy") == "ACTIVE":
            continue
        shutil.copy2(directory / "latency.json", directory / "latency-passive.json")
        threshold = mapping(
            read_report(directory / "metrics_val.json")["operating_threshold"], "threshold"
        )
        command(
            "tools.select_model",
            "benchmark-native",
            "--model",
            str(package),
            "--dataset",
            str(arguments.dataset),
            "--prefix",
            str(arguments.prefix),
            "--model-id",
            str(descriptor["model_id"]),
            "--confidence",
            str(threshold["value"]),
            "--calibration",
            str(arguments.calibration),
            "--out",
            str(directory / "latency.json"),
        )
        latency = read_report(directory / "latency.json")
        test = mapping(read_report(directory / "metrics_test.json")["pigeon"], "pigeon")
        samples, distribution = processing_samples(
            latency, calibration_factor(read_report(arguments.calibration), "xcng-n1")
        )
        reaction = reaction_for_recall(samples, number(test["recall_at_threshold"], "recall"))
        reaction.update(
            {
                "processing_host": "estimated Pi5",
                "distribution": distribution,
                "recall_source": "test operating recall as per-frame proxy",
                "omp_wait_policy": "ACTIVE",
            }
        )
        write_json(directory / "reaction.json", reaction)
        if descriptor["seed"] == 0:
            comparison = [
                "tools.bench_ncnn",
                "run",
                "--model",
                str(package),
                "--input-hw",
                *map(str, cast(list[int], descriptor["input_hw"])),
                "--model-id",
                str(descriptor["model_id"]).replace("/", "-"),
                "--host-profile",
                "xcng",
                "--prefix",
                str(arguments.prefix),
                "--format",
                str(descriptor["precision"]),
                "--confidence",
                str(min(number(threshold["value"], "threshold"), 1.0)),
                "--output",
                str(directory / "ws4-bridge"),
            ]
            if descriptor["decode"] == "yolo26":
                comparison.append("--nms-free")
            command(*comparison)
        print(f"ACTIVE latency completed {descriptor['model_id']}", flush=True)


def score_styles(arguments: argparse.Namespace) -> None:
    for descriptor_path in sorted(arguments.candidates.rglob("candidate.json")):
        descriptor = read_report(descriptor_path)
        if descriptor["precision"] not in arguments.precisions:
            continue
        if (descriptor_path.parent / "metrics_styles.json").exists():
            complete_sidecars(descriptor_path.parent)
            continue
        package = (
            arguments.exports
            / str(descriptor["recipe"])
            / f"seed{descriptor['seed']}"
            / str(descriptor["decode"])
            / f"{descriptor['precision']}_ncnn_model"
        )
        if not (package / "model.ncnn.bin").is_file():
            continue
        with taxonomy_view(arguments.dataset, str(descriptor["taxonomy"])) as aligned:
            score_export_styles(descriptor_path.parent, descriptor, aligned, arguments.exports)


def score_export_styles(
    directory: Path,
    descriptor: dict[str, JsonValue],
    dataset: Path,
    exports: Path,
) -> None:
    package = (
        exports
        / str(descriptor["recipe"])
        / f"seed{descriptor['seed']}"
        / str(descriptor["decode"])
        / f"{descriptor['precision']}_ncnn_model"
    )
    output = directory / "predictions_styles.json"
    if not output.exists():
        command(
            "tools.select_model",
            "predict-native",
            "--model",
            str(package),
            "--dataset",
            str(dataset),
            "--split",
            "test",
            "--model-id",
            str(descriptor["model_id"]),
            "--out",
            str(output),
        )
    write_json(
        directory / "metrics_styles.json",
        evaluate(
            dataset,
            "test",
            output,
            threshold_from=directory / "metrics_val.json",
        ),
    )
    print(f"styles completed {descriptor['model_id']}", flush=True)


def predict_references(arguments: argparse.Namespace) -> None:
    with taxonomy_view(arguments.test_dataset, arguments.taxonomy) as aligned:
        if arguments.styles_dataset:
            with taxonomy_view(arguments.styles_dataset, arguments.taxonomy) as styles:
                predict_aligned_references(
                    argparse.Namespace(
                        **{
                            **vars(arguments),
                            "test_dataset": aligned,
                            "styles_dataset": styles,
                        }
                    )
                )
        else:
            predict_aligned_references(
                argparse.Namespace(**{**vars(arguments), "test_dataset": aligned})
            )


def predict_aligned_references(arguments: argparse.Namespace) -> None:
    directory = arguments.output / arguments.recipe / f"seed{arguments.seed}"
    for decode in arguments.decodes:
        reference = directory / decode / "pt"
        reference.mkdir(parents=True, exist_ok=True)
        splits = [("val", "val", arguments.dataset), ("test", "test", arguments.test_dataset)]
        if arguments.styles_dataset:
            splits.append(("styles", "test", arguments.styles_dataset))
        for name, split, dataset in splits:
            output = reference / f"predictions_{name}.json"
            if output.exists():
                continue
            command(
                "tools.predict_to_coco",
                "--model",
                str(arguments.checkpoint),
                "--dataset",
                str(dataset),
                "--split",
                split,
                "--out",
                str(output),
                "--imgsz",
                *map(str, arguments.input_hw),
                "--decode",
                decode,
                "--device",
                arguments.device,
                "--batch",
                str(arguments.batch),
            )
        score(reference, arguments.dataset, arguments.test_dataset)
        if arguments.styles_dataset:
            write_json(
                reference / "metrics_styles.json",
                evaluate(
                    arguments.styles_dataset,
                    "test",
                    reference / "predictions_styles.json",
                    threshold_from=reference / "metrics_val.json",
                ),
            )


def fetch_candidate(arguments: argparse.Namespace) -> None:
    from cluster.sweep import remote

    receipt = read_report(arguments.output / "submissions" / f"{arguments.recipe}.json")
    workspace = str(receipt["workspace"])
    remote(["test", "-f", workspace + "/DONE"])
    (arguments.output / "results").mkdir(parents=True, exist_ok=True)
    (arguments.output / "exports").mkdir(parents=True, exist_ok=True)
    for name in ("results", "exports"):
        subprocess.run(
            [
                "rsync",
                "-e",
                RSYNC_REMOTE_SHELL,
                "-aL",
                f"columba-cluster:{workspace}/{name}/",
                str(arguments.output / name) + "/",
            ],
            check=True,
        )


def add_candidate(arguments: argparse.Namespace) -> None:
    import shlex

    from cluster.sweep import current_queue, deploy, remote, remote_root

    if not arguments.recipe or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in arguments.recipe
    ):
        raise ValueError("recipe must be a lower-case recipe id, without path components")
    from cluster.recipe import component

    component(arguments.dataset_name, "dataset")
    output = arguments.output.resolve()
    root = remote_root()
    workspace = root / "scratch/ws7/additions" / arguments.recipe
    checkpoint_root = output / "checkpoints" / arguments.recipe
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for seed in sorted(REQUIRED_SEEDS):
        try:
            remote(["test", "-f", str(root / "runs" / arguments.recipe / f"seed{seed}" / "DONE")])
        except RuntimeError as error:
            raise ValueError(
                f"{arguments.recipe}/seed{seed} is not DONE; wait for its final checkpoint"
            ) from error
    subprocess.run(
        [
            "rsync",
            "-e",
            RSYNC_REMOTE_SHELL,
            "-a",
            "--include=*/",
            "--include=best.pt",
            "--exclude=*",
            f"columba-cluster:{root}/runs/{arguments.recipe}/",
            str(checkpoint_root) + "/",
        ],
        check=True,
    )
    for seed in sorted(REQUIRED_SEEDS):
        export_candidate(
            argparse.Namespace(
                checkpoint=checkpoint_root / f"seed{seed}/weights/best.pt",
                recipe=arguments.recipe,
                seed=seed,
                input_hw=arguments.input_hw,
                taxonomy=arguments.taxonomy,
                decodes=arguments.decodes,
                output=output / "exports",
            )
        )
    if arguments.reference_data_root:
        for seed in sorted(REQUIRED_SEEDS):
            predict_references(
                argparse.Namespace(
                    checkpoint=checkpoint_root / f"seed{seed}/weights/best.pt",
                    recipe=arguments.recipe,
                    seed=seed,
                    input_hw=arguments.input_hw,
                    taxonomy=arguments.taxonomy,
                    decodes=arguments.decodes,
                    output=output / "references",
                    dataset=arguments.reference_data_root / arguments.dataset_name,
                    test_dataset=arguments.reference_data_root / "columba-test-v2",
                    styles_dataset=arguments.reference_data_root / "columba-test-v2-styles",
                    device="cpu",
                    batch=1,
                )
            )
    checkout, _ = deploy(root)
    remote(["mkdir", "-p", str(workspace / "exports"), str(workspace / "results")])
    subprocess.run(
        [
            "rsync",
            "-e",
            RSYNC_REMOTE_SHELL,
            "-a",
            str(output / "exports" / arguments.recipe),
            f"columba-cluster:{workspace}/exports/",
        ],
        check=True,
    )
    if arguments.reference_data_root:
        remote(["mkdir", "-p", str(workspace / "references")])
        subprocess.run(
            [
                "rsync",
                "-e",
                RSYNC_REMOTE_SHELL,
                "-a",
                str(output / "references" / arguments.recipe),
                f"columba-cluster:{workspace}/references/",
            ],
            check=True,
        )
    single_recipe_arguments = (
        ("--single-recipe", arguments.single_recipe) if arguments.single_recipe else ()
    )
    reference_arguments = (
        ("--references", str(workspace / "references")) if arguments.reference_data_root else ()
    )
    evaluation = shlex.join(
        [
            "python",
            "-m",
            "tools.select_model",
            "evaluate",
            "--recipe",
            arguments.recipe,
            "--input-hw",
            *map(str, arguments.input_hw),
            "--taxonomy",
            arguments.taxonomy,
            "--decodes",
            *arguments.decodes,
            *single_recipe_arguments,
            *reference_arguments,
        ]
    )
    script = f"""#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodelist=xcng0
#SBATCH --nice=1000
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
set -euo pipefail
workspace={shlex.quote(str(workspace))}
checkout={shlex.quote(str(checkout))}
stage="/tmp/cz-$SLURM_JOB_ID"
mkdir -p "$stage"
cleanup() {{
    local exit_status=$?
    trap - EXIT
    if [[ -d "$stage/exports" ]]; then
        if ! rsync -a "$stage/exports/" "$workspace/exports/"; then
            exit_status=1
        fi
    fi
    rm -rf -- "$stage"
    if [[ "$exit_status" == 0 ]]; then
        date -u > "$workspace/DONE"
    fi
    exit "$exit_status"
}}
trap cleanup EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=ACTIVE
export TMPDIR="$stage" YOLO_CONFIG_DIR="$stage/settings" SLURM_EXPORT_ENV=ALL
export PATH="$HOME/columba/scratch/ws8/venv/bin:$PATH"
export PYTHONPATH="$HOME/columba/scratch/ws8/round3/repo/native/build/linux-aarch64:$checkout"
cd "$checkout"
cp "$HOME/columba/data/columba-test-v2.tar.zst" "$stage/"
tar --zstd -xf "$stage/columba-test-v2.tar.zst" -C "$stage"
tar --zstd -xf "$HOME/columba/data/{arguments.dataset_name}.tar.zst" -C "$stage" \\
    {arguments.dataset_name}/DATASET.json {arguments.dataset_name}/data.yaml \\
    {arguments.dataset_name}/manifest.jsonl {arguments.dataset_name}/images/val \\
    {arguments.dataset_name}/labels/val
python -m tools.build_dataset verify-freeze "$stage/columba-test-v2" \\
    data_sources/columba-test-v2.sha256
cp -r "$workspace/exports" "$stage/exports"
tar --zstd -xf "$HOME/columba/data/columba-test-v2-styles.tar.zst" -C "$stage"
for seed in 0 1 2; do
    srun --cpu-bind=cores {evaluation} --seed "$seed" \\
        --checkpoint "$HOME/columba/runs/{arguments.recipe}/seed$seed/weights/best.pt" \\
        --exports "$stage/exports" --output "$workspace/results" \\
        --dataset "$stage/{arguments.dataset_name}" --test-dataset "$stage/columba-test-v2" \\
        --prefix "$HOME/columba/scratch/ws4/ncnn-xcng"
done
srun --cpu-bind=cores python -m tools.select_model styles --candidates "$workspace/results" \\
    --exports "$stage/exports" --dataset "$stage/columba-test-v2-styles"
"""
    print(
        json.dumps(
            [{"job_id": job.identifier, "state": job.state} for job in current_queue() if job.owned]
        )
    )
    payload = {
        "budget": {"gpu_jobs": 8, "cpu_jobs": 2, "queued_jobs": 24},
        "submissions": [
            {
                "name": "cz-ws7-add-" + hashlib.sha256(arguments.recipe.encode()).hexdigest()[:12],
                "lane": "arm",
                "constraint": None,
                "script": script,
                "run_path": str(workspace),
            }
        ],
    }
    response = remote(
        [
            "sh",
            "-c",
            (
                f"cd {shlex.quote(str(checkout))} && python -m cluster.slurm submit "
                f"--payload {shlex.quote(json.dumps(payload))}"
            ),
        ]
    )
    print(response)
    write_json(
        output / "submissions" / f"{arguments.recipe}.json",
        {
            "workspace": str(workspace),
            "response": json.loads(response),
            "script": script,
        },
    )


def package_candidate(arguments: argparse.Namespace) -> None:
    from tools.prepare_models import prepare_models

    decision = read_report(arguments.decision)
    selected = decision.get("package_candidate")
    eligible = isinstance(selected, str)
    if not eligible and arguments.diagnostic:
        selected = decision.get("diagnostic_candidate")
    if not isinstance(selected, str):
        raise ValueError("selection is HOLD; no candidate passed all gates")
    descriptor = read_report(Path(selected) / "candidate.json")
    source = (
        arguments.exports
        / str(descriptor["recipe"])
        / f"seed{descriptor['seed']}"
        / str(descriptor["decode"])
        / f"{descriptor['precision']}_ncnn_model"
    )
    model_id = str(descriptor["model_id"]).replace("/", "-") + "-provisional"
    repository = Path(__file__).resolve().parents[1]
    destination = repository / "outputs/ws7/packages" / model_id
    destination.mkdir(parents=True, exist_ok=True)
    files: list[JsonValue] = []
    names = ["model.ncnn.param", "model.ncnn.bin", "metadata.yaml"]
    names.extend(
        name
        for name in ("calibration.json", "calibration.table", "export.json")
        if (source / name).is_file()
    )
    for name in names:
        shutil.copy2(source / name, destination / name)
        if name == "metadata.yaml":
            import yaml

            metadata = mapping(yaml.safe_load((destination / name).read_text()), "metadata")
            columba = mapping(metadata["columba"], "columba")
            columba["model_id"] = model_id
            metadata["columba"] = columba
            (destination / name).write_text(yaml.safe_dump(metadata, sort_keys=False))
        payload = (destination / name).read_bytes()
        files.append(
            {"file": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        )
    manifest_path = repository / "deploy/model_manifest.json"
    manifest = read_report(manifest_path)
    entry: dict[str, JsonValue] = {
        "model_id": model_id,
        "provisional": True,
        "promotion_status": "dry-run-only" if eligible else "blocked-by-gates",
        "eligible": eligible,
        "failed_gates": [] if eligible else decision["diagnostic_failures"],
        "selection": str(arguments.decision),
        "operating_threshold": read_report(Path(selected) / "metrics_val.json")[
            "operating_threshold"
        ],
        "decode": descriptor["decode"],
        "input_hw": descriptor["input_hw"],
        "quantization": descriptor["precision"],
        "taxonomy": descriptor["taxonomy"],
        "source": str(destination.relative_to(repository)),
        "runtime_directory": f"models/{model_id}",
        "files": files,
    }
    entries = manifest.get("provisional_models", [])
    if not isinstance(entries, list):
        raise ValueError("provisional_models must be a list")
    manifest["provisional_models"] = [
        item for item in entries if mapping(item, "entry").get("model_id") != model_id
    ] + [entry]
    write_json(manifest_path, manifest)
    model_settings(destination, cast(list[int], descriptor["input_hw"]), str(descriptor["decode"]))
    verified = prepare_models(repository, model_id)
    decision["package_verification"] = {
        "model_id": model_id,
        "eligible": eligible,
        "provisional": True,
        "directory": str(verified.relative_to(repository)),
        "source": str(destination.relative_to(repository)),
        "manifest": "deploy/model_manifest.json",
        "status": "SHA-256 and byte counts verified",
    }
    write_json(arguments.decision, decision)
    if arguments.report:
        write_decision_report(arguments.report, decision)
    print(verified)


def write_decision_report(path: Path, decision: dict[str, JsonValue]) -> None:
    arms = [mapping(arm, "arm") for arm in cast(list[JsonValue], decision["arms"])]
    baseline = mapping(decision["baseline"], "baseline")
    lines = [
        "# WS7 provisional export decision",
        "",
        f"Provisional selection: `{decision['selected'] or 'HOLD: no arm passes every gate'}`.",
        "Final selection waits for distilled and species candidates.",
        f"Coverage: {len(arms)} configurations and "
        f"{sum(len(cast(list[JsonValue], arm['seeds'])) for arm in arms)} seed/export cases.",
        f"Comparison baseline: `{baseline['recipe']}`, measured native fp16 384x640 test R@P90 "
        f"{number(baseline['test_mean'], 'baseline mean'):.4f} +/- "
        f"{number(baseline['test_std'], 'baseline sd'):.4f}, "
        "independently val-thresholded per seed.",
        "Styles is a secondary measured column. Test-v2 still supplies every promotion",
        "gate. Paired v2/v2s arms within 0.01 validation recall and 5% measured latency",
        "use styles recall to break the tie. The 5% band treats host timing noise",
        "as equal latency.",
        "",
        "Primary accuracy is measured using columba_native package-default compute on xcng0.",
        "Native styles accuracy and ACTIVE latency use xcng0/xcng1; C3/C7 record each host.",
        "Val/test CPU PyTorch references use xcng0 for the first seed's two heads, then Mac",
        "vexcalibur. Mac versus xcng0 passes all 1,408 first-reference image comparisons.",
        "All PyTorch styles references use Mac vexcalibur CPU.",
        "Every export independently reproduces parity against its own saved CPU reference.",
        "Styles scores are measured on WS2's synthetic camera-style copies.",
        "These rectangular deploy-size measurements decide the section-1 and false-alarm",
        "gates. WS5 square-input leaderboard scores are context, not deployment scores.",
        "C7 latency uses OMP_WAIT_POLICY=ACTIVE; PASSIVE results remain diagnostics.",
        "WS10 will decide the deployed wait policy on the Pi, including pipeline contention.",
        "The launcher is unchanged.",
        "Pi 5 latency is estimated with WS4 cal-v0-published, xcng times x 1.527110713.",
        "Reaction is simulated by WS4 with 30 FPS, k2/n3, 20,000 trials, seed 42,",
        "independent hits and per-frame recall proxied by test recall at each export's",
        "fresh validation-selected threshold. Temporal correlation is unmeasured.",
        "",
        "Arms rank by three-seed mean validation R@P90; test is a promotion gate.",
        "Choose the highest-val-recall seed among val FAR passes, then apply test,",
        "latency, reaction and export gates to that seed alone. No test-driven retry.",
        "Every seed's FAR remains visible as robustness context.",
        "FP16 and int8 each permit at most 0.01 loss in val/test R@P90, AP50 and",
        "operating recall versus same-head rectangular PyTorch. This is an explicit",
        "confirmed accuracy policy. FP16 parity requires strict fp32 parity of the same",
        "graph plus that accuracy gate. Strict fp16 per-image parity stays diagnostic.",
        "Strict comparison uses count equality, IoU >=0.8 / score delta <=0.05;",
        "int8 diagnostics use IoU >=0.7 / score delta <=0.10, at confidence 0.25.",
        "Coordinator exclusions: int8 and one-to-one heads are not deployment candidates.",
        "",
        "Calibration uses 200 deterministic val images, centred deployment letterbox,",
        "RGB /255 and padding 114. Selected SHA-256s are checked against frozen test.",
        "No calibration image or operating threshold comes from test.",
        "",
        "## Arm results",
        "",
        "Accuracy values below are measured on xcng; frame p95 is estimated Pi 5;",
        "reaction p95 is simulated. Timing uses the val-chosen seed, or worst seed",
        "for diagnostic configurations with no val-eligible seed. Input sizes are H x W.",
        "",
        (
            "| Arm / decode / precision / input | Val R@P90 | Test R@P90 mean +/- sd | "
            "Test AP50 | Test operating R | PT styles R@P90 | Export styles R@P90 | "
            "Pi frame p95 ms | "
            "Reaction p95 ms | Shipped seed | Pass |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in arms:
        styles = arm["styles_recall_mean"]
        reference_styles = arm["reference_styles_recall_mean"]
        reaction_p95_ms = arm["reaction_p95_ms"]
        styles_display = "pending" if styles is None else f"{number(styles, 'styles'):.4f}"
        reference_styles_display = (
            "pending" if reference_styles is None else f"{number(reference_styles, 'styles'):.4f}"
        )
        reaction_display = (
            "unconfirmed"
            if reaction_p95_ms is None
            else f"{number(reaction_p95_ms, 'reaction'):.2f}"
        )
        lines.append(
            f"| {arm['id']} | {number(arm['val_recall_mean'], 'val'):.4f} | "
            f"{number(arm['test_recall_mean'], 'test'):.4f} +/- "
            f"{number(arm['test_recall_std'], 'std'):.4f} | "
            f"{number(arm['test_ap50_mean'], 'AP'):.4f} | "
            f"{number(arm['test_operating_recall_mean'], 'recall'):.4f} | "
            f"{reference_styles_display} | "
            f"{styles_display} | "
            f"{number(arm['pi5_p95_ms'], 'latency'):.2f} | "
            f"{reaction_display} | {arm['shipped_seed']} | "
            f"{'yes' if arm['passes'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Precision comparisons",
            "",
            "Mean deltas are export minus same-head rectangular CPU PyTorch.",
            "Accuracy and strict per-image diagnostic parity count passing seeds independently.",
            "",
            "| Arm / decode / precision | Test R@P90 delta | Test AP50 delta | "
            "Accuracy gate | Strict parity diagnostic |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        seeds = [mapping(value, "seed") for value in cast(list[JsonValue], arm["seeds"])]
        precision_deltas = [mapping(seed["accuracy_deltas"], "deltas") for seed in seeds]
        recall_delta = statistics.mean(
            number(delta["test_recall_at_p90"], "R") for delta in precision_deltas
        )
        ap_delta = statistics.mean(number(delta["test_ap50"], "AP") for delta in precision_deltas)
        accuracy_passes = sum(seed["accuracy_gate_passed"] is True for seed in seeds)
        parity_passes = sum(seed["parity_passed"] is True for seed in seeds)
        lines.append(
            f"| {arm['id']} | {recall_delta:+.4f} | {ap_delta:+.4f} | "
            f"{accuracy_passes}/{len(seeds)} | {parity_passes}/{len(seeds)} |"
        )
    lines.extend(
        [
            "",
            "## Head-swap comparison",
            "",
            "Deltas are one-to-one minus standard NMS, matched by recipe, precision",
            "and deploy size. Latency compares val-chosen-seed estimated Pi frame p95.",
            "",
            "| Arm / precision | Val R@P90 delta | Test R@P90 delta | Styles R@P90 delta | "
            "Frame p95 delta ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        if arm["decode"] != "yolo26":
            continue
        standard = next(
            (
                candidate
                for candidate in arms
                if candidate["recipe"] == arm["recipe"]
                and candidate["precision"] == arm["precision"]
                and candidate["input_hw"] == arm["input_hw"]
                and candidate["decode"] == "yolo11"
            ),
            None,
        )
        if standard is None:
            continue
        differences = [
            f"{number(arm[field], field) - number(standard[field], field):+.4f}"
            if arm[field] is not None and standard[field] is not None
            else "pending"
            for field in (
                "val_recall_mean",
                "test_recall_mean",
                "styles_recall_mean",
                "pi5_p95_ms",
            )
        ]
        lines.append(f"| {arm['recipe']} / {arm['precision']} | " + " | ".join(differences) + " |")
    lines.extend(
        [
            "",
            "## Individual exports",
            "",
            "Deltas are export minus same-head rectangular PyTorch, measured on xcng.",
            "Each export reselects its threshold on val. No square-training metric",
            "is substituted for rectangular inference accuracy.",
            "",
            (
                "| Candidate | Val R@P90 | Test R@P90 | Test R@P90 delta | Test AP50 delta | "
                "Val empty / confuser FAR | Test empty / confuser FAR | Val threshold | "
                "Strict parity failed images |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        for value in cast(list[JsonValue], arm["seeds"]):
            seed = mapping(value, "seed")
            deltas = mapping(seed["accuracy_deltas"], "deltas")
            lines.append(
                f"| {seed['model_id']} | {metric(seed, 'val', 'recall_at_p90'):.4f} | "
                f"{metric(seed, 'test', 'recall_at_p90'):.4f} | "
                f"{number(deltas['test_recall_at_p90'], 'delta'):+.4f} | "
                f"{number(deltas['test_ap50'], 'delta'):+.4f} | "
                f"{seed['val_empty_far']} / {seed['val_confuser_far']} | "
                f"{seed['empty_far']} / {seed['confuser_far']} | "
                f"{number(seed['threshold'], 'threshold'):.6f} | {seed['parity_failures']} |"
            )
    lines.extend(["", "## Rejected gates", ""])
    for arm in arms:
        if not arm["passes"]:
            lines.append(
                f"- `{arm['id']}`: " + "; ".join(map(str, cast(list[JsonValue], arm["failures"])))
            )
    lines.extend(
        [
            "",
            f"Admissible Pareto front: `{decision['admissible_front']}`.",
            "",
            "No measured Pi 5 result exists. The calibration factor and the mapping",
            "from still-image recall to temporal hits remain estimates and assumptions.",
            "",
            "Reproduction commands, candidate addition, resource handling and policy",
            "requests are in [WS7-methods.md](WS7-methods.md).",
            "",
        ]
    )
    if decision.get("package_verification") is not None:
        verified = mapping(decision["package_verification"], "package verification")
        lines.extend(
            [
                "## Package dry run",
                "",
                f"`{verified['model_id']}` is provisional. {verified['status']}.",
                f"Package source: `{verified['source']}`. "
                f"Verified copy: `{verified['directory']}`.",
                f"Manifest: `{verified['manifest']}`. "
                f"Eligible for promotion: `{verified['eligible']}`.",
                "",
            ]
        )
    if decision.get("baseline_measurements") is not None:
        measurements = mapping(decision["baseline_measurements"], "baseline measurements")
        lines.extend(["## Rectangular baseline", "", str(measurements["description"]), ""])
        lines.extend(cast(list[str], measurements["markdown"]))
    selected = next((arm for arm in arms if arm["id"] == decision["selected"]), None)
    if selected is not None:
        lines.extend(
            [
                "## Selected export gates",
                "",
                f"Shipped seed {selected['shipped_seed']} was chosen on val only. "
                "The three-seed means rank the arm; the rows below gate its shipped export.",
                "",
                "| Gate | Value | Requirement | Pass |",
                "|---|---:|---|---|",
            ]
        )
        for value in cast(list[JsonValue], selected["gates"]):
            gate = mapping(value, "gate")
            observed = gate["value"]
            display = f"{observed:.6f}" if isinstance(observed, float) else str(observed)
            lines.append(
                f"| {gate['gate']} | {display} | {gate['requirement']} | "
                f"{'yes' if gate['passed'] else 'no'} |"
            )
        lines.append("")
    if decision.get("aspect_ratio") is not None:
        aspect = mapping(decision["aspect_ratio"], "aspect")
        lines.extend(aspect_markdown(aspect))
        if aspect.get("interpretation") is not None:
            lines.extend([str(aspect["interpretation"]), ""])
    lines.extend(
        [
            "## Quantisation diagnostic",
            "",
            "A plausible cause of the 800-arm int8 collapse is full-graph KL post-training",
            "quantisation sensitivity, including the detection head, to model activation ranges.",
            "Both sizes used the same 200 val IDs and RGB /255 preprocessing. The 800 and",
            "640 checkpoints also differ, so this is not a controlled input-shape comparison.",
            "No specific cause is established. Int8 is excluded; no further investigation was run.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def audit_candidates(arguments: argparse.Namespace) -> None:
    import math

    from tools.quantize_ncnn import CALIBRATION_IMAGES, digest

    validation = load_dataset(arguments.dataset, "val")
    frozen = load_dataset(arguments.test_dataset, "test")
    validation_hashes = {image.image_id: digest(image.path) for image in validation.images}
    frozen_hashes = {digest(image.path) for image in frozen.images}
    verified = quantised = 0
    for marker in sorted(arguments.candidates.rglob("candidate.json")):
        directory = marker.parent
        descriptor = read_report(marker)
        candidate_record(directory)
        latency = read_report(directory / "latency.json")
        samples = latency.get("samples_ms")
        if not isinstance(samples, list) or len(samples) < TIMED_ITERATIONS:
            raise ValueError(f"{directory}: C7 needs at least 200 paired samples")
        calculated = summary([number(value, "sample") for value in samples])
        measured = mapping(latency["end_to_end_ms"], "timings")
        estimated = mapping(latency["estimated_pi5"], "estimated Pi")
        if not math.isclose(number(calculated["p95"], "p95"), number(measured["p95"], "p95")):
            raise ValueError(f"{directory}: C7 p95 differs from its paired samples")
        if not math.isclose(
            number(estimated["end_to_end_p95_ms"], "Pi p95"),
            number(measured["p95"], "host p95") * number(estimated["factor"], "factor"),
        ):
            raise ValueError(f"{directory}: Pi estimate differs from calibrated host p95")
        package = (
            arguments.exports
            / str(descriptor["recipe"])
            / f"seed{descriptor['seed']}"
            / str(descriptor["decode"])
            / f"{descriptor['precision']}_ncnn_model"
        )
        for filename, field in (
            ("model.ncnn.bin", "weights_sha256"),
            ("model.ncnn.param", "parameters_sha256"),
        ):
            if digest(package / filename) != latency[field]:
                raise ValueError(f"{directory}: benchmark model hash differs from package")
        validation_threshold = mapping(
            read_report(directory / "metrics_val.json")["operating_threshold"], "threshold"
        )
        for name in ("test", "styles"):
            report = read_report(directory / f"metrics_{name}.json")
            if (
                mapping(report["operating_threshold"], "threshold")["value"]
                != validation_threshold["value"]
            ):
                raise ValueError(f"{directory}: {name} threshold differs from val")
        reproduced = parity(
            directory.parent / "pt/predictions_test.json",
            directory / "predictions_test.json",
            str(descriptor["precision"]),
            [image.image_id for image in frozen.images],
        )
        recorded = read_report(directory / "parity.json")
        for field in ("passed", "failures", "count_mismatches"):
            if reproduced[field] != recorded[field]:
                raise ValueError(f"{directory}: parity {field} does not reproduce")
        if descriptor["precision"] == "int8":
            calibration = read_report(package / "calibration.json")
            images = calibration["images"]
            if (
                calibration["split"] != "val"
                or not isinstance(images, list)
                or len(images) != CALIBRATION_IMAGES
            ):
                raise ValueError(f"{directory}: requires 200 recorded val calibration images")
            for value in images:
                image = mapping(value, "calibration image")
                image_id = str(image["id"])
                if image_id not in validation_hashes:
                    raise ValueError(f"calibration image is not in val: {image_id}")
                actual = validation_hashes[image_id]
                if actual != image["sha256"] or actual in frozen_hashes:
                    raise ValueError(f"calibration provenance/test exclusion fails: {image_id}")
            quantised += 1
        verified += 1
    audit_report: dict[str, JsonValue] = {
        "label": "measured",
        "host": platform.node(),
        "candidates_verified": verified,
        "parity_reports_reproduced": verified,
        "int8_calibrations_verified": quantised,
        "calibration_images_each": CALIBRATION_IMAGES,
        "test_calibration_overlap": False,
        "checks": (
            "native compute, own val thresholds, C7 samples/scaling, "
            "package hashes, C3 parity, val-only calibration"
        ),
    }
    write_json(arguments.out, audit_report)
    print(json.dumps(audit_report))


def aspect_bucket(width_px: int, height_px: int) -> str:
    ratio = width_px / height_px
    if ratio < 1:
        return "portrait"
    if ratio < SQUARE_LANDSCAPE_BOUNDARY_RATIO:
        return "square"
    if ratio < WIDE_LANDSCAPE_BOUNDARY_RATIO:
        return "4:3"
    return "approx-16:9"


def geometry_summary(metrics: dict[str, JsonValue]) -> dict[str, JsonValue]:
    pigeon = mapping(metrics["pigeon"], "pigeon")
    return {
        "recall_at_p90": pigeon["recall_at_p90"],
        "small_recall_at_p90": pigeon["small_recall_at_p90"],
        "recall_at_full_test_p90": pigeon["recall_at_threshold"],
        "counts": metrics["counts"],
    }


def aspect_report(arguments: argparse.Namespace) -> None:
    dataset = load_dataset(arguments.dataset, "test")
    paths = (arguments.square_predictions, arguments.rectangular_predictions)
    metadata = [read_report(sidecar_path(path)) for path in paths]
    for report, shape in zip(metadata, ([800, 800], [448, 800]), strict=True):
        if report.get("format") != "pt" or report.get("input_hw") != shape:
            raise ValueError("aspect comparison requires PyTorch 800x800 and 448x800 C3")
    for field in ("model_path", "device", "decode", "host"):
        if metadata[0].get(field) != metadata[1].get(field):
            raise ValueError(f"aspect comparison requires matching {field}")
    predictions = [read_predictions(path, dataset) for path in paths]
    shapes = ((800, 800), (448, 800))
    thresholds = [
        number(select_operating_threshold(dataset, detected, shape)["value"], "threshold")
        for detected, shape in zip(predictions, shapes, strict=True)
    ]
    buckets: list[JsonValue] = []
    for bucket in ("all", *ASPECT_BUCKETS):
        images = tuple(
            image
            for image in dataset.images
            if bucket == "all" or aspect_bucket(image.width_px, image.height_px) == bucket
        )
        if not images:
            continue
        subset = Dataset(dataset.name, dataset.names, images, dataset.id_source)
        image_ids = {image.image_id for image in images}
        subsets = [
            tuple(p for p in detected if p.image_id in image_ids) for detected in predictions
        ]
        scales = [
            min(448 / image.height_px, 800 / image.width_px)
            / min(800 / image.height_px, 800 / image.width_px)
            for image in images
            for annotation in image.annotations
            if annotation.category_id == 0
        ]
        square = geometry_summary(metric_group(subset, subsets[0], shapes[0], thresholds[0]))
        rectangular = geometry_summary(metric_group(subset, subsets[1], shapes[1], thresholds[1]))
        fixed_small = geometry_summary(metric_group(subset, subsets[1], shapes[0], thresholds[1]))
        buckets.append(
            {
                "bucket": bucket,
                "pigeon_images": sum(
                    any(a.category_id == 0 for a in image.annotations) for image in images
                ),
                "pigeon_weighted_content_scale_ratio": statistics.mean(scales) if scales else None,
                "square": square,
                "rectangular": rectangular,
                "rectangular_fixed_square_small_cohort": fixed_small,
            }
        )
    report = {
        "label": "measured",
        "host": platform.node(),
        "framework": "ultralytics",
        "device": metadata[0]["device"],
        "model_path": metadata[0]["model_path"],
        "square_predictions": str(paths[0]),
        "rectangular_predictions": str(paths[1]),
        "square_input_hw": [800, 800],
        "rectangular_input_hw": [448, 800],
        "bucket_boundaries": (
            "portrait <1; square [1,1.1547); 4:3 [1.1547,1.5396); "
            "approx-16:9 >=1.5396, includes wider images"
        ),
        "small_definition": "WS1 projected short side <32 px; fixed cohort uses 800x800 for both",
        "global_test_p90_thresholds_diagnostic_only": cast(list[JsonValue], thresholds),
        "threshold_policy": (
            "subset R@P90 is diagnostic; no operating threshold or promotion gate changes"
        ),
        "buckets": buckets,
    }
    write_json(arguments.out, report)
    arguments.report.write_text("\n".join(aspect_markdown(report)))


def aspect_markdown(report: dict[str, JsonValue]) -> list[str]:
    lines = [
        "## Aspect-ratio diagnostic, PyTorch only",
        "",
        f"Measured on `{report['host']}`, CPU, same checkpoint and preprocessing implementation.",
        "Inputs are 800x800 versus 448x800. The wide bucket includes ratios >=1.5396;",
        "4:3 includes [1.1547,1.5396), square [1,1.1547), portrait <1.",
        "Small means projected short side <32 px under WS1. Its cohort changes with input size.",
        "The final column holds the square-defined small cohort fixed for both predictions.",
        "Subset R@P90 can use a different diagnostic precision-90 score cut in each bucket.",
        "No validation operating threshold or promotion gate is changed.",
        "",
        "| Bucket | Images / pigeon images / boxes | Content scale ratio | R@P90 square / rect | "
        "Small R@P90 square / rect | Small boxes square / rect | Rect small recall, fixed cohort |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for value in cast(list[JsonValue], report["buckets"]):
        bucket = mapping(value, "bucket")
        square = mapping(bucket["square"], "square")
        rectangular = mapping(bucket["rectangular"], "rectangular")
        fixed = mapping(bucket["rectangular_fixed_square_small_cohort"], "fixed")
        counts = mapping(square["counts"], "counts")
        rect_counts = mapping(rectangular["counts"], "counts")
        scale = bucket["pigeon_weighted_content_scale_ratio"]
        scale_display = "n/a" if scale is None else f"{number(scale, 'scale'):.3f}"
        lines.append(
            f"| {bucket['bucket']} | {counts['images']} / {bucket['pigeon_images']} / "
            f"{counts['pigeon_boxes']} | {scale_display} | "
            f"{number(square['recall_at_p90'], 'R'):.4f} / "
            f"{number(rectangular['recall_at_p90'], 'R'):.4f} | "
            f"{small_recall_display(square)} / {small_recall_display(rectangular)} | "
            f"{counts['small_pigeon_boxes']} / {rect_counts['small_pigeon_boxes']} | "
            f"{small_recall_display(fixed)} |"
        )
    lines.extend(
        [
            "",
            "At each shape's full-test P90 cut, diagnostic matched pigeon boxes are:",
            "",
            "| Bucket | Square matched | Rectangular matched | Lost matches |",
            "|---|---:|---:|---:|",
        ]
    )
    for value in cast(list[JsonValue], report["buckets"]):
        bucket = mapping(value, "bucket")
        square = mapping(bucket["square"], "square")
        rectangular = mapping(bucket["rectangular"], "rectangular")
        count = number(mapping(square["counts"], "counts")["pigeon_boxes"], "boxes")
        square_matches = round(number(square["recall_at_full_test_p90"], "R") * count)
        rectangular_matches = round(number(rectangular["recall_at_full_test_p90"], "R") * count)
        lines.append(
            f"| {bucket['bucket']} | {square_matches} | {rectangular_matches} | "
            f"{square_matches - rectangular_matches} |"
        )
    return lines + [""]


def small_recall_display(metrics: dict[str, JsonValue]) -> str:
    if mapping(metrics["counts"], "counts")["small_pigeon_boxes"] == 0:
        return "n/a"
    return f"{number(metrics['small_recall_at_p90'], 'R'):.4f}"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="action", required=True)
    for action in ("export", "evaluate", "references"):
        sub = commands.add_parser(action)
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--recipe", required=True)
        sub.add_argument("--seed", type=int, required=True)
        sub.add_argument("--input-hw", type=int, nargs=2, required=True)
        sub.add_argument("--taxonomy", choices=("single", "species"), default="single")
        sub.add_argument(
            "--decodes", nargs="+", choices=("yolo11", "yolo26"), default=["yolo11", "yolo26"]
        )
        sub.add_argument("--output", type=Path, required=True)
        if action == "evaluate":
            sub.add_argument("--exports", type=Path, required=True)
            sub.add_argument("--references", type=Path)
            sub.add_argument("--single-recipe", help="matched single-taxonomy arm for PLAN 5.4")
            sub.add_argument("--dataset", type=Path, required=True)
            sub.add_argument("--test-dataset", type=Path, required=True)
            sub.add_argument("--prefix", type=Path, required=True)
            sub.add_argument(
                "--calibration", type=Path, default=Path("experiments/latency/calibration.json")
            )
        if action == "references":
            sub.add_argument("--dataset", type=Path, required=True)
            sub.add_argument("--test-dataset", type=Path, required=True)
            sub.add_argument("--styles-dataset", type=Path)
            sub.add_argument("--device", default="cpu")
            sub.add_argument("--batch", type=int, default=1)
    for action in ("predict-native", "benchmark-native"):
        sub = commands.add_parser(action)
        sub.add_argument("--model", type=Path, required=True)
        sub.add_argument("--model-id", required=True)
        sub.add_argument("--dataset", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
        if action == "predict-native":
            sub.add_argument("--split", choices=("val", "test"), required=True)
        else:
            sub.add_argument("--prefix", type=Path, required=True)
            sub.add_argument("--confidence", type=float, required=True)
            sub.add_argument("--calibration", type=Path, required=True)
    sub = commands.add_parser(
        "add", help="fetch/export all three seeds and submit one guarded ARM bundle"
    )
    sub.add_argument("--recipe", required=True)
    sub.add_argument("--input-hw", type=int, nargs=2, required=True)
    sub.add_argument("--taxonomy", choices=("single", "species"), default="single")
    sub.add_argument("--dataset-name", default="columba-train-v2")
    sub.add_argument("--single-recipe")
    sub.add_argument(
        "--reference-data-root",
        type=Path,
        default=Path.home() / "columba-data",
        help="optional Mac C1 root for CPU reference precomputation",
    )
    sub.add_argument(
        "--decodes", nargs="+", choices=("yolo11", "yolo26"), default=["yolo11", "yolo26"]
    )
    sub.add_argument("--output", type=Path, default=Path("outputs/ws7"))
    sub = commands.add_parser("audit", help="verify saved C3/C4/C7 data and real package bytes")
    sub.add_argument("--candidates", type=Path, required=True)
    sub.add_argument("--exports", type=Path, required=True)
    sub.add_argument("--dataset", type=Path, required=True)
    sub.add_argument("--test-dataset", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True)
    sub = commands.add_parser("fetch", help="fetch a completed add bundle and its C8 packages")
    sub.add_argument("--recipe", required=True)
    sub.add_argument("--output", type=Path, default=Path("outputs/ws7"))
    sub = commands.add_parser("remeasure")
    sub.add_argument("--candidates", type=Path, required=True)
    sub.add_argument("--exports", type=Path, required=True)
    sub.add_argument("--dataset", type=Path, required=True)
    sub.add_argument("--prefix", type=Path, required=True)
    sub.add_argument(
        "--calibration", type=Path, default=Path("experiments/latency/calibration.json")
    )
    sub = commands.add_parser("styles")
    sub.add_argument("--candidates", type=Path, required=True)
    sub.add_argument("--exports", type=Path, required=True)
    sub.add_argument("--dataset", type=Path, required=True)
    sub.add_argument(
        "--precisions",
        nargs="+",
        choices=("fp32", "fp16", "int8"),
        default=["fp32", "fp16", "int8"],
    )
    sub = commands.add_parser("package")
    sub.add_argument("--decision", type=Path, required=True)
    sub.add_argument("--report", type=Path)
    sub.add_argument(
        "--diagnostic",
        action="store_true",
        help="package a blocked candidate for hash-check dry run while selection stays HOLD",
    )
    sub.add_argument("--exports", type=Path, required=True)
    sub = commands.add_parser("aspect")
    sub.add_argument("--dataset", type=Path, required=True)
    sub.add_argument("--square-predictions", type=Path, required=True)
    sub.add_argument("--rectangular-predictions", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True)
    sub.add_argument("--report", type=Path, required=True)
    sub = commands.add_parser("select")
    sub.add_argument(
        "candidates",
        nargs="+",
        type=Path,
        help="candidate directories or roots, searched recursively",
    )
    sub.add_argument("--report", type=Path)
    sub.add_argument(
        "--baseline-metrics", type=Path, default=Path("experiments/reports/WS7-baseline/fp16")
    )
    sub.add_argument(
        "--aspect-report", type=Path, default=Path("experiments/reports/WS7-aspect.json")
    )
    sub.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    arguments = parser().parse_args()
    if arguments.action == "export":
        export_candidate(arguments)
    elif arguments.action == "evaluate":
        evaluate_candidate(arguments)
    elif arguments.action == "references":
        predict_references(arguments)
    elif arguments.action == "styles":
        score_styles(arguments)
    elif arguments.action == "remeasure":
        remeasure_latency(arguments)
    elif arguments.action == "predict-native":
        native_predictions(arguments)
    elif arguments.action == "benchmark-native":
        benchmark_native(arguments)
    elif arguments.action == "add":
        add_candidate(arguments)
    elif arguments.action == "fetch":
        fetch_candidate(arguments)
    elif arguments.action == "audit":
        audit_candidates(arguments)
    elif arguments.action == "package":
        package_candidate(arguments)
    elif arguments.action == "aspect":
        aspect_report(arguments)
    else:
        directories = sorted(
            {
                path.parent.resolve()
                for root in arguments.candidates
                for path in root.rglob("candidate.json")
            }
        )
        if not directories:
            raise SystemExit("no candidate.json files found")
        decision = select_candidates(directories, baseline_scores(arguments.baseline_metrics))
        baseline_summary = arguments.baseline_metrics.parent / "measurements.json"
        if baseline_summary.exists():
            decision["baseline_measurements"] = read_report(baseline_summary)
        if arguments.aspect_report.exists():
            decision["aspect_ratio"] = read_report(arguments.aspect_report)
        write_json(arguments.output, decision)
        if arguments.report:
            write_decision_report(arguments.report, decision)
        print(
            json.dumps({"selected": decision["selected"], "pareto_front": decision["pareto_front"]})
        )


if __name__ == "__main__":
    main()
