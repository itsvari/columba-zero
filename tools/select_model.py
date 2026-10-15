#!/usr/bin/env python
"""Export, evaluate and select C8 candidates using C4 accuracy and C7 latency.

Run export on the Mac, evaluate in an allocated xcng job, then select locally.
Each evaluation directory is self-contained and can be supplied to select later.
Selection ranks three-seed arms by validation recall; test is a promotion gate.
"""

from __future__ import annotations

import argparse
import csv
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
from pathlib import Path
from typing import cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.json_data import JsonValue
from tools.bench_ncnn import apply_calibration, calibration_factor, read_report, summary
from tools.compare_exports import ExportPrediction, match_predictions
from tools.evaluate_predictions import evaluate, load_dataset, mapping, number, sidecar_path
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


def affinity_cores() -> list[int]:
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []


@contextmanager
def taxonomy_view(dataset: Path, taxonomy: str) -> Iterator[Path]:
    from cluster.staging import align_test_taxonomy

    if taxonomy == "single":
        yield dataset
        return
    with tempfile.TemporaryDirectory(prefix="cz-taxonomy-") as temporary:
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


def write_json(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


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
    detector = columba_native.Detector(str(arguments.model.resolve()), THREAD_COUNT)
    if tuple(detector.names[index] for index in range(len(detector.names))) != dataset.names:
        raise ValueError("native model names differ from dataset")
    predictions: list[JsonValue] = []
    for image in dataset.images:
        frame = cv2.imread(str(image.path))
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
            "compute_flags": detector.compute_flags,
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
    import columba_native
    import cv2

    from tools.export_ncnn import fingerprint

    cv2.setNumThreads(1)
    detector = columba_native.Detector(str(arguments.model.resolve()), THREAD_COUNT)
    dataset = load_dataset(arguments.dataset, "val")
    frames = [cv2.imread(str(image.path)) for image in dataset.images[:WARMUP_ITERATIONS]]
    if any(frame is None for frame in frames):
        raise ValueError("unreadable benchmark frame")
    frames = [cv2.resize(frame, (640, 480)) for frame in frames if frame is not None]
    stages: dict[str, list[float]] = {
        stage: [] for stage in ("preprocess", "extract", "postprocess", "total")
    }
    wall_ms: list[float] = []
    for index in range(WARMUP_ITERATIONS + TIMED_ITERATIONS):
        started_s = time.perf_counter()
        rows, timings = detector.detect_with_timings(
            frames[index % len(frames)], arguments.confidence
        )
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
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", "runtime_default"),
        "compute": detector.compute,
        "compute_flags": detector.compute_flags,
        "camera_hw": [480, 640],
        "confidence": arguments.confidence,
        "weights_sha256": fingerprint(arguments.model / "model.ncnn.bin"),
        "parameters_sha256": fingerprint(arguments.model / "model.ncnn.param"),
        "int8_calibration": read_report(arguments.model / "calibration.json")
        if (arguments.model / "calibration.json").exists()
        else None,
        "end_to_end_boundary": (
            "native letterbox, inference, decode/NMS, Python return and bool decision"
        ),
        "frame_source": "first 20 validation photographs resized to camera shape, cycling",
    }
    write_json(arguments.out, apply_calibration(report, read_report(arguments.calibration)))


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
            reaction = simulate_reaction(
                samples,
                ReactionSettings(recall=max(0.001, number(test["recall_at_threshold"], "recall"))),
            )
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
    return {
        **descriptor,
        "directory": str(directory),
        "val": validation["pigeon"],
        "test": test["pigeon"],
        "empty_far": test["empty_false_alarm_rate"],
        "confuser_far": test["confuser_false_alarm_rate"],
        "pi5_p95_ms": mapping(latency["estimated_pi5"], "estimated Pi")["end_to_end_p95_ms"],
        "reaction_p95_ms": mapping(reaction["reaction_ms"], "reaction")["p95"],
        "threshold": threshold["value"],
        "host": native["host"],
        "reference_host": reference_meta["host"],
        "reference_device": reference_meta["device"],
        "parity_passed": read_report(directory / "parity.json")["passed"],
        "accuracy_gate_passed": read_report(directory / "accuracy_gate.json")["passed"],
        "accuracy_deltas": read_report(directory / "accuracy_gate.json")["deltas"],
        "parity_failures": read_report(directory / "parity.json")["failures"],
        "styles": read_report(directory / "metrics_styles.json")["pigeon"]
        if (directory / "metrics_styles.json").exists()
        else None,
    }


def baseline_scores(leaderboard: Path) -> tuple[float, ...]:
    with leaderboard.open() as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["recipe_id"] == "baseline-repro-p50" and row["status"] == "complete"
        ]
    if {int(row["seed"]) for row in rows} != REQUIRED_SEEDS or len(rows) != len(REQUIRED_SEEDS):
        raise ValueError("baseline-repro-p50 requires exactly seeds 0, 1, 2")
    return tuple(float(row["test_recall_at_p90"]) for row in rows)


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
            < number(reference["test_recall_mean"], "recall") - MAXIMUM_ACCURACY_DROP
        ):
            reasons.append("species pigeon recall is more than 0.01 below matched single")
        if reference is not None:
            comparison = [
                mapping(seed, "seed") for seed in cast(list[JsonValue], reference["seeds"])
            ]
            for field in ("empty_far", "confuser_far"):
                if statistics.mean(number(seed[field], field) for seed in seeds) > statistics.mean(
                    number(seed[field], field) for seed in comparison
                ):
                    reasons.append(f"species mean {field} is worse than matched single")
            if (
                number(arm["pi5_p95_ms"], "latency")
                > number(reference["pi5_p95_ms"], "latency") * 1.05
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
        and abs(
            number(arm["val_recall_mean"], "recall") - number(selected["val_recall_mean"], "recall")
        )
        <= 0.01
        and max(number(arm["pi5_p95_ms"], "latency"), number(selected["pi5_p95_ms"], "latency"))
        <= min(number(arm["pi5_p95_ms"], "latency"), number(selected["pi5_p95_ms"], "latency"))
        * 1.05
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


def select_candidates(directories: list[Path], baseline: tuple[float, ...]) -> dict[str, JsonValue]:
    candidates = [candidate_record(directory) for directory in directories]
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
        for seed in seeds:
            for field, limit in (
                ("empty_far", EMPTY_FAR_LIMIT),
                ("confuser_far", CONFUSER_FAR_LIMIT),
                ("pi5_p95_ms", FRAME_P95_LIMIT_MS),
                ("reaction_p95_ms", REACTION_P95_LIMIT_MS),
            ):
                if seed[field] is None or number(seed[field], field) > limit:
                    reasons.append(f"seed{seed['seed']} fails {field}")
            if seed["parity_passed"] is not True:
                reasons.append(f"seed{seed['seed']} fails export parity")
            if seed["accuracy_gate_passed"] is not True:
                reasons.append(f"seed{seed['seed']} fails deployed precision accuracy")
        representative = max(
            seeds, key=lambda seed: (metric(seed, "val", "recall_at_p90"), -cast(int, seed["seed"]))
        )
        arms.append(
            {
                "id": "/".join(key[:3]) + "/" + key[3],
                "recipe": key[0],
                "input_hw": seeds[0]["input_hw"],
                "decode": key[1],
                "precision": key[2],
                "val_recall_mean": statistics.mean(
                    metric(seed, "val", "recall_at_p90") for seed in seeds
                ),
                "test_recall_mean": mean_recall,
                "test_recall_std": standard_deviation,
                "test_ap50_mean": statistics.mean(metric(seed, "test", "ap50") for seed in seeds),
                "styles_recall_mean": statistics.mean(
                    metric(seed, "styles", "recall_at_p90") for seed in seeds
                )
                if all(seed.get("styles") is not None for seed in seeds)
                else None,
                "test_operating_recall_mean": statistics.mean(
                    metric(seed, "test", "recall_at_threshold") for seed in seeds
                ),
                "pi5_p95_ms": max(number(seed["pi5_p95_ms"], "latency") for seed in seeds),
                "reaction_p95_ms": max(
                    number(seed["reaction_p95_ms"], "reaction") for seed in seeds
                ),
                "passes": not reasons,
                "failures": [str(reason) for reason in reasons],
                "representative": representative["directory"],
                "seeds": [dict(seed) for seed in seeds],
            }
        )
    enforce_species_gate(arms)
    eligible = [arm for arm in arms if arm["passes"]]
    front = pareto_front(eligible)
    selected = styles_tie_break(eligible)
    return {
        "provisional": True,
        "baseline": {
            "recipe": "baseline-repro-p50",
            "test_mean": baseline_mean,
            "test_std": baseline_std,
        },
        "ranking": "three-seed validation R@P90, latency tie-break; test used for promotion only",
        "false_alarm_policy": "each seed must pass, not just the arm average",
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
        "awaiting": ["round2-distill", "round3-species"],
    }


def remeasure_latency(arguments: argparse.Namespace) -> None:
    if os.environ.get("OMP_WAIT_POLICY", "").upper() != "ACTIVE":
        raise ValueError("latency gate must run with OMP_WAIT_POLICY=ACTIVE, matching deployment")
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
        reaction = simulate_reaction(
            samples,
            ReactionSettings(recall=max(0.001, number(test["recall_at_threshold"], "recall"))),
        )
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
                str(threshold["value"]),
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
    subprocess.run(
        [
            "rsync",
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
                "-a",
                str(output / "references" / arguments.recipe),
                f"columba-cluster:{workspace}/references/",
            ],
            check=True,
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
    if [[ -d "$stage/exports" ]]; then
        rsync -a "$stage/exports/" "$workspace/exports/"
    fi
    rm -rf -- "$stage"
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
    taskset -c 0-3 {evaluation} --seed "$seed" \\
        --checkpoint "$HOME/columba/runs/{arguments.recipe}/seed$seed/weights/best.pt" \\
        --exports "$stage/exports" --output "$workspace/results" \\
        --dataset "$stage/{arguments.dataset_name}" --test-dataset "$stage/columba-test-v2" \\
        --prefix "$HOME/columba/scratch/ws4/ncnn-xcng"
done
date -u > "$workspace/DONE"
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
        "promotion_status": "dry-run-only",
        "selection": str(arguments.decision),
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
    print(prepare_models(repository, model_id))


def write_decision_report(path: Path, decision: dict[str, JsonValue]) -> None:
    arms = [mapping(arm, "arm") for arm in cast(list[JsonValue], decision["arms"])]
    lines = [
        "# WS7 provisional export decision",
        "",
        f"Provisional selection: `{decision['selected'] or 'HOLD: no arm passes every gate'}`.",
        "Final selection waits for distilled and species candidates.",
        "Styles is a secondary measured column. Test-v2 still supplies every promotion",
        "gate. Paired v2/v2s arms within 0.01 validation recall and 5% measured latency",
        "use styles recall to break the tie. The 5% band treats host timing noise",
        "as equal latency.",
        "",
        "Accuracy is measured using columba_native package-default compute on xcng.",
        "These 16:9 deploy-size measurements decide the section-1 and false-alarm",
        "gates. WS5 square-input leaderboard scores are context, not deployment scores.",
        "Latency uses OMP_WAIT_POLICY=ACTIVE. PASSIVE results are retained as diagnostics.",
        "Pi 5 latency is estimated with WS4 cal-v0-published, xcng times x 1.527110713.",
        "Reaction is simulated by WS4 with 30 FPS, k2/n3, 20,000 trials, seed 42,",
        "independent hits and per-frame recall proxied by test recall at each export's",
        "fresh validation-selected threshold. Temporal correlation is unmeasured.",
        "",
        "Arms rank by three-seed mean validation R@P90; test is a promotion gate.",
        "Every seed must pass false-alarm, parity, precision-accuracy and time gates.",
        "FP16 and int8 each permit at most 0.01 loss in val/test R@P90, AP50 and",
        "operating recall versus same-head rectangular PyTorch. This is an explicit",
        "WS7 accuracy policy. Parity independently retains count equality and",
        "compare_exports.py IoU >=0.8 / score delta <=0.05 for fp32/fp16,",
        "documented IoU >=0.7 / score delta <=0.10 for int8, at confidence 0.25.",
        "",
        "Calibration uses 200 deterministic val images, centred deployment letterbox,",
        "RGB /255 and padding 114. Selected SHA-256s are checked against frozen test.",
        "No calibration image or operating threshold comes from test.",
        "",
        "## Arm results",
        "",
        "Accuracy values below are measured on xcng; frame p95 is estimated Pi 5;",
        "reaction p95 is simulated. Input sizes are H x W.",
        "",
        (
            "| Arm / decode / precision / input | Val R@P90 | Test R@P90 mean +/- sd | "
            "Test AP50 | Test operating R | Styles R@P90 | Pi frame p95 ms | "
            "Reaction p95 ms | Pass |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in arms:
        lines.append(
            f"| {arm['id']} | {number(arm['val_recall_mean'], 'val'):.4f} | "
            f"{number(arm['test_recall_mean'], 'test'):.4f} +/- "
            f"{number(arm['test_recall_std'], 'std'):.4f} | "
            f"{number(arm['test_ap50_mean'], 'AP'):.4f} | "
            f"{number(arm['test_operating_recall_mean'], 'recall'):.4f} | "
            f"{arm['styles_recall_mean']} | "
            f"{number(arm['pi5_p95_ms'], 'latency'):.2f} | "
            f"{number(arm['reaction_p95_ms'], 'reaction'):.2f} | "
            f"{'yes' if arm['passes'] else 'no'} |"
        )
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
                "Empty FAR | Confuser FAR | Val threshold | Parity failed images |"
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
                f"{seed['empty_far']} | {seed['confuser_far']} | "
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
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


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
        help="optional Mac C1 root for CPU reference precomputation",
    )
    sub.add_argument(
        "--decodes", nargs="+", choices=("yolo11", "yolo26"), default=["yolo11", "yolo26"]
    )
    sub.add_argument("--output", type=Path, default=Path("outputs/ws7"))
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
    sub.add_argument("--exports", type=Path, required=True)
    sub = commands.add_parser("select")
    sub.add_argument(
        "candidates",
        nargs="+",
        type=Path,
        help="candidate directories or roots, searched recursively",
    )
    sub.add_argument("--report", type=Path)
    sub.add_argument("--leaderboard", type=Path, default=Path("experiments/leaderboard.csv"))
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
    elif arguments.action == "package":
        package_candidate(arguments)
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
        decision = select_candidates(directories, baseline_scores(arguments.leaderboard))
        write_json(arguments.output, decision)
        if arguments.report:
            write_decision_report(arguments.report, decision)
        print(
            json.dumps({"selected": decision["selected"], "pareto_front": decision["pareto_front"]})
        )


if __name__ == "__main__":
    main()
