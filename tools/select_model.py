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
import time
from collections import defaultdict
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
        "compute": detector.compute,
        "compute_flags": detector.compute_flags,
        "camera_hw": [480, 640],
        "confidence": arguments.confidence,
        "weights_sha256": fingerprint(arguments.model / "model.ncnn.bin"),
        "parameters_sha256": fingerprint(arguments.model / "model.ncnn.param"),
        "end_to_end_boundary": (
            "native letterbox, inference, decode/NMS, Python return and bool decision"
        ),
        "frame_source": "first 20 validation photographs resized to camera shape, cycling",
    }
    write_json(arguments.out, apply_calibration(report, read_report(arguments.calibration)))


def evaluate_candidate(arguments: argparse.Namespace) -> None:
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
    if native.get("framework") != "columba_native" or not str(native.get("host", "")).startswith(
        "xcng"
    ):
        raise ValueError(f"{directory}: deployed native xcng accuracy is required")
    if latency.get("backend") != "columba_native":
        raise ValueError(f"{directory}: native end-to-end C7 is required")
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
        "parity_passed": read_report(directory / "parity.json")["passed"],
        "accuracy_gate_passed": read_report(directory / "accuracy_gate.json")["passed"],
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
                "decode": key[1],
                "precision": key[2],
                "val_recall_mean": statistics.mean(
                    metric(seed, "val", "recall_at_p90") for seed in seeds
                ),
                "test_recall_mean": mean_recall,
                "test_recall_std": standard_deviation,
                "test_ap50_mean": statistics.mean(metric(seed, "test", "ap50") for seed in seeds),
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
    eligible = [arm for arm in arms if arm["passes"]]
    front = [
        arm["id"]
        for arm in eligible
        if not any(
            number(other["val_recall_mean"], "recall") >= number(arm["val_recall_mean"], "recall")
            and number(other["pi5_p95_ms"], "latency") <= number(arm["pi5_p95_ms"], "latency")
            and (
                other["val_recall_mean"] != arm["val_recall_mean"]
                or other["pi5_p95_ms"] != arm["pi5_p95_ms"]
            )
            for other in eligible
        )
    ]
    selected = max(
        eligible,
        key=lambda arm: (
            number(arm["val_recall_mean"], "recall"),
            -number(arm["pi5_p95_ms"], "latency"),
        ),
        default=None,
    )
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
        "pareto_front": front,
        "selected": selected["id"] if selected else None,
        "package_candidate": selected["representative"] if selected else None,
        "awaiting": ["round2-distill", "round3-species"],
    }


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
    for name in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml"):
        shutil.copy2(source / name, destination / name)
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


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="action", required=True)
    for action in ("export", "evaluate"):
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
            sub.add_argument("--dataset", type=Path, required=True)
            sub.add_argument("--test-dataset", type=Path, required=True)
            sub.add_argument("--prefix", type=Path, required=True)
            sub.add_argument(
                "--calibration", type=Path, default=Path("experiments/latency/calibration.json")
            )
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
    sub.add_argument("--leaderboard", type=Path, default=Path("experiments/leaderboard.csv"))
    sub.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    arguments = parser().parse_args()
    if arguments.action == "export":
        export_candidate(arguments)
    elif arguments.action == "evaluate":
        evaluate_candidate(arguments)
    elif arguments.action == "predict-native":
        native_predictions(arguments)
    elif arguments.action == "benchmark-native":
        benchmark_native(arguments)
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
        print(
            json.dumps({"selected": decision["selected"], "pareto_front": decision["pareto_front"]})
        )


if __name__ == "__main__":
    main()
