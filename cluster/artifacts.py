"""Atomic C6 persistence and leaderboard updates."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from app.json_data import JsonValue

LEADERBOARD_COLUMNS = (
    "recipe_id",
    "seed",
    "status",
    "dataset",
    "taxonomy",
    "model",
    "train_imgsz",
    "host",
    "gpu",
    "epochs_done",
    "val_recall_at_p90",
    "test_recall_at_p90",
    "test_small_recall_at_p90",
    "test_ap50",
    "test_ap50_95",
    "test_empty_far",
    "test_confuser_far",
    "operating_threshold",
    "git_commit",
    "run_path",
)
REQUIRED_ARTIFACTS = (
    "run.json",
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "predictions_val.json",
    "predictions_val.meta.json",
    "predictions_test.json",
    "predictions_test.meta.json",
    "metrics_val.json",
    "metrics_test.json",
)
OPTIONAL_ARTIFACTS = ("TRAINED",)


def atomic_json(path: Path, value: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, JsonValue]:
    value: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def copy_artifacts(source: Path, destination: Path) -> None:
    for relative_path in (*REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS):
        original = source / relative_path
        if not original.is_file():
            continue
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        shutil.copy2(original, temporary)
        temporary.replace(target)


def mark_done(directory: Path) -> None:
    missing = [name for name in REQUIRED_ARTIFACTS if not (directory / name).is_file()]
    report = read_json(directory / "run.json")
    if missing or report.get("evaluation") != "complete" or report.get("exit_status") != 0:
        raise RuntimeError(f"cannot write DONE: incomplete C6 artifacts {missing}")
    temporary = directory / ".DONE.tmp"
    temporary.write_text("complete\n", encoding="utf-8")
    temporary.replace(directory / "DONE")


def metric_value(metrics: Mapping[str, JsonValue], key: str, group: str | None = None) -> str:
    source = metrics.get(group, {}) if group else metrics
    if not isinstance(source, dict):
        return ""
    value = source.get(key)
    return "" if value is None else str(value)


def leaderboard_row(directory: Path) -> dict[str, str]:
    report = read_json(directory / "run.json")
    validation = (
        read_json(directory / "metrics_val.json")
        if (directory / "metrics_val.json").exists()
        else {}
    )
    test = (
        read_json(directory / "metrics_test.json")
        if (directory / "metrics_test.json").exists()
        else {}
    )
    row = {column: str(report.get(column, "")) for column in LEADERBOARD_COLUMNS}
    row.update(
        {
            "status": str(report.get("status", "failed")),
            "run_path": str(directory),
            "val_recall_at_p90": metric_value(validation, "recall_at_p90", "pigeon"),
            "test_recall_at_p90": metric_value(test, "recall_at_p90", "pigeon"),
            "test_small_recall_at_p90": metric_value(test, "small_recall_at_p90", "pigeon"),
            "test_ap50": metric_value(test, "ap50", "pigeon"),
            "test_ap50_95": metric_value(test, "ap50_95", "pigeon"),
            "test_empty_far": metric_value(test, "empty_false_alarm_rate"),
            "test_confuser_far": metric_value(test, "confuser_false_alarm_rate"),
            "operating_threshold": metric_value(validation, "value", "operating_threshold"),
        }
    )
    return row


def update_leaderboard(path: Path, rows: list[dict[str, str]]) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing: list[dict[str, str]] = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != list(LEADERBOARD_COLUMNS):
                    raise ValueError("leaderboard header does not match C6")
                existing = list(reader)
        indexed = {(row["recipe_id"], row["seed"]): row for row in existing}
        indexed.update({(row["recipe_id"], row["seed"]): row for row in rows})
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=LEADERBOARD_COLUMNS)
            writer.writeheader()
            writer.writerows(cast(list[dict[str, str]], list(indexed.values())))
        temporary.replace(path)
