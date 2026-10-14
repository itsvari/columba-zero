"""Execute a C5 recipe/seed and publish honest C6 artifacts."""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Protocol, TypedDict, cast

from app.json_data import JsonValue
from cluster.artifacts import atomic_json, copy_artifacts, mark_done, read_json
from cluster.hardware import configure_ultralytics, describe_hardware, select_batch
from cluster.recipe import Recipe, load_recipe, recipe_sha256, resolve_teacher
from cluster.requeue import REQUEUE_EXIT_STATUS, RequeueRequested
from cluster.staging import sha256_file, stage_dataset, stage_weights

CHECKPOINT_PERIOD_EPOCHS = 5
NOMINAL_BATCH_SIZE = 64
ARM_BATCH_SIZE = 8


class Checkpoint(TypedDict):
    epoch: int
    train_args: dict[str, JsonValue]


class Trainer(Protocol):
    epoch: int
    batch_size: int
    epoch_time: float
    save_dir: Path


class TrainingModel(Protocol):
    def add_callback(self, event: str, callback: Callable[[Trainer], None]) -> None: ...
    def train(self, **arguments: JsonValue) -> None: ...


@dataclass(frozen=True)
class TrainingContext:
    recipe: Recipe
    seed: int
    dataset: Path
    base_weights: Path
    teacher: Path | None
    local_run: Path
    persistent_run: Path
    batch_size: int
    device: str
    stop_after_epoch: int | None = None
    requeue_requested: Callable[[], bool] = lambda: False


class Adapter(Protocol):
    def train(self, context: TrainingContext, report: dict[str, JsonValue]) -> None: ...


def interrupt_job(signal_number: int, frame: FrameType | None) -> None:
    raise InterruptedError(f"job received signal {signal_number}")


def now() -> str:
    return datetime.now(UTC).isoformat()


def checkpoint_complete(checkpoint: Path, target_epochs: int) -> bool:
    import torch

    saved = cast(Checkpoint, torch.load(checkpoint, map_location="cpu", weights_only=False))
    epoch = saved.get("epoch", -1)
    return isinstance(epoch, int) and (epoch == -1 or epoch + 1 >= target_epochs)


def relocate_checkpoint(checkpoint: Path, context: TrainingContext) -> dict[str, JsonValue]:
    import torch

    saved = cast(Checkpoint, torch.load(checkpoint, map_location="cpu", weights_only=False))
    arguments = saved.get("train_args")
    if not isinstance(arguments, dict):
        raise ValueError("checkpoint has no train_args")
    previous_paths = {
        "previous_data": arguments.get("data"),
        "previous_distill_model": arguments.get("distill_model"),
    }
    arguments.update(
        {
            "data": str(context.dataset / "data.yaml"),
            "project": str(context.local_run.parent),
            "name": context.local_run.name,
            "save_dir": str(context.local_run),
            "distill_model": str(context.teacher) if context.teacher else None,
        }
    )
    torch.save(saved, checkpoint)
    return {
        **previous_paths,
        "data": arguments["data"],
        "distill_model": arguments["distill_model"],
        "checkpoint": str(checkpoint),
    }


def reconcile_completed_training(directory: Path, report: dict[str, JsonValue]) -> None:
    results = directory / "results.csv"
    if not results.is_file():
        raise ValueError("completed checkpoint is missing results.csv")
    with results.open(newline="", encoding="utf-8") as source:
        epochs = [
            int(row["epoch"].strip()) for row in csv.DictReader(source, skipinitialspace=True)
        ]
    report["epochs_done"] = max(epochs, default=0)
    timings = report.get("seconds_per_epoch", [])
    if isinstance(timings, list):
        report["seconds_per_epoch"] = timings[: max(epochs, default=0)]


class UltralyticsAdapter:
    def train(self, context: TrainingContext, report: dict[str, JsonValue]) -> None:
        from ultralytics import YOLO

        copy_artifacts(context.persistent_run, context.local_run)
        checkpoint = context.local_run / "weights/last.pt"
        epochs = cast(int, context.recipe.train["epochs"])
        if checkpoint.exists() and checkpoint_complete(checkpoint, epochs):
            report["training"] = "already complete"
            previous = (
                read_json(context.persistent_run / "run.json")
                if (context.persistent_run / "run.json").is_file()
                else {}
            )
            report["batch_size"] = previous.get("batch_size", report.get("batch_size"))
            reconcile_completed_training(context.local_run, report)
            return
        resuming = checkpoint.exists()
        if resuming:
            report["checkpoint_relocation"] = relocate_checkpoint(checkpoint, context)
        model = cast(TrainingModel, YOLO(str(checkpoint if resuming else context.base_weights)))

        def persist(trainer: Trainer) -> None:
            report["epochs_done"] = trainer.epoch + 1
            report["batch_size"] = trainer.batch_size
            report["heartbeat_at"] = now()
            atomic_json(context.local_run / "run.json", report)
            copy_artifacts(context.local_run, context.persistent_run)

        def record_epoch(trainer: Trainer) -> None:
            # final_eval fires this callback again with an incremented epoch and no new checkpoint.
            if trainer.epoch + 1 != report.get("epochs_done"):
                return
            timings = report.setdefault("seconds_per_epoch", [])
            if isinstance(timings, list):
                timings.append(trainer.epoch_time)
            persist(trainer)
            if context.stop_after_epoch == trainer.epoch + 1:
                raise InterruptedError(f"smoke stopped after saved epoch {trainer.epoch + 1}")
            forced_epoch = os.environ.get("COLUMBA_SIGNAL_AFTER_EPOCH")
            if (
                forced_epoch
                and context.recipe.identifier.startswith("smoke-")
                and int(os.environ.get("SLURM_RESTART_COUNT", "0")) == 0
                and trainer.epoch + 1 == int(forced_epoch)
            ):
                os.kill(os.getpid(), signal.SIGUSR1)
            if context.requeue_requested():
                raise RequeueRequested("checkpoint saved for Slurm requeue")

        model.add_callback("on_model_save", persist)
        model.add_callback("on_fit_epoch_end", record_epoch)
        arguments: dict[str, JsonValue] = {
            **context.recipe.train,
            "nbs": NOMINAL_BATCH_SIZE,
            "batch": context.batch_size,
            "seed": context.seed,
            "data": str(context.dataset / "data.yaml"),
            "device": context.device,
            "project": str(context.local_run.parent),
            "name": context.local_run.name,
            "save_dir": str(context.local_run),
            "exist_ok": True,
            "save": True,
            "save_period": context.recipe.train.get("save_period", CHECKPOINT_PERIOD_EPOCHS),
            "distill_model": str(context.teacher) if context.teacher else None,
        }
        if resuming:
            arguments["resume"] = str(checkpoint)
        report["resumed"] = resuming
        model.train(**arguments)
        (context.local_run / "TRAINED").write_text("complete\n", encoding="utf-8")
        report["training"] = "complete"
        reconcile_completed_training(context.local_run, report)
        copy_artifacts(context.local_run, context.persistent_run)


def adapter_for(framework: str) -> Adapter:
    if framework == "ultralytics":
        return UltralyticsAdapter()
    module = importlib.import_module(f"training.adapters.{framework}")
    return cast(Adapter, module.adapter)


def evaluation_commands(
    context: TrainingContext, test_dataset: Path, repository: Path
) -> list[list[str]]:
    commands: list[list[str]] = []
    for split, dataset in (("val", context.dataset), ("test", test_dataset)):
        predictions = context.local_run / f"predictions_{split}.json"
        metrics = context.local_run / f"metrics_{split}.json"
        commands.append(
            [
                sys.executable,
                str(repository / "tools/predict_to_coco.py"),
                "--model",
                str(context.local_run / "weights/best.pt"),
                "--dataset",
                str(dataset),
                "--split",
                split,
                "--out",
                str(predictions),
                "--imgsz",
                str(context.recipe.train["imgsz"]),
                "--conf",
                "0.001",
                "--iou",
                "0.7",
                "--device",
                context.device,
                "--model-id",
                f"{context.recipe.identifier}/seed{context.seed}",
                "--host",
                str(os.uname().nodename),
            ]
        )
        threshold_arguments = (
            ["--select-threshold"]
            if split == "val"
            else ["--threshold-from", str(context.local_run / "metrics_val.json")]
        )
        commands.append(
            [
                sys.executable,
                str(repository / "tools/evaluate_predictions.py"),
                "--dataset",
                str(dataset),
                "--split",
                split,
                "--predictions",
                str(predictions),
                "--out",
                str(metrics),
                *threshold_arguments,
            ]
        )
    return commands


def evaluate(context: TrainingContext, test_dataset: Path, repository: Path) -> str:
    missing = [
        name
        for name in ("predict_to_coco.py", "evaluate_predictions.py")
        if not (repository / "tools" / name).is_file()
    ]
    if missing:
        if context.recipe.allow_missing_evaluation:
            return "skipped: tools missing"
        raise FileNotFoundError(f"evaluation tools missing: {missing}")
    for command in evaluation_commands(context, test_dataset, repository):
        subprocess.run(command, check=True)
    return "complete"


def validate_existing_run(directory: Path, recipe: Recipe, checksum: str) -> None:
    if not (directory / "run.json").is_file():
        return
    existing = read_json(directory / "run.json")
    if existing.get("recipe_sha256") != checksum or existing.get("dataset") != recipe.dataset:
        raise ValueError("recipe changed under an existing run; use a new recipe id")


def run(
    recipe_path: Path, seed: int, root: Path, repository: Path, stop_after_epoch: int | None = None
) -> int:
    import fcntl

    recipe = load_recipe(recipe_path)
    recipe.constraint_for_seed(seed)
    if stop_after_epoch is not None:
        if not recipe.identifier.startswith("smoke-") or not 0 < stop_after_epoch < cast(
            int, recipe.train["epochs"]
        ):
            raise ValueError("--stop-after-epoch is restricted to unfinished smoke epochs")
    if not root.resolve().is_relative_to((Path.home() / "columba").resolve()):
        raise ValueError("cluster storage must be inside ~/columba")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdecimal() or not os.environ.get("SLURM_JOB_NAME", "").startswith("cz-"):
        raise ValueError("runner requires a cz- Slurm job")
    runs_root = Path(os.environ.get("COLUMBA_RUNS_ROOT", str(root / "runs"))).resolve()
    if not runs_root.is_relative_to(root.resolve()):
        raise ValueError("COLUMBA_RUNS_ROOT must be inside COLUMBA_ROOT")
    persistent = runs_root / recipe.identifier / f"seed{seed}"
    persistent.mkdir(parents=True, exist_ok=True)
    with (persistent / ".run.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checksum = recipe_sha256(recipe_path)
        validate_existing_run(persistent, recipe, checksum)
        if (persistent / "DONE").is_file():
            return 0
        return execute(
            recipe, seed, root, repository, persistent, checksum, job_id, stop_after_epoch
        )


def publish_report(local_run: Path, persistent: Path, report: dict[str, JsonValue]) -> int:
    atomic_json(local_run / "run.json", report)
    copy_artifacts(local_run, persistent)
    if report["status"] != "complete":
        return cast(int, report["exit_status"])
    try:
        mark_done(persistent)
        return 0
    except RuntimeError as error:
        report.update({"status": "failed", "exit_status": 1, "error": str(error)})
        atomic_json(persistent / "run.json", report)
        return 1


def execute(
    recipe: Recipe,
    seed: int,
    root: Path,
    repository: Path,
    persistent: Path,
    checksum: str,
    job_id: str,
    stop_after_epoch: int | None = None,
) -> int:
    scratch = Path("/tmp") / f"cz-{job_id}"
    local_run = scratch / "runs" / recipe.identifier / f"seed{seed}"
    local_run.mkdir(parents=True, exist_ok=True)
    configure_ultralytics(scratch)
    previous = read_json(persistent / "run.json") if (persistent / "run.json").exists() else {}
    report: dict[str, JsonValue] = {
        **previous,
        "recipe_id": recipe.identifier,
        "recipe_sha256": checksum,
        "seed": seed,
        "dataset": recipe.dataset,
        "taxonomy": recipe.taxonomy,
        "model": recipe.base_model,
        "train_imgsz": recipe.train["imgsz"],
        "framework": recipe.framework,
        "git_commit": os.environ.get("COLUMBA_GIT_COMMIT", "unknown"),
        "source_snapshot": repository.name,
        "host": platform.node(),
        "python_executable": sys.executable,
        "venv_path": sys.prefix,
        "job_id": job_id,
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "start": previous.get("start", now()),
        "end": None,
        "status": "running",
        "epochs_done": previous.get("epochs_done", 0),
        "exit_status": None,
        "evaluation": "pending",
        "timing_label": "measured",
        "attempt_started_at": now(),
        "error": None,
    }
    started_s = time.monotonic()
    requeue_requested = threading.Event()
    if os.environ.get("COLUMBA_REQUEUE_REQUESTED") == "true":
        requeue_requested.set()
    signal.signal(signal.SIGUSR1, lambda signal_number, frame: requeue_requested.set())
    signal.signal(signal.SIGTERM, interrupt_job)
    signal.signal(signal.SIGINT, interrupt_job)
    exit_status = 1
    try:
        report.update(describe_hardware(recipe.lane == "nv"))
        atomic_json(persistent / "run.json", report)
        dataset, dataset_checksum = stage_dataset(recipe.dataset, root / "data", scratch / "data")
        test_dataset, test_checksum = stage_dataset(
            recipe.test_dataset, root / "data", scratch / "data"
        )
        report.update(
            {
                "dataset_archive_sha256": dataset_checksum,
                "test_dataset": recipe.test_dataset,
                "test_dataset_archive_sha256": test_checksum,
            }
        )
        weights = stage_weights(recipe.base_model, root / "weights", scratch / "weights")
        report["base_model_sha256"] = sha256_file(weights)
        if recipe.framework == "ultralytics" and recipe.train.get("amp", True):
            stage_weights("yolo26n.pt", root / "weights", scratch / "weights")
        teacher_source = resolve_teacher(recipe.teacher, persistent.parents[1])
        teacher = None
        if teacher_source:
            teacher = (scratch / "weights/teacher.pt").resolve()
            shutil.copy2(teacher_source, teacher)
            report["teacher_sha256"] = sha256_file(teacher)
        report["data_yaml"] = str(dataset / "data.yaml")
        report["teacher"] = recipe.teacher
        report["distill_model"] = str(teacher) if teacher else None
        memory_gb = cast(float, report["gpu_memory_gb"])
        batch_size = select_batch(
            memory_gb, recipe.base_model, cast(int, recipe.train["imgsz"]), teacher is not None
        )
        if recipe.lane == "arm":
            batch_size = ARM_BATCH_SIZE
        elif recipe.batch_override is not None:
            batch_size = recipe.batch_override
        report["batch_size"] = batch_size
        report["initial_batch_size"] = batch_size
        context = TrainingContext(
            recipe,
            seed,
            dataset,
            weights,
            teacher,
            local_run,
            persistent,
            batch_size,
            "0" if recipe.lane == "nv" else "cpu",
            stop_after_epoch,
            requeue_requested.is_set,
        )
        adapter_for(recipe.framework).train(context, report)
        report["evaluation"] = evaluate(context, test_dataset, repository)
        report["status"] = (
            "complete" if report["evaluation"] == "complete" else "evaluation_missing"
        )
        exit_status = 0
    except RequeueRequested as error:
        report["status"] = "requeue_pending"
        report["error"] = str(error)
        exit_status = REQUEUE_EXIT_STATUS
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr)
    finally:
        report.update(
            {
                "end": now(),
                "exit_status": exit_status,
                "attempt_wall_time_s": time.monotonic() - started_s,
            }
        )
        attempt: dict[str, JsonValue] = {
            key: report.get(key)
            for key in (
                "job_id",
                "host",
                "attempt_started_at",
                "end",
                "status",
                "exit_status",
                "error",
                "epochs_done",
                "resumed",
                "batch_size",
                "architecture",
                "gpu",
                "data_yaml",
                "distill_model",
                "checkpoint_relocation",
            )
        }
        attempts = report.get("attempts", [])
        report["attempts"] = [*(attempts if isinstance(attempts, list) else []), attempt]
        try:
            exit_status = publish_report(local_run, persistent, report)
        finally:
            shutil.rmtree(scratch)
    return exit_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stop-after-epoch", type=int)
    arguments = parser.parse_args()
    root = Path(os.environ.get("COLUMBA_ROOT", str(Path.home() / "columba")))
    raise SystemExit(
        run(
            arguments.recipe.resolve(), arguments.seed, root, Path.cwd(), arguments.stop_after_epoch
        )
    )


if __name__ == "__main__":
    main()
