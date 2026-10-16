"""Bundle ARM exports or WS4 benchmarks into one low-priority job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Protocol, cast

from app.json_data import JsonValue
from cluster.artifacts import atomic_json
from cluster.hardware import configure_ultralytics, describe_hardware
from cluster.recipe import resolve_teacher


class ExportModel(Protocol):
    def export(self, **arguments: JsonValue) -> str: ...


def export_model(
    model_id: str, root: Path, scratch: Path, destination: Path
) -> dict[str, JsonValue]:
    from ultralytics import YOLO

    source = resolve_teacher(model_id, root / "runs")
    if source is None:
        raise ValueError("model reference required")
    weights = scratch / "weights" / model_id / "best.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, weights)
    started_s = time.monotonic()
    exported = Path(
        cast(ExportModel, YOLO(str(weights.resolve()))).export(
            format="ncnn", imgsz=[384, 640], device="cpu", quantize=16
        )
    )
    output = destination / model_id
    shutil.copytree(exported, output, dirs_exist_ok=True)
    return {
        "model_id": model_id,
        "export": str(output),
        "wall_time_s": time.monotonic() - started_s,
        "input_hw": [384, 640],
        "status": "complete",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", required=True, help="JSON list of C5 teacher-style model references"
    )
    parser.add_argument("--action", choices=("export", "benchmark"), required=True)
    arguments = parser.parse_args()
    models = cast(list[str], json.loads(arguments.models))
    root = Path.home() / "columba"
    for model in models:
        resolve_teacher(model, root / "runs")
    digest = hashlib.sha256(json.dumps([arguments.action, *models]).encode()).hexdigest()[:12]
    destination = root / "runs/arm-bundles" / digest
    destination.mkdir(parents=True, exist_ok=True)
    scratch = Path("/tmp") / f"cz-{os.environ['SLURM_JOB_ID']}"
    configure_ultralytics(scratch)
    report: dict[str, JsonValue] = {
        "status": "running",
        "models": cast(list[JsonValue], models),
        "action": arguments.action,
        "label": "measured",
    }
    exit_status = 1
    try:
        report.update(describe_hardware(False))
        atomic_json(destination / "run.json", report)
        if arguments.action == "export":
            report["results"] = [
                cast(JsonValue, export_model(model, root, scratch, destination)) for model in models
            ]
        else:
            benchmark = Path.cwd() / "tools/bench_ncnn.py"
            if not benchmark.is_file():
                raise FileNotFoundError("tools/bench_ncnn.py is missing from the checkout")
            command_file = root / "benchmark-commands.json"
            commands = cast(list[list[str]], json.loads(command_file.read_text()))
            for command in commands:
                if not command or Path(command[0]).name != "bench_ncnn.py":
                    raise ValueError("bundle commands must target tools/bench_ncnn.py")
                subprocess.run([sys.executable, str(benchmark), *command[1:]], check=True)
        report["status"] = "complete"
        exit_status = 0
    except Exception as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        print(report["error"], file=sys.stderr)
    finally:
        report["exit_status"] = exit_status
        atomic_json(destination / "run.json", report)
        shutil.rmtree(scratch, ignore_errors=True)
    if exit_status == 0:
        (destination / "DONE").write_text("complete\n", encoding="utf-8")
    raise SystemExit(exit_status)


if __name__ == "__main__":
    main()
