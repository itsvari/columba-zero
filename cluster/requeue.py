"""Bound checkpoint restarts without a polling process."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
from pathlib import Path
from typing import cast

from app.json_data import JsonValue
from cluster.artifacts import atomic_json, read_json

MAX_REQUEUES = 10
REQUEUE_EXIT_STATUS = 75


class RequeueRequested(InterruptedError):
    pass


def requeue_identifier(job_id: str, array_id: str | None, task_id: str | None) -> str:
    if not job_id.isdecimal():
        raise ValueError("invalid Slurm job id")
    if array_id is None and task_id is None:
        return job_id
    if array_id is None or task_id is None or not array_id.isdecimal() or not task_id.isdecimal():
        raise ValueError("invalid Slurm array identity")
    return f"{array_id}_{task_id}"


def update_ledger(directory: Path, job_id: str, restart_count: int, reason: str) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".requeue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "DONE").exists():
            return False
        path = directory / "requeue.json"
        ledger = read_json(path) if path.exists() else {}
        report = read_json(directory / "run.json") if (directory / "run.json").exists() else {}
        epoch = cast(int, report.get("epochs_done", 0))
        previous_epoch = cast(int, ledger.get("last_epoch", 0))
        count = cast(int, ledger.get("count", 0))
        recorded = (
            cast(int, ledger.get("recorded_restart_count", 0))
            if ledger.get("job_id") == job_id
            else 0
        )
        device_path = directory / "device_faults.json"
        device_events = (
            cast(list[dict[str, JsonValue]], read_json(device_path).get("events", []))
            if device_path.exists()
            else []
        )
        device_restart = any(
            entry.get("job_id") == job_id
            and entry.get("expected_restart_count") == restart_count
            and entry.get("status") in {"reserved", "requeued", "repair_pending"}
            for entry in device_events
        )
        if reason == "start" and device_restart and restart_count > recorded:
            if (directory / "HOLD").exists():
                raise RuntimeError((directory / "HOLD").read_text())
            atomic_json(path, {**ledger, "job_id": job_id, "recorded_restart_count": restart_count})
            return False
        if reason == "start" and restart_count <= recorded:
            if (directory / "HOLD").exists():
                raise RuntimeError((directory / "HOLD").read_text())
            return False
        increment = 1 if reason != "start" else restart_count - recorded
        failure = None
        if count + increment > MAX_REQUEUES:
            failure = f"requeue limit {MAX_REQUEUES} reached"
        elif epoch <= previous_epoch or not (directory / "weights/last.pt").is_file():
            failure = f"no saved epoch progressed since requeue: {previous_epoch} -> {epoch}"
        if failure:
            (directory / "HOLD").write_text(failure + "\n")
            atomic_json(path, {**ledger, "hold": failure, "observed_epoch": epoch})
            raise RuntimeError(failure)
        entries = cast(list[JsonValue], ledger.get("entries", []))
        entry: dict[str, JsonValue] = {
            "job_id": job_id,
            "epoch": epoch,
            "reason": reason,
            "restart_count": restart_count,
        }
        atomic_json(
            path,
            {
                "count": count + increment,
                "last_epoch": epoch,
                "job_id": job_id,
                "recorded_restart_count": restart_count + (reason != "start"),
                "entries": [*entries, entry],
            },
        )
        return reason != "start"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "signal"))
    parser.add_argument("--run-path", type=Path, required=True)
    arguments = parser.parse_args()
    if not os.environ.get("SLURM_JOB_NAME", "").startswith("cz-"):
        raise ValueError("requeue requires a cz- allocation")
    if not arguments.run_path.resolve().is_relative_to((Path.home() / "columba/runs").resolve()):
        raise ValueError("requeue ledger must be inside ~/columba/runs")
    job_id = os.environ["SLURM_JOB_ID"]
    target = requeue_identifier(
        job_id, os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    )
    requeue = update_ledger(
        arguments.run_path,
        job_id,
        int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        arguments.action,
    )
    if requeue:
        subprocess.run(["scontrol", "requeue", target], check=True)


if __name__ == "__main__":
    main()
