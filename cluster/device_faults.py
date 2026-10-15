"""Quarantine invisible GPUs and repair their Slurm tasks without checkpoint retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from app.json_data import JsonValue
from cluster.artifacts import atomic_json, read_json
from cluster.locking import submission_lock
from cluster.requeue import requeue_identifier

REPOSITORY = Path(__file__).resolve().parents[1]
DEVICE_FAILURE_EXIT_STATUS = 6
MAX_ERROR_CHARACTERS = 4000
MAX_FAILURE_LOG_BYTES = 64 * 1024
NODE_PATTERN = re.compile(r"xgp[d-k][0-9]+|xcng[01]")
FAULT_PATTERNS = (
    "no devices were found",
    "no cuda-capable device",
    "no cuda gpus are available",
    "cuda driver initialization failed",
    "cuda initialization error",
    "cuda initialization failed",
    "cuda error: initialization error",
    "cuda unknown error",
    "cuda error: unknown error",
    "cuda driver error: initialization error",
    "torch.cuda.is_available() is false",
    "gpu visibility count",
    "nvidia-smi unavailable",
)


class DeviceVisibilityError(RuntimeError):
    pass


def policy() -> tuple[float, int]:
    values = read_json(REPOSITORY / "cluster/device_policy.json")
    hours, retries = values.get("bad_node_ttl_hours"), values.get("max_retries_per_run")
    if (
        type(hours) not in {int, float}
        or cast(float, hours) <= 0
        or type(retries) is not int
        or retries < 1
    ):
        raise ValueError("device policy requires positive TTL hours and retry count")
    return float(cast(float, hours)), retries


def recognised_fault(text: str) -> bool:
    return any(pattern in text.lower() for pattern in FAULT_PATTERNS)


def active_bad_nodes(root: Path, now: datetime | None = None) -> tuple[str, ...]:
    path = root / "state/bad_nodes.json"
    if not path.exists():
        return ()
    entries = cast(dict[str, dict[str, JsonValue]], read_json(path).get("nodes", {}))
    instant = now or datetime.now(UTC)
    return tuple(
        sorted(
            node
            for node, entry in entries.items()
            if NODE_PATTERN.fullmatch(node)
            and not entry.get("simulated")
            and datetime.fromisoformat(cast(str, entry["expires_at"])) > instant
        )
    )


def combined_exclusions(
    root: Path, extra: Sequence[str] = (), static_path: Path | None = None
) -> tuple[str, ...]:
    path = static_path or REPOSITORY / "cluster/excluded_nodes.txt"
    static = (
        [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    nodes = tuple(sorted(set(filter(None, [*static, *extra, *active_bad_nodes(root)]))))
    if any(not NODE_PATTERN.fullmatch(node) for node in nodes):
        raise ValueError("invalid project node exclusion")
    return nodes


def expected_gpu_count() -> int:
    value = os.environ.get("SLURM_GPUS_ON_NODE", "1")
    if not value.isdecimal() or int(value) < 1:
        raise ValueError("SLURM_GPUS_ON_NODE must specify a positive count")
    return int(value)


def check_visibility(expected_count: int, nvidia_only: bool = False) -> dict[str, JsonValue]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise DeviceVisibilityError(f"nvidia-smi unavailable: {error}") from error
    identifiers = [
        line for line in result.stdout.splitlines() if line.strip().startswith(("GPU-", "MIG-"))
    ]
    if result.returncode or len(identifiers) < expected_count:
        raise DeviceVisibilityError(
            f"GPU visibility count {len(identifiers)} < {expected_count}, "
            f"nvidia-smi exit {result.returncode}: {result.stdout} {result.stderr}"
        )
    report: dict[str, JsonValue] = {
        "expected_gpu_count": expected_count,
        "nvidia_gpu_count": len(identifiers),
        "nvidia_smi": result.stdout,
    }
    if nvidia_only:
        return report
    import torch

    try:
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
        if not available or count != expected_count:
            raise DeviceVisibilityError(
                f"torch.cuda.is_available() is false or GPU visibility count {count} "
                f"!= {expected_count}; no CUDA-capable device"
            )
        initialise = cast(Callable[[], None], torch.cuda.init)
        initialise()
        for index in range(count):
            torch.cuda.get_device_properties(index)
    except RuntimeError as error:
        if isinstance(error, DeviceVisibilityError) or recognised_fault(str(error)):
            raise DeviceVisibilityError(str(error)) from error
        raise
    return {**report, "cuda_available": available, "cuda_device_count": count}


def reserve_retry(
    root: Path,
    run_path: Path,
    node: str,
    target: str,
    job_id: str,
    restart_count: int,
    error: str,
    simulated: bool = False,
) -> dict[str, JsonValue]:
    if (
        not run_path.resolve().is_relative_to((root / "runs").resolve())
        or not NODE_PATTERN.fullmatch(node)
        or not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", target)
    ):
        raise ValueError("invalid device-fault run, node or task")
    hours, maximum = policy()
    instant = datetime.now(UTC)
    event_id = f"{target}:{restart_count}"
    with submission_lock(root):
        path = run_path / "device_faults.json"
        ledger = read_json(path) if path.exists() else {"retry_count": 0, "events": []}
        events = cast(list[dict[str, JsonValue]], ledger.get("events", []))
        existing = next((entry for entry in events if entry["event_id"] == event_id), None)
        if existing is not None:
            return existing
        count = cast(int, ledger.get("retry_count", 0))
        entry: dict[str, JsonValue] = {
            "event_id": event_id,
            "node": node,
            "time": instant.isoformat(),
            "target": target,
            "job_id": job_id,
            "restart_count": restart_count,
            "expected_restart_count": restart_count + 1,
            "error": error[:MAX_ERROR_CHARACTERS],
            "retry_count": min(count + 1, maximum),
            "status": "reserved" if count < maximum else "exhausted",
            "simulated": simulated,
        }
        bad_path = root / "state/bad_nodes.json"
        bad: dict[str, JsonValue] = (
            read_json(bad_path) if bad_path.exists() else {"version": 1, "nodes": {}}
        )
        nodes = cast(dict[str, JsonValue], bad["nodes"])
        previous = cast(dict[str, JsonValue], nodes.get(node, {}))
        if previous.get("simulated", False) != simulated:
            previous = {}
        nodes[node] = {
            "node": node,
            "time": instant.isoformat(),
            "first_seen": previous.get("first_seen", instant.isoformat()),
            "expires_at": (instant + timedelta(hours=hours)).isoformat(),
            "job": target,
            "error": error[:MAX_ERROR_CHARACTERS],
            "count": cast(int, previous.get("count", 0)) + 1,
            "simulated": simulated,
        }
        atomic_json(bad_path, bad)
        atomic_json(
            path,
            {
                "retry_count": entry["retry_count"],
                "events": cast(list[JsonValue], [*events, entry]),
            },
        )
        report_path = run_path / "run.json"
        report = (
            read_json(report_path)
            if report_path.exists()
            else cast(
                dict[str, JsonValue], json.loads(os.environ.get("COLUMBA_RUN_METADATA", "{}"))
            )
        )
        atomic_json(
            report_path,
            {
                **report,
                "job_id": job_id,
                "array_job_id": target.split("_")[0] if "_" in target else None,
                "array_task_id": target.split("_")[1] if "_" in target else None,
                "epochs_done": report.get("epochs_done", 0),
                "host": node,
                "architecture": platform.machine(),
                "slurm_restart_count": restart_count,
                "status": "device_fault",
                "device_fault_retries": entry["retry_count"],
                "exit_status": DEVICE_FAILURE_EXIT_STATUS,
                "error": error[:MAX_ERROR_CHARACTERS],
                "evaluation": "not run: device visibility failed",
                "end": instant.isoformat(),
            },
        )
        if entry["status"] == "exhausted":
            (run_path / "HOLD").write_text(
                f"device-fault retry budget {maximum} exhausted: {error}\n"
            )
        return entry


def update_event(root: Path, run_path: Path, event_id: str, updates: dict[str, JsonValue]) -> None:
    with submission_lock(root):
        path = run_path / "device_faults.json"
        ledger = read_json(path)
        for entry in cast(list[dict[str, JsonValue]], ledger["events"]):
            if entry["event_id"] == event_id:
                entry.update(updates)
        atomic_json(path, ledger)


def clear_bad_node(root: Path, node: str) -> None:
    with submission_lock(root):
        path = root / "state/bad_nodes.json"
        if path.exists():
            ledger = read_json(path)
            cast(dict[str, JsonValue], ledger["nodes"]).pop(node, None)
            atomic_json(path, ledger)


def owned_job(target: str) -> dict[str, str]:
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", target):
        raise ValueError("invalid job identity")
    result = subprocess.run(
        ["scontrol", "show", "job", target, "-o"], check=True, text=True, capture_output=True
    )
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise ValueError("expected exactly one Slurm task")
    fields = dict(re.findall(r"(\w+)=([^\s]*)", rows[0]))
    if not fields.get("JobName", "").startswith("cz-") or (
        "UserId" in fields and fields["UserId"].split("(")[0] != os.environ.get("USER")
    ):
        raise ValueError("device repair only accepts owned cz- jobs")
    return fields


def exclusion_update(root: Path, target: str, node: str, fields: dict[str, str]) -> list[str]:
    from cluster.gpu_selection import expand_nodes

    existing = fields.get("ExcNodeList", "(null)")
    nodes = () if existing == "(null)" else expand_nodes(existing)
    exclusions = combined_exclusions(root, [*nodes, node])
    return [
        "scontrol",
        "update",
        f"JobId={target}",
        "ReqNodeList=",
        f"ExcNodeList={','.join(exclusions)}",
    ]


def repair(root: Path, run_path: Path, event_id: str) -> None:
    ledger = read_json(run_path / "device_faults.json")
    entry = next(
        item
        for item in cast(list[dict[str, JsonValue]], ledger["events"])
        if item["event_id"] == event_id
    )
    target = cast(str, entry["target"])
    fields = owned_job(target)
    throttle = fields.get("ArrayTaskThrottle", entry.get("array_throttle"))
    if (run_path / "DONE").exists():
        return
    update_event(
        root,
        run_path,
        event_id,
        {"failed_job_state": fields.get("JobState"), "failed_exit_code": fields.get("ExitCode")},
    )
    subprocess.run(["scontrol", "requeuehold", target], check=True, capture_output=True, text=True)
    fields = owned_job(target)
    subprocess.run(
        exclusion_update(root, target, cast(str, entry["node"]), fields),
        check=True,
        capture_output=True,
        text=True,
    )
    if throttle is not None:
        subprocess.run(
            [
                "scontrol",
                "update",
                f"JobId={target.split('_')[0]}",
                f"ArrayTaskThrottle={throttle}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    update_event(
        root,
        run_path,
        event_id,
        {"status": "requeued", "repair_job_id": os.environ.get("SLURM_JOB_ID")},
    )
    subprocess.run(["scontrol", "release", target], check=True, capture_output=True, text=True)


def handle_fault(root: Path, run_path: Path, error: str, simulated: bool = False) -> None:
    from cluster.gpu_policy import load_policy
    from cluster.slurm import ACTIVE_STATES, Budget, queue

    job_id = os.environ["SLURM_JOB_ID"]
    target = requeue_identifier(
        job_id, os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    )
    entry = reserve_retry(
        root,
        run_path,
        platform.node().split(".")[0],
        target,
        job_id,
        int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        error,
        simulated,
    )
    if entry["status"] == "exhausted":
        return
    fields = owned_job(target)
    throttle = fields.get("ArrayTaskThrottle")
    if throttle:
        update_event(
            root, run_path, cast(str, entry["event_id"]), {"array_throttle": int(throttle)}
        )
    result = subprocess.run(
        exclusion_update(root, target, cast(str, entry["node"]), fields),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        update_event(root, run_path, cast(str, entry["event_id"]), {"status": "requeued"})
        subprocess.run(["scontrol", "requeue", target], check=True, capture_output=True, text=True)
        return
    with submission_lock(root):
        jobs = queue()
        if (
            len(jobs) >= load_policy().max_submit_jobs
            or sum(job.owned for job in jobs) >= Budget().queued_jobs
            or sum(
                job.owned
                and not job.gpu
                and job.state in ACTIVE_STATES
                and job.identifier != target
                for job in jobs
            )
            >= Budget().cpu_jobs
        ):
            raise RuntimeError("no CPU/queued budget for finite device repair")
        digest = hashlib.sha256(f"{run_path}:{entry['event_id']}".encode()).hexdigest()[:16]
        command = (
            f"cd {shlex.quote(str(REPOSITORY))}\n"
            f"python -m cluster.device_faults repair --run-path {shlex.quote(str(run_path))} "
            f"--event {shlex.quote(str(entry['event_id']))}"
        )
        script = "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n"
        completed = subprocess.run(
            [
                "sbatch",
                "--parsable",
                f"--nice={load_policy().nice}",
                "--export=NONE",
                "--partition=normal",
                "--nodelist=xcng0",
                "--nodes=1",
                "--cpus-per-task=1",
                "--mem=2G",
                "--time=1",
                f"--job-name=cz-device-repair-{digest}",
                f"--dependency=afterany:{target}",
                f"--output={root}/logs/%x-%j.log",
            ],
            input=script,
            check=True,
            capture_output=True,
            text=True,
        )
        repair_id = completed.stdout.strip().split(";")[0]
        if not repair_id.isdecimal():
            raise RuntimeError("invalid device repair job id")
        update_event(
            root,
            run_path,
            cast(str, entry["event_id"]),
            {
                "status": "repair_pending",
                "repair_job_id": repair_id,
                "running_update_error": result.stderr.strip(),
            },
        )


def fault_evidence(
    root: Path, run_path: Path, report: dict[str, JsonValue]
) -> tuple[str, str] | None:
    path = run_path / "device_faults.json"
    target = (
        f"{report['array_job_id']}_{report['array_task_id']}"
        if report.get("array_job_id") is not None and report.get("array_task_id") is not None
        else str(report.get("job_id", ""))
    )
    error = str(report.get("error", ""))
    if error.startswith(
        ("ValueError:", "FileNotFoundError:", "KeyError:", "TypeError:", "AssertionError:")
    ):
        return None
    if path.exists():
        events = cast(list[dict[str, JsonValue]], read_json(path).get("events", []))
        matching = [
            entry
            for entry in events
            if entry.get("target") == target
            and str(entry.get("job_id")) == str(report.get("job_id"))
            and entry.get("restart_count") == report.get("slurm_restart_count", 0)
        ]
        if matching:
            return cast(str, matching[-1]["node"]), cast(str, matching[-1]["error"])
    node = str(report.get("host", "")).split(".")[0]
    if recognised_fault(error) and NODE_PATTERN.fullmatch(node):
        return node, error
    early_shell_failure = (
        report.get("exit_status") == DEVICE_FAILURE_EXIT_STATUS
        or "shell or bootstrap failed" in error
    )
    if not early_shell_failure or not NODE_PATTERN.fullmatch(node):
        return None
    identifiers = {target, str(report.get("job_id", ""))}
    for identifier in identifiers:
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", identifier):
            continue
        for log_path in (root / "logs").glob(f"cz-*-{identifier}.log"):
            with log_path.open("rb") as source:
                source.seek(0, 2)
                source.seek(max(0, source.tell() - MAX_FAILURE_LOG_BYTES))
                text = source.read().decode("utf-8", errors="replace")
            marker = f"CZ_DEVICE_ATTEMPT {target}:{report.get('slurm_restart_count', 0)}"
            if marker in text:
                text = text.rsplit(marker, 1)[1]
            elif "CZ_DEVICE_ATTEMPT " in text or report.get("slurm_restart_count", 0):
                continue
            if recognised_fault(text):
                return node, text[-MAX_ERROR_CHARACTERS:]
    return None


def node_resources(node: str) -> tuple[str, int]:
    if not re.fullmatch(r"xgp[d-k][0-9]+", node):
        raise ValueError("GPU probe requires an approved GPU node")
    result = subprocess.run(
        ["scontrol", "show", "node", node, "-o"], check=True, capture_output=True, text=True
    )
    fields = dict(re.findall(r"(\w+)=([^\s]*)", result.stdout))
    resources = re.findall(r"(?:^|,)gpu:([^:,()]+):([0-9]+)", fields.get("Gres", ""))
    if len(resources) != 1:
        raise ValueError("probe requires one typed GPU GRES class per node")
    resource_type, count = resources[0]
    if int(count) < 1:
        raise ValueError("GPU node has no configured devices")
    return resource_type, int(count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "repair"))
    parser.add_argument("--nvidia-only", action="store_true")
    parser.add_argument("--simulate-once", action="store_true")
    parser.add_argument("--run-path", type=Path)
    parser.add_argument("--event")
    arguments = parser.parse_args()
    root = (Path.home() / "columba").resolve()
    if not os.environ.get("SLURM_JOB_NAME", "").startswith("cz-"):
        raise ValueError("device checks require a cz- allocation")
    run_path = arguments.run_path or Path(os.environ["COLUMBA_RUN_PATH"])
    if not run_path.resolve().is_relative_to(root / "runs"):
        raise ValueError("device ledger must stay in project runs")
    if arguments.action == "repair":
        try:
            repair(root, run_path, arguments.event)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            update_event(
                root,
                run_path,
                arguments.event,
                {"status": "repair_failed", "repair_error": str(error)},
            )
            (run_path / "HOLD").write_text(f"device repair failed: {error}\n")
            raise
        return
    if arguments.simulate_once and (
        not os.environ["SLURM_JOB_NAME"].startswith("cz-ws3-device-")
        or "smoke-" not in str(run_path.relative_to(root / "runs"))
    ):
        raise ValueError("simulation is restricted to WS3 smoke allocations")
    target = requeue_identifier(
        os.environ["SLURM_JOB_ID"],
        os.environ.get("SLURM_ARRAY_JOB_ID"),
        os.environ.get("SLURM_ARRAY_TASK_ID"),
    )
    print(f"CZ_DEVICE_ATTEMPT {target}:{os.environ.get('SLURM_RESTART_COUNT', '0')}", flush=True)
    try:
        if arguments.simulate_once:
            if int(os.environ.get("SLURM_RESTART_COUNT", "0")) == 0:
                raise DeviceVisibilityError("simulated: No devices were found")
            print(json.dumps({"simulated": True, "device_check": "passed after retry"}))
        else:
            print(json.dumps(check_visibility(expected_gpu_count(), arguments.nvidia_only)))
    except DeviceVisibilityError as error:
        print(str(error), flush=True)
        try:
            handle_fault(root, run_path, str(error), arguments.simulate_once)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as repair_error:
            (run_path / "HOLD").write_text(
                f"device repair could not be scheduled: {repair_error}\n"
            )
            print(str(repair_error), flush=True)
        raise SystemExit(DEVICE_FAILURE_EXIT_STATUS) from error


if __name__ == "__main__":
    main()
