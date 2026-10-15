"""Submit typed copies of each run; Slurm QoS and an atomic start claim pick the winner."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

from app.json_data import JsonValue
from cluster.arrays import ArrayTask, active_arrays, active_run_names, submit_arrays, task_for
from cluster.artifacts import atomic_json, read_json
from cluster.device_faults import combined_exclusions
from cluster.gpu_policy import load_policy
from cluster.gpu_selection import GpuOffer, GpuRequest, gpu_inventory, request_for
from cluster.locking import submission_lock
from cluster.recipe import load_recipe
from cluster.slurm import (
    ACTIVE_STATES,
    GPU_RUNNING_LIMIT,
    Budget,
    Job,
    queue,
    resource_type_for_job,
)

SMALL_BACKLOG_RUNS = 8
LARGE_BACKLOG_WIDTH = 2


def fanout_width(run_count: int, available_tasks: int, preferred_width: int) -> int:
    if run_count < 1:
        return 0
    if available_tasks < run_count:
        raise RuntimeError(
            "fan-out exceeds submitted-task budget; submit a smaller recipe/seed batch"
        )
    maximum = preferred_width if run_count <= SMALL_BACKLOG_RUNS else LARGE_BACKLOG_WIDTH
    return min(maximum, available_tasks // run_count)


def requests_for(
    task: ArrayTask,
    offers: Sequence[GpuOffer],
    constraint: str | None = None,
    required_node: str | None = None,
) -> tuple[GpuRequest, ...]:
    recipe = load_recipe(Path(task.recipe_path))
    selected = (
        None
        if constraint in {"any", "null"}
        else constraint or recipe.constraint_for_seed(task.seed)
    )
    policy = load_policy()
    preferences = (
        (selected,)
        if selected or required_node
        else policy.preferences(
            recipe.base_model,
            cast(int, recipe.train["imgsz"]),
            recipe.teacher is not None or "teacher" in recipe.identifier,
            recipe.estimated_gpu_hours,
        )
    )
    requests: dict[str, GpuRequest] = {}
    for preference in preferences:
        choice = "a100-80" if preference == "xgpj0" else "t4" if preference == "nv" else preference
        automatic_nv = (
            selected is None and required_node is None and choice in {"t4", "titanv", "titanrtx"}
        )
        matched = tuple(
            offer for offer in offers if not automatic_nv or offer.resource_type == "nv"
        )
        if automatic_nv:
            choice = next(
                (
                    feature
                    for offer in matched
                    for feature in ("t4", "titanv", "titanrtx")
                    if feature in offer.features
                ),
                None,
            )
            if choice is None:
                continue
        try:
            request = request_for(choice, matched, required_node=required_node)
        except ValueError:
            if selected or required_node:
                raise
            continue
        resource_type = request.resources.split(":")[1]
        partitions = tuple(
            partition
            for partition in ("gpu-long", "gpu")
            if any(
                offer.partition == partition
                and offer.resource_type == resource_type
                and (required_node is None or offer.node == required_node)
                and (resource_type != "nv" or automatic_nv or request.constraint in offer.features)
                for offer in matched
            )
            and (partition != "gpu" or policy.allow_short_partition)
        )
        if not partitions:
            continue
        request = replace(
            request,
            partitions=partitions,
            time_limit="03:00:00" if "gpu" in partitions else "3-00:00:00",
            constraint="t4|titanv|titanrtx" if automatic_nv else request.constraint,
        )
        requests.setdefault(resource_type, request)
    if not requests:
        raise ValueError("no preferred GPU inventory for fan-out")
    return tuple(requests.values())


def narrow_requests(
    requests: tuple[GpuRequest, ...],
    width: int,
    offers: Sequence[GpuOffer],
    jobs: Sequence[Job],
    planned: dict[str, int],
) -> tuple[GpuRequest, ...]:
    if width >= len(requests):
        return requests
    limits = load_policy().type_limits

    def headroom(request: GpuRequest) -> int:
        resource_type = request.resources.split(":")[1]
        free_by_node = {
            offer.node: offer.free_units for offer in offers if offer.resource_type == resource_type
        }
        held = sum(
            job.gpu_count
            for job in jobs
            if job.owned
            and job.state in {"RUNNING", "CONFIGURING", "COMPLETING"}
            and resource_type_for_job(job) == resource_type
        )
        return min(sum(free_by_node.values()), limits[resource_type] - held) - planned.get(
            resource_type, 0
        )

    primary = next((request for request in requests if headroom(request) > 0), requests[0])
    resource_type = primary.resources.split(":")[1]
    planned[resource_type] = planned.get(resource_type, 0) + 1
    alternatives = tuple(request for request in requests if request != primary)
    return (primary, *alternatives[: width - 1])


def submit_runs(
    root: Path,
    checkout: Path,
    tasks: Sequence[ArrayTask],
    gpu_limit: int = GPU_RUNNING_LIMIT,
    constraint: str | None = None,
    required_node: str | None = None,
    reserved_jobs: int = 0,
    resume: bool = False,
) -> list[dict[str, JsonValue]]:
    root = root.resolve()
    if root != (Path.home() / "columba").resolve() or not checkout.resolve().is_relative_to(
        root / "checkouts"
    ):
        raise ValueError("fan-out must use ~/columba and a pinned checkout")
    if gpu_limit != GPU_RUNNING_LIMIT:
        return submit_arrays(
            root, checkout, tasks, gpu_limit, constraint, required_node, reserved_jobs
        )
    if len({task.relative_run for task in tasks}) != len(tasks):
        raise ValueError("duplicate recipe/seed in fan-out")
    policy = load_policy()
    with submission_lock(root):
        jobs = queue()
        active = active_run_names(root, jobs) | {
            job.name for job in jobs if job.owned and job.state in ACTIVE_STATES
        }
        missing = [
            task
            for task in tasks
            if (resume or task.run_name not in active)
            and not (root / "runs" / task.relative_run / "DONE").exists()
        ]
        desired = {task.run_name for task in tasks}
        receipts = [
            record
            for record in active_arrays(root, jobs)
            if any(
                item["run_name"] in desired
                for item in cast(
                    list[dict[str, JsonValue]],
                    read_json(Path(cast(str, record["manifest"])))["tasks"],
                )
            )
        ]
        if not missing:
            return receipts
        available = (
            min(
                policy.max_submit_jobs - len(jobs),
                Budget().queued_jobs - sum(job.owned for job in jobs),
            )
            - reserved_jobs
        )
        offers = tuple(
            offer for offer in gpu_inventory() if offer.node not in combined_exclusions(root)
        )
        choices = {
            task.relative_run: requests_for(task, offers, constraint, required_node)
            for task in missing
        }
        width = fanout_width(
            len(missing), available, max(len(values) for values in choices.values())
        )
        footprint = subprocess.run(
            ["du", "-sk", str(root)], check=True, capture_output=True, text=True
        )
        if int(footprint.stdout.split()[0]) >= 80 * 1024 * 1024:
            raise RuntimeError("project storage limit reached")
        results = list(receipts) if not resume else []
        release_ids: list[str] = []
        planned: dict[str, int] = {}
        try:
            for task in missing:
                for request in narrow_requests(
                    choices[task.relative_run], width, offers, jobs, planned
                ):
                    payload: dict[str, JsonValue] = {
                        "checkout": str(checkout),
                        "request": cast(dict[str, JsonValue], asdict(request)),
                        "tasks": cast(list[JsonValue], [asdict(task)]),
                        "kind": "fanout",
                        "constraint": constraint,
                        "required_node": required_node,
                    }
                    # checkpoint segments keep the exact shared name across new copies.
                    segment = (
                        read_json(root / "runs" / task.relative_run / "requeue.json")
                        if (root / "runs" / task.relative_run / "requeue.json").exists()
                        else {}
                    )
                    payload["segment"] = segment.get("count", 0)
                    payload["resume"] = resume
                    digest = hashlib.sha256(
                        json.dumps(payload, sort_keys=True).encode()
                    ).hexdigest()[:16]
                    directory = root / "arrays" / digest
                    payload["directory"] = str(directory)
                    existing_path = directory / "submission.json"
                    if existing_path.exists():
                        existing_record = read_json(existing_path)
                        if any(
                            job.identifier.split("_")[0] == str(existing_record["array_id"])
                            for job in jobs
                        ):
                            results.append(existing_record)
                            continue
                    atomic_json(directory / "manifest.json", payload)
                    script = (checkout / "cluster/templates/nv.sbatch").read_text()
                    exports = (
                        f"export COLUMBA_CHECKOUT={shlex.quote(str(checkout))}\n"
                        "export COLUMBA_ARRAY_MANIFEST="
                        f"{shlex.quote(str(directory / 'manifest.json'))}\n"
                        "export COLUMBA_FANOUT=true\n"
                        'eval "$(python -m cluster.arrays task-env '
                        '--manifest "$COLUMBA_ARRAY_MANIFEST")"\n'
                    )
                    script = script.replace(
                        "set -euo pipefail",
                        "set -euo pipefail\ncd " + shlex.quote(str(checkout)) + "\n" + exports,
                    )
                    command = [
                        "sbatch",
                        "--parsable",
                        "--hold",
                        f"--nice={policy.nice}",
                        "--export=NONE",
                        "--nodes=1",
                        f"--job-name={task.run_name}",
                        "--array=0-0",
                        f"--output={root}/logs/%x-%A_%a.log",
                        *request.arguments(),
                    ]
                    exclusions = combined_exclusions(
                        root, static_path=checkout / "cluster/excluded_nodes.txt"
                    )
                    if exclusions:
                        command.append("--exclude=" + ",".join(exclusions))
                    if required_node:
                        command.append("--nodelist=" + required_node)
                    if resume:
                        command.append("--begin=now+1minute")
                    if task.recipe_id.startswith("smoke-"):
                        script = script.replace("#SBATCH --signal=B:USR1@600\n", "")
                        command.extend(["--time=00:00:40", "--cpus-per-task=2", "--mem=8G"])
                    completed = subprocess.run(
                        command, input=script, check=True, capture_output=True, text=True
                    )
                    identifier = completed.stdout.strip().split(";")[0]
                    if not identifier.isdecimal():
                        raise RuntimeError("invalid fan-out job id")
                    record: dict[str, JsonValue] = {
                        "array_id": identifier,
                        "throttle": 1,
                        "resource_type": request.resources.split(":")[1],
                        "task_count": 1,
                        "manifest": str(directory / "manifest.json"),
                        "kind": "fanout",
                    }
                    atomic_json(directory / "submission.json", record)
                    results.append(record)
                    release_ids.append(identifier)
        finally:
            if release_ids:
                subprocess.run(
                    ["scontrol", "release", ",".join(release_ids)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
        return results


def refanout() -> list[dict[str, JsonValue]]:
    root = Path.home() / "columba"
    manifest_path = Path(os.environ["COLUMBA_ARRAY_MANIFEST"])
    manifest = read_json(manifest_path)
    if manifest.get("kind") != "fanout":
        return []
    item = cast(list[dict[str, JsonValue]], manifest["tasks"])[0]
    task = task_for(Path(cast(str, item["recipe_path"])), cast(int, item["seed"]))
    return submit_runs(
        root,
        Path(cast(str, manifest["checkout"])),
        [task],
        constraint=cast(str | None, manifest.get("constraint")),
        required_node=cast(str | None, manifest.get("required_node")),
        resume=True,
    )
