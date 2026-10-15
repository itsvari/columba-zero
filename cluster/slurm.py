"""Slurm policy and an atomic remote submission/cancellation boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

from cluster.device_faults import combined_exclusions, node_resources
from cluster.gpu_policy import available_request, load_policy
from cluster.gpu_selection import (
    GPU_CONSTRAINTS,
    GPU_NODE_PATTERN,
    PARTITION_TIME_LIMIT_MINUTES,
    GpuOffer,
    GpuRequest,
    gpu_inventory,
    request_for,
)

# per-user QoS max (8 GPUs); raised from 4 on 15 Oct 2026 when the user freed the account.
GPU_RUNNING_LIMIT = 8
CPU_RUNNING_LIMIT = 2
QUEUED_LIMIT = 24  # QoS allows 32 submitted; keep headroom
GPU_CPU_LIMIT = 8
GPU_MEMORY_LIMIT_GB = 64
PROJECT_STORAGE_LIMIT_KB = 80 * 1024 * 1024
ACTIVE_STATES = frozenset(
    {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING"}
)
GPU_FEATURES = frozenset(GPU_CONSTRAINTS)


@dataclass(frozen=True)
class Job:
    identifier: str
    name: str
    state: str
    partition: str
    resources: str
    node_count: int = 1
    submitted_at: datetime | None = None

    @property
    def owned(self) -> bool:
        return self.name.startswith("cz-")

    @property
    def gpu_count(self) -> int:
        counts = re.findall(r"(?:^|,)(?:gres/)?gpu(?::([^,:=()]+))?[:=]([0-9]+)", self.resources)
        if counts:
            generic = sum(int(count) for resource_type, count in counts if not resource_type)
            typed = sum(int(count) for resource_type, count in counts if resource_type)
            return max(generic, typed) * self.node_count
        # unknown GPU requests fail closed instead of assuming a single device.
        return GPU_RUNNING_LIMIT if "gpu" in self.resources or "gpu" in self.partition else 0

    @property
    def gpu(self) -> bool:
        return self.gpu_count > 0


@dataclass(frozen=True)
class Budget:
    gpu_jobs: int = GPU_RUNNING_LIMIT
    cpu_jobs: int = CPU_RUNNING_LIMIT
    queued_jobs: int = QUEUED_LIMIT

    def __post_init__(self) -> None:
        if (
            not 0 <= self.gpu_jobs <= GPU_RUNNING_LIMIT
            or not 0 <= self.cpu_jobs <= CPU_RUNNING_LIMIT
        ):
            raise ValueError("budget may lower but cannot raise the project caps")
        if not 0 <= self.queued_jobs <= QUEUED_LIMIT:
            raise ValueError(f"queued budget cannot exceed {QUEUED_LIMIT}")

    def allows(self, jobs: Sequence[Job], lane: str, gpus_requested: int = 1) -> bool:
        owned = [job for job in jobs if job.owned and job.state in ACTIVE_STATES]
        gpu_count = sum(job.gpu_count for job in owned)
        cpu_count = sum(not job.gpu for job in owned)
        if len(owned) >= self.queued_jobs:
            return False
        # pending jobs reserve a slot so simultaneous starts cannot exceed the running cap.
        return (
            gpu_count + gpus_requested <= self.gpu_jobs
            if lane == "nv"
            else cpu_count < self.cpu_jobs
        )


class Submission(TypedDict):
    name: str
    lane: Literal["nv", "arm"]
    constraint: str | None
    script: str
    run_path: str
    excluded_nodes: NotRequired[list[str]]
    required_node: NotRequired[str]
    base_model: NotRequired[str]
    image_size_px: NotRequired[int]
    teacher: NotRequired[bool]
    est_gpu_hours: NotRequired[float]
    time_limit_minutes: NotRequired[int]
    cpu_count: NotRequired[int]
    memory_limit_gb: NotRequired[int]
    gpu_count: NotRequired[int]


def parse_queue(output: str) -> tuple[Job, ...]:
    jobs: list[Job] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        columns = line.strip().split("|")
        if len(columns) not in {5, 6, 7}:
            raise ValueError(f"unexpected squeue row: {line}")
        node_count = int(columns[5]) if len(columns) >= 6 else 1
        if node_count < 1:
            raise ValueError("squeue node count must be positive")
        identifier, name, state, partition, resources = columns[:5]
        submitted_at = datetime.fromisoformat(columns[6]) if len(columns) == 7 else None
        jobs.append(Job(identifier, name, state, partition, resources, node_count, submitted_at))
    return tuple(jobs)


def queue() -> tuple[Job, ...]:
    completed = subprocess.run(
        [
            "squeue",
            "--array",
            "-u",
            os.environ["USER"],
            "--noheader",
            "--format=%i|%j|%T|%P|%b|%D|%V",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return parse_queue(completed.stdout)


def job_name(recipe_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{recipe_id}/seed{seed}".encode()).hexdigest()[:12]
    return f"cz-{recipe_id[:55]}-s{seed}-{digest}"


def gpu_resource_arguments(submission: Submission) -> tuple[str, ...]:
    arguments: list[str] = []
    values = (
        (submission.get("cpu_count"), "cpus-per-task", GPU_CPU_LIMIT),
        (submission.get("memory_limit_gb"), "mem", GPU_MEMORY_LIMIT_GB),
    )
    for value, flag, maximum in values:
        if value is None:
            continue
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError("GPU resource override exceeds the template limits")
        suffix = "G" if flag == "mem" else ""
        arguments.append(f"--{flag}={value}{suffix}")
    return tuple(arguments)


def request_arguments(request: GpuRequest, submission: Submission) -> tuple[str, ...]:
    arguments = request.arguments()
    minutes = submission.get("time_limit_minutes")
    if minutes is None:
        return arguments
    maximum_minutes = min(
        PARTITION_TIME_LIMIT_MINUTES[partition] for partition in request.partitions
    )
    if not 0 < minutes <= maximum_minutes:
        raise ValueError("job time must fit its selected partition")
    return tuple(
        f"--time={minutes}" if argument.startswith("--time=") else argument
        for argument in arguments
    )


def resource_type_for_job(job: Job) -> str | None:
    matched = re.search(r"(?:gres/)?gpu:([^:,=]+)[:=][0-9]+", job.resources)
    return matched.group(1) if matched else None


def scheduling_request(
    submission: Submission,
    offers: Sequence[GpuOffer],
    avoid: tuple[str, str] | None = None,
    jobs: Sequence[Job] = (),
) -> tuple[tuple[str, ...], str] | None:
    policy = load_policy()
    if not submission["name"].startswith("cz-ws3-probe-"):
        submission = {
            **submission,
            "excluded_nodes": list(
                combined_exclusions(Path.home() / "columba", submission.get("excluded_nodes", []))
            ),
        }
    offers = tuple(
        offer
        for offer in offers
        if sum(
            job.gpu_count
            for job in jobs
            if job.owned
            and job.state in ACTIVE_STATES
            and resource_type_for_job(job) == offer.resource_type
        )
        < policy.type_limits.get(offer.resource_type, 0)
    )
    if not offers:
        return None
    preferences = policy.preferences(
        submission.get("base_model", "yolo11n.pt"),
        submission.get("image_size_px", 640),
        submission.get("teacher", False),
        submission.get("est_gpu_hours", 0),
    )
    automatic = (
        submission["constraint"] is None and "required_node" not in submission
    ) or avoid is not None
    if automatic:
        selected = available_request(
            offers,
            preferences,
            excluded_nodes=submission.get("excluded_nodes", []),
            allow_short_partition=policy.allow_short_partition,
            avoid=avoid,
        )
        if selected is None:
            return None
        request, node = selected
        return (
            *request_arguments(request, submission),
            f"--nodelist={node}",
        ), f"{request.resources}@{','.join(request.partitions)}"
    request = request_for(
        submission["constraint"],
        offers,
        required_node=submission.get("required_node"),
        excluded_nodes=submission.get("excluded_nodes", []),
    )
    count = submission.get("gpu_count", 1)
    if count != 1:
        request = replace(request, resources=request.resources.rsplit(":", 1)[0] + f":{count}")
    return request_arguments(
        request, submission
    ), f"{request.resources}@{','.join(request.partitions)}"


def pending_target(job: Job, submission: Submission) -> tuple[str, str]:
    typed = re.search(r"gpu:([^:,]+):[0-9]+", job.resources)
    choice = typed.group(1) if typed else "unknown"
    if choice == "nv":
        choice = submission["constraint"] or "nv"
    if choice == "h200-141":
        choice = "h200"
    return choice, job.partition


def reserve_submitted_capacity(
    offers: Sequence[GpuOffer], command: Sequence[str]
) -> tuple[GpuOffer, ...]:
    resources = next(
        argument.removeprefix("--gres=") for argument in command if argument.startswith("--gres=")
    )
    feature = next(
        (
            argument.removeprefix("--constraint=")
            for argument in command
            if argument.startswith("--constraint=")
        ),
        None,
    )
    nodes = next(
        (
            argument.removeprefix("--nodelist=").split(",")
            for argument in reversed(command)
            if argument.startswith("--nodelist=")
        ),
        None,
    )
    resource_type = resources.split(":")[1] if resources.count(":") == 2 else None
    eligible = next(
        (
            offer
            for offer in offers
            if offer.free_units > 0
            and (nodes is None or offer.node in nodes)
            and (resource_type is None or offer.resource_type == resource_type)
            and (feature is None or feature in offer.features)
        ),
        None,
    )
    if eligible is None:
        return tuple(offers)
    return tuple(
        replace(offer, free_units=max(0, offer.free_units - int(resources.rsplit(":", 1)[1])))
        if offer.node == eligible.node and offer.resource_type == eligible.resource_type
        else offer
        for offer in offers
    )


def submit_pending(
    submissions: Sequence[Submission],
    budget: Budget,
    retarget_pending_after_hours: float | None = None,
) -> list[dict[str, str]]:
    if retarget_pending_after_hours is not None and retarget_pending_after_hours <= 0:
        raise ValueError("pending re-target timeout must be positive")
    results: list[dict[str, str]] = []
    offers = gpu_inventory() if any(item["lane"] == "nv" for item in submissions) else ()
    for submission in submissions:
        if not submission["name"].startswith("cz-"):
            raise ValueError("only cz- jobs may be submitted")
        if submission["lane"] not in {"nv", "arm"} or submission[
            "constraint"
        ] not in GPU_FEATURES | {None}:
            raise ValueError("unapproved lane or GPU constraint")
        if (
            not Path(submission["run_path"])
            .resolve()
            .is_relative_to((Path.home() / "columba").resolve())
        ):
            raise ValueError("run paths must stay inside ~/columba")
        excluded_nodes = (
            submission.get("excluded_nodes", [])
            if submission["name"].startswith("cz-ws3-probe-")
            else list(
                combined_exclusions(Path.home() / "columba", submission.get("excluded_nodes", []))
            )
        )
        submission = {**submission, "excluded_nodes": excluded_nodes}
        if any(not re.fullmatch(r"xgp[d-k][0-9]+|xcng[01]", node) for node in excluded_nodes):
            raise ValueError("excluded nodes must belong to approved GPU or ARM lanes")
        required_node = submission.get("required_node")
        if required_node and (
            submission["lane"] != "nv" or not GPU_NODE_PATTERN.fullmatch(required_node)
        ):
            raise ValueError("GPU jobs may only pin approved GPU nodes")
        resource_arguments = (
            gpu_resource_arguments(submission) if submission["lane"] == "nv" else ()
        )
        if submission["name"].startswith("cz-ws3-probe-"):
            resource_type, gpu_count = node_resources(cast(str, required_node))
            if gpu_count > load_policy().type_limits.get(resource_type, 0):
                raise ValueError(
                    "all-GPU probe exceeds the per-type QoS cap; administrator help required"
                )
            submission = {**submission, "gpu_count": gpu_count}
        jobs = queue()
        if (Path(submission["run_path"]) / "DONE").is_file():
            results.append({"name": submission["name"], "status": "done"})
            continue
        from cluster.arrays import active_run_names

        if submission["name"] in active_run_names(Path.home() / "columba", jobs):
            results.append({"name": submission["name"], "status": "active array task"})
            continue
        active = next(
            (job for job in jobs if job.name == submission["name"] and job.state in ACTIVE_STATES),
            None,
        )
        scheduling = None
        retargeted_from = None
        if active:
            expired = (
                retarget_pending_after_hours is not None
                and submission["lane"] == "nv"
                and active.state == "PENDING"
                and active.submitted_at is not None
                and (datetime.now(active.submitted_at.tzinfo) - active.submitted_at).total_seconds()
                >= retarget_pending_after_hours * 3600
            )
            scheduling = (
                scheduling_request(submission, offers, pending_target(active, submission), jobs)
                if expired
                else None
            )
            if scheduling is None:
                results.append({"name": submission["name"], "status": "active"})
                continue
            # Slurm applies the state filter atomically; a job that just started is never cancelled.
            subprocess.run(["scancel", "--state=PENDING", active.identifier], check=True)
            jobs = queue()
            if any(job.name == submission["name"] and job.state in ACTIVE_STATES for job in jobs):
                results.append({"name": submission["name"], "status": "active"})
                continue
            retargeted_from = active.identifier
        if not budget.allows(jobs, submission["lane"], submission.get("gpu_count", 1)):
            results.append({"name": submission["name"], "status": "budget full"})
            continue
        command = [
            "sbatch",
            "--parsable",
            f"--nice={load_policy().nice}",
            f"--job-name={submission['name']}",
            f"--output={Path.home()}/columba/logs/%x-%j.log",
            "--export=NONE",
        ]
        if excluded_nodes:
            command.append(f"--exclude={','.join(excluded_nodes)}")
        required_node = submission.get("required_node")
        if required_node:
            if submission["lane"] != "nv" or not GPU_NODE_PATTERN.fullmatch(required_node):
                raise ValueError("GPU jobs may only pin approved GPU nodes")
            if retargeted_from is None:
                command.append(f"--nodelist={required_node}")
        if submission["lane"] == "nv":
            command.append("--nodes=1")
            command.extend(resource_arguments)
            scheduling = scheduling or scheduling_request(submission, offers, jobs=jobs)
            if scheduling is None:
                results.append({"name": submission["name"], "status": "no free preferred GPU"})
                continue
            command.extend(scheduling[0])
        try:
            completed = subprocess.run(
                command, input=submission["script"], text=True, check=True, capture_output=True
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"sbatch failed: {error.stderr.strip()}") from error
        if submission["lane"] == "nv":
            offers = reserve_submitted_capacity(offers, command)
        identifier = completed.stdout.strip().split(";")[0]
        if not identifier.isdecimal():
            raise RuntimeError(f"unexpected sbatch response: {completed.stdout}")
        result = {"name": submission["name"], "status": "submitted", "job_id": identifier}
        if scheduling:
            result["gpu_request"] = scheduling[1]
        if retargeted_from:
            result["retargeted_from"] = retargeted_from
        results.append(result)
    return results


def cancel_jobs(identifiers: Sequence[str]) -> list[str]:
    jobs = {job.identifier: job for job in queue()}
    for identifier in identifiers:
        if identifier not in jobs or not jobs[identifier].owned:
            raise ValueError(f"refusing to cancel job outside cz- namespace: {identifier}")
    if identifiers:
        subprocess.run(["scancel", *identifiers], check=True)
    return list(identifiers)


def main() -> None:
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "cancel", "status"))
    parser.add_argument("--payload", default="{}")
    arguments = parser.parse_args()
    payload = cast(
        dict[str, str | float | None | list[Submission] | list[str] | dict[str, int]],
        json.loads(arguments.payload),
    )
    root = Path.home() / "columba"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with (root / ".submission.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if arguments.action == "submit":
            storage = subprocess.run(
                ["du", "-sk", str(root)], check=True, text=True, capture_output=True
            )
            if int(storage.stdout.split()[0]) >= PROJECT_STORAGE_LIMIT_KB:
                raise RuntimeError("~/columba has reached the 80 GiB storage limit")
            budget = Budget(**cast(dict[str, int], payload["budget"]))
            print(
                json.dumps(
                    submit_pending(
                        cast(list[Submission], payload["submissions"]),
                        budget,
                        cast(float | None, payload.get("retarget_pending_after_hours")),
                    )
                )
            )
        elif arguments.action == "cancel":
            print(json.dumps(cancel_jobs(cast(list[str], payload["job_ids"]))))
        else:
            print(
                json.dumps(
                    [
                        {"job_id": job.identifier, "name": job.name, "state": job.state}
                        for job in queue()
                        if job.owned
                    ]
                )
            )


if __name__ == "__main__":
    main()
