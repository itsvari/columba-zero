"""Slurm policy and an atomic remote submission/cancellation boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

GPU_RUNNING_LIMIT = 4
CPU_RUNNING_LIMIT = 2
QUEUED_LIMIT = 12
PROJECT_STORAGE_LIMIT_KB = 80 * 1024 * 1024
ACTIVE_STATES = frozenset(
    {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING"}
)
GPU_FEATURES = frozenset({"t4", "titanrtx", "titanv"})


@dataclass(frozen=True)
class Job:
    identifier: str
    name: str
    state: str
    partition: str
    resources: str

    @property
    def owned(self) -> bool:
        return self.name.startswith("cz-")

    @property
    def gpu(self) -> bool:
        return "gpu" in self.resources or "gpu" in self.partition


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
            raise ValueError("queued budget cannot exceed 12")

    def allows(self, jobs: Sequence[Job], lane: str) -> bool:
        owned = [job for job in jobs if job.owned and job.state in ACTIVE_STATES]
        gpu_count = sum(job.gpu for job in owned)
        cpu_count = len(owned) - gpu_count
        if len(owned) >= self.queued_jobs:
            return False
        # pending jobs reserve a slot so simultaneous starts cannot exceed the running cap.
        return gpu_count < self.gpu_jobs if lane == "nv" else cpu_count < self.cpu_jobs


class Submission(TypedDict):
    name: str
    lane: Literal["nv", "arm"]
    constraint: str | None
    script: str
    run_path: str
    excluded_nodes: NotRequired[list[str]]


def parse_queue(output: str) -> tuple[Job, ...]:
    jobs: list[Job] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        columns = line.strip().split("|")
        if len(columns) != 5:
            raise ValueError(f"unexpected squeue row: {line}")
        jobs.append(Job(*columns))
    return tuple(jobs)


def queue() -> tuple[Job, ...]:
    completed = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "--noheader", "--format=%i|%j|%T|%P|%b"],
        check=True,
        text=True,
        capture_output=True,
    )
    return parse_queue(completed.stdout)


def job_name(recipe_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{recipe_id}/seed{seed}".encode()).hexdigest()[:12]
    return f"cz-{recipe_id[:55]}-s{seed}-{digest}"


def submit_pending(submissions: Sequence[Submission], budget: Budget) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
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
        jobs = queue()
        if (Path(submission["run_path"]) / "DONE").is_file():
            results.append({"name": submission["name"], "status": "done"})
            continue
        if any(job.name == submission["name"] and job.state in ACTIVE_STATES for job in jobs):
            results.append({"name": submission["name"], "status": "active"})
            continue
        if not budget.allows(jobs, submission["lane"]):
            results.append({"name": submission["name"], "status": "budget full"})
            continue
        command = [
            "sbatch",
            "--parsable",
            "--nice=1000",
            f"--job-name={submission['name']}",
            f"--output={Path.home()}/columba/logs/%x-%j.log",
            "--export=NONE",
        ]
        excluded_nodes = submission.get("excluded_nodes", [])
        if any(not re.fullmatch(r"xgp[def][0-9]+|xcng[01]", node) for node in excluded_nodes):
            raise ValueError("excluded nodes must belong to permitted nv or ARM lanes")
        if excluded_nodes:
            command.append(f"--exclude={','.join(excluded_nodes)}")
        if submission["constraint"]:
            command.append(f"--constraint={submission['constraint']}")
        completed = subprocess.run(
            command, input=submission["script"], text=True, check=True, capture_output=True
        )
        identifier = completed.stdout.strip().split(";")[0]
        if not identifier.isdecimal():
            raise RuntimeError(f"unexpected sbatch response: {completed.stdout}")
        results.append({"name": submission["name"], "status": "submitted", "job_id": identifier})
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
        dict[str, str | list[Submission] | list[str] | dict[str, int]],
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
                json.dumps(submit_pending(cast(list[Submission], payload["submissions"]), budget))
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
