"""Account for array claims inside the shared GPU allowance and typed QoS ceilings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from app.json_data import JsonValue
from cluster.gpu_policy import load_policy
from cluster.slurm import ACTIVE_STATES, GPU_RUNNING_LIMIT, Job, resource_type_for_job


def claimed_gpus(
    records: Sequence[dict[str, JsonValue]], jobs: Sequence[Job], skip_parent: str | None = None
) -> tuple[int, dict[str, int]]:
    active = [job for job in jobs if job.owned and job.gpu and job.state in ACTIVE_STATES]
    managed = {str(record["array_id"]): record for record in records}
    total = 0
    typed: dict[str, int] = {}
    for parent in {job.identifier.split("_")[0] for job in active}:
        if parent == skip_parent:
            continue
        members = [job for job in active if job.identifier.split("_")[0] == parent]
        record = managed.get(parent)
        maximum = cast(int, record["throttle"]) if record else sum(job.gpu_count for job in members)
        total += min(maximum, sum(job.gpu_count for job in members))
        for resource_type in {resource_type_for_job(job) for job in members}:
            if resource_type is None:
                continue
            count = sum(
                job.gpu_count for job in members if resource_type_for_job(job) == resource_type
            )
            typed[resource_type] = typed.get(resource_type, 0) + min(maximum, count)
    return total, typed


def allocate_throttles(
    counts: Mapping[str, int],
    resources: Mapping[str, str],
    jobs: Sequence[Job],
    parents: Mapping[str, str],
    gpu_limit: int,
) -> dict[str, int]:
    if not 0 < gpu_limit <= GPU_RUNNING_LIMIT:
        raise ValueError("array concurrency must be within 1..8")
    policy = load_policy()
    unmanaged = [job for job in jobs if job.identifier.split("_")[0] not in parents.values()]
    total, typed = claimed_gpus((), unmanaged)
    allocation = dict.fromkeys(counts, 0)
    for key, parent in parents.items():
        running = sum(
            job.gpu_count
            for job in jobs
            if job.owned
            and job.identifier.split("_")[0] == parent
            and job.state != "PENDING"
            and job.state in ACTIVE_STATES
        )
        allocation[key] = running
        total += running
        typed[resources[key]] = typed.get(resources[key], 0) + running
    while total < gpu_limit:
        eligible = [
            key
            for key in counts
            if allocation[key] < counts[key]
            and typed.get(resources[key], 0) < policy.type_limits[resources[key]]
        ]
        if not eligible:
            break
        for key in eligible:
            resource_type = resources[key]
            if (
                total >= gpu_limit
                or typed.get(resource_type, 0) >= policy.type_limits[resource_type]
            ):
                continue
            allocation[key] += 1
            typed[resource_type] = typed.get(resource_type, 0) + 1
            total += 1
    if any(value == 0 for value in allocation.values()):
        raise ValueError("no claimed GPU slot for an array; cannot set a zero/unlimited throttle")
    return allocation
