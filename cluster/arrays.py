"""Finite GPU arrays, immutable task manifests and cluster-side submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

from app.json_data import JsonValue
from cluster.artifacts import atomic_json, read_json
from cluster.device_faults import combined_exclusions, owned_job
from cluster.gpu_policy import load_policy
from cluster.gpu_selection import (
    PARTITION_TIME_LIMIT_MINUTES,
    GpuOffer,
    GpuRequest,
    gpu_inventory,
    request_for,
)
from cluster.locking import submission_lock
from cluster.recipe import load_recipe, recipe_sha256
from cluster.slurm import ACTIVE_STATES, GPU_RUNNING_LIMIT, Budget, Job, job_name, queue

REPOSITORY = Path(__file__).resolve().parents[1]
MAX_ARRAY_SIZE = 1001
STORAGE_LIMIT_KIB = 80 * 1024 * 1024
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24


@dataclass(frozen=True)
class ArrayTask:
    recipe_path: str
    recipe_id: str
    seed: int
    run_name: str
    recipe_sha256: str
    signal_after_epoch: int | None = None

    @property
    def relative_run(self) -> str:
        return f"{self.recipe_id}/seed{self.seed}"


def task_for(path: Path, seed: int, signal_after_epoch: int | None = None) -> ArrayTask:
    recipe = load_recipe(path)
    recipe.constraint_for_seed(seed)
    if recipe.lane != "nv":
        raise ValueError("training arrays require GPU recipes")
    if signal_after_epoch is not None and (
        not recipe.identifier.startswith("smoke-")
        or not 0 < signal_after_epoch < cast(int, recipe.train["epochs"])
    ):
        raise ValueError("forced signal requires an unfinished smoke epoch")
    return ArrayTask(
        str(path.resolve()),
        recipe.identifier,
        seed,
        job_name(recipe.identifier, seed),
        recipe_sha256(path),
        signal_after_epoch,
    )


def manifests(root: Path) -> tuple[dict[str, JsonValue], ...]:
    return tuple(read_json(path) for path in sorted((root / "arrays").glob("*/manifest.json")))


def active_arrays(root: Path, jobs: Sequence[Job]) -> tuple[dict[str, JsonValue], ...]:
    active = {
        job.identifier.split("_")[0] for job in jobs if job.owned and job.state in ACTIVE_STATES
    }
    return tuple(
        record
        for path in sorted((root / "arrays").glob("*/submission.json"))
        for record in (read_json(path),)
        if str(record["array_id"]) in active
    )


def persist_throttle(record: dict[str, JsonValue], throttle: int) -> None:
    subprocess.run(
        ["scontrol", "update", f"JobId={record['array_id']}", f"ArrayTaskThrottle={throttle}"],
        check=True,
        capture_output=True,
        text=True,
    )
    record["throttle"] = throttle
    atomic_json(Path(cast(str, record["manifest"])).parent / "submission.json", record)


def active_run_names(root: Path, jobs: Sequence[Job]) -> set[str]:
    identifiers = {job.identifier for job in jobs if job.owned and job.state in ACTIVE_STATES}
    names: set[str] = set()
    for manifest in manifests(root):
        record_path = Path(cast(str, manifest["directory"])) / "submission.json"
        if record_path.exists():
            array_id = str(read_json(record_path)["array_id"])
        else:
            matching = next(
                (job for job in jobs if job.name == f"cz-array-{record_path.parent.name}"), None
            )
            if matching is None:
                continue
            array_id = matching.identifier.split("_")[0]
        for index, task in enumerate(cast(list[dict[str, JsonValue]], manifest["tasks"])):
            if f"{array_id}_{index}" in identifiers or array_id in identifiers:
                names.add(cast(str, task["run_name"]))
    return names


def shares(counts: dict[str, int], limits: dict[str, int], total: int) -> dict[str, int]:
    if not 0 < total <= GPU_RUNNING_LIMIT:
        raise ValueError("array concurrency must be within 1..8")
    allocated = dict.fromkeys(counts, 0)
    while sum(allocated.values()) < total:
        eligible = [key for key in counts if allocated[key] < min(counts[key], limits[key])]
        if not eligible:
            break
        for key in eligible:
            if sum(allocated.values()) >= total:
                break
            allocated[key] += 1
    return allocated


def place_request(
    request: GpuRequest,
    offers: Sequence[GpuOffer],
    allow_short: bool,
    required_node: str | None = None,
) -> GpuRequest:
    resource_type = request.resources.split(":")[1]
    matched = [
        offer
        for offer in offers
        if offer.resource_type == resource_type
        and (required_node is None or offer.node == required_node)
        and (
            resource_type != "nv"
            or request.constraint == "t4|titanv|titanrtx"
            or request.constraint in offer.features
        )
    ]
    partitions = ("gpu-long", "gpu") if allow_short else ("gpu-long",)
    available = next(
        (
            partition
            for partition in partitions
            if any(offer.partition == partition and offer.free_units > 0 for offer in matched)
        ),
        None,
    )
    if available:
        return replace(
            request,
            partitions=(available,),
            time_limit="03:00:00" if available == "gpu" else "3-00:00:00",
        )
    return request


def group_tasks(
    tasks: Sequence[ArrayTask],
    offers: Sequence[GpuOffer],
    constraint: str | None = None,
    required_node: str | None = None,
) -> dict[str, tuple[GpuRequest, tuple[ArrayTask, ...]]]:
    policy = load_policy()
    offers = tuple(
        offer for offer in offers if offer.node not in combined_exclusions(Path.home() / "columba")
    )
    grouped: dict[str, tuple[GpuRequest, list[ArrayTask]]] = {}
    heavy_types = [key for key in policy.array_gpu_slots if key != "nv"]
    heavy_index = 0
    for task in tasks:
        recipe = load_recipe(Path(task.recipe_path))
        choice = (
            None
            if constraint in {"any", "null"}
            else constraint or recipe.constraint_for_seed(task.seed)
        )
        if choice is None and required_node is None:
            heavy = (
                policy.preferences(
                    recipe.base_model,
                    cast(int, recipe.train["imgsz"]),
                    recipe.teacher is not None or "teacher" in recipe.identifier,
                    recipe.estimated_gpu_hours,
                )
                == policy.heavy_preferences
            )
            resource_type = (
                heavy_types[heavy_index % len(heavy_types)] if heavy and heavy_types else "nv"
            )
            heavy_index += heavy
            choice = (
                next(
                    feature
                    for offer in offers
                    if offer.resource_type == "nv"
                    for feature in sorted(offer.features)
                    if feature in {"t4", "titanv", "titanrtx"}
                )
                if resource_type == "nv"
                else "h200"
                if resource_type == "h200-141"
                else resource_type
            )
            request = request_for(choice, offers)
            if resource_type == "nv":
                request = replace(request, constraint="t4|titanv|titanrtx")
            if resource_type == "a100-80" and policy.allow_short_partition:
                partitions = tuple(
                    partition
                    for partition in ("gpu-long", "gpu")
                    if any(
                        offer.resource_type == resource_type and offer.partition == partition
                        for offer in offers
                    )
                )
                request = replace(request, partitions=partitions, time_limit="03:00:00")
        else:
            request = request_for(choice, offers, required_node=required_node)
        request = place_request(request, offers, policy.allow_short_partition, required_node)
        key = json.dumps(asdict(request), sort_keys=True)
        if key not in grouped:
            grouped[key] = (request, [])
        grouped[key][1].append(task)
    return {key: (request, tuple(members)) for key, (request, members) in grouped.items()}


def submit_arrays(
    root: Path,
    checkout: Path,
    tasks: Sequence[ArrayTask],
    gpu_limit: int = GPU_RUNNING_LIMIT,
    constraint: str | None = None,
    required_node: str | None = None,
    reserved_jobs: int = 0,
) -> list[dict[str, JsonValue]]:
    policy = load_policy()
    root = root.resolve()
    if root != (Path.home() / "columba").resolve() or not checkout.resolve().is_relative_to(
        root / "checkouts"
    ):
        raise ValueError("arrays must use ~/columba and its pinned checkouts")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with submission_lock(root):
        control = root / "arrays/concurrency.json"
        if control.exists():
            gpu_limit = min(gpu_limit, cast(int, read_json(control)["gpu_limit"]))
        jobs = queue()
        existing = active_arrays(root, jobs)
        active = active_run_names(root, jobs) | {
            job.name for job in jobs if job.owned and job.state in ACTIVE_STATES
        }
        missing = tuple(
            task
            for task in tasks
            if task.run_name not in active
            and not (root / "runs" / task.relative_run / "DONE").exists()
        )
        if len({task.relative_run for task in tasks}) != len(tasks):
            raise ValueError("duplicate recipe/seed in array")
        desired_names = {task.run_name for task in tasks}
        receipts = [
            record
            for record in existing
            if any(
                cast(str, task["run_name"]) in desired_names
                for task in cast(
                    list[dict[str, JsonValue]],
                    read_json(Path(cast(str, record["manifest"])))["tasks"],
                )
            )
        ]
        if not missing:
            return receipts
        if (
            len(jobs) + len(missing) + reserved_jobs > policy.max_submit_jobs
            or sum(job.owned for job in jobs) + len(missing) + reserved_jobs > Budget().queued_jobs
        ):
            raise RuntimeError("array tasks exceed project/association submitted-job cap")
        footprint = subprocess.run(
            ["du", "-sk", str(root)], check=True, capture_output=True, text=True
        )
        if int(footprint.stdout.split()[0]) >= STORAGE_LIMIT_KIB:
            raise RuntimeError("project storage limit reached")
        grouped = group_tasks(missing, gpu_inventory(), constraint, required_node)
        counts = {key: len(members) for key, (_, members) in grouped.items()}
        counts.update(
            {
                f"existing:{record['array_id']}": cast(int, record["task_count"])
                for record in existing
            }
        )
        type_groups: dict[str, list[str]] = {}
        for key, (request, _) in grouped.items():
            resource_type = request.resources.split(":")[1]
            type_groups.setdefault(resource_type, []).append(key)
        for record in existing:
            type_groups.setdefault(cast(str, record["resource_type"]), []).append(
                f"existing:{record['array_id']}"
            )
        limits: dict[str, int] = {}
        for resource_type, keys in type_groups.items():
            capacity = min(
                policy.type_limits[resource_type],
                policy.array_gpu_slots.get(resource_type, policy.type_limits[resource_type]),
            )
            limits.update(
                shares({key: counts[key] for key in keys}, dict.fromkeys(keys, capacity), capacity)
            )
        throttles = shares(counts, limits, gpu_limit)
        if any(value == 0 for value in throttles.values()):
            raise RuntimeError("too many GPU groups for concurrency budget; split the submission")
        for record in existing:
            persist_throttle(record, throttles[f"existing:{record['array_id']}"])
        results: list[dict[str, JsonValue]] = receipts
        for key, (request, members) in grouped.items():
            if len(members) > MAX_ARRAY_SIZE:
                raise ValueError("array exceeds MaxArraySize")
            payload: dict[str, JsonValue] = {
                "checkout": str(checkout),
                "request": cast(dict[str, JsonValue], asdict(request)),
                "tasks": cast(list[JsonValue], [asdict(task) for task in members]),
            }
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
            directory = root / "arrays" / digest
            payload["directory"] = str(directory)
            atomic_json(directory / "manifest.json", payload)
            script = (checkout / "cluster/templates/nv.sbatch").read_text()
            exports = (
                f"export COLUMBA_CHECKOUT={shlex.quote(str(checkout))}\n"
                f"export COLUMBA_ARRAY_MANIFEST={shlex.quote(str(directory / 'manifest.json'))}\n"
            )
            exports += (
                'eval "$(python -m cluster.arrays task-env --manifest "$COLUMBA_ARRAY_MANIFEST")"\n'
            )
            script = script.replace(
                "set -euo pipefail",
                "set -euo pipefail\ncd " + shlex.quote(str(checkout)) + "\n" + exports,
            )
            command = [
                "sbatch",
                "--parsable",
                "--nice=1000",
                "--export=NONE",
                "--nodes=1",
                f"--job-name=cz-array-{digest}",
                f"--array=0-{len(members) - 1}%{throttles[key]}",
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
            if all(task.recipe_id.startswith("smoke-") for task in members):
                command.extend(["--time=20", "--cpus-per-task=4", "--mem=16G"])
            # Slurm holds the backlog; QoS enforces total/type caps while legacy jobs finish.
            completed = subprocess.run(
                command, input=script, check=True, capture_output=True, text=True
            )
            identifier = completed.stdout.strip().split(";")[0]
            if not identifier.isdecimal():
                raise RuntimeError("invalid sbatch array id")
            submission_record: dict[str, JsonValue] = {
                "array_id": identifier,
                "throttle": throttles[key],
                "resource_type": request.resources.split(":")[1],
                "task_count": len(members),
                "manifest": str(directory / "manifest.json"),
            }
            atomic_json(directory / "submission.json", submission_record)
            results.append(submission_record)
        return results


def throttle_arrays(root: Path, gpu_limit: int) -> list[dict[str, JsonValue]]:
    if root.resolve() != (Path.home() / "columba").resolve():
        raise ValueError("array controls must stay in ~/columba")
    policy = load_policy()
    with submission_lock(root):
        records = active_arrays(root, queue())
        counts = {str(item["array_id"]): cast(int, item["task_count"]) for item in records}
        type_groups: dict[str, list[str]] = {}
        for item in records:
            type_groups.setdefault(cast(str, item["resource_type"]), []).append(
                str(item["array_id"])
            )
        limits: dict[str, int] = {}
        for resource_type, identifiers in type_groups.items():
            capacity = policy.type_limits[resource_type]
            limits.update(
                shares(
                    {key: counts[key] for key in identifiers},
                    dict.fromkeys(identifiers, capacity),
                    capacity,
                )
            )
        allocation = shares(counts, limits, gpu_limit)
        if any(value == 0 for value in allocation.values()):
            raise ValueError("more active arrays than slots; cannot assign zero/unlimited throttle")
        atomic_json(root / "arrays/concurrency.json", {"gpu_limit": gpu_limit})
        for record in records:
            identifier = str(record["array_id"])
            persist_throttle(record, allocation[identifier])
        return list(records)


def retarget_array(
    root: Path, identifier: str, constraint: str, throttle: int | None = None
) -> dict[str, JsonValue]:
    with submission_lock(root):
        fields = owned_job(identifier)
        parent = fields.get("ArrayJobId")
        if not parent:
            raise ValueError("retarget-array requires an array or array task")
        jobs = [
            job
            for job in queue()
            if job.identifier == identifier or job.identifier.startswith(identifier + "_")
        ]
        if not jobs or any(not job.owned or job.state != "PENDING" for job in jobs):
            raise ValueError("retarget-array accepts only currently PENDING owned array tasks")
        current = int(fields.get("ArrayTaskThrottle", "0"))
        selected = current if throttle is None else throttle
        if selected < 1 or selected > GPU_RUNNING_LIMIT:
            raise ValueError("array throttle must be within 1..8")
        offers = tuple(
            offer for offer in gpu_inventory() if offer.node not in combined_exclusions(root)
        )
        request = place_request(
            request_for(constraint, offers), offers, load_policy().allow_short_partition
        )
        resource_type = request.resources.split(":")[1]
        if selected > load_policy().type_limits[resource_type]:
            raise ValueError("array throttle exceeds destination GPU QoS cap")
        active = active_arrays(root, queue())
        control = root / "arrays/concurrency.json"
        overall = (
            cast(int, read_json(control)["gpu_limit"]) if control.exists() else GPU_RUNNING_LIMIT
        )
        others = [record for record in active if str(record["array_id"]) != parent]
        if selected + sum(cast(int, record["throttle"]) for record in others) > overall:
            raise ValueError("retarget throttle exceeds total array concurrency")
        if (
            selected
            + sum(
                cast(int, record["throttle"])
                for record in others
                if record["resource_type"] == resource_type
            )
            > load_policy().type_limits[resource_type]
        ):
            raise ValueError("retarget throttle exceeds combined GPU type cap")
        maximum_minutes = min(
            PARTITION_TIME_LIMIT_MINUTES[partition] for partition in request.partitions
        )
        original = fields.get("TimeLimit", request.time_limit)
        days, clock = original.split("-", 1) if "-" in original else ("0", original)
        components = [int(value) for value in clock.split(":")]
        original_minutes = (
            int(days) * HOURS_PER_DAY * MINUTES_PER_HOUR
            + components[0] * MINUTES_PER_HOUR
            + components[1]
        )
        updated = False
        try:
            subprocess.run(
                [
                    "scontrol",
                    "update",
                    f"JobId={identifier}",
                    f"Partition={','.join(request.partitions)}",
                    f"TresPerNode=gres/{request.resources}",
                    f"Features={request.constraint or ''}",
                    "ReqNodeList=",
                    f"ExcNodeList={','.join(combined_exclusions(root))}",
                    f"TimeLimit={min(original_minutes, maximum_minutes)}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            updated = True
        finally:
            restored = selected if updated else min(current, selected)
            subprocess.run(
                ["scontrol", "update", f"JobId={parent}", f"ArrayTaskThrottle={restored}"],
                check=True,
                capture_output=True,
                text=True,
            )
        for path in (root / "arrays").glob("*/submission.json"):
            record = read_json(path)
            if str(record["array_id"]) == parent:
                record.update(
                    {
                        "throttle": selected,
                        "resource_type": resource_type,
                        "retarget": {
                            "task": identifier,
                            "constraint": constraint,
                            "partition": ",".join(request.partitions),
                        },
                    }
                )
                atomic_json(path, record)
        return {
            "array_id": parent,
            "task": identifier,
            "gpu_request": request.resources,
            "partition": ",".join(request.partitions),
            "throttle": selected,
        }


def task_environment(path: Path, index: int) -> dict[str, str]:
    manifest = read_json(path)
    tasks = cast(list[dict[str, JsonValue]], manifest["tasks"])
    task = tasks[index]
    recipe_path = Path(cast(str, task["recipe_path"])).resolve()
    checkout = Path(cast(str, manifest["checkout"])).resolve()
    root = Path.home() / "columba"
    if not recipe_path.is_relative_to(checkout) and not recipe_path.is_relative_to(
        root / "generated-recipes"
    ):
        raise ValueError("recipe outside pinned checkout/generated recipes")
    if recipe_sha256(recipe_path) != task["recipe_sha256"]:
        raise ValueError("array recipe changed since submission")
    recipe = load_recipe(recipe_path)
    seed = cast(int, task["seed"])
    recipe.constraint_for_seed(seed)
    ready = read_json(checkout / ".columba-ready.json")
    metadata: dict[str, JsonValue] = {
        "recipe_id": recipe.identifier,
        "seed": seed,
        "dataset": recipe.dataset,
        "taxonomy": recipe.taxonomy,
        "model": recipe.base_model,
        "train_imgsz": recipe.train["imgsz"],
        "recipe_sha256": task["recipe_sha256"],
        "git_commit": ready["commit"],
        "source_snapshot": checkout.name,
    }
    values = {
        "COLUMBA_ROOT": str(root),
        "COLUMBA_RECIPE": str(recipe_path),
        "COLUMBA_SEED": str(seed),
        "COLUMBA_RUN_PATH": str(root / "runs" / recipe.identifier / f"seed{seed}"),
        "COLUMBA_GIT_COMMIT": str(ready["commit"]),
        "COLUMBA_RUN_METADATA": json.dumps(metadata),
    }
    if task.get("signal_after_epoch") is not None:
        values["COLUMBA_SIGNAL_AFTER_EPOCH"] = str(task["signal_after_epoch"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "task-env", "throttle", "retarget"))
    parser.add_argument("--payload")
    parser.add_argument("--manifest", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "task-env":
        for name, value in task_environment(
            arguments.manifest, int(os.environ["SLURM_ARRAY_TASK_ID"])
        ).items():
            print(f"export {name}={shlex.quote(value)}")
        return
    payload = cast(dict[str, JsonValue], json.loads(arguments.payload))
    if arguments.action == "throttle":
        print(json.dumps(throttle_arrays(Path.home() / "columba", cast(int, payload["gpu_limit"]))))
        return
    if arguments.action == "retarget":
        print(
            json.dumps(
                retarget_array(
                    Path.home() / "columba",
                    str(payload["job"]),
                    str(payload["constraint"]),
                    cast(int | None, payload.get("throttle")),
                )
            )
        )
        return
    tasks = [
        task_for(
            Path(cast(str, item["recipe_path"])),
            cast(int, item["seed"]),
            cast(int | None, item.get("signal_after_epoch")),
        )
        for item in cast(list[dict[str, JsonValue]], payload["tasks"])
    ]
    print(
        json.dumps(
            submit_arrays(
                Path.home() / "columba",
                REPOSITORY,
                tasks,
                cast(int, payload.get("gpu_limit", GPU_RUNNING_LIMIT)),
                cast(str | None, payload.get("constraint")),
                cast(str | None, payload.get("required_node")),
            )
        )
    )


if __name__ == "__main__":
    main()
