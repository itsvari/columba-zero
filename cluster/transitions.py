"""Short, explicit dependency jobs that apply confirmed mechanical round rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import cast

import yaml

from app.json_data import JsonValue
from cluster.arrays import REPOSITORY, submit_arrays, task_for
from cluster.artifacts import (
    REQUIRED_ARTIFACTS,
    atomic_json,
    leaderboard_row,
    read_json,
    update_leaderboard,
)
from cluster.locking import submission_lock
from cluster.recipe import Recipe, load_recipe, validate_recipe
from cluster.slurm import ACTIVE_STATES, Budget, queue

METRIC_EPSILON = 1e-12
MAX_RECOVERY_ATTEMPTS = 2


class HoldRequired(ValueError):
    pass


@dataclass(frozen=True)
class Rules:
    identifier: str
    approval: str
    inputs: tuple[Path, ...]
    seeds: tuple[int, ...]
    top_k: int
    tie_epsilon: float
    minimum_small_gain: float
    latency_records: Mapping[str, Path]
    tie_order: tuple[str, ...]
    training: Mapping[str, JsonValue]
    nano_imgsz: int | None
    stage: str
    selection_from: str | None
    teachers: tuple[Path, ...]
    teacher_tie_order: tuple[str, ...]
    next_rules: Path | None


@dataclass(frozen=True)
class Candidate:
    recipe_path: Path
    recipe: Recipe
    recall: float
    latency_ms: float


def safe_identifier(value: JsonValue, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value):
        raise ValueError(f"{field} must be a safe identifier")
    return value


def load_rules(path: Path, checkout: Path = REPOSITORY) -> Rules:
    values = cast(dict[str, JsonValue], yaml.safe_load(path.read_text()))
    if not isinstance(values, dict) or values.get("version") != 1:
        raise ValueError("round rules require version 1")
    allowed = {
        "version",
        "id",
        "approval",
        "confirmed_by",
        "notes",
        "inputs",
        "excluded_references",
        "seeds",
        "top_k",
        "tie_epsilon",
        "minimum_small_gain",
        "latency_records",
        "tie_order",
        "training",
        "nano_imgsz",
        "stage",
        "selection_from",
        "teachers",
        "teacher_tie_order",
        "next_rules",
    }
    if set(values) - allowed:
        raise ValueError("unknown round rule fields")
    identifier = safe_identifier(values.get("id"), "id")
    approval = values.get("approval")
    if approval not in {"draft", "confirmed"} or (
        approval == "confirmed" and not values.get("confirmed_by")
    ):
        raise ValueError("rules require draft or named confirmation")

    def paths(field: str) -> tuple[Path, ...]:
        items = values.get(field, [])
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError(f"{field} must list recipe paths")
        resolved = tuple((checkout / cast(str, item)).resolve() for item in items)
        if any(not item.is_relative_to(checkout.resolve()) for item in resolved):
            raise ValueError("rules may only reference pinned checkout files")
        return resolved

    seeds = values.get("seeds")
    training = values.get("training")
    top_k = values.get("top_k")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("seeds must be nonnegative integers")
    if type(top_k) is not int or top_k < 1 or not isinstance(training, dict):
        raise ValueError("top_k and training are required")
    if (
        training.get("nbs") != 64
        or type(training.get("epochs")) is not int
        or cast(int, training["epochs"]) < 1
    ):
        raise ValueError("training requires positive epochs and nbs 64")
    stage = values.get("stage", "plain")
    if stage not in {"plain", "distill"}:
        raise ValueError("only plain/distill transitions are automatic")
    records = values.get("latency_records", {})
    if not isinstance(records, dict) or any(
        not isinstance(value, str) for value in records.values()
    ):
        raise ValueError("latency_records must map model names to C7 files")
    latency = {key: (checkout / cast(str, value)).resolve() for key, value in records.items()}
    if any(not value.is_relative_to(checkout.resolve()) for value in latency.values()):
        raise ValueError("latency records must stay in pinned checkout")
    tie_order = values.get("tie_order", [])
    teacher_order = values.get("teacher_tie_order", [])
    if (
        not isinstance(tie_order, list)
        or not isinstance(teacher_order, list)
        or any(not isinstance(item, str) for item in [*tie_order, *teacher_order])
    ):
        raise ValueError("tie orders must list recipe ids")
    tie_epsilon = values.get("tie_epsilon", 0.01)
    minimum_gain = values.get("minimum_small_gain", 0.04)
    if (
        not isinstance(tie_epsilon, (int, float))
        or not isinstance(minimum_gain, (int, float))
        or not 0 <= tie_epsilon <= 1
        or not 0 <= minimum_gain <= 1
    ):
        raise ValueError("metric margins must be within 0..1")
    nano_imgsz = values.get("nano_imgsz")
    if nano_imgsz is not None and (type(nano_imgsz) is not int or nano_imgsz < 1):
        raise ValueError("nano_imgsz must be positive or null")
    next_rules = values.get("next_rules")
    next_path = (checkout / cast(str, next_rules)).resolve() if next_rules else None
    if next_path and not next_path.is_relative_to(checkout.resolve()):
        raise ValueError("next rules outside checkout")
    selection_from = values.get("selection_from")
    if selection_from is not None:
        selection_from = safe_identifier(selection_from, "selection_from")
    return Rules(
        identifier,
        approval,
        paths("inputs"),
        tuple(cast(list[int], seeds)),
        top_k,
        float(tie_epsilon),
        float(minimum_gain),
        latency,
        tuple(cast(list[str], tie_order)),
        training,
        nano_imgsz,
        stage,
        selection_from,
        paths("teachers"),
        tuple(cast(list[str], teacher_order)),
        next_path,
    )


def completed_metric(root: Path, recipe: Recipe, seed: int) -> float:
    directory = root / "runs" / recipe.identifier / f"seed{seed}"
    if not (directory / "DONE").is_file() or any(
        not (directory / name).is_file() for name in REQUIRED_ARTIFACTS
    ):
        raise HoldRequired(f"incomplete run: {recipe.identifier}/seed{seed}")
    report = read_json(directory / "run.json")
    if report.get("status") != "complete" or report.get("evaluation") != "complete":
        raise HoldRequired(f"failed/evaluation-missing run: {recipe.identifier}/seed{seed}")
    metrics = read_json(directory / "metrics_val.json")
    pigeon = metrics.get("pigeon")
    value = pigeon.get("recall_at_p90") if isinstance(pigeon, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise HoldRequired("missing/nonfinite val recall_at_p90")
    return float(value)


def collect_candidates(root: Path, rules: Rules) -> tuple[Candidate, ...]:
    candidates: list[Candidate] = []
    for path in rules.inputs:
        recipe = load_recipe(path)
        recall = mean(completed_metric(root, recipe, seed) for seed in recipe.seeds)
        if recipe.base_model not in rules.latency_records:
            raise HoldRequired(f"latency missing for {recipe.base_model}")
        latency = read_json(rules.latency_records[recipe.base_model])
        estimated = latency.get("estimated_pi5")
        duration = estimated.get("end_to_end_p95_ms") if isinstance(estimated, dict) else None
        if (
            not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
            or latency.get("input_hw") != [384, 640]
        ):
            raise HoldRequired("missing/invalid estimated Pi latency at 384x640")
        candidates.append(Candidate(path, recipe, recall, float(duration)))
    if not candidates:
        raise HoldRequired("no declared completed candidates")
    return tuple(candidates)


def select_candidates(candidates: Sequence[Candidate], rules: Rules) -> tuple[Candidate, ...]:
    eligible: list[Candidate] = []
    for candidate in candidates:
        size = re.search(r"([ns])\.pt$", candidate.recipe.base_model)
        if not size:
            raise HoldRequired("round selection supports declared n/s models only")
        if size.group(1) == "s":
            nanos = [
                item.recall
                for item in candidates
                if item.recipe.dataset == candidate.recipe.dataset
                and item.recipe.base_model.endswith("n.pt")
            ]
            if not nanos:
                raise HoldRequired("s candidate lacks same-dataset nano comparison")
            if candidate.recall <= max(nanos) + rules.minimum_small_gain + METRIC_EPSILON:
                continue
        eligible.append(candidate)
    ranked = sorted(eligible, key=lambda item: -item.recall)
    result: list[Candidate] = []
    while ranked:
        leading = ranked[0].recall
        tied = [
            item for item in ranked if leading - item.recall <= rules.tie_epsilon + METRIC_EPSILON
        ]
        ranked = [item for item in ranked if item not in tied]
        tied.sort(key=lambda item: item.latency_ms)
        for first, second in zip(tied, tied[1:], strict=False):
            if abs(first.latency_ms - second.latency_ms) <= METRIC_EPSILON and (
                first.recipe.identifier not in rules.tie_order
                or second.recipe.identifier not in rules.tie_order
            ):
                raise HoldRequired("metric/latency tie not covered by predeclared order")
        tied.sort(
            key=lambda item: (
                item.latency_ms,
                rules.tie_order.index(item.recipe.identifier)
                if item.recipe.identifier in rules.tie_order
                else -1,
            )
        )
        result.extend(tied)
    if len(result) < rules.top_k:
        raise HoldRequired("fewer eligible finished combinations than top_k")
    return tuple(result[: rules.top_k])


def best_teacher(root: Path, rules: Rules) -> str:
    values: list[tuple[float, str]] = []
    for path in rules.teachers:
        recipe = load_recipe(path)
        if len(recipe.seeds) != 1:
            raise HoldRequired("teacher must have one declared seed")
        values.append(
            (
                completed_metric(root, recipe, recipe.seeds[0]),
                f"{recipe.identifier}/seed{recipe.seeds[0]}",
            )
        )
    if len(values) < 2:
        raise HoldRequired("both teachers must finish")
    highest = max(value for value, _ in values)
    tied = [name for value, name in values if abs(highest - value) <= METRIC_EPSILON]
    if len(tied) == 1:
        return tied[0]
    if any(name.split("/")[0] not in rules.teacher_tie_order for name in tied):
        raise HoldRequired("teacher tie not covered by rules")
    return min(tied, key=lambda name: rules.teacher_tie_order.index(name.split("/")[0]))


def generate_recipes(
    selected: Sequence[Candidate], rules: Rules, destination: Path, teacher: str | None = None
) -> tuple[Path, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for candidate in selected:
        sizes = [640]
        if (
            rules.stage == "plain"
            and rules.nano_imgsz
            and candidate.recipe.base_model.endswith("n.pt")
        ):
            sizes.append(rules.nano_imgsz)
        for size in sizes:
            original = cast(dict[str, JsonValue], yaml.safe_load(candidate.recipe_path.read_text()))
            prefix = re.sub(r"-640-r1$", "", candidate.recipe.identifier)
            identifier = f"{prefix}-{size}-{rules.identifier}"
            train = {**candidate.recipe.train, **rules.training, "imgsz": size}
            deploy_sizes = (
                [[352, 608], [384, 640]]
                if candidate.recipe.base_model.endswith("s.pt")
                else [[352, 608], [384, 640], [448, 800]]
            )
            generated: dict[str, JsonValue] = {
                **original,
                "id": identifier,
                "seeds": list(rules.seeds),
                "gpu_constraint": None,
                "teacher": teacher,
                "train": train,
                "deploy_sizes": cast(list[JsonValue], deploy_sizes),
                "notes": (
                    f"Mechanical {rules.identifier}, source {candidate.recipe.identifier}; "
                    "selection on val only."
                ),
            }
            generated.pop("seed_gpu_constraints", None)
            validate_recipe(generated)
            path = destination / f"{identifier}.yaml"
            text = yaml.safe_dump(generated, sort_keys=False)
            if path.exists() and path.read_text() != text:
                raise HoldRequired("generated recipe would change an existing run identity")
            path.write_text(text)
            paths.append(path)
    return tuple(paths)


def require_owned_dependencies(identifiers: Sequence[str]) -> None:
    if not identifiers:
        raise ValueError("transition needs explicit dependencies")
    for identifier in identifiers:
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", identifier):
            raise ValueError("invalid dependency job id")
        result = subprocess.run(
            [
                "sacct",
                "-j",
                identifier,
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,JobName%100",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        names = [
            row.split("|")[1] for row in result.stdout.splitlines() if "." not in row.split("|")[0]
        ]
        if not names or any(not name.startswith("cz-") for name in names):
            raise ValueError(f"dependency outside cz- namespace: {identifier}")


def arm_transition(
    root: Path,
    checkout: Path,
    rules_path: Path,
    after: Sequence[str],
    teacher_after: Sequence[str] = (),
    upstream: str | None = None,
    prepare_only: bool = False,
) -> dict[str, JsonValue]:
    with submission_lock(root):
        return _arm_transition(
            root, checkout, rules_path, after, teacher_after, upstream, prepare_only
        )


def _arm_transition(
    root: Path,
    checkout: Path,
    rules_path: Path,
    after: Sequence[str],
    teacher_after: Sequence[str] = (),
    upstream: str | None = None,
    prepare_only: bool = False,
) -> dict[str, JsonValue]:
    if root.resolve() != (
        Path.home() / "columba"
    ).resolve() or not checkout.resolve().is_relative_to(root.resolve() / "checkouts"):
        raise ValueError("transitions must use pinned project checkouts")
    rules = load_rules(rules_path, checkout)
    if rules.next_rules and not teacher_after:
        raise ValueError("second transition needs both teacher dependencies")
    if not prepare_only and rules.approval != "confirmed":
        raise ValueError("draft rules require WS5 confirmation before arming")
    if (
        not prepare_only
        and rules.next_rules
        and load_rules(rules.next_rules, checkout).approval != "confirmed"
    ):
        raise ValueError("second-stage rules also require confirmation")
    require_owned_dependencies([*after, *teacher_after])
    identity = hashlib.sha256(
        json.dumps(
            [
                rules.identifier,
                hashlib.sha256(rules_path.read_bytes()).hexdigest(),
                list(after),
                list(teacher_after),
                upstream,
            ]
        ).encode()
    ).hexdigest()[:16]
    directory = root / "transitions" / identity
    record_path = directory / "submission.json"
    if record_path.exists():
        return read_json(record_path)
    jobs = queue()
    if not prepare_only and (
        sum(job.owned and job.state in ACTIVE_STATES for job in jobs) >= Budget().queued_jobs
        or sum(job.owned and not job.gpu and job.state in ACTIVE_STATES for job in jobs)
        >= Budget().cpu_jobs
    ):
        raise HoldRequired("no CPU/queued budget for transition")
    plan: dict[str, JsonValue] = {
        "rules": str(rules_path),
        "checkout": str(checkout),
        "after": list(after),
        "teacher_after": list(teacher_after),
        "upstream": upstream,
        "directory": str(directory),
        "round": rules.identifier,
    }
    atomic_json(directory / "plan.json", plan)
    if prepare_only:
        return {
            "status": "prepared, not armed",
            "plan": str(directory / "plan.json"),
            "after": list(after),
        }
    script = f"""#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodelist=xcng0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
trap 'rm -rf -- "/tmp/cz-${{SLURM_JOB_ID:?}}"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd {shlex.quote(str(checkout))}
bash cluster/bootstrap.sh --cpu
environment="$(bash cluster/bootstrap.sh --cpu --print-env)"
plan={shlex.quote(str(directory / "plan.json"))}
"$environment/bin/python" -m cluster.transitions execute --plan "$plan"
"""
    active_ids = {job.identifier for job in jobs if job.owned}
    active_array_ids = {identifier.split("_")[0] for identifier in active_ids}
    effective_after = [
        identifier for identifier in after if identifier in active_ids | active_array_ids
    ]
    command = [
        "sbatch",
        "--parsable",
        "--nice=1000",
        "--export=NONE",
        f"--job-name=cz-transition-{rules.identifier}-{identity}",
        f"--output={root}/logs/%x-%j.log",
    ]
    if effective_after:
        command.append(f"--dependency=afterany:{':'.join(effective_after)}")
    completed = subprocess.run(
        command,
        input=script,
        check=True,
        capture_output=True,
        text=True,
    )
    identifier = completed.stdout.strip().split(";")[0]
    if not identifier.isdecimal():
        raise RuntimeError("invalid transition job id")
    record: dict[str, JsonValue] = {
        "job_id": identifier,
        "round": rules.identifier,
        "after": list(after),
        "effective_after": cast(list[JsonValue], effective_after),
        "plan": str(directory / "plan.json"),
    }
    atomic_json(record_path, record)
    return record


def terminal_state(job_id: str) -> str:
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
        raise HoldRequired("missing valid job id for checkpoint recovery")
    result = subprocess.run(
        ["sacct", "-j", job_id, "--noheader", "--parsable2", "--format=JobIDRaw,State"],
        check=True,
        capture_output=True,
        text=True,
    )
    states = [
        line.split("|")[1].split()[0].rstrip("+")
        for line in result.stdout.splitlines()
        if "." not in line.split("|")[0]
    ]
    return states[0] if states else "UNKNOWN"


def recover_interrupted_inputs(
    root: Path, checkout: Path, rules: Rules, plan: dict[str, JsonValue], directory: Path
) -> bool:
    pending: list[tuple[Path, int, str, int]] = []
    ledger_path = root / "transitions/recovery" / f"{rules.identifier}.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"runs": {}}
    entries = cast(dict[str, JsonValue], ledger["runs"])
    for path in rules.inputs:
        recipe = load_recipe(path)
        for seed in recipe.seeds:
            relative = f"{recipe.identifier}/seed{seed}"
            run_path = root / "runs" / relative
            if (run_path / "DONE").exists():
                continue
            report = read_json(run_path / "run.json") if (run_path / "run.json").exists() else {}
            job_id = (
                f"{report['array_job_id']}_{report['array_task_id']}"
                if report.get("array_job_id") is not None
                and report.get("array_task_id") is not None
                else str(report.get("job_id", ""))
            )
            state = terminal_state(job_id)
            if (
                state not in {"TIMEOUT", "PREEMPTED", "NODE_FAIL"}
                or not (run_path / "weights/last.pt").exists()
            ):
                raise HoldRequired(f"{relative}: {state}, no mechanical recovery allowed")
            epoch = cast(int, report.get("epochs_done", 0))
            previous = cast(dict[str, JsonValue], entries.get(relative, {}))
            if cast(int, previous.get("count", 0)) >= MAX_RECOVERY_ATTEMPTS or epoch <= cast(
                int, previous.get("last_epoch", 0)
            ):
                raise HoldRequired(f"{relative}: recovery limit/no checkpoint progress")
            pending.append((path, seed, relative, epoch))
    if not pending:
        return False
    with submission_lock(root):
        jobs = queue()
        if (
            sum(job.owned and not job.gpu and job.state in ACTIVE_STATES for job in jobs)
            >= Budget().cpu_jobs
        ):
            raise HoldRequired("no CPU budget to re-arm recovered transition")
        arrays = submit_arrays(
            root, checkout, [task_for(path, seed) for path, seed, _, _ in pending]
        )
        if not arrays:
            raise HoldRequired("recovery tasks already active; explicit dependencies required")
        for _, _, relative, epoch in pending:
            previous = cast(dict[str, JsonValue], entries.get(relative, {}))
            entries[relative] = {
                "count": cast(int, previous.get("count", 0)) + 1,
                "last_epoch": epoch,
                "array_ids": [item["array_id"] for item in arrays],
            }
        atomic_json(ledger_path, {"round": rules.identifier, "runs": entries})
        successor = arm_transition(
            root,
            checkout,
            Path(cast(str, plan["rules"])),
            [os.environ["SLURM_JOB_ID"], *(str(item["array_id"]) for item in arrays)],
            cast(list[str], plan.get("teacher_after", [])),
            cast(str | None, plan.get("upstream")),
        )
        atomic_json(
            directory / "run.json",
            {
                "status": "recovering",
                "round": rules.identifier,
                "recovered_runs": [item[2] for item in pending],
                "arrays": cast(list[JsonValue], arrays),
                "next_transition": successor,
            },
        )
        (directory / "DONE").write_text("recovery submitted\n")
    return True


def consolidate(root: Path, rules: Rules) -> None:
    rows: list[dict[str, str]] = []
    for path in [*rules.inputs, *rules.teachers]:
        recipe = load_recipe(path)
        for seed in recipe.seeds:
            run_path = root / "runs" / recipe.identifier / f"seed{seed}"
            if (run_path / "run.json").exists():
                row = leaderboard_row(run_path)
                row["run_path"] = f"runs/{recipe.identifier}/seed{seed}"
                rows.append(row)
    leaderboard = (
        root
        / "experiments"
        / ("leaderboard-smoke.csv" if rules.identifier.startswith("smoke-") else "leaderboard.csv")
    )
    update_leaderboard(leaderboard, rows)


def execute_transition(plan_path: Path) -> None:
    root = (Path.home() / "columba").resolve()
    if not plan_path.resolve().is_relative_to(root / "transitions"):
        raise ValueError("transition plan must stay in ~/columba/transitions")
    directory = plan_path.parent
    if (directory / "DONE").exists() or (directory / "HOLD").exists():
        return
    round_identifier = directory.name
    try:
        plan = read_json(plan_path)
        checkout = Path(cast(str, plan["checkout"]))
        rules = load_rules(Path(cast(str, plan["rules"])), checkout)
        round_identifier = rules.identifier
        if rules.approval != "confirmed":
            raise HoldRequired("rules lost their confirmation")
        if rules.next_rules and not plan.get("teacher_after"):
            raise HoldRequired("teacher dependencies were not declared at arming")
        if recover_interrupted_inputs(root, checkout, rules, plan, directory):
            return
        consolidate(root, rules)
        candidates = collect_candidates(root, rules)
        selected = select_candidates(candidates, rules)
        if rules.stage == "distill":
            upstream = plan.get("upstream")
            if not isinstance(upstream, str):
                raise HoldRequired("distillation needs successful plain transition")
            upstream_report = read_json(root / "transitions" / upstream / "run.json")
            expected_ids = cast(list[str], upstream_report["selected"])
            selected = tuple(item for item in candidates if item.recipe.identifier in expected_ids)
            if len(selected) != rules.top_k:
                raise HoldRequired("plain selected combinations do not match distillation rules")
            for generated_path in cast(list[str], upstream_report["recipes"]):
                generated_recipe = load_recipe(Path(generated_path))
                for seed in generated_recipe.seeds:
                    completed_metric(root, generated_recipe, seed)
        teacher = best_teacher(root, rules) if rules.stage == "distill" else None
        atomic_json(
            directory / "ranking.json",
            {
                "label": "measured validation / estimated Pi latency",
                "candidates": [
                    {
                        "recipe_id": item.recipe.identifier,
                        "val_recall_at_p90": item.recall,
                        "estimated_pi_p95_ms": item.latency_ms,
                    }
                    for item in candidates
                ],
                "selected": [item.recipe.identifier for item in selected],
                "teacher": teacher,
            },
        )
        paths = generate_recipes(
            selected, rules, root / "generated-recipes" / rules.identifier, teacher
        )
        tasks = [task_for(path, seed) for path in paths for seed in load_recipe(path).seeds]
        with submission_lock(root):
            current_jobs = queue()
            if (
                rules.next_rules
                and sum(
                    job.owned and not job.gpu and job.state in ACTIVE_STATES for job in current_jobs
                )
                >= Budget().cpu_jobs
            ):
                raise HoldRequired("no CPU budget for declared second transition")
            arrays = submit_arrays(root, checkout, tasks)
            report: dict[str, JsonValue] = {
                "status": "submitted",
                "round": rules.identifier,
                "selected": [item.recipe.identifier for item in selected],
                "recipes": [str(path) for path in paths],
                "arrays": cast(list[JsonValue], arrays),
                "teacher": teacher,
            }
            if rules.next_rules:
                teacher_after = cast(list[str], plan.get("teacher_after", []))
                if not teacher_after:
                    raise HoldRequired("teacher dependencies were not declared at arming")
                atomic_json(directory / "run.json", report)
                report["next_transition"] = arm_transition(
                    root,
                    checkout,
                    rules.next_rules,
                    [
                        os.environ["SLURM_JOB_ID"],
                        *(str(item["array_id"]) for item in arrays),
                        *teacher_after,
                    ],
                    upstream=directory.name,
                )
            atomic_json(directory / "run.json", report)
            (directory / "DONE").write_text("submitted\n")
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        atomic_json(directory / "ranking.json", {"status": "HOLD", "reason": str(error)})
        (directory / "HOLD").write_text(str(error) + "\n")
        atomic_json(
            directory / "run.json",
            {"status": "HOLD", "round": round_identifier, "reason": str(error)},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("arm", "execute"))
    parser.add_argument("--payload")
    parser.add_argument("--plan", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "execute":
        if not os.environ.get("SLURM_JOB_NAME", "").startswith("cz-"):
            raise ValueError("transition execution needs cz- compute job")
        execute_transition(arguments.plan)
        return
    payload = cast(dict[str, JsonValue], json.loads(arguments.payload))
    print(
        json.dumps(
            arm_transition(
                Path.home() / "columba",
                REPOSITORY,
                Path(cast(str, payload["rules"])),
                cast(list[str], payload["after"]),
                cast(list[str], payload.get("teacher_after", [])),
                prepare_only=bool(payload.get("prepare_only", False)),
            )
        )
    )


if __name__ == "__main__":
    main()
