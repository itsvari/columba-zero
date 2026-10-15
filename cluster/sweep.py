"""Plan, submit, inspect, fetch and cancel C5 runs over BatchMode SSH."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from app.json_data import JsonValue
from cluster.artifacts import LEADERBOARD_COLUMNS, leaderboard_row, update_leaderboard
from cluster.gpu_selection import GPU_CONSTRAINTS, GPU_NODE_PATTERN, selector
from cluster.recipe import Recipe, load_recipe, recipe_sha256
from cluster.slurm import (
    ACTIVE_STATES,
    CPU_RUNNING_LIMIT,
    GPU_RUNNING_LIMIT,
    QUEUED_LIMIT,
    Budget,
    Job,
    Submission,
    job_name,
    parse_queue,
)

SSH_HOST = "columba-cluster"
REPOSITORY = Path(__file__).resolve().parents[1]
SMOKE_TIME_LIMIT_MINUTES = 10
SMOKE_CPU_COUNT = 4
SMOKE_MEMORY_LIMIT_GB = 16
PROBE_TIME_LIMIT_MINUTES = 1
EXCLUDED_NODES_FILE = REPOSITORY / "cluster/excluded_nodes.txt"


def excluded_nodes(extra: Sequence[str] = (), path: Path = EXCLUDED_NODES_FILE) -> list[str]:
    """Return the default exclusion list from `path` followed by any extra nodes."""
    listed = (line.split("#", 1)[0].strip() for line in path.read_text("utf-8").splitlines())
    return list(dict.fromkeys([*(node for node in listed if node), *extra]))


def remote(command: Sequence[str], input_bytes: bytes | None = None) -> str:
    try:
        completed = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                SSH_HOST,
                shlex.join(command),
            ],
            input=input_bytes,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.decode().strip()) from error
    return completed.stdout.decode()


def remote_root() -> Path:
    return Path(
        remote(["python", "-c", "from pathlib import Path; print(Path.home() / 'columba')"]).strip()
    )


def current_queue() -> tuple[Job, ...]:
    return parse_queue(
        remote(["sh", "-c", 'squeue --array -u "$USER" --noheader --format="%i|%j|%T|%P|%b|%D|%V"'])
    )


@dataclass(frozen=True)
class DesiredRun:
    recipe_path: Path
    recipe: Recipe
    seed: int

    @property
    def name(self) -> str:
        return job_name(self.recipe.identifier, self.seed)

    @property
    def relative_run(self) -> str:
        return f"{self.recipe.identifier}/seed{self.seed}"


def desired_runs(paths: Sequence[Path]) -> tuple[DesiredRun, ...]:
    runs = tuple(
        DesiredRun(path.resolve(), recipe, seed)
        for path in paths
        for recipe in (load_recipe(path),)
        for seed in recipe.seeds
    )
    if len({run.relative_run for run in runs}) != len(runs):
        raise ValueError("duplicate recipe/seed pairs")
    return runs


def inspect_runs(root: Path, runs: Sequence[DesiredRun]) -> dict[str, dict[str, JsonValue]]:
    relative_runs = [run.relative_run for run in runs]
    script = (
        "import json,pathlib; root=pathlib.Path(" + repr(str(root / "runs")) + "); "
        "names=" + repr(relative_runs) + "; "
        "print(json.dumps({name:{'done':(root/name/'DONE').is_file(),"
        "'report':json.loads((root/name/'run.json').read_text()) "
        "if (root/name/'run.json').is_file() else {}} for name in names}))"
    )
    return cast(dict[str, dict[str, JsonValue]], json.loads(remote(["python", "-c", script])))


def plan_runs(
    runs: Sequence[DesiredRun], done: set[str], jobs: Sequence[Job]
) -> list[dict[str, str]]:
    active = {job.name: job for job in jobs if job.owned and job.state in ACTIVE_STATES}
    return [
        {
            "run": run.relative_run,
            "name": run.name,
            "state": "done"
            if run.relative_run in done
            else active[run.name].state
            if run.name in active
            else "missing",
            "lane": run.recipe.lane,
            "constraint": run.recipe.constraint_for_seed(run.seed) or "any GPU",
        }
        for run in runs
    ]


def deploy(root: Path) -> tuple[Path, str]:
    tracked = (
        subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in sorted(set(tracked) - {""}):
            path = REPOSITORY / name
            if path.is_file():
                archive.add(path, arcname=name)
    content = buffer.getvalue()
    digest = hashlib.sha256(content).hexdigest()[:16]
    checkout = root / "checkouts" / digest
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True, text=True, capture_output=True
    ).stdout.strip()
    publication_code = (REPOSITORY / "cluster/deployment.py").read_text()
    remote(
        [
            "python",
            "-c",
            publication_code,
            "--root",
            str(root / "checkouts"),
            "--digest",
            digest,
            "--commit",
            commit,
        ],
        content,
    )
    return checkout, commit


def submission_for(
    run: DesiredRun,
    root: Path,
    checkout: Path,
    commit: str,
    stop_after_epoch: int | None = None,
    constraint: str | None = None,
    required_node: str | None = None,
) -> Submission:
    """Build one submission; `constraint` replaces the recipe's GPU type for this submission."""
    if constraint is not None:
        selector(None if constraint in {"any", "null"} else constraint)
        if run.recipe.lane != "nv":
            raise ValueError("--constraint needs a GPU recipe")
    if required_node and (run.recipe.lane != "nv" or not GPU_NODE_PATTERN.fullmatch(required_node)):
        raise ValueError("--node needs a GPU recipe and an approved GPU node")
    try:
        relative_recipe = run.recipe_path.relative_to(REPOSITORY)
    except ValueError as error:
        raise ValueError("recipes must be inside this checkout") from error
    template = (REPOSITORY / "cluster/templates" / f"{run.recipe.lane}.sbatch").read_text()
    metadata = {
        "recipe_id": run.recipe.identifier,
        "seed": run.seed,
        "dataset": run.recipe.dataset,
        "taxonomy": run.recipe.taxonomy,
        "model": run.recipe.base_model,
        "train_imgsz": run.recipe.train["imgsz"],
        "recipe_sha256": recipe_sha256(run.recipe_path),
        "git_commit": commit,
        "epochs_done": 0,
        "source_snapshot": checkout.name,
    }
    if constraint is not None:
        metadata["constraint_override"] = constraint
    exports = "\n".join(
        f"export {key}={shlex.quote(value)}"
        for key, value in {
            "COLUMBA_CHECKOUT": str(checkout),
            "COLUMBA_RECIPE": str(checkout / relative_recipe),
            "COLUMBA_SEED": str(run.seed),
            "COLUMBA_ROOT": str(root),
            "COLUMBA_GIT_COMMIT": commit,
            "COLUMBA_RUN_PATH": str(root / "runs" / run.relative_run),
            "COLUMBA_RUN_METADATA": json.dumps(metadata),
        }.items()
    )
    if stop_after_epoch is not None:
        if not run.recipe.identifier.startswith("smoke-") or not 0 < stop_after_epoch < cast(
            int, run.recipe.train["epochs"]
        ):
            raise ValueError("--stop-after-epoch requires an unfinished smoke epoch")
        exports += f"\nexport COLUMBA_STOP_AFTER_EPOCH={stop_after_epoch}"
    script = template.replace("set -euo pipefail", f"set -euo pipefail\n{exports}")
    submission: Submission = {
        "name": run.name,
        "lane": run.recipe.lane,
        "constraint": (None if constraint in {"any", "null"} else constraint)
        if constraint is not None
        else run.recipe.constraint_for_seed(run.seed),
        "script": script,
        "run_path": str(root / "runs" / run.relative_run),
        "base_model": run.recipe.base_model,
        "image_size_px": cast(int, run.recipe.train["imgsz"]),
        "teacher": run.recipe.teacher is not None or "teacher" in run.recipe.identifier,
        "est_gpu_hours": run.recipe.estimated_gpu_hours,
    }
    if run.recipe.identifier.startswith("smoke-"):
        submission["time_limit_minutes"] = SMOKE_TIME_LIMIT_MINUTES
        if run.recipe.lane == "nv":
            submission["cpu_count"] = SMOKE_CPU_COUNT
            submission["memory_limit_gb"] = SMOKE_MEMORY_LIMIT_GB
    if required_node:
        submission["required_node"] = required_node
    return submission


def fetch(root: Path, runs: Sequence[DesiredRun], destination: Path, leaderboard: Path) -> None:
    rows: list[dict[str, str]] = []
    for run in runs:
        local = destination / run.relative_run
        local.mkdir(parents=True, exist_ok=True)
        source = str(root / "runs" / run.relative_run) + "/"
        completed = subprocess.run(
            [
                "rsync",
                "-a",
                "--prune-empty-dirs",
                "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=20",
                "--include=weights/",
                "--include=weights/best.pt",
                "--include=weights/last.pt",
                "--include=*.json",
                "--include=results.csv",
                "--include=DONE",
                "--include=TRAINED",
                "--include=HOLD",
                "--exclude=*",
                f"{SSH_HOST}:{source}",
                str(local) + "/",
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip())
        if (local / "run.json").exists():
            row = leaderboard_row(local)
            production_leaderboard = any(
                path.name == "leaderboard.csv" and path.parent.name == "experiments"
                for path in (leaderboard, leaderboard.resolve())
            )
            smoke = run.recipe.identifier.startswith("smoke-") or row["recipe_id"].startswith(
                "smoke-"
            )
            if not smoke or not production_leaderboard:
                rows.append(row)
    if rows or not leaderboard.exists():
        update_leaderboard(leaderboard, rows)


def fetch_orchestration(root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    names = ("arrays", "transitions", "generated-recipes", "state")
    remote(["mkdir", "-p", *(str(root / name) for name in names)])
    for name in names:
        completed = subprocess.run(
            [
                "rsync",
                "-a",
                "--prune-empty-dirs",
                "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=20",
                "--include=*/",
                "--include=*.json",
                "--include=*.yaml",
                "--include=DONE",
                "--include=HOLD",
                "--exclude=*",
                f"{SSH_HOST}:{root / name}",
                str(destination) + "/",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stderr:
            print(completed.stderr.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "plan",
            "submit",
            "status",
            "fetch",
            "cancel",
            "arm-bundle",
            "gpu-probe",
            "submit-arrays",
            "arm-transition",
            "prepare-transition",
            "throttle-arrays",
            "retarget-array",
        ),
    )
    parser.add_argument("--recipes", nargs="+", type=Path, default=[])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--gpu-limit", type=int, default=GPU_RUNNING_LIMIT)
    parser.add_argument("--cpu-limit", type=int, default=CPU_RUNNING_LIMIT)
    parser.add_argument("--queue-limit", type=int, default=QUEUED_LIMIT)
    parser.add_argument("--job-ids", nargs="+", default=[])
    parser.add_argument(
        "--exclude-nodes",
        nargs="+",
        default=[],
        help="nodes to exclude in addition to cluster/excluded_nodes.txt",
    )
    parser.add_argument("--probe-node", default="xgpe6")
    parser.add_argument("--node", help="submit: pin an approved GPU node, including xgpj0")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument(
        "--constraint",
        choices=(*GPU_CONSTRAINTS, "any", "null"),
        help="submit: override the recipe's GPU type (recipe hash unchanged; use with --seeds)",
    )
    parser.add_argument("--destination", type=Path, default=REPOSITORY / "experiments/runs")
    parser.add_argument(
        "--leaderboard", type=Path, default=REPOSITORY / "experiments/leaderboard.csv"
    )
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--bundle-action", choices=("export", "benchmark"), default="export")
    parser.add_argument(
        "--retarget-pending-after-hours",
        type=float,
        help="cancel expired PENDING requested cz- runs and re-target to free preferred GPUs",
    )
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--after", nargs="+", default=[])
    parser.add_argument("--teacher-after", nargs="+", default=[])
    parser.add_argument("--signal-after-epoch", type=int)
    parser.add_argument("--clear-on-success", action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--throttle", type=int)
    parser.add_argument("--upstream")
    parser.add_argument("--generated-round", nargs="+", default=[])
    parser.add_argument(
        "--state-destination", type=Path, default=REPOSITORY / "outputs/ws3/cluster-state"
    )
    arguments = parser.parse_args()
    if (
        arguments.retarget_pending_after_hours is not None
        and arguments.retarget_pending_after_hours <= 0
    ):
        parser.error("pending re-target timeout must be positive")
    budget = Budget(arguments.gpu_limit, arguments.cpu_limit, arguments.queue_limit)
    runs = desired_runs(arguments.recipes)
    if arguments.seeds is not None:
        if not set(arguments.seeds).issubset({run.seed for run in runs}):
            parser.error("--seeds must be present in the requested recipes")
        runs = tuple(run for run in runs if run.seed in arguments.seeds)
    root = remote_root()
    if arguments.action in {
        "submit-arrays",
        "arm-transition",
        "prepare-transition",
        "throttle-arrays",
        "retarget-array",
    }:
        if arguments.action in {"arm-transition", "prepare-transition"}:
            from cluster.transitions import load_rules

            if arguments.rules is None or (
                arguments.action == "arm-transition"
                and load_rules(arguments.rules).approval != "confirmed"
            ):
                parser.error("arm-transition requires WS5-confirmed rules")
        checkout, commit = deploy(root)
        if arguments.action == "retarget-array":
            if (
                not arguments.job
                or not arguments.constraint
                or arguments.constraint in {"any", "null"}
            ):
                parser.error("retarget-array requires --job and a typed --constraint")
            orchestration_payload = {
                "job": arguments.job,
                "constraint": arguments.constraint,
                "throttle": arguments.throttle,
            }
            module, action = "cluster.arrays", "retarget"
        elif arguments.action == "throttle-arrays":
            orchestration_payload = {"gpu_limit": arguments.gpu_limit}
            module, action = "cluster.arrays", "throttle"
        elif arguments.action == "submit-arrays":
            tasks = [
                {
                    "recipe_path": str(checkout / run.recipe_path.relative_to(REPOSITORY)),
                    "seed": run.seed,
                    "signal_after_epoch": arguments.signal_after_epoch,
                }
                for run in runs
            ]
            orchestration_payload = {
                "tasks": tasks,
                "gpu_limit": arguments.gpu_limit,
                "constraint": arguments.constraint,
                "required_node": arguments.node,
            }
            module, action = "cluster.arrays", "submit"
        else:
            orchestration_payload = {
                "rules": str(checkout / arguments.rules.resolve().relative_to(REPOSITORY)),
                "after": arguments.after,
                "teacher_after": arguments.teacher_after,
                "prepare_only": arguments.action == "prepare-transition",
                "upstream": arguments.upstream,
            }
            module, action = "cluster.transitions", "arm"
        print(
            remote(
                [
                    "sh",
                    "-c",
                    f"cd {shlex.quote(str(checkout))} && python -m {module} {action} "
                    f"--payload {shlex.quote(json.dumps(orchestration_payload))}",
                ]
            )
        )
        return
    jobs = current_queue()
    state = inspect_runs(root, runs)
    orchestration = json.loads(
        remote(
            [
                "python",
                "-c",
                (REPOSITORY / "cluster/state.py").read_text(),
                "--root",
                str(root),
                "--jobs",
                json.dumps({job.identifier: job.state for job in jobs if job.owned}),
            ]
        )
    )
    done = {name for name, details in state.items() if details["done"]}
    if arguments.action in {"plan", "status"}:
        array_jobs = {
            str(task["run_name"]): str(task["state"])
            for task in orchestration["arrays"]
            if task["state"] in ACTIVE_STATES
        }
        planned_jobs = (
            *jobs,
            *(Job("array", name, phase, "gpu", "gpu:1") for name, phase in array_jobs.items()),
        )
        for item in plan_runs(runs, done, planned_jobs):
            if arguments.action == "status":
                report = state[item["run"]].get("report", {})
                if isinstance(report, dict):
                    item["report_status"] = str(report.get("status", "absent"))
                    item["epochs_done"] = str(report.get("epochs_done", 0))
            print(json.dumps(item))
        if not runs:
            for job in jobs:
                if job.owned:
                    metadata = asdict(job)
                    metadata["submitted_at"] = (
                        job.submitted_at.isoformat() if job.submitted_at else None
                    )
                    print(json.dumps(metadata))
        if arguments.action == "status":
            for item in [
                *orchestration["arrays"],
                *orchestration["transitions"],
                *orchestration.get("bad_nodes", []),
                *orchestration.get("device_runs", []),
            ]:
                print(json.dumps(item))
        return
    if arguments.action == "fetch":
        fetch_orchestration(root, arguments.state_destination)
        generated_paths: list[Path] = []
        for identifier in arguments.generated_round:
            if Path(identifier).name != identifier or identifier in {".", ".."}:
                parser.error("--generated-round requires a round identifier")
            generated_paths.extend(
                sorted(
                    (arguments.state_destination / "generated-recipes" / identifier).glob("*.yaml")
                )
            )
        runs = (*runs, *desired_runs(generated_paths))
        if generated_paths:
            state = inspect_runs(root, runs)
        available = [run for run in runs if state[run.relative_run].get("report")]
        fetch(root, available, arguments.destination, arguments.leaderboard)
        print(
            f"Fetched {len(runs)} requested runs; C6 header has {len(LEADERBOARD_COLUMNS)} columns"
        )
        return
    if arguments.action == "cancel":
        if not arguments.job_ids:
            parser.error("cancel requires explicit --job-ids")
        if any(
            identifier not in {job.identifier for job in jobs if job.owned}
            for identifier in arguments.job_ids
        ):
            raise ValueError("cancel accepts only currently queued cz- job ids")
    checkout, commit = deploy(root)
    if arguments.action == "arm-bundle":
        submissions = [bundle_submission(arguments.models, arguments.bundle_action, root, checkout)]
    elif arguments.action == "gpu-probe":
        submissions = [
            probe_submission(arguments.probe_node, root, checkout, arguments.clear_on_success)
        ]
    else:
        submissions = [
            submission_for(
                run,
                root,
                checkout,
                commit,
                arguments.stop_after_epoch,
                arguments.constraint,
                arguments.node,
            )
            for run in runs
            if run.relative_run not in done
        ]
    for submission in submissions:
        submission["excluded_nodes"] = (
            [] if arguments.action == "gpu-probe" else excluded_nodes(arguments.exclude_nodes)
        )
    payload: dict[str, JsonValue] = {
        "budget": cast(dict[str, JsonValue], asdict(budget)),
        "retarget_pending_after_hours": arguments.retarget_pending_after_hours,
        "submissions": cast(list[JsonValue], submissions),
        "job_ids": cast(list[JsonValue], arguments.job_ids),
    }
    print(
        remote(
            [
                "sh",
                "-c",
                f"cd {shlex.quote(str(checkout))} && python -m cluster.slurm "
                f"{'cancel' if arguments.action == 'cancel' else 'submit'} "
                f"--payload {shlex.quote(json.dumps(payload))}",
            ]
        )
    )


def probe_submission(
    node: str, root: Path, checkout: Path, clear_on_success: bool = False
) -> Submission:
    if not GPU_NODE_PATTERN.fullmatch(node):
        raise ValueError("probe node must be an approved GPU node")
    template = (REPOSITORY / "cluster/templates/gpu-probe.sbatch").read_text()
    destination = root / "runs/gpu-probes" / node
    exports = f"export COLUMBA_CHECKOUT={shlex.quote(str(checkout))}\n"
    exports += f"export COLUMBA_RUN_PATH={shlex.quote(str(destination))}\n"
    exports += f"export COLUMBA_PROBE_CLEAR_ON_SUCCESS={str(clear_on_success).lower()}\n"
    return {
        "name": f"cz-ws3-probe-{node}",
        "time_limit_minutes": PROBE_TIME_LIMIT_MINUTES,
        "lane": "nv",
        "constraint": None,
        "required_node": node,
        "run_path": str(destination),
        "script": template.replace("set -euo pipefail", f"set -euo pipefail\n{exports}"),
    }


def bundle_submission(
    models: Sequence[str], action: Literal["export", "benchmark"], root: Path, checkout: Path
) -> Submission:
    from cluster.recipe import resolve_teacher

    if not models:
        raise ValueError("arm-bundle requires --models <recipe_id>/seed<k> ...")
    for model in models:
        resolve_teacher(model, root / "runs")
    digest = hashlib.sha256(json.dumps([action, *models]).encode()).hexdigest()[:12]
    template = (REPOSITORY / "cluster/templates/arm-bundle.sbatch").read_text()
    exports = f"export COLUMBA_CHECKOUT={shlex.quote(str(checkout))}\n"
    exports += f"export COLUMBA_MODELS={shlex.quote(json.dumps(list(models)))}\n"
    exports += f"export COLUMBA_ACTION={shlex.quote(action)}\n"
    exports += f"export COLUMBA_RUN_PATH={shlex.quote(str(root / 'runs/arm-bundles' / digest))}\n"
    exports += (
        "export COLUMBA_RUN_METADATA="
        + shlex.quote(
            json.dumps(
                {
                    "recipe_id": f"arm-{action}-{digest}",
                    "seed": "",
                    "action": action,
                    "source_snapshot": checkout.name,
                }
            )
        )
        + "\n"
    )
    return {
        "name": f"cz-arm-{action}-{digest}",
        "lane": "arm",
        "constraint": None,
        "script": template.replace("set -euo pipefail", f"set -euo pipefail\n{exports}"),
        "run_path": str(root / "runs" / "arm-bundles" / digest),
    }


if __name__ == "__main__":
    main()
