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
from cluster.recipe import Recipe, load_recipe, recipe_sha256
from cluster.slurm import ACTIVE_STATES, Budget, Job, Submission, job_name, parse_queue

SSH_HOST = "columba-cluster"
REPOSITORY = Path(__file__).resolve().parents[1]


def remote(command: Sequence[str], input_bytes: bytes | None = None) -> str:
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", SSH_HOST, shlex.join(command)],
        input=input_bytes,
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode()


def remote_root() -> Path:
    return Path(
        remote(["python", "-c", "from pathlib import Path; print(Path.home() / 'columba')"]).strip()
    )


def current_queue() -> tuple[Job, ...]:
    return parse_queue(
        remote(["sh", "-c", 'squeue -u "$USER" --noheader --format="%i|%j|%T|%P|%b"'])
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
            "constraint": run.recipe.constraint_for_seed(run.seed) or "any nv",
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
    remote(["mkdir", "-p", str(checkout), str(root / "logs")])
    remote(["tar", "-xf", "-", "-C", str(checkout)], content)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True, text=True, capture_output=True
    ).stdout.strip()
    return checkout, commit


def submission_for(run: DesiredRun, root: Path, checkout: Path, commit: str) -> Submission:
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
    script = template.replace("set -euo pipefail", f"set -euo pipefail\n{exports}")
    return {
        "name": run.name,
        "lane": run.recipe.lane,
        "constraint": run.recipe.constraint_for_seed(run.seed),
        "script": script,
        "run_path": str(root / "runs" / run.relative_run),
    }


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
                "ssh -o BatchMode=yes",
                "--include=weights/",
                "--include=weights/best.pt",
                "--include=weights/last.pt",
                "--include=*.json",
                "--include=results.csv",
                "--include=DONE",
                "--include=TRAINED",
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
            rows.append(leaderboard_row(local))
    update_leaderboard(leaderboard, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("plan", "submit", "status", "fetch", "cancel", "arm-bundle")
    )
    parser.add_argument("--recipes", nargs="+", type=Path, default=[])
    parser.add_argument("--gpu-limit", type=int, default=4)
    parser.add_argument("--cpu-limit", type=int, default=2)
    parser.add_argument("--queue-limit", type=int, default=12)
    parser.add_argument("--job-ids", nargs="+", default=[])
    parser.add_argument("--exclude-nodes", nargs="+", default=[])
    parser.add_argument("--destination", type=Path, default=REPOSITORY / "experiments/runs")
    parser.add_argument(
        "--leaderboard", type=Path, default=REPOSITORY / "experiments/leaderboard.csv"
    )
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--bundle-action", choices=("export", "benchmark"), default="export")
    arguments = parser.parse_args()
    budget = Budget(arguments.gpu_limit, arguments.cpu_limit, arguments.queue_limit)
    runs = desired_runs(arguments.recipes)
    root = remote_root()
    jobs = current_queue()
    state = inspect_runs(root, runs)
    done = {name for name, details in state.items() if details["done"]}
    if arguments.action in {"plan", "status"}:
        for item in plan_runs(runs, done, jobs):
            if arguments.action == "status":
                report = state[item["run"]].get("report", {})
                if isinstance(report, dict):
                    item["report_status"] = str(report.get("status", "absent"))
                    item["epochs_done"] = str(report.get("epochs_done", 0))
            print(json.dumps(item))
        if not runs:
            for job in jobs:
                if job.owned:
                    print(json.dumps(asdict(job)))
        return
    if arguments.action == "fetch":
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
    else:
        submissions = [
            submission_for(run, root, checkout, commit)
            for run in runs
            if run.relative_run not in done
        ]
    for submission in submissions:
        submission["excluded_nodes"] = arguments.exclude_nodes
    payload: dict[str, JsonValue] = {
        "budget": cast(dict[str, JsonValue], asdict(budget)),
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
