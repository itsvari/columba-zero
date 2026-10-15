"""Claim one recipe/seed before bootstrap; losing copies do no training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from app.json_data import JsonValue
from cluster.artifacts import atomic_json, read_json
from cluster.locking import submission_lock
from cluster.requeue import requeue_identifier

DUPLICATE_START_EXIT_STATUS = 76
LIVE_STATES = frozenset({"RUNNING", "COMPLETING", "CONFIGURING"})


def sibling_jobs(name: str) -> tuple[tuple[str, str], ...]:
    if not name.startswith("cz-") or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("claim requires an exact cz- job name")
    result = subprocess.run(
        ["squeue", "--array", "-u", os.environ["USER"], "--noheader", "--format=%i|%j|%T"],
        check=True,
        text=True,
        capture_output=True,
    )
    return tuple(
        (identifier, state)
        for line in result.stdout.splitlines()
        for identifier, observed_name, state in (line.strip().split("|"),)
        if observed_name == name and re.fullmatch(r"[0-9]+(?:_[0-9]+)?", identifier)
    )


def acquire(root: Path, run_path: Path, target: str, name: str) -> bool:
    root = root.resolve()
    if not run_path.resolve().is_relative_to(root / "runs") or not re.fullmatch(
        r"[0-9]+(?:_[0-9]+)?", target
    ):
        raise ValueError("invalid run claim path or job")
    identity = hashlib.sha256(str(run_path.resolve()).encode()).hexdigest()[:16]
    claim_path = root / "claims" / f"{identity}.json"
    with submission_lock(root):
        previous = read_json(claim_path) if claim_path.exists() else {}
        jobs = sibling_jobs(name)
        live = {identifier for identifier, state in jobs if state in LIVE_STATES}
        if target not in live:
            raise ValueError("claimant must be an owned live copy with the exact shared name")
        owner = str(previous.get("owner", ""))
        duplicate = (owner != target and owner in live) or (run_path / "DONE").exists()
        event: dict[str, JsonValue] = {
            "event": "duplicate_start" if duplicate else "claim_acquired",
            "job": target,
            "previous_owner": owner,
            "time": datetime.now(UTC).isoformat(),
        }
        events = cast(list[JsonValue], previous.get("events", []))
        atomic_json(
            claim_path,
            {
                **previous,
                "owner": owner if duplicate else target,
                "run_path": str(run_path),
                "name": name,
                "events": [*events, event],
            },
        )
        print(json.dumps(event), flush=True)
        if duplicate:
            return False
        pending = [
            identifier for identifier, state in jobs if state == "PENDING" and identifier != target
        ]
        if pending:
            subprocess.run(
                ["scancel", "--state=PENDING", "--user=" + os.environ["USER"], "--name=" + name],
                check=True,
                text=True,
                capture_output=True,
            )
        run_path.mkdir(parents=True, exist_ok=True)
        report = read_json(run_path / "run.json") if (run_path / "run.json").exists() else {}
        history = cast(list[JsonValue], report.get("fanout_claims", []))
        atomic_json(
            run_path / "run.json",
            {
                **report,
                "fanout_claims": [
                    *history,
                    {**event, "cancelled_pending_siblings": cast(list[JsonValue], pending)},
                ],
            },
        )
        return True


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    target = requeue_identifier(
        os.environ["SLURM_JOB_ID"],
        os.environ.get("SLURM_ARRAY_JOB_ID"),
        os.environ.get("SLURM_ARRAY_TASK_ID"),
    )
    won = acquire(
        Path.home() / "columba",
        Path(os.environ["COLUMBA_RUN_PATH"]),
        target,
        os.environ["SLURM_JOB_NAME"],
    )
    raise SystemExit(0 if won else DUPLICATE_START_EXIT_STATUS)


if __name__ == "__main__":
    main()
