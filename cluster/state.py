"""Read array, restart and transition state using only the standard library."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import TypeAlias, cast

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def mapping(path: Path) -> dict[str, JsonValue]:
    value = cast(JsonValue, json.loads(path.read_text()))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def historical_states(root: Path) -> dict[str, str]:
    identifiers = {
        str(record.get("array_id", record.get("job_id", "")))
        for directory in ("arrays", "transitions")
        for path in (root / directory).glob("*/submission.json")
        for record in (mapping(path),)
    }
    if not identifiers:
        return {}
    if any(not identifier.isdecimal() for identifier in identifiers):
        raise ValueError("invalid recorded job identity")
    completed = subprocess.run(
        [
            "sacct",
            "-j",
            ",".join(sorted(identifiers)),
            "--noheader",
            "--parsable2",
            "--format=JobID,State,JobName%100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        identifier: state.split()[0].rstrip("+")
        for row in completed.stdout.splitlines()
        for identifier, state, name, *_ in (row.split("|"),)
        if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", identifier) and name.startswith("cz-")
    }


def snapshot(root: Path, job_states: dict[str, str]) -> dict[str, JsonValue]:
    arrays: list[JsonValue] = []
    for path in sorted((root / "arrays").glob("*/manifest.json")):
        manifest = mapping(path)
        record_path = path.parent / "submission.json"
        if not record_path.exists():
            continue
        record = mapping(record_path)
        identifier = str(record["array_id"])
        for index, task in enumerate(cast(list[dict[str, JsonValue]], manifest["tasks"])):
            relative = f"{task['recipe_id']}/seed{task['seed']}"
            directory = (root / "runs" / relative).resolve()
            if not directory.is_relative_to((root / "runs").resolve()):
                raise ValueError("invalid task run path")
            ledger = (
                mapping(directory / "requeue.json") if (directory / "requeue.json").exists() else {}
            )
            report = mapping(directory / "run.json") if (directory / "run.json").exists() else {}
            hold = (
                (directory / "HOLD").read_text().strip() if (directory / "HOLD").exists() else None
            )
            state = (
                "DONE"
                if (directory / "DONE").exists()
                else job_states.get(
                    f"{identifier}_{index}", str(report.get("status", "not queued"))
                )
            )
            arrays.append(
                {
                    "kind": "array_task",
                    "array_id": identifier,
                    "task_id": index,
                    "run": relative,
                    "run_name": task["run_name"],
                    "state": state,
                    "requeue_count": ledger.get("count", 0),
                    "hold": hold,
                    "throttle": record["throttle"],
                    "resource_type": record["resource_type"],
                    "epochs_done": report.get("epochs_done", 0),
                }
            )
    transitions: list[JsonValue] = []
    for path in sorted((root / "transitions").glob("*/plan.json")):
        plan = mapping(path)
        report = mapping(path.parent / "run.json") if (path.parent / "run.json").exists() else {}
        record = (
            mapping(path.parent / "submission.json")
            if (path.parent / "submission.json").exists()
            else {}
        )
        hold = (
            (path.parent / "HOLD").read_text().strip() if (path.parent / "HOLD").exists() else None
        )
        transitions.append(
            {
                "kind": "transition",
                "id": path.parent.name,
                "round": plan["round"],
                "job_id": record.get("job_id"),
                "state": job_states.get(
                    str(record.get("job_id")), report.get("status", "prepared")
                ),
                "hold": hold,
                "report": report,
                "after": plan["after"],
            }
        )
    return {"arrays": arrays, "transitions": transitions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", default="{}")
    arguments = parser.parse_args()
    if arguments.root.resolve() != (Path.home() / "columba").resolve():
        raise ValueError("state must stay under ~/columba")
    current = cast(dict[str, str], json.loads(arguments.jobs))
    print(json.dumps(snapshot(arguments.root, {**historical_states(arguments.root), **current})))


if __name__ == "__main__":
    main()
