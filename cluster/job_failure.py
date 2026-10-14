"""Record bootstrap or shell failures even before the vision environment exists."""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from app.json_data import JsonValue
from cluster.artifacts import atomic_json, read_json


def record_failure(run_path: Path, metadata: dict[str, JsonValue], exit_status: int) -> None:
    if not run_path.resolve().is_relative_to((Path.home() / "columba").resolve()):
        raise ValueError("failure reports must stay inside ~/columba")
    report_path = run_path / "run.json"
    previous = read_json(report_path) if report_path.is_file() else {}
    job_id = os.environ["SLURM_JOB_ID"]
    if previous.get("job_id") == job_id and previous.get("status") == "failed":
        return
    report: dict[str, JsonValue] = {
        **previous,
        **metadata,
        "job_id": job_id,
        "epochs_done": previous.get("epochs_done", metadata.get("epochs_done", 0)),
        "host": platform.node(),
        "status": "failed",
        "exit_status": exit_status,
        "end": datetime.now(UTC).isoformat(),
        "evaluation": "not run: job failed",
        "error": "job shell or bootstrap failed; see the cz- Slurm log",
    }
    atomic_json(report_path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-status", type=int, required=True)
    arguments = parser.parse_args()
    if not os.environ.get("SLURM_JOB_NAME", "").startswith("cz-"):
        raise ValueError("only cz- failures may be recorded")
    metadata = cast(dict[str, JsonValue], json.loads(os.environ["COLUMBA_RUN_METADATA"]))
    record_failure(Path(os.environ["COLUMBA_RUN_PATH"]), metadata, arguments.exit_status)


if __name__ == "__main__":
    main()
