"""Run the software checks required by the pan-tilt demonstration plan."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SOURCE_DIRECTORIES = ("app", "tests", "tools", "cluster", "training", "native")
CHECKS = (
    ("pytest",),
    ("ruff", "check", *SOURCE_DIRECTORIES),
    ("ruff", "format", "--check", *SOURCE_DIRECTORIES),
    ("mypy",),
)


def verify() -> int:
    repository = Path(__file__).resolve().parents[1]
    failed_checks: list[str] = []
    for arguments in CHECKS:
        command = (sys.executable, "-m", *arguments)
        print(f"Running {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=repository, check=False)
        if result.returncode:
            failed_checks.append(" ".join(arguments))
    if failed_checks:
        print(f"Failed checks: {'; '.join(failed_checks)}", file=sys.stderr)
        return 1
    print("All software checks passed. Hardware acceptance remains pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(verify())
