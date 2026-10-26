"""Export retained demonstration JPEGs without rerunning detection."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.demo_recording import (
    DEFAULT_EXPORT_TIMEOUT_S,
    DEFAULT_FRAME_DURATION_S,
    DEFAULT_MAX_BYTES,
    export_recording,
)


@dataclass
class ExportOptions:
    spool: Path = Path()
    output: Path = Path()
    timeout_s: float = DEFAULT_EXPORT_TIMEOUT_S
    frame_duration_s: float = DEFAULT_FRAME_DURATION_S
    max_bytes: int = DEFAULT_MAX_BYTES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spool", type=Path, help="run's demo directory containing index.jsonl")
    parser.add_argument("output", type=Path, help="destination MP4")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_EXPORT_TIMEOUT_S)
    parser.add_argument(
        "--frame-duration-s",
        type=float,
        default=DEFAULT_FRAME_DURATION_S,
        help="final frame hold; use 1 / source FPS for replay, 1 / preview FPS for live",
    )
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    options = ExportOptions()
    parser.parse_args(namespace=options)
    try:
        result = export_recording(
            options.spool,
            options.output,
            timeout_s=options.timeout_s,
            default_frame_duration_s=options.frame_duration_s,
            max_bytes=options.max_bytes,
        )
    except KeyboardInterrupt:
        print("Export cancelled; source frames retained.", file=sys.stderr)
        raise SystemExit(130) from None
    print(json.dumps(result.as_dict(), indent=2, allow_nan=False))
    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
