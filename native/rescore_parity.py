"""Recompare saved parity predictions without running either inference backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import cast

from app.json_data import JsonValue
from native.verify_native import (
    DEGENERATE_BOX_RULE,
    ParityRecord,
    compare_predictions,
    comparison_diagnostics,
)


def prediction_digest(records: list[ParityRecord]) -> str:
    predictions = [
        {"image": record["image"], "reference": record["reference"], "native": record["native"]}
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(predictions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def rescore_report(source: Path, destination: Path) -> dict[str, JsonValue]:
    contents = source.read_bytes()
    report = cast(dict[str, JsonValue], json.loads(contents))
    previous = cast(list[ParityRecord], report["records"])
    digest = prediction_digest(previous)
    records = [
        compare_predictions(
            record["image"],
            record["reference"],
            record["native"],
            minimum_iou=cast(float, report["min_iou"]),
            maximum_confidence_delta=cast(float, report["max_confidence_delta"]),
        )
        for record in previous
    ]
    assert prediction_digest(records) == digest
    failures = sum(not record["passed"] for record in records)
    report.update(
        {
            "records": cast(JsonValue, records),
            "failures": failures,
            "strict_failures": sum(not record["strict_passed"] for record in records),
            "diagnostics": cast(JsonValue, comparison_diagnostics(records)),
            "degenerate_box_rule": DEGENERATE_BOX_RULE,
            "rescore": {
                "source_report_sha256": hashlib.sha256(contents).hexdigest(),
                "prediction_sha256": digest,
                "comparison_host": platform.node(),
                "comparison_code_sha256": hashlib.sha256(
                    Path(__file__).with_name("verify_native.py").read_bytes()
                ).hexdigest(),
                "method": "current native.verify_native comparison; original predictions unchanged",
            },
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = rescore_report(arguments.source, arguments.output)
    passed = report["failures"] == 0
    print(json.dumps({"passed": passed, "output": str(arguments.output)}))
    raise SystemExit(1 if not passed and not report.get("informational", False) else 0)


if __name__ == "__main__":
    main()
