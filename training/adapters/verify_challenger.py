"""Check an NCNN package against framework-generated, real-image NPZ references.

Each reference contains CHW image, raw packed output, framework detections
in original pixels, and camera_hw. No framework or network access is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy

from app.json_data import JsonValue
from tools.bench_ncnn import Engine
from training.adapters.bench_challenger import MODEL_IDS, Framework, decode
from training.adapters.postprocess import Values

BOX_TOLERANCE_PX = 1.0
SCORE_TOLERANCE = 0.01
CLASS_COLUMN = 5
SCORE_COLUMN = 4


def compare_detections(actual: Values, expected: Values) -> tuple[float, float]:
    if actual.shape != expected.shape:
        raise ValueError(f"detection count differs: {actual.shape} vs {expected.shape}")
    remaining = list(range(len(actual)))
    maximum_box_error_px = 0.0
    maximum_score_error = 0.0
    for detection in expected:
        eligible = [
            index for index in remaining if actual[index, CLASS_COLUMN] == detection[CLASS_COLUMN]
        ]
        if not eligible:
            raise ValueError("detection classes differ")
        matched = min(
            eligible,
            key=lambda index: float(numpy.max(numpy.abs(actual[index, :4] - detection[:4]))),
        )
        remaining.remove(matched)
        maximum_box_error_px = max(
            maximum_box_error_px, float(numpy.max(numpy.abs(actual[matched, :4] - detection[:4])))
        )
        maximum_score_error = max(
            maximum_score_error, abs(float(actual[matched, SCORE_COLUMN] - detection[SCORE_COLUMN]))
        )
    return maximum_box_error_px, maximum_score_error


def verify(
    framework: Framework,
    model: Path,
    prefix: Path,
    references: list[Path],
    input_hw: tuple[int, int],
    precision: str = "fp32",
) -> list[dict[str, JsonValue]]:
    engine = Engine(prefix, model, precision)
    results: list[dict[str, JsonValue]] = []
    try:
        for path in references:
            with numpy.load(path, allow_pickle=False) as reference:
                image = numpy.ascontiguousarray(reference["image"], dtype=numpy.float32)
                expected_raw = numpy.asarray(reference["raw"], dtype=numpy.float32)
                expected_detections = numpy.asarray(reference["detections"], dtype=numpy.float32)
                camera_dimensions = reference["camera_hw"].tolist()
            if image.shape != (3, *input_hw) or expected_raw.ndim != 2:
                raise ValueError("invalid reference shapes")
            camera_hw = (int(camera_dimensions[0]), int(camera_dimensions[1]))
            prediction, _ = engine.infer(image, numpy.empty(expected_raw.size, dtype=numpy.float32))
            if prediction.shape != expected_raw.shape:
                raise ValueError("export output shape mismatch")
            actual = decode(framework, prediction, camera_hw, input_hw, 0.05)
            adapter_reference = decode(framework, expected_raw, camera_hw, input_hw, 0.05)
            adapter_box_error_px, adapter_score_error = compare_detections(
                adapter_reference, expected_detections
            )
            try:
                box_error_px, score_error = compare_detections(actual, expected_detections)
            except ValueError as error:
                results.append(
                    {
                        "reference": path.name,
                        "detections": len(actual),
                        "reason": str(error),
                        "passed": False,
                    }
                )
                continue
            passed = (
                max(box_error_px, adapter_box_error_px) <= BOX_TOLERANCE_PX
                and max(score_error, adapter_score_error) <= SCORE_TOLERANCE
                and len(expected_detections) > 0
            )
            results.append(
                {
                    "reference": path.name,
                    "detections": len(actual),
                    "raw_max_abs_error": float(numpy.max(numpy.abs(prediction - expected_raw))),
                    "box_max_abs_error_px": box_error_px,
                    "score_max_abs_error": score_error,
                    "adapter_box_max_abs_error_px": adapter_box_error_px,
                    "adapter_score_max_abs_error": adapter_score_error,
                    "passed": passed,
                }
            )
    finally:
        engine.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--input-hw", type=int, nargs=2, required=True)
    parser.add_argument("--format", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("references", type=Path, nargs="+")
    arguments = parser.parse_args()
    results = verify(
        cast(Framework, arguments.framework),
        arguments.model,
        arguments.prefix,
        arguments.references,
        tuple(arguments.input_hw),
        arguments.format,
    )
    arguments.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if all(result["passed"] for result in results) else 1)


if __name__ == "__main__":
    main()
