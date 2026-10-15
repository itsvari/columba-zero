from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.json_data import JsonValue
from tools.evaluate_predictions import Dataset, ImageRecord
from tools.quantize_ncnn import calibration_images
from tools.select_model import accuracy_gate, parity, select_candidates

BASELINE = (0.21, 0.24, 0.28)


def candidate(recipe: str, seed: int, validation_recall: float = 0.8) -> dict[str, JsonValue]:
    return {
        "recipe": recipe,
        "seed": seed,
        "decode": "yolo11",
        "precision": "fp16",
        "input_hw": [448, 800],
        "val": {"recall_at_p90": validation_recall},
        "test": {"recall_at_p90": 0.78, "ap50": 0.85, "recall_at_threshold": 0.75},
        "empty_far": 0.0,
        "confuser_far": 0.04,
        "pi5_p95_ms": 52.0,
        "reaction_p95_ms": 350.0,
        "parity_passed": True,
        "accuracy_gate_passed": True,
        "directory": f"{recipe}/seed{seed}",
    }


class ExportSelectionTests(unittest.TestCase):
    def selection(self, records: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
        with patch("tools.select_model.candidate_record", side_effect=records):
            return select_candidates([Path(str(index)) for index in range(len(records))], BASELINE)

    def test_selection_ranks_validation_and_uses_test_only_as_gate(self) -> None:
        records = [
            candidate(recipe, seed, recall)
            for recipe, recall in (("high-val", 0.81), ("high-test", 0.79))
            for seed in range(3)
        ]
        for record in records[3:]:
            record["test"] = {"recall_at_p90": 0.99, "ap50": 0.99, "recall_at_threshold": 0.99}
        decision = self.selection(records)
        self.assertTrue(str(decision["selected"]).startswith("high-val/"))

    def test_incomplete_seeds_and_single_failed_gate_prevent_promotion(self) -> None:
        for field, invalid in (
            ("parity_passed", False),
            ("accuracy_gate_passed", False),
            ("empty_far", 0.03),
            ("confuser_far", 0.06),
            ("pi5_p95_ms", 101.0),
            ("reaction_p95_ms", 501.0),
        ):
            with self.subTest(field=field):
                records = [candidate("arm", seed) for seed in range(3)]
                records[0][field] = invalid
                self.assertIsNone(self.selection(records)["selected"])
        self.assertIsNone(self.selection([candidate("arm", 0), candidate("arm", 1)])["selected"])

    def test_baseline_improvement_must_exceed_larger_seed_deviation(self) -> None:
        records = [candidate("arm", seed) for seed in range(3)]
        for record in records:
            record["test"] = {"recall_at_p90": 0.26, "ap50": 0.6, "recall_at_threshold": 0.3}
        self.assertIsNone(self.selection(records)["selected"])

    def test_deployed_precision_accuracy_is_an_explicit_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, exported = root / "pt", root / "fp16"
            reference.mkdir()
            exported.mkdir()
            for split in ("val", "test"):
                source = {
                    "pigeon": {"recall_at_p90": 0.8, "ap50": 0.85, "recall_at_threshold": 0.75}
                }
                (reference / f"metrics_{split}.json").write_text(json.dumps(source))
                (exported / f"metrics_{split}.json").write_text(json.dumps(source))
            self.assertTrue(accuracy_gate(reference, exported)["passed"])
            (exported / "metrics_test.json").write_text(
                json.dumps(
                    {"pigeon": {"recall_at_p90": 0.78, "ap50": 0.85, "recall_at_threshold": 0.75}}
                )
            )
            self.assertFalse(accuracy_gate(reference, exported)["passed"])

    def test_int8_parity_has_documented_tolerances_and_still_requires_equal_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference, exported = Path(temporary) / "pt.json", Path(temporary) / "int8.json"
            source = {"image_id": "bird", "category_id": 0, "bbox": [0, 0, 100, 100], "score": 0.9}
            reference.write_text(json.dumps([source]))
            exported.write_text(json.dumps([{**source, "score": 0.82}]))
            self.assertFalse(parity(reference, exported, "fp16", ["bird"])["passed"])
            self.assertTrue(parity(reference, exported, "int8", ["bird"])["passed"])
            exported.write_text("[]")
            self.assertFalse(parity(reference, exported, "int8", ["bird"])["passed"])

    def test_calibration_is_deterministic_and_rejects_test_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for index in range(5):
                path = root / str(index)
                path.write_bytes(bytes([index]))
                records.append(
                    ImageRecord(str(index), path, 10, 10, "original", "fixture", str(index), (), ())
                )
            validation = Dataset("training", ("pigeon",), tuple(records[:4]), "manifest")
            frozen = Dataset("test", ("pigeon",), (records[-1],), "manifest")
            selected = calibration_images(validation, frozen, 2, 42)
            self.assertEqual(selected, calibration_images(validation, frozen, 2, 42))
            overlapping = replace(frozen, images=(records[0],))
            with self.assertRaisesRegex(ValueError, "overlaps frozen test"):
                calibration_images(validation, overlapping, 4, 42)
            with self.assertRaisesRegex(ValueError, "must fit"):
                calibration_images(validation, frozen, 200, 42)


if __name__ == "__main__":
    unittest.main()
