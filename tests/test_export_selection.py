from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

from app.json_data import JsonValue
from tools.evaluate_predictions import Dataset, ImageRecord, load_dataset, mapping
from tools.prepare_models import prepare_models
from tools.quantize_ncnn import calibration_images
from tools.select_model import (
    accuracy_gate,
    add_candidate,
    aspect_bucket,
    baseline_scores,
    candidate_record,
    latest_frame_rate,
    package_final_candidate,
    parity,
    reaction_for_recall,
    select_candidates,
    taxonomy_view,
)

BASELINE = (0.21, 0.24, 0.28)


def candidate(recipe: str, seed: int, validation_recall: float = 0.8) -> dict[str, JsonValue]:
    return {
        "recipe": recipe,
        "seed": seed,
        "decode": "yolo11",
        "precision": "fp16",
        "input_hw": [448, 800],
        "val": {"recall_at_p90": validation_recall, "ap50": 0.85},
        "test": {"recall_at_p90": 0.78, "ap50": 0.85, "recall_at_threshold": 0.75},
        "empty_far": 0.0,
        "confuser_far": 0.04,
        "val_empty_far": 0.0,
        "val_confuser_far": 0.04,
        "pi5_p95_ms": 52.0,
        "reaction_p95_ms": 350.0,
        "parity_passed": True,
        "fp32_parity_passed": True,
        "accuracy_gate_passed": True,
        "directory": f"{recipe}/seed{seed}",
    }


class ExportSelectionTests(unittest.TestCase):
    def selection(self, records: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
        with patch("tools.select_model.candidate_record", side_effect=records):
            return select_candidates([Path(str(index)) for index in range(len(records))], BASELINE)

    def test_species_taxonomy_view_keeps_manifest_images_inside_immutable_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "columba-test-v2"
            (dataset / "images/test").mkdir(parents=True)
            (dataset / "labels/test").mkdir(parents=True)
            image = dataset / "images/test/bird.jpg"
            image.write_bytes(b"image")
            (dataset / "labels/test/bird.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            original = "names: [pigeon]\ntest: images/test\n"
            (dataset / "data.yaml").write_text(original)
            (dataset / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "id": "bird",
                        "file": "images/test/bird.jpg",
                        "split": "test",
                        "width": 100,
                        "height": 100,
                        "classes_present": ["pigeon"],
                        "style": "original",
                        "source": "test",
                        "group": "bird",
                    }
                )
                + "\n"
            )
            with taxonomy_view(dataset, "species") as view:
                loaded = load_dataset(view, "test")
                self.assertEqual(loaded.names[0], "pigeon")
                self.assertGreater(len(loaded.names), 1)
                self.assertTrue(loaded.images[0].path.is_relative_to(view))
                self.assertEqual(loaded.images[0].annotations[0].category_id, 0)
            self.assertEqual((dataset / "data.yaml").read_text(), original)
            self.assertEqual(image.read_bytes(), b"image")

    def test_aspect_buckets_cover_canonical_shapes(self) -> None:
        for width_px, height_px, expected in (
            (1920, 1080, "approx-16:9"),
            (640, 480, "4:3"),
            (640, 640, "square"),
            (480, 640, "portrait"),
        ):
            self.assertEqual(aspect_bucket(width_px, height_px), expected)

    def test_baseline_requires_three_rectangular_frozen_test_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in range(3):
                directory = root / f"seed{seed}"
                directory.mkdir()
                (directory / "metrics_test.json").write_text(
                    json.dumps(
                        {
                            "dataset": "columba-test-v2",
                            "input_hw": [384, 640],
                            "pigeon": {"recall_at_p90": BASELINE[seed]},
                        }
                    )
                )
            self.assertEqual(baseline_scores(root), BASELINE)
            report = root / "seed2/metrics_test.json"
            content = json.loads(report.read_text())
            content["input_hw"] = [640, 640]
            report.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "rectangular"):
                baseline_scores(root)

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
            ("fp32_parity_passed", False),
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

    def test_shipped_seed_uses_validation_only_and_never_retries_test_failures(self) -> None:
        records = [candidate("arm", seed, recall) for seed, recall in enumerate((0.81, 0.82, 0.80))]
        records[0]["confuser_far"] = 0.06
        decision = self.selection(records)
        self.assertTrue(str(decision["package_candidate"]).endswith("seed1"))
        records[1]["confuser_far"] = 0.06
        self.assertIsNone(self.selection(records)["selected"])
        records[1]["val_confuser_far"] = 0.06
        self.assertIsNone(self.selection(records)["selected"])
        records[0]["confuser_far"] = 0.04
        self.assertTrue(str(self.selection(records)["package_candidate"]).endswith("seed0"))

    def test_fp16_strict_parity_is_diagnostic_but_fp32_and_accuracy_are_gates(self) -> None:
        records = [candidate("arm", seed) for seed in range(3)]
        records[0]["parity_passed"] = False
        self.assertIsNotNone(self.selection(records)["selected"])
        records[0]["fp32_parity_passed"] = False
        self.assertIsNone(self.selection(records)["selected"])

    def test_int8_and_one_to_one_remain_excluded_even_with_passing_metrics(self) -> None:
        for field, value in (("precision", "int8"), ("decode", "yolo26")):
            records = [candidate("arm", seed) for seed in range(3)]
            for record in records:
                record[field] = value
            self.assertIsNone(self.selection(records)["selected"])

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

    def test_final_selection_keeps_fp32_as_companion_only(self) -> None:
        records = [candidate("arm", seed) for seed in range(3)]
        companions = [candidate("arm", seed, 0.81) for seed in range(3)]
        for record in companions:
            record["precision"] = "fp32"
        with patch("tools.select_model.candidate_record", side_effect=records + companions):
            decision = select_candidates(
                [Path(str(index)) for index in range(6)], BASELINE, final=True
            )
        self.assertIn("/fp16/", str(decision["selected"]))
        self.assertFalse(decision["provisional"])
        self.assertEqual(decision["awaiting"], [])

    def test_validation_ties_use_false_alarms_then_ap50_before_seed_index(self) -> None:
        records = [candidate("arm", seed) for seed in range(3)]
        records[2]["val_confuser_far"] = 0.03
        self.assertEqual(
            mapping(cast(list[JsonValue], self.selection(records)["arms"])[0], "arm")[
                "shipped_seed"
            ],
            2,
        )
        records[1]["val_confuser_far"] = 0.03
        records[2]["val_empty_far"] = 0.01
        self.assertEqual(
            mapping(cast(list[JsonValue], self.selection(records)["arms"])[0], "arm")[
                "shipped_seed"
            ],
            1,
        )
        records[2]["val_empty_far"] = 0.0
        mapping(records[2]["val"], "val")["ap50"] = 0.86
        self.assertEqual(
            mapping(cast(list[JsonValue], self.selection(records)["arms"])[0], "arm")[
                "shipped_seed"
            ],
            2,
        )

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

    def test_add_builds_valid_guarded_arm_script_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            arguments = argparse.Namespace(
                recipe="yolo26n-v2s-species-800-r3",
                input_hw=[448, 800],
                taxonomy="species",
                dataset_name="columba-train-v2s-species",
                decodes=["yolo11", "yolo26"],
                output=Path(temporary),
                reference_data_root=None,
                precisions=["fp32", "fp16"],
                node="xcng1",
                single_recipe="yolo26n-v2s-800-round2-plain",
            )
            with (
                patch("cluster.sweep.remote_root", return_value=Path("/home/worker/columba")),
                patch(
                    "cluster.sweep.deploy",
                    return_value=(Path("/home/worker/columba/checkout"), "commit"),
                ),
                patch("cluster.sweep.current_queue", return_value=()),
                patch(
                    "cluster.sweep.remote",
                    side_effect=["", "", "", "", '[{"status":"budget full"}]'],
                ) as remote,
                patch("tools.select_model.export_candidate"),
                patch("tools.select_model.subprocess.run"),
            ):
                add_candidate(arguments)
            payload = json.loads(shlex.split(remote.call_args.args[0][2])[-1])
            self.assertEqual(payload["budget"], {"gpu_jobs": 8, "cpu_jobs": 2, "queued_jobs": 24})
            script = payload["submissions"][0]["script"]
            subprocess.run(["bash", "-n"], input=script, text=True, check=True, capture_output=True)
            self.assertIn("OMP_WAIT_POLICY=ACTIVE", script)
            self.assertIn("--precisions fp32 fp16", script)
            self.assertIn("#SBATCH --nodelist=xcng1", script)
            self.assertIn("--single-recipe yolo26n-v2s-800-round2-plain", script)
            self.assertIn("tools.select_model styles", script)
            self.assertIn("#SBATCH --nice=1000", script)

    def test_hold_can_name_a_diagnostic_package_without_promoting_it(self) -> None:
        records = [candidate("arm", seed) for seed in range(3)]
        records[0]["confuser_far"] = 0.06
        decision = self.selection(records)
        self.assertIsNone(decision["selected"])
        self.assertIsNone(decision["package_candidate"])
        self.assertIsNotNone(decision["diagnostic_candidate"])
        self.assertTrue(decision["diagnostic_failures"])

    def test_latest_frame_rate_limits_service_and_drops_stale_frames(self) -> None:
        self.assertAlmostEqual(latest_frame_rate([40.0]), 25.0)
        self.assertAlmostEqual(latest_frame_rate([100.0]), 10.0)
        self.assertAlmostEqual(latest_frame_rate([20.0]), 30.0, places=3)

    def test_zero_recall_has_no_fabricated_reaction_quantile(self) -> None:
        reaction = reaction_for_recall([30.0, 40.0], 0.0)
        self.assertIsNone(mapping(reaction["reaction_ms"], "reaction")["p95"])
        records = [candidate("arm", seed) for seed in range(3)]
        records[0]["reaction_p95_ms"] = None
        self.assertIsNone(self.selection(records)["selected"])

    def test_primary_plain_pick_is_not_duplicated_as_its_own_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            arguments = argparse.Namespace(
                exports=Path("exports"),
                decision=repository / "decision.json",
                report=None,
                fallback_candidate=Path("plain/seed1"),
                fallback_exports=Path("old-exports"),
            )
            decision: dict[str, JsonValue] = {
                "provisional": False,
                "package_candidate": "plain/seed1",
                "arms": [],
            }
            entry: dict[str, JsonValue] = {
                "model_id": "plain-seed1",
                "measured_model_id": "plain/seed1",
                "source": "tracked/plain",
                "runtime_directory": "models/plain",
                "files": [],
            }
            with (
                patch("tools.select_model.__file__", str(repository / "tools/select_model.py")),
                patch("tools.select_model.durable_package_entry", return_value=entry) as package,
                patch(
                    "tools.select_model.candidate_record", return_value={"model_id": "plain/seed1"}
                ),
                patch(
                    "tools.prepare_models.prepare_models", return_value=repository / "models/plain"
                ),
            ):
                package_final_candidate(arguments, decision)
            package.assert_called_once()
            manifest = json.loads((repository / "deploy/model_manifest.json").read_text())
            self.assertEqual(manifest["fallback_models"], [])

    def test_provisional_manifest_verifies_c8_without_switching_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            source = repository / "candidate"
            source.mkdir()
            records = []
            for name in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml"):
                payload = name.encode()
                (source / name).write_bytes(payload)
                records.append(
                    {
                        "file": name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                    }
                )
            manifest = repository / "deploy/model_manifest.json"
            manifest.parent.mkdir()
            manifest.write_text(
                json.dumps(
                    {
                        "source": "incumbent",
                        "runtime_directory": "models/incumbent",
                        "files": [],
                        "provisional_models": [
                            {
                                "model_id": "candidate-provisional",
                                "provisional": True,
                                "source": "candidate",
                                "runtime_directory": "models/candidate",
                                "files": records,
                            }
                        ],
                    }
                )
            )
            prepared = prepare_models(repository, "candidate-provisional")
            self.assertEqual((prepared / "model.ncnn.bin").read_bytes(), b"model.ncnn.bin")
            self.assertFalse((repository / "models/incumbent").exists())
            (source / "model.ncnn.bin").write_bytes(b"corrupt")
            with self.assertRaisesRegex(SystemExit, "source checksum mismatch"):
                prepare_models(repository, "candidate-provisional")
            with self.assertRaisesRegex(SystemExit, "expected one model entry"):
                prepare_models(repository, "unknown")

    def test_species_taxonomy_view_preserves_immutable_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "columba-test-v2"
            dataset.mkdir()
            configuration = dataset / "data.yaml"
            configuration.write_text("names: [pigeon]\n")
            original = configuration.read_bytes()
            with taxonomy_view(dataset, "species") as aligned:
                self.assertNotEqual(aligned, dataset)
                self.assertIn("spotted_dove", (aligned / "data.yaml").read_text())
                self.assertEqual(configuration.read_bytes(), original)
            self.assertFalse(aligned.exists())

    def test_candidate_requires_matching_native_accuracy_and_active_latency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "fp16"
            directory.mkdir()
            reference = directory.parent / "pt"
            reference.mkdir()
            flags: dict[str, JsonValue] = {
                key: True for key in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic")
            }
            reports: dict[str, JsonValue] = {
                "candidate": {
                    "model_id": "arm/seed0/fp16",
                    "precision": "fp16",
                    "input_hw": [384, 640],
                },
                "metrics_val": {
                    "split": "val",
                    "input_hw": [384, 640],
                    "pigeon": {},
                    "empty_false_alarm_rate": 0.0,
                    "confuser_false_alarm_rate": 0.04,
                    "operating_threshold": {"value": 0.5, "source": "selected"},
                },
                "metrics_test": {
                    "split": "test",
                    "dataset": "columba-test-v2",
                    "input_hw": [384, 640],
                    "pigeon": {},
                    "empty_false_alarm_rate": 0.0,
                    "confuser_false_alarm_rate": 0.0,
                    "operating_threshold": {"value": 0.5, "source": "metrics_val.json"},
                },
                "latency": {
                    "backend": "columba_native",
                    "input_hw": [384, 640],
                    "model_id": "arm/seed0/fp16",
                    "format": "ncnn-fp16",
                    "compute_flags": flags,
                    "omp_wait_policy": "ACTIVE",
                    "estimated_pi5": {"end_to_end_p95_ms": 50.0},
                },
                "reaction": {"reaction_ms": {"p95": 300.0}},
                "parity": {"passed": True, "failures": 0},
                "accuracy_gate": {"passed": True, "deltas": {}},
                "predictions_test.meta": {
                    "framework": "columba_native",
                    "host": "xcng0",
                    "input_hw": [384, 640],
                    "model_id": "arm/seed0/fp16",
                    "format": "ncnn-fp16",
                    "compute_flags": flags,
                    "ncnn_compute": "runtime_default",
                },
            }
            for name, report in reports.items():
                (directory / f"{name}.json").write_text(json.dumps(report))
            (reference / "predictions_test.meta.json").write_text(
                json.dumps({"host": "vexcalibur", "device": "cpu"})
            )
            self.assertTrue(candidate_record(directory)["accuracy_gate_passed"])
            invalid = mapping(reports["latency"], "latency")
            invalid["omp_wait_policy"] = "PASSIVE"
            (directory / "latency.json").write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "remeasure latency"):
                candidate_record(directory)
            invalid["omp_wait_policy"] = "ACTIVE"
            invalid["input_hw"] = [640, 640]
            (directory / "latency.json").write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "mismatched candidate input"):
                candidate_record(directory)
            invalid["input_hw"] = [384, 640]
            (directory / "latency.json").write_text(json.dumps(invalid))
            styles = mapping(reports["metrics_test"], "test").copy()
            styles["dataset"] = "columba-test-v2-styles"
            styles["operating_threshold"] = {"value": 0.6}
            (directory / "metrics_styles.json").write_text(json.dumps(styles))
            (directory / "predictions_styles.meta.json").write_text(
                json.dumps(reports["predictions_test.meta"])
            )
            with self.assertRaisesRegex(ValueError, "styles threshold differs"):
                candidate_record(directory)
            styles["operating_threshold"] = {"value": 0.5}
            (directory / "metrics_styles.json").write_text(json.dumps(styles))
            self.assertIsNotNone(candidate_record(directory)["styles"])

    def test_styles_breaks_matched_data_ties_without_changing_test_gates(self) -> None:
        records = [
            candidate(recipe, seed, recall)
            for recipe, recall in (("yolo26n-v2-800", 0.802), ("yolo26n-v2s-800", 0.800))
            for seed in range(3)
        ]
        for record in records:
            record["styles"] = {"recall_at_p90": 0.7 if "v2s" in str(record["recipe"]) else 0.6}
        self.assertTrue(str(self.selection(records)["selected"]).startswith("yolo26n-v2s-800/"))
        for record in records[3:]:
            record["confuser_far"] = 0.06
        self.assertTrue(str(self.selection(records)["selected"]).startswith("yolo26n-v2-800/"))
        for record in records[3:]:
            record["confuser_far"] = 0.04
            record["input_hw"] = [384, 640]
        self.assertTrue(str(self.selection(records)["selected"]).startswith("yolo26n-v2-800/"))

    def test_species_requires_matched_single_and_pigeon_accuracy_gate(self) -> None:
        single = [candidate("single", seed, 0.79) for seed in range(3)]
        species = [candidate("species", seed, 0.81) for seed in range(3)]
        for record in species:
            record["taxonomy"] = "species"
            record["single_recipe"] = "single"
        self.assertTrue(str(self.selection(single + species)["selected"]).startswith("species/"))
        self.assertIsNone(self.selection(species)["selected"])
        for record in species:
            record["test"] = {"recall_at_p90": 0.76, "ap50": 0.85, "recall_at_threshold": 0.75}
        self.assertTrue(str(self.selection(single + species)["selected"]).startswith("single/"))

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
