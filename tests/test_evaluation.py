from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import yaml

from app.json_data import JsonValue
from experiments.incumbent.rescore import (
    HistoricalImage,
    coco_splits,
    primary_exposure,
)
from tools.evaluate_predictions import (
    NO_DETECTION_THRESHOLD,
    Box,
    evaluate,
    load_dataset,
    mapping,
    sidecar_path,
)
from tools.predict_to_coco import (
    Letterbox,
    configure_ultralytics,
    decode_output,
    letterbox,
    model_settings,
    predict_to_coco,
)

TEST_WIDTH_PX = 200
TEST_HEIGHT_PX = 100
RECTANGULAR_INPUT_HW = (384, 640)
PERFECT_BOX: Box = (10.0, 10.0, 20.0, 20.0)
SECOND_BOX: Box = (80.0, 10.0, 40.0, 40.0)
FALSE_BOX: Box = (150.0, 50.0, 20.0, 20.0)


@dataclass(frozen=True)
class FixtureImage:
    image_id: str
    boxes: tuple[tuple[int, Box], ...] = ()
    present: tuple[str, ...] = ()
    style: str = "original"
    source: str = "source-a"
    split: str = "val"


def detection(
    image_id: str, bbox: Box = PERFECT_BOX, score: float = 0.9, category_id: int = 0
) -> dict[str, JsonValue]:
    return {"image_id": image_id, "category_id": category_id, "bbox": list(bbox), "score": score}


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.predictions = self.root / "predictions_val.json"
        sidecar_path(self.predictions).write_text(json.dumps({"input_hw": RECTANGULAR_INPUT_HW}))

    def dataset(self, images: Sequence[FixtureImage], names: tuple[str, ...] = ("pigeon",)) -> None:
        import cv2

        (self.root / "data.yaml").write_text(
            yaml.safe_dump({"names": list(names), "val": "images/val", "test": "images/test"})
        )
        manifest: list[dict[str, JsonValue]] = []
        for image in images:
            image_path = self.root / "images" / image.split / f"file-{image.image_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(image_path), np.zeros((TEST_HEIGHT_PX, TEST_WIDTH_PX, 3), dtype=np.uint8)
            )
            label = self.root / "labels" / image.split / f"file-{image.image_id}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(
                "\n".join(
                    f"{category} {(bbox[0] + bbox[2] / 2) / TEST_WIDTH_PX} "
                    f"{(bbox[1] + bbox[3] / 2) / TEST_HEIGHT_PX} "
                    f"{bbox[2] / TEST_WIDTH_PX} {bbox[3] / TEST_HEIGHT_PX}"
                    for category, bbox in image.boxes
                )
            )
            manifest.append(
                {
                    "id": image.image_id,
                    "file": str(image_path.relative_to(self.root)),
                    "split": image.split,
                    "width": TEST_WIDTH_PX,
                    "height": TEST_HEIGHT_PX,
                    "style": image.style,
                    "source": image.source,
                    "group": f"group-{image.image_id}",
                    "classes_present": list(image.present),
                }
            )
        (self.root / "manifest.jsonl").write_text(
            "\n".join(json.dumps(record) for record in manifest)
        )

    def run_evaluation(
        self, predictions: Sequence[dict[str, JsonValue]], **options: object
    ) -> dict[str, JsonValue]:
        self.predictions.write_text(json.dumps(predictions))
        split = str(options.pop("split", "val"))
        threshold = options.pop("threshold", None)
        selection = bool(options.pop("select_threshold", False))
        threshold_from = options.pop("threshold_from", None)
        classes = options.pop("classes", None)
        self.assertFalse(options)
        return evaluate(
            self.root,
            split,
            self.predictions,
            threshold=float(threshold) if isinstance(threshold, (float, int)) else None,
            select_threshold=selection,
            threshold_from=Path(str(threshold_from)) if threshold_from is not None else None,
            classes=Path(str(classes)) if classes is not None else None,
        )

    def test_perfect_predictions_and_missing_prediction_images(self) -> None:
        self.dataset(
            [FixtureImage("pigeon", ((0, PERFECT_BOX),), ("pigeon",)), FixtureImage("empty")]
        )
        report = self.run_evaluation([detection("pigeon")], select_threshold=True)
        pigeon = mapping(report["pigeon"], "pigeon")
        for metric in (
            "ap50",
            "ap50_95",
            "recall_at_p90",
            "precision_at_threshold",
            "recall_at_threshold",
        ):
            self.assertAlmostEqual(float(str(pigeon[metric])), 1.0)
        self.assertEqual(report["empty_false_alarm_rate"], 0.0)
        self.assertIsNone(report["confuser_false_alarm_rate"])
        self.assertEqual(mapping(report["operating_threshold"], "threshold")["value"], 0.9)
        self.assertEqual(mapping(report["counts"], "counts")["groups"], 2)

    def test_known_precision_recall_curve_and_coco_ap(self) -> None:
        self.dataset([FixtureImage("image", ((0, PERFECT_BOX), (0, SECOND_BOX)), ("pigeon",))])
        report = self.run_evaluation(
            [
                detection("image", PERFECT_BOX, 0.9),
                detection("image", FALSE_BOX, 0.8),
                detection("image", SECOND_BOX, 0.7),
            ],
            threshold=0.7,
        )
        pigeon = mapping(report["pigeon"], "pigeon")
        expected_ap = (51 + 50 * (2 / 3)) / 101
        self.assertAlmostEqual(float(str(pigeon["ap50"])), expected_ap)
        self.assertAlmostEqual(float(str(pigeon["ap50_95"])), expected_ap)
        self.assertEqual(pigeon["recall_at_p90"], 0.5)
        self.assertAlmostEqual(float(str(pigeon["precision_at_threshold"])), 2 / 3)
        self.assertEqual(pigeon["recall_at_threshold"], 1.0)

    def test_duplicate_prediction_is_a_false_positive(self) -> None:
        self.dataset([FixtureImage("image", ((0, PERFECT_BOX),), ("pigeon",))])
        report = self.run_evaluation(
            [detection("image", score=0.9), detection("image", score=0.8)], threshold=0.8
        )
        self.assertEqual(mapping(report["pigeon"], "pigeon")["precision_at_threshold"], 0.5)

    def test_equal_scores_enter_together(self) -> None:
        self.dataset([FixtureImage("image", ((0, PERFECT_BOX),), ("pigeon",))])
        report = self.run_evaluation(
            [detection("image"), detection("image", FALSE_BOX)], select_threshold=True
        )
        self.assertEqual(mapping(report["pigeon"], "pigeon")["recall_at_p90"], 0.0)
        operating = mapping(report["operating_threshold"], "threshold")
        self.assertEqual(operating["value"], NO_DETECTION_THRESHOLD)
        self.assertFalse(operating["attainable"])

    def test_small_uses_letterboxed_short_side_and_strict_boundary(self) -> None:
        small: Box = (10.0, 10.0, 9.0, 20.0)
        boundary: Box = (80.0, 10.0, 10.0, 20.0)
        self.dataset([FixtureImage("image", ((0, small), (0, boundary)), ("pigeon",))])
        report = self.run_evaluation(
            [detection("image", small, 0.9), detection("image", boundary, 0.8)]
        )
        self.assertEqual(mapping(report["counts"], "counts")["small_pigeon_boxes"], 1)
        self.assertEqual(mapping(report["pigeon"], "pigeon")["small_recall_at_p90"], 1.0)

    def test_small_recall_uses_overall_pigeon_precision(self) -> None:
        small: Box = (10.0, 10.0, 9.0, 20.0)
        self.dataset([FixtureImage("image", ((0, small), (0, SECOND_BOX)), ("pigeon",))])
        report = self.run_evaluation(
            [
                detection("image", SECOND_BOX, 0.9),
                detection("image", FALSE_BOX, 0.8),
                detection("image", small, 0.7),
            ]
        )
        self.assertEqual(mapping(report["pigeon"], "pigeon")["recall_at_p90"], 0.5)
        self.assertEqual(mapping(report["pigeon"], "pigeon")["small_recall_at_p90"], 0.0)

    def test_empty_and_unlabelled_single_taxonomy_confusers(self) -> None:
        self.dataset(
            [
                FixtureImage("empty-a"),
                FixtureImage("empty-b"),
                FixtureImage("confuser-a", present=("spotted_dove",)),
                FixtureImage("confuser-b", present=("house_crow",)),
            ]
        )
        report = self.run_evaluation(
            [
                detection("empty-a", score=0.8),
                detection("empty-b", score=0.4),
                detection("confuser-a", score=0.8),
            ],
            threshold=0.5,
        )
        self.assertEqual(report["empty_false_alarm_rate"], 0.5)
        self.assertEqual(report["confuser_false_alarm_rate"], 0.5)
        self.assertIsNone(mapping(report["pigeon"], "pigeon")["ap50"])
        self.assertEqual(mapping(report["pigeon"], "pigeon")["recall_at_p90"], 0.0)

    def test_threshold_selection_chooses_lowest_eligible_score(self) -> None:
        boxes = tuple((0, (float(index * 10), 0.0, 8.0, 8.0)) for index in range(10))
        self.dataset([FixtureImage("image", boxes, ("pigeon",))])
        predictions = [detection("image", bbox, 0.9) for _, bbox in boxes]
        report = self.run_evaluation(
            [*predictions, detection("image", FALSE_BOX, 0.3), detection("image", SECOND_BOX, 0.2)],
            select_threshold=True,
        )
        self.assertEqual(mapping(report["operating_threshold"], "threshold")["value"], 0.3)
        self.assertAlmostEqual(
            float(str(mapping(report["pigeon"], "pigeon")["precision_at_threshold"])), 10 / 11
        )

    def test_test_reuses_validation_threshold_and_refuses_selection(self) -> None:
        self.dataset(
            [
                FixtureImage("val", ((0, PERFECT_BOX),), ("pigeon",)),
                FixtureImage("test", ((0, PERFECT_BOX),), ("pigeon",), split="test"),
            ]
        )
        validation = self.run_evaluation([detection("val", score=0.6)], select_threshold=True)
        metrics = self.root / "metrics_val.json"
        metrics.write_text(json.dumps(validation))
        report = self.run_evaluation(
            [detection("test", score=0.59)], split="test", threshold_from=metrics
        )
        self.assertEqual(mapping(report["pigeon"], "pigeon")["recall_at_threshold"], 0.0)
        with self.assertRaisesRegex(ValueError, "only permitted on val"):
            self.run_evaluation([detection("test")], split="test", select_threshold=True)
        validation["split"] = "test"
        metrics.write_text(json.dumps(validation))
        with self.assertRaisesRegex(ValueError, "selected on val"):
            self.run_evaluation([detection("test")], split="test", threshold_from=metrics)

    def test_no_predictions_and_unattainable_sentinel_roundtrip(self) -> None:
        self.dataset(
            [
                FixtureImage("val", ((0, PERFECT_BOX),), ("pigeon",)),
                FixtureImage("test", ((0, PERFECT_BOX),), ("pigeon",), split="test"),
            ]
        )
        report = self.run_evaluation([], select_threshold=True)
        pigeon = mapping(report["pigeon"], "pigeon")
        self.assertEqual(pigeon["ap50"], 0.0)
        self.assertEqual(pigeon["precision_at_threshold"], 0.0)
        self.assertEqual(pigeon["recall_at_p90"], 0.0)
        metrics = self.root / "metrics_val.json"
        metrics.write_text(json.dumps(report, allow_nan=False))
        test = self.run_evaluation(
            [detection("test", score=1.0)], split="test", threshold_from=metrics
        )
        self.assertEqual(mapping(test["pigeon"], "pigeon")["recall_at_threshold"], 0.0)

    def test_breakdowns_recompute_with_shared_threshold(self) -> None:
        self.dataset(
            [
                FixtureImage("a", ((0, PERFECT_BOX),), ("pigeon",)),
                FixtureImage(
                    "b", ((0, PERFECT_BOX),), ("pigeon",), style="noir_night", source="source-b"
                ),
            ]
        )
        report = self.run_evaluation([detection("a")], threshold=0.8)
        styles = mapping(report["by_style"], "by_style")
        sources = mapping(report["by_source"], "by_source")
        self.assertEqual(mapping(styles["original"], "original")["pigeon_recall_at_p90"], 1.0)
        self.assertEqual(mapping(styles["noir_night"], "night")["pigeon_recall_at_p90"], 0.0)
        self.assertEqual(mapping(sources["source-a"], "source-a")["pigeon_recall_at_p90"], 1.0)
        self.assertEqual(mapping(sources["source-b"], "source-b")["pigeon_recall_at_p90"], 0.0)

    def test_species_class_ap_and_class_aware_matching(self) -> None:
        self.dataset(
            [
                FixtureImage(
                    "image", ((0, PERFECT_BOX), (1, SECOND_BOX)), ("pigeon", "spotted_dove")
                )
            ],
            names=("pigeon", "spotted_dove"),
        )
        classes = self.root / "classes.yaml"
        classes.write_text("taxonomies:\n  test:\n    names: [pigeon, spotted_dove]\n")
        report = self.run_evaluation(
            [detection("image", PERFECT_BOX), detection("image", SECOND_BOX, category_id=1)],
            classes=classes,
        )
        self.assertAlmostEqual(
            float(str(mapping(report["per_class_ap50"], "per_class_ap50")["spotted_dove"])), 1.0
        )
        self.assertEqual(report["confuser_false_alarm_rate"], None)

    def test_unknown_ids_invalid_categories_and_nonfinite_predictions_fail(self) -> None:
        self.dataset([FixtureImage("image")])
        invalid = [
            detection("absent"),
            detection("image", category_id=1),
            detection("image", score=float("nan")),
            detection("image", (0, 0, -1, 10)),
            {**detection("image"), "score": True},
        ]
        for prediction in invalid:
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                self.run_evaluation([prediction])

    def test_plain_yolo_stems_and_missing_meta_require_input_size(self) -> None:
        self.dataset([FixtureImage("image", ((0, PERFECT_BOX),), ("pigeon",))])
        (self.root / "manifest.jsonl").unlink()
        report = self.run_evaluation([detection("file-image")])
        self.assertEqual(report["image_id_source"], "image_stem")
        self.predictions.write_text("[]")
        sidecar_path(self.predictions).unlink()
        with self.assertRaises(FileNotFoundError):
            evaluate(self.root, "val", self.predictions)
        report = evaluate(self.root, "val", self.predictions, input_hw=RECTANGULAR_INPUT_HW)
        self.assertEqual(report["input_hw"], list(RECTANGULAR_INPUT_HW))

    def test_rectangular_letterbox_colour_padding_and_original_coordinates(self) -> None:
        frame = np.full((TEST_HEIGHT_PX, TEST_WIDTH_PX, 3), (10, 20, 30), dtype=np.uint8)
        tensor, transform = letterbox(frame, RECTANGULAR_INPUT_HW)
        self.assertEqual(tensor.shape, (3, *RECTANGULAR_INPUT_HW))
        self.assertEqual(transform.top_px, 32)
        self.assertEqual(transform.left_px, 0)
        np.testing.assert_allclose(tensor[:, 0, 0], np.full(3, 114 / 255))
        np.testing.assert_allclose(tensor[:, 32, 0], np.asarray([30, 20, 10]) / 255)
        np.testing.assert_allclose(transform.original_box((32, 64, 64, 64)), PERFECT_BOX)

    def test_raw_yolo26_topk_has_no_nms_and_yolo11_is_class_aware(self) -> None:
        output = np.asarray(
            [
                [64, 64, 64],
                [96, 96, 96],
                [64, 64, 64],
                [64, 64, 64],
                [0.9, 0.8, 0.1],
                [0.1, 0.2, 0.7],
            ],
            dtype=np.float32,
        )
        transform = Letterbox(3.2, 0, 32, TEST_WIDTH_PX, TEST_HEIGHT_PX)
        nms = decode_output(
            output, "image", ("pigeon", "spotted_dove"), transform, "yolo11", 0.3, 0.7, 300
        )
        topk = decode_output(
            output, "image", ("pigeon", "spotted_dove"), transform, "yolo26", 0.3, 0.7, 2
        )
        self.assertEqual([prediction.category_id for prediction in nms], [0, 1])
        self.assertEqual([prediction.category_id for prediction in topk], [0, 0])
        np.testing.assert_allclose(topk[0].bbox, PERFECT_BOX)

    def test_c8_metadata_and_legacy_model_settings(self) -> None:
        package = self.root / "model"
        package.mkdir()
        for name in ("model.ncnn.param", "model.ncnn.bin"):
            (package / name).write_text("placeholder")
        metadata: dict[str, JsonValue] = {
            "task": "detect",
            "imgsz": list(RECTANGULAR_INPUT_HW),
            "names": {"0": "pigeon"},
            "columba": {
                "input_hw": list(RECTANGULAR_INPUT_HW),
                "decode": "yolo26",
                "quantization": "fp16",
            },
        }
        metadata_path = package / "metadata.yaml"
        metadata_path.write_text(yaml.safe_dump(metadata))
        settings = model_settings(package, None, None)
        self.assertEqual(settings.input_hw, RECTANGULAR_INPUT_HW)
        self.assertEqual(settings.decode, "yolo26")
        self.assertEqual(settings.format, "ncnn-fp16")
        with self.assertRaisesRegex(ValueError, "differs from NCNN"):
            model_settings(package, [640], None)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            model_settings(package, None, "yolo11")
        metadata.pop("columba")
        metadata_path.write_text(yaml.safe_dump(metadata))
        self.assertEqual(model_settings(package, None, None).decode, "yolo11")

    def test_mocked_pytorch_model_writes_manifest_ids_and_sidecar(self) -> None:
        self.dataset([FixtureImage("manifest-id", ((0, PERFECT_BOX),), ("pigeon",))])
        weights = self.root / "model.pt"
        weights.write_bytes(b"mock")
        box = Mock()
        box.xyxy = [Mock(tolist=Mock(return_value=[10, 10, 30, 30]))]
        box.cls.item.return_value = 0
        box.conf.item.return_value = 0.9
        model = Mock(names={0: "pigeon"})
        model.predict.return_value = [Mock(names={0: "pigeon"}, boxes=[box])]
        factory = Mock(return_value=model)
        with (
            patch.dict("sys.modules", {"ultralytics": Mock(YOLO=factory), "torch": Mock()}),
            patch("tools.predict_to_coco.configure_ultralytics"),
        ):
            metadata = predict_to_coco(
                weights,
                self.root,
                "val",
                self.predictions,
                imgsz=list(RECTANGULAR_INPUT_HW),
                model_id="test-model",
                host="unit-test",
            )
        values = json.loads(self.predictions.read_text())
        self.assertEqual(values[0]["image_id"], "manifest-id")
        self.assertEqual(values[0]["bbox"], list(PERFECT_BOX))
        self.assertEqual(metadata["input_hw"], list(RECTANGULAR_INPUT_HW))
        self.assertEqual(metadata["image_id_source"], "manifest")
        self.assertEqual(
            json.loads(sidecar_path(self.predictions).read_text())["model_id"], "test-model"
        )
        self.assertEqual(model.predict.call_args.kwargs["imgsz"], list(RECTANGULAR_INPUT_HW))
        self.assertFalse(model.predict.call_args.kwargs["rect"])

    def test_coco_ap_uses_300_detections_rather_than_default_100(self) -> None:
        image_count = 150
        boxes = tuple(
            (0, (float(index % 20 * 10), float(index // 20 * 10), 5.0, 5.0))
            for index in range(image_count)
        )
        self.dataset([FixtureImage("image", boxes, ("pigeon",))])
        report = self.run_evaluation([detection("image", bbox) for _, bbox in boxes])
        self.assertAlmostEqual(float(str(mapping(report["pigeon"], "pigeon")["ap50"])), 1.0)

    def test_duplicate_manifest_ids_fail(self) -> None:
        self.dataset([FixtureImage("image")])
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(manifest.read_text() + "\n" + manifest.read_text())
        with self.assertRaisesRegex(ValueError, "duplicate manifest"):
            load_dataset(self.root, "val")

    def test_new_ultralytics_settings_disable_sync_and_use_scratch(self) -> None:
        settings_root = self.root / "configuration"
        with patch.dict("os.environ", {"YOLO_CONFIG_DIR": str(settings_root)}):
            configure_ultralytics()
        settings = json.loads((settings_root / "Ultralytics/settings.json").read_text())
        self.assertFalse(settings["sync"])
        for key in ("datasets_dir", "weights_dir", "runs_dir"):
            self.assertTrue(Path(settings[key]).is_relative_to(settings_root.resolve()))

    def test_mocked_c8_ncnn_public_api_topk_and_fp32_compute(self) -> None:
        self.dataset([FixtureImage("image", ((0, PERFECT_BOX),), ("pigeon",))])
        package = self.root / "ncnn"
        package.mkdir()
        for name in ("model.ncnn.param", "model.ncnn.bin"):
            (package / name).write_text("mock")
        (package / "metadata.yaml").write_text(
            yaml.safe_dump(
                {
                    "task": "detect",
                    "imgsz": list(RECTANGULAR_INPUT_HW),
                    "names": ["pigeon"],
                    "columba": {
                        "decode": "yolo26",
                        "input_hw": list(RECTANGULAR_INPUT_HW),
                        "quantization": "fp16",
                    },
                }
            )
        )
        output = np.asarray([[64, 64], [96, 96], [64, 64], [64, 64], [0.9, 0.8]], dtype=np.float32)
        extractor = Mock()
        extractor.input.return_value = 0
        extractor.extract.return_value = (0, output)
        network = Mock()
        network.load_param.return_value = 0
        network.load_model.return_value = 0
        extractor_context = Mock(
            __enter__=Mock(return_value=extractor), __exit__=Mock(return_value=False)
        )
        network.create_extractor.return_value = extractor_context
        network_context = Mock(
            __enter__=Mock(return_value=network), __exit__=Mock(return_value=False)
        )
        with patch.dict("sys.modules", {"ncnn": Mock(Net=Mock(return_value=network_context))}):
            metadata = predict_to_coco(package, self.root, "val", self.predictions, ncnn_fp32=True)
        values = json.loads(self.predictions.read_text())
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0]["image_id"], "image")
        np.testing.assert_allclose(values[0]["bbox"], PERFECT_BOX)
        self.assertFalse(network.opt.use_fp16_storage)
        self.assertFalse(network.opt.use_fp16_packed)
        self.assertFalse(network.opt.use_fp16_arithmetic)
        self.assertEqual(metadata["format"], "ncnn-fp16")
        self.assertEqual(metadata["ncnn_compute"], "fp32")
        self.assertIsNone(metadata["nms_iou"])

    def test_subset_filters_images_and_predictions_after_full_split_validation(self) -> None:
        self.dataset(
            [
                FixtureImage("unseen", ((0, PERFECT_BOX),), ("pigeon",)),
                FixtureImage("seen", ((0, PERFECT_BOX),), ("pigeon",)),
                FixtureImage("negative"),
            ]
        )
        self.predictions.write_text(
            json.dumps(
                [
                    detection("unseen"),
                    detection("seen", FALSE_BOX),
                    detection("negative"),
                ]
            )
        )
        subset = self.root / "unseen.txt"
        subset.write_text("unseen\nnegative\n")
        report = evaluate(self.root, "val", self.predictions, subset=subset, threshold=0.5)
        self.assertEqual(mapping(report["counts"], "counts")["images"], 2)
        self.assertEqual(mapping(report["pigeon"], "pigeon")["precision_at_threshold"], 0.5)
        self.assertEqual(report["empty_false_alarm_rate"], 1.0)
        self.assertEqual(report["subset"], str(subset))
        self.predictions.write_text(json.dumps([detection("not-in-split")]))
        with self.assertRaisesRegex(ValueError, "prediction image not in dataset split"):
            evaluate(self.root, "val", self.predictions, subset=subset)

    def test_subset_rejects_unknown_and_empty_ids(self) -> None:
        self.dataset([FixtureImage("image")])
        self.predictions.write_text("[]")
        subset = self.root / "subset.txt"
        for ids in ("absent\n", "\n"):
            with self.subTest(ids=ids):
                subset.write_text(ids)
                with self.assertRaisesRegex(ValueError, "subset"):
                    evaluate(self.root, "val", self.predictions, subset=subset)

    def test_contamination_train_precedes_val_and_old_test_is_unseen(self) -> None:
        original = HistoricalImage(
            "image", "source", "test", "unseen", "stem", "scene", "sha", "hash", "mirror"
        )
        validation = HistoricalImage(
            "validation", "source", "valid", "seen_val", "stem", "scene", "sha", "hash", "mirror"
        )
        training = HistoricalImage(
            "training", "source", "train", "seen_train", "stem", "scene", "sha", "hash", "mirror"
        )
        self.assertEqual(primary_exposure([original]), "unseen")
        self.assertEqual(primary_exposure([original, validation]), "seen_val")
        self.assertEqual(primary_exposure([validation, training]), "seen_train")
        self.assertEqual(primary_exposure([]), "unseen")

    def test_coco_reconstruction_uses_all_birds_before_licence_filter(self) -> None:
        annotation = self.root / "instances_val2017.json"
        bird_count = 20
        annotation.write_text(
            json.dumps(
                {
                    "images": [{"id": index} for index in range(bird_count + 1)],
                    "categories": [{"id": 16, "name": "bird"}, {"id": 1, "name": "person"}],
                    "annotations": [
                        {"id": index, "image_id": index, "category_id": 16}
                        for index in range(bird_count)
                    ]
                    + [{"id": bird_count, "image_id": bird_count, "category_id": 1}],
                }
            )
        )
        splits, original_order = coco_splits(annotation)
        self.assertEqual(len(original_order), bird_count)
        self.assertNotIn(bird_count, splits)
        self.assertEqual(sum(split == "train" for split in splits.values()), 16)
        self.assertEqual(sum(split == "valid" for split in splits.values()), 2)
        self.assertEqual(sum(split == "test" for split in splits.values()), 2)
        self.assertEqual(coco_splits(annotation), (splits, original_order))


if __name__ == "__main__":
    unittest.main()
