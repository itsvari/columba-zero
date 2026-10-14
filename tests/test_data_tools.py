from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy
import yaml
from PIL import Image, ImageOps

from tools.build_dataset import (
    DatasetError,
    archive_dataset,
    build_test,
    build_train,
    freeze,
    split_groups,
    validate_dataset,
    verify_freeze,
    yolo_lines,
)
from tools.dataset_audit import main as audit_main
from tools.dedupe_images import assign_groups, hamming, near_duplicate_pairs, perceptual_hashes
from tools.fetch_sources import (
    RateLimiter,
    license_allowed,
    photo_url,
    roboflow_original_stem,
    select_coco_images,
    yolo_to_xyxy,
)

IMAGE_PX = 96
PHASH_MAX_DISTANCE = 10
TEST_EXCLUSION_DISTANCE = 14


def noise_image(seed: int) -> Image.Image:
    """A blocky random image: distinct seeds give unrelated perceptual hashes."""
    rng = numpy.random.default_rng(seed)
    blocks = rng.integers(0, 255, size=(8, 8, 3), dtype=numpy.uint8)
    return Image.fromarray(blocks).resize((IMAGE_PX, IMAGE_PX), Image.Resampling.NEAREST)


def write_source(
    root: Path, name: str, images: list[tuple[str, Image.Image, dict[str, Any]]]
) -> None:
    source = root / "raw" / name
    (source / "images").mkdir(parents=True)
    with open(source / "provenance.jsonl", "w", encoding="utf-8") as handle:
        for image_id, image, fields in images:
            image.save(source / "images" / f"{image_id}.png")
            record = {
                "id": image_id,
                "file": f"images/{image_id}.png",
                "source": name,
                "source_id": image_id,
                "scene_id": f"{name}-{image_id}",
                "license": "CC-BY-4.0",
                "attribution": f"tester, {name}",
                "url": "https://example.invalid",
                "width": IMAGE_PX,
                "height": IMAGE_PX,
                "boxes": [{"class": "pigeon", "xyxy": [10, 10, 40, 40]}],
                "label_origin": "source",
                "classes_present": ["pigeon"],
                **fields,
            }
            handle.write(json.dumps(record) + "\n")


class FetchSourceTests(unittest.TestCase):
    def test_licence_filter_allows_only_cc0_by_and_by_sa(self) -> None:
        for allowed in ("CC0-1.0", "CC-BY-2.0", "CC-BY-4.0", "CC-BY-SA-2.0", "CC-BY-SA-4.0"):
            self.assertTrue(license_allowed(allowed), allowed)
        for excluded in (
            "CC-BY-NC-2.0",
            "CC-BY-NC-SA-2.0",
            "CC-BY-NC-ND-2.0",
            "CC-BY-ND-2.0",
            "all-rights-reserved",
            "Flickr-no-known-copyright-restrictions",
            "",
            None,
        ):
            self.assertFalse(license_allowed(excluded), excluded)
        self.assertTrue(license_allowed("US-Government-Work", ["US-Government-Work"]))

    def test_coco_selection_filters_licences_and_categories(self) -> None:
        annotations = {
            "categories": [{"id": 16, "name": "bird"}, {"id": 3, "name": "car"}],
            "images": [
                {"id": 1, "license": 4},
                {"id": 2, "license": 2},
                {"id": 3, "license": 5},
                {"id": 4, "license": 4},
            ],
            "annotations": [
                {"image_id": 1, "category_id": 16, "bbox": [0, 0, 5, 5]},
                {"image_id": 2, "category_id": 16, "bbox": [0, 0, 5, 5]},
                {"image_id": 3, "category_id": 3, "bbox": [0, 0, 5, 5]},
                {"image_id": 4, "category_id": 3, "bbox": [0, 0, 5, 5]},
                {"image_id": 4, "category_id": 16, "bbox": [0, 0, 5, 5]},
            ],
        }
        birds, licences = select_coco_images(
            {"require_any": ["bird"], "box_categories": ["bird"]}, annotations
        )
        self.assertEqual([image["id"] for image, _ in birds], [1, 4])
        self.assertEqual(licences["dropped:CC-BY-NC-2.0"], 1)
        self.assertEqual(len(birds[1][1]), 1)
        streets, _ = select_coco_images(
            {"require_any": ["car"], "exclude_any": ["bird"], "box_categories": []}, annotations
        )
        self.assertEqual([image["id"] for image, _ in streets], [3])

    def test_roboflow_copies_share_the_original_stem(self) -> None:
        self.assertEqual(roboflow_original_stem("IMG_12_jpg.rf.0a1b2c3d4e5f"), "IMG_12")
        self.assertEqual(roboflow_original_stem("a_b_png.rf.ffff"), "a_b")
        self.assertEqual(roboflow_original_stem("plain"), "plain")
        self.assertEqual(yolo_to_xyxy("0 0.5 0.5 0.5 1.0", 100, 50), (0, [25.0, 0.0, 75.0, 50.0]))
        self.assertIsNone(yolo_to_xyxy("0 0.1 0.1 0.2 0.2 0.3 0.3", 10, 10))

    def test_rate_limiter_spaces_requests(self) -> None:
        now = [100.0]
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        limiter = RateLimiter(1.1, clock=lambda: now[0], sleep=sleep)
        limiter.wait()
        now[0] += 0.4
        limiter.wait()
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 0.7)

    def test_photo_url_switches_size(self) -> None:
        url = "https://inaturalist-open-data.s3.amazonaws.com/photos/1/square.jpg"
        self.assertTrue(photo_url(url, "large").endswith("/photos/1/large.jpg"))


class DedupeTests(unittest.TestCase):
    def test_banded_search_matches_brute_force(self) -> None:
        rng = random.Random(3)
        base = [rng.getrandbits(64) for _ in range(40)]
        hashes = {}
        for index, value in enumerate(base):
            hashes[f"a{index:02d}"] = f"{value:016x}"
            flipped = value
            for bit in rng.sample(range(64), rng.randint(0, 9)):
                flipped ^= 1 << bit
            hashes[f"b{index:02d}"] = f"{flipped:016x}"
        expected = sorted(
            (left, right, hamming(hashes[left], hashes[right]))
            for left in hashes
            for right in hashes
            if left < right and hamming(hashes[left], hashes[right]) <= 6
        )
        self.assertEqual(near_duplicate_pairs(hashes, 6), expected)

    def test_groups_join_scenes_exact_copies_near_copies_and_mirrors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            images = {
                "a": noise_image(1),
                "a_mirror": ImageOps.mirror(noise_image(1)),
                "a_bright": Image.eval(noise_image(1), lambda value: min(255, value + 6)),
                "b": noise_image(2),
                "c": noise_image(3),
                "d": noise_image(4),
            }
            items = []
            for name, image in images.items():
                image.save(folder / f"{name}.png")
                phash, mirror = perceptual_hashes(folder / f"{name}.png")
                items.append({"id": name, "sha256": name, "phash": phash, "phash_mirror": mirror})
            items.append({**items[3], "id": "b_copy"})  # same bytes as b
            items[4]["scene_id"] = "scene-1"  # c and d come from one Roboflow original
            items[5]["scene_id"] = "scene-1"
            groups, clusters = assign_groups(items, PHASH_MAX_DISTANCE)
        self.assertEqual(groups["a"], groups["a_mirror"])
        self.assertEqual(groups["a"], groups["a_bright"])
        self.assertEqual(groups["b"], groups["b_copy"])
        self.assertEqual(groups["c"], groups["d"])
        self.assertEqual(len({groups["a"], groups["b"], groups["c"]}), 3)
        self.assertEqual(len(clusters), 3)
        reversed_groups, _ = assign_groups(list(reversed(items)), PHASH_MAX_DISTANCE)
        self.assertEqual(groups, reversed_groups)

    def test_split_keeps_groups_together(self) -> None:
        items = [
            {"id": f"i{index}", "group": f"g{index // 3}", "source": f"s{index % 2}"}
            for index in range(300)
        ]
        assignment = split_groups(items, 0.2, seed=0)
        self.assertEqual(set(assignment.values()), {"train", "val"})
        val_images = sum(1 for item in items if assignment[item["group"]] == "val")
        self.assertTrue(55 <= val_images <= 70, val_images)
        self.assertEqual(assignment, split_groups(list(reversed(items)), 0.2, seed=0))


class BuildDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.environment = patch.dict(os.environ, {"COLUMBA_DATA_ROOT": str(self.root)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.directory.cleanup)
        write_source(
            self.root,
            "pigeons",
            [(f"p{index}", noise_image(100 + index), {}) for index in range(12)]
            + [
                # p0's mirrored copy under another name, and p1's Roboflow sibling.
                ("p0-mirror", ImageOps.mirror(noise_image(100)), {}),
                ("p1-copy", noise_image(300), {"scene_id": "pigeons-p1"}),
            ],
        )
        write_source(
            self.root,
            "streets",
            [
                (f"s{index}", noise_image(200 + index), {"boxes": [], "classes_present": []})
                for index in range(3)
            ],
        )
        selection = self.root / "selection.jsonl"
        selection.write_text(
            "\n".join(
                json.dumps(line)
                for line in [
                    {"id": "p0", "kind": "pigeon"},
                    {"id": "p1", "kind": "pigeon"},
                    {"id": "s0", "kind": "empty"},
                    {"id": "s1", "kind": "other_bird", "classes_present": ["spotted_dove"]},
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        self.test_spec: dict[str, Any] = {
            "name": "tiny-test",
            "kind": "test",
            "taxonomy": "single",
            "sources": [{"name": "pigeons"}, {"name": "streets"}],
            "selection": str(selection),
            "phash_max_distance": PHASH_MAX_DISTANCE,
        }
        self.train_spec: dict[str, Any] = {
            "name": "tiny-train",
            "kind": "train",
            "taxonomy": "single",
            "sources": [{"name": "pigeons", "require_boxes": True}, {"name": "streets"}],
            "test_dataset": "tiny-test",
            "phash_max_distance": PHASH_MAX_DISTANCE,
            "test_exclusion_distance": TEST_EXCLUSION_DISTANCE,
            "val_fraction": 0.25,
            "seed": 0,
        }

    def manifest(self, name: str) -> list[dict[str, Any]]:
        lines = (self.root / name / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def test_test_and_train_builds_meet_c1_without_leakage(self) -> None:
        test = build_test(self.test_spec, self.root)
        summary = json.loads((test / "DATASET.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["counts"]["test"]["images"], 4)
        self.assertEqual(summary["counts"]["test"]["backgrounds"], 2)
        self.assertEqual(summary["composition"]["empty_images"], 1)
        self.assertEqual(summary["composition"]["confuser_images"], 1)
        records = {record["id"]: record for record in self.manifest("tiny-test")}
        self.assertEqual(records["s1"]["classes_present"], ["spotted_dove"])
        self.assertEqual(records["s0"]["classes_present"], [])
        self.assertEqual(yaml.safe_load((test / "data.yaml").read_text())["names"], ["pigeon"])

        train = build_train(self.train_spec, self.root)
        train_records = self.manifest("tiny-train")
        train_ids = {record["id"] for record in train_records}
        # Test images, p0's mirror (phash) and p1's sibling (scene id) never reach training.
        self.assertFalse(train_ids & {"p0", "p1", "s0", "s1", "p0-mirror", "p1-copy"})
        self.assertEqual(train_ids, {f"p{index}" for index in range(2, 12)} | {"s2"})
        self.assertEqual({record["split"] for record in train_records}, {"train", "val"})
        data = yaml.safe_load((train / "data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["test"], "../tiny-test/images/test")
        self.assertNotIn("path", data)
        report = json.loads((train / "DATASET.json").read_text())["build_report"]
        self.assertGreater(report["min_phash_distance_to_test"], TEST_EXCLUSION_DISTANCE)
        self.assertEqual(report["per_source"]["pigeons"]["dropped_test_group"], 4)
        validate_dataset(train)

    def test_audit_accepts_c1_test_and_train_datasets(self) -> None:
        test = build_test(self.test_spec, self.root)
        train = build_train(self.train_spec, self.root)
        for dataset in (test, train):
            with (
                patch("sys.argv", ["dataset_audit.py", str(dataset / "data.yaml")]),
                patch("builtins.print"),
            ):
                audit_main()
        manifest = train / "manifest.jsonl"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace('"CC-BY-4.0"', '"CC-BY-NC-4.0"'),
            encoding="utf-8",
        )
        with (
            patch("sys.argv", ["dataset_audit.py", str(train / "data.yaml")]),
            patch("builtins.print"),
            self.assertRaises(SystemExit),
        ):
            audit_main()

    def test_archive_is_reproducible_and_freeze_detects_changes(self) -> None:
        test = build_test(self.test_spec, self.root)
        first = archive_dataset(test)
        archive = self.root / "tiny-test.tar.zst"
        self.assertEqual((self.root / "tiny-test.tar.zst.sha256").read_text().split()[0], first)
        self.assertEqual(json.loads((test / "DATASET.json").read_text())["archive_sha256"], first)
        with self.assertRaises(DatasetError):
            archive_dataset(test)
        freeze_file = self.root / "tiny-test.sha256"
        freeze(test, freeze_file)
        self.assertIn(f"{first}  tiny-test.tar.zst", freeze_file.read_text())
        self.assertEqual(verify_freeze(test, freeze_file), [])
        label = test / "labels" / "test" / "p0.txt"
        label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        self.assertEqual(verify_freeze(test, freeze_file), ["changed tiny-test/labels/test/p0.txt"])
        self.assertTrue(archive.exists())

    def test_validation_rejects_unpublishable_records(self) -> None:
        test = build_test(self.test_spec, self.root)
        manifest_path = test / "manifest.jsonl"
        original = manifest_path.read_text(encoding="utf-8")
        cases = {
            "pseudo_unreviewed": (
                '"label_origin": "source"',
                '"label_origin": "pseudo_unreviewed"',
            ),
            "not allowed": ('"license": "CC-BY-4.0"', '"license": "CC-BY-NC-2.0"'),
            "unknown names": ('"classes_present": ["spotted_dove"]', '"classes_present": ["dodo"]'),
            "missing from classes_present": (
                '"classes_present": ["pigeon"]',
                '"classes_present": []',
            ),
        }
        for expected, (old, new) in cases.items():
            manifest_path.write_text(original.replace(old, new), encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, expected):
                validate_dataset(test, check_files=False)
        manifest_path.write_text(original, encoding="utf-8")
        data_path = test / "data.yaml"
        data_path.write_text("names: [bird]\ntest: images/test\n", encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "taxonomy"):
            validate_dataset(test, check_files=False)

    def test_selection_kinds_are_checked(self) -> None:
        selection = Path(self.test_spec["selection"])
        selection.write_text(json.dumps({"id": "s0", "kind": "pigeon"}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "no pigeon boxes"):
            build_test(self.test_spec, self.root)
        selection.write_text(
            json.dumps({"id": "s0", "kind": "empty", "classes_present": ["other_bird"]}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(DatasetError, "empty scene"):
            build_test(self.test_spec, self.root)

    def test_yolo_lines_clip_and_skip_degenerate_boxes(self) -> None:
        lines = yolo_lines(
            [
                {"class": "pigeon", "xyxy": [-10, 0, 50, 100]},
                {"class": "pigeon", "xyxy": [5, 5, 5.5, 50]},
            ],
            ["pigeon"],
            100,
            100,
        )
        self.assertEqual(lines, ["0 0.250000 0.500000 0.500000 1.000000"])


if __name__ == "__main__":
    unittest.main()
