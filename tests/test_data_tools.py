from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy
import yaml
from PIL import Image, ImageOps

from tools.auto_label import auto_label, label_policy, reviewed_labels, summarize_review
from tools.build_dataset import (
    DatasetError,
    archive_dataset,
    build_test,
    build_test_styles,
    build_train,
    composite_license,
    freeze,
    image_kind,
    split_by_kind,
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
from tools.style_augment import (
    STYLES,
    apply_style,
    barrel_destination_points,
    barrel_source_coordinates,
    box_edge_points,
    choose_style,
    remap_box,
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

    def test_val_mix_follows_the_test_composition_and_train_only_sources(self) -> None:
        items: list[dict[str, Any]] = []
        for index in range(600):
            kind = ("pigeon", "other_bird", "empty")[index % 3]
            present = {"pigeon": ["pigeon"], "other_bird": ["zebra_dove"], "empty": []}[kind]
            source = "nea" if kind == "pigeon" and index % 9 == 0 else f"s-{kind}"
            items.append(
                {
                    "id": f"i{index}",
                    "group": f"g{index % 3}-{index // 6}",  # two images of one kind
                    "source": source,
                    "_present": present,
                }
            )
        mix = {"pigeon": 0.5, "other_bird": 0.2, "empty": 0.3}
        assignment, report = split_by_kind(items, 0.2, 0, mix, ["nea"])
        val = [item for item in items if assignment[item["group"]] == "val"]
        kinds = Counter(image_kind(item["_present"]) for item in val)
        self.assertAlmostEqual(kinds["empty"] / len(val), 0.3, delta=0.05)
        self.assertAlmostEqual(kinds["other_bird"] / len(val), 0.2, delta=0.05)
        self.assertFalse(any(item["source"] == "nea" for item in val))
        self.assertEqual(report["val_mix_target"], mix)

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


class DatasetFixture(unittest.TestCase):
    """Tiny raw sources and specs shared by the build tests (no tests of its own)."""

    def make_fixture(self) -> None:
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


class BuildDatasetTests(DatasetFixture):
    def setUp(self) -> None:
        self.make_fixture()

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


class AutoLabelTests(unittest.TestCase):
    def test_policy_drops_missed_uncertain_and_mixed_birds(self) -> None:
        box = [10.0, 10.0, 30.0, 30.0]
        far = [60.0, 60.0, 80.0, 80.0]
        kept = label_policy([(0.9, box)], "spotted_dove", 0.35, 0.1)
        self.assertEqual(kept["decision"], "keep")
        self.assertEqual(kept["boxes"][0]["class"], "spotted_dove")
        self.assertEqual(label_policy([], "pigeon", 0.35, 0.1)["flags"], ["no_bird"])
        uncertain = label_policy([(0.9, box), (0.2, far)], "pigeon", 0.35, 0.1)
        self.assertEqual(uncertain["decision"], "drop")
        overlapping = label_policy(
            [(0.9, box), (0.2, [11.0, 11.0, 31.0, 31.0])], "pigeon", 0.35, 0.1
        )
        self.assertEqual(overlapping["decision"], "keep")
        flock = label_policy([(0.9, box), (0.8, far)], "pigeon", 0.35, 0.1)
        self.assertEqual((flock["decision"], flock["flags"]), ("keep", ["multi_bird"]))
        mixed = label_policy([(0.9, box), (0.8, far)], "javan_myna", 0.35, 0.1)
        self.assertEqual(mixed["decision"], "drop")

    def test_labels_resume_and_need_a_passed_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_source(
                root,
                "inat-test",
                [
                    (f"i{index}", noise_image(index), {"extra": {"taxon": "pigeon"}, "boxes": None})
                    for index in range(4)
                ],
            )
            provenance = root / "raw" / "inat-test" / "provenance.jsonl"
            calls: list[int] = []

            def detector(paths: Any) -> list[list[tuple[float, list[float]]]]:
                calls.append(len(paths))
                return [
                    [(0.9, [1.0, 1.0, 20.0, 20.0])] if "i3" not in str(p) else [] for p in paths
                ]

            auto_label(provenance, detector, "mock.pt", batch_size=3)
            auto_label(provenance, detector, "mock.pt", batch_size=3)
            self.assertEqual(calls, [3, 1])
            labels = {
                r["id"]: r
                for r in map(
                    json.loads, (provenance.parent / "auto_labels.jsonl").read_text().splitlines()
                )
            }
            self.assertEqual(labels["i3"]["decision"], "drop")
            self.assertEqual(labels["i0"]["status"], "pseudo_unreviewed")
            review_dir = root / "review"
            review_dir.mkdir()
            self.assertIsNone(reviewed_labels(provenance, review_dir))
            review = review_dir / "inat-test.review.jsonl"
            lines = [{"id": f"r{n}", "boxes": 1} for n in range(199)]
            lines.append({"id": "i1", "boxes": 1, "wrong_boxes": 1, "drop": True})
            review.write_text("".join(json.dumps(line) + "\n" for line in lines))
            summary = summarize_review(review)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["box_error_rate"], 0.005)
            accepted = reviewed_labels(provenance, review_dir)
            assert accepted is not None
            self.assertEqual(sorted(accepted), ["i0", "i2"])
            self.assertEqual(accepted["i0"]["status"], "pseudo_reviewed")
            lines[:12] = [{"id": f"w{n}", "boxes": 1, "wrong_boxes": 1} for n in range(12)]
            review.write_text("".join(json.dumps(line) + "\n" for line in lines))
            self.assertFalse(summarize_review(review)["passed"])
            self.assertIsNone(reviewed_labels(provenance, review_dir))


class StyleAugmentTests(unittest.TestCase):
    def test_barrel_box_mapping_inverts_the_pixel_mapping(self) -> None:
        width, height, k = 640, 384, 0.25
        points = numpy.array([[100.0, 50.0], [320.0, 192.0], [600.0, 300.0], [10.0, 380.0]])
        destination = barrel_destination_points(points, width, height, k)
        back_x, back_y = barrel_source_coordinates(
            destination[:, 0], destination[:, 1], width, height, k
        )
        numpy.testing.assert_allclose(numpy.stack([back_x, back_y], axis=1), points, atol=1e-6)
        corners = numpy.array([[0.0, 0.0], [640.0, 384.0]])
        numpy.testing.assert_allclose(barrel_destination_points(corners, width, height, k), corners)
        # Barrel: off-centre points move outwards; a centred box stays centred and grows.
        box = remap_box([300.0, 172.0, 340.0, 212.0], width, height, k)
        assert box is not None
        self.assertAlmostEqual((box[0] + box[2]) / 2, 320.0, places=3)
        self.assertGreater(box[2] - box[0], 40.0)
        self.assertIsNone(remap_box([0.0, 180.0, 4.0, 200.0], width, height, k))

    def test_styles_are_deterministic_and_move_boxes_consistently(self) -> None:
        image = noise_image(5).resize((160, 96))
        canvas = noise_image(6)
        boxes = [{"class": "pigeon", "xyxy": [40.0, 30.0, 80.0, 60.0]}]
        for style in STYLES:
            first = apply_style(image, boxes, style, 0, "img", canvas)
            second = apply_style(image, boxes, style, 0, "img", canvas)
            self.assertEqual(first[0].tobytes(), second[0].tobytes(), style)
            self.assertEqual(first[1], second[1])
            self.assertEqual(first[0].size, (160, 96))
            other_seed = apply_style(image, boxes, style, 1, "img", canvas)
            self.assertNotEqual(first[0].tobytes(), other_seed[0].tobytes(), style)
        night = apply_style(image, boxes, "noir_night", 0, "img")
        pixels = numpy.asarray(night[0])
        self.assertTrue((pixels[..., 0] == pixels[..., 2]).all())
        self.assertEqual(night[1], boxes)
        wide_image, wide_boxes, params = apply_style(image, boxes, "wide", 0, "img", canvas)
        zoom, (left, top), k = params["zoom"], params["offset"], params["barrel_k"]
        expected = barrel_destination_points(
            box_edge_points([left + 40 * zoom, top + 30 * zoom, left + 80 * zoom, top + 60 * zoom]),
            160,
            96,
            k,
        )
        self.assertAlmostEqual(
            wide_boxes[0]["xyxy"][0], float(numpy.min(expected[:, 0])), delta=0.6
        )
        self.assertLess(wide_boxes[0]["xyxy"][2] - wide_boxes[0]["xyxy"][0], 40 * 0.62 + 2)
        self.assertEqual(
            choose_style({"a": 1, "b": 1}, 0, "x"), choose_style({"b": 1, "a": 1}, 0, "x")
        )
        picks = Counter(choose_style({"a": 3, "b": 1}, 0, f"i{n}") for n in range(400))
        self.assertTrue(250 < picks["a"] < 350, picks)


class PhaseTwoBuildTests(DatasetFixture):
    def setUp(self) -> None:
        self.make_fixture()
        species = ["spotted_dove", "zebra_dove"]
        write_source(
            self.root,
            "inat",
            [
                (
                    f"n{index}",
                    noise_image(400 + index),
                    {
                        "boxes": None,
                        "label_origin": None,
                        "classes_present": [species[index % 2]],
                        "extra": {"taxon": species[index % 2]},
                    },
                )
                for index in range(6)
            ],
        )
        provenance = self.root / "raw" / "inat" / "provenance.jsonl"
        auto_label(
            provenance, lambda paths: [[(0.9, [5.0, 5.0, 50.0, 50.0])] for _ in paths], "m.pt"
        )
        self.review_dir = self.root / "review"
        self.review_dir.mkdir()
        review = [{"id": f"n{n % 6}-{n}", "boxes": 1} for n in range(200)]
        (self.review_dir / "inat.review.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in review)
        )
        summarize_review(self.review_dir / "inat.review.jsonl")
        coco_review = self.root / "coco-review.jsonl"
        coco_review.write_text(
            json.dumps({"id": "c0", "verdict": "pigeon"})
            + "\n"
            + json.dumps({"id": "c1", "verdict": "other_bird"})
            + "\n"
        )
        write_source(
            self.root,
            "coco",
            [
                (
                    f"c{index}",
                    noise_image(500 + index),
                    {
                        "boxes": [{"class": "bird", "xyxy": [8, 8, 30, 30]}],
                        "classes_present": ["bird"],
                    },
                )
                for index in range(3)
            ],
        )
        self.train_spec["sources"] = [
            {"name": "pigeons", "require_boxes": True},
            {"name": "streets"},
            {"name": "inat", "labels": "auto"},
            {"name": "coco", "image_review": str(coco_review)},
        ]
        patcher = patch("tools.build_dataset.REVIEW_DIRECTORY", self.review_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        build_test(self.test_spec, self.root)

    def test_single_v2_uses_reviews_and_keeps_birds_in_classes_present(self) -> None:
        build_train(self.train_spec, self.root)
        records = {record["id"]: record for record in self.manifest("tiny-train")}
        self.assertNotIn("c2", records)  # not reviewed
        self.assertEqual(records["c1"]["classes_present"], ["other_bird"])
        self.assertEqual(records["n0"]["classes_present"], ["spotted_dove"])
        self.assertEqual(records["n0"]["label_origin"], "pseudo_reviewed")
        dataset = self.root / "tiny-train"

        def label(record: dict[str, Any]) -> str:
            return (dataset / "labels" / str(record["split"]) / f"{record['id']}.txt").read_text()

        self.assertEqual(label(records["n0"]), "")
        self.assertEqual(label(records["c1"]), "")
        self.assertTrue(label(records["c0"]).startswith("0 "))

    def test_species_folds_rare_classes_and_style_copies_follow_their_original(self) -> None:
        spec = {**self.train_spec, "name": "tiny-species", "taxonomy": "species"}
        spec["style_copies"] = {"weights": {"noir_night": 1, "noir_day": 1, "wide": 1}, "seed": 0}
        build_train(spec, self.root)
        dataset = self.root / "tiny-species"
        summary = json.loads((dataset / "DATASET.json").read_text())
        self.assertEqual(summary["folded_classes"][:2], ["asian_glossy_starling", "asian_koel"])
        self.assertIn("spotted_dove", summary["folded_classes"])
        self.assertEqual(summary["counts"]["train"]["boxes"]["spotted_dove"], 0)
        records = self.manifest("tiny-species")
        originals = {r["id"]: r for r in records if r["style"] == "original"}
        copies = [r for r in records if r["style"] != "original"]
        train_originals = [r for r in originals.values() if r["split"] == "train"]
        self.assertEqual(len(copies), len(train_originals))
        for copy in copies:
            original = originals[copy["derived_from"]]
            self.assertEqual((copy["split"], copy["group"]), ("train", original["group"]))
            self.assertEqual(copy["classes_present"], original["classes_present"])
        validate_dataset(dataset)

    def test_test_styles_copy_every_test_image_in_every_style(self) -> None:
        spec = {
            "name": "tiny-test-styles",
            "kind": "test_styles",
            "taxonomy": "single",
            "from_dataset": "tiny-test",
            "styles": list(STYLES),
            "seed": 0,
        }
        dataset = build_test_styles(spec, self.root)
        records = self.manifest("tiny-test-styles")
        self.assertEqual(len(records), 4 * len(STYLES))
        self.assertEqual({r["derived_from"] for r in records}, {"p0", "p1", "s0", "s1"})
        summary = json.loads((dataset / "DATASET.json").read_text())
        self.assertEqual(summary["composition"]["wide"]["empty_images"], 1)
        wide_s1 = next(r for r in records if r["id"] == "s1-wide")
        self.assertEqual(wide_s1["style_params"]["canvas"], "s0")  # bird-free, non-SA canvas
        self.assertEqual(composite_license("CC0-1.0", "CC-BY-2.0"), "CC-BY-2.0")
        self.assertEqual(composite_license("CC-BY-SA-4.0", "CC-BY-2.0"), "CC-BY-SA-4.0")
