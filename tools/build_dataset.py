#!/usr/bin/env python
"""Build, validate and archive C1 datasets from fetched sources.

    python -m tools.build_dataset build data_sources/builds/columba-test-v2.yaml
    python -m tools.build_dataset validate ~/columba-data/columba-test-v2
    python -m tools.build_dataset archive ~/columba-data/columba-test-v2
    python -m tools.build_dataset freeze ~/columba-data/columba-test-v2 \
        data_sources/columba-test-v2.sha256

A build spec names the dataset, its kind (`test` or `train`), taxonomy and sources
(`raw/<source>/provenance.jsonl`, written by tools/fetch_sources.py). Test builds take an
explicit, reviewed selection; train builds take every eligible image and split it into
train/val by group. Groups come from tools/dedupe_images.py (SHA-256, mirror-aware phash
within `phash_max_distance`, shared scene ids), so a group never crosses splits.
Train builds drop every image in a group with a test image and every image whose phash
(or mirrored phash) is within `test_exclusion_distance` of any test image.

`classes_present` lists every bird in an image by species-taxonomy name (`other_bird` when
the species is unknown), even when the dataset's taxonomy does not box it (CONTRACTS.md
C4 v1.1: `[]` = empty scene, only non-pigeon names = confuser image).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from tools.dedupe_images import (
    all_hashes,
    assign_groups,
    file_sha256,
    hash_source,
    nearest_distances,
    perceptual_hashes,
)
from tools.fetch_sources import data_root, license_allowed, read_provenance

REPOSITORY = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = REPOSITORY / "training" / "classes.yaml"
MANIFEST_FIELDS = (
    "id",
    "split",
    "file",
    "source",
    "source_id",
    "group",
    "license",
    "attribution",
    "sha256",
    "phash",
    "width",
    "height",
    "style",
    "derived_from",
    "label_origin",
    "classes_present",
)
STYLES = {"original", "noir_night", "noir_day", "wide"}
PUBLISHABLE_LABEL_ORIGINS = {"human", "source", "pseudo_reviewed"}
SELECTION_KINDS = {"pigeon", "other_bird", "empty"}
TEST_SPLITS = ("test",)
TRAIN_SPLITS = ("train", "val")
MIN_BOX_PX = 1.0
ZSTD_LEVEL = 10
ARCHIVE_MTIME = 0
TEST_DATASET_RELATIVE = "../{test}/images/test"
REVIEW_DIRECTORY = Path("data_sources/review")
STYLE_MAX_SIDE = 1280
STYLE_WORKERS = 4

Record = dict[str, Any]


class DatasetError(ValueError):
    pass


# --- taxonomy and specs ---------------------------------------------------------------------


def load_taxonomy(path: Path = TAXONOMY_PATH) -> Record:
    return cast(Record, yaml.safe_load(path.read_text(encoding="utf-8")))


def taxonomy_names(taxonomy: Record, name: str) -> list[str]:
    return list(taxonomy["taxonomies"][name]["names"])


def bird_names(taxonomy: Record) -> set[str]:
    return set(taxonomy_names(taxonomy, "species"))


def load_spec(path: Path) -> Record:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise DatasetError(f"{path}: build spec must be a mapping")
    if spec.get("kind") not in ("test", "train", "test_styles"):
        raise DatasetError(f"{path}: kind must be test, train or test_styles")
    spec["sources"] = [
        source if isinstance(source, dict) else {"name": source}
        for source in spec.get("sources", [])
    ]
    return cast(Record, spec)


def git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "tools",
                "data_sources",
                "training",
            ],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


# --- loading source images ------------------------------------------------------------------


def load_source_items(root: Path, source_name: str) -> list[Record]:
    provenance = root / "raw" / source_name / "provenance.jsonl"
    hashes = hash_source(provenance)
    items = []
    for record in read_provenance(provenance):
        item = {**record, **hashes[record["id"]], "file": record["file"]}
        item["path"] = str(provenance.parent / record["file"])
        items.append(item)
    return items


def drop_exact_duplicates(items: Sequence[Record]) -> tuple[list[Record], list[Record]]:
    """Keep the first image (by id) of every set of byte-identical files."""
    kept: dict[str, Record] = {}
    dropped = []
    for item in sorted(items, key=lambda record: record["id"]):
        if item["sha256"] in kept:
            dropped.append(item)
        else:
            kept[item["sha256"]] = item
    return sorted(kept.values(), key=lambda record: record["id"]), dropped


# --- labels ---------------------------------------------------------------------------------


def yolo_lines(boxes: Iterable[Record], names: Sequence[str], width: int, height: int) -> list[str]:
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
            continue
        index = names.index(box["class"])
        lines.append(
            f"{index} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    return lines


def mapped_boxes(item: Record, class_map: dict[str, str | None]) -> list[Record]:
    boxes = []
    for box in item.get("boxes") or []:
        name = class_map.get(box["class"], box["class"])
        if name is not None:
            boxes.append({"class": name, "xyxy": box["xyxy"]})
    return boxes


def exif_orientation(path: Path) -> int:
    from PIL import Image

    with Image.open(path) as image:
        return int(image.getexif().get(0x0112, 1))


# --- splitting ------------------------------------------------------------------------------


def split_groups(
    items: Sequence[Record], val_fraction: float, seed: int, stratify_key: str = "source"
) -> dict[str, str]:
    """Assign whole groups to train or val.

    Per stratum (the most common `stratify_key` value among a group's images), groups are
    shuffled with `seed`, then taken largest first; each goes to val while that keeps val
    closer to `val_fraction` of the stratum's images than leaving it out would.
    """
    members: dict[str, list[Record]] = defaultdict(list)
    for item in items:
        members[item["group"]].append(item)
    strata: dict[str, list[str]] = defaultdict(list)
    for group, records in sorted(members.items()):
        values = Counter(str(record[stratify_key]) for record in records)
        stratum = min(values, key=lambda value: (-values[value], value))
        strata[stratum].append(group)
    assignment: dict[str, str] = {}
    for stratum, groups in sorted(strata.items()):
        rng = random.Random(f"{seed}:{stratum}")
        rng.shuffle(groups)
        groups.sort(key=lambda group: -len(members[group]))  # stable: shuffled within a size
        target = val_fraction * sum(len(members[group]) for group in groups)
        in_val = 0
        for group in groups:
            size = len(members[group])
            if abs(in_val + size - target) < abs(in_val - target):
                assignment[group] = "val"
                in_val += size
            else:
                assignment[group] = "train"
    return assignment


def image_kind(present: Sequence[str]) -> str:
    """C4 image kinds: `pigeon` (a pigeon present), `other_bird` (only other birds), `empty`."""
    if "pigeon" in present:
        return "pigeon"
    return "other_bird" if present else "empty"


def split_by_kind(
    items: Sequence[Record],
    val_fraction: float,
    seed: int,
    val_mix: dict[str, float] | None,
    train_only_sources: Iterable[str] = (),
) -> tuple[dict[str, str], Record]:
    """Group split whose val split has a chosen image-kind mix (e.g. test-v2's).

    A group's kind is the strongest kind among its images (pigeon > other_bird > empty).
    Pigeon groups go to val with `val_fraction`; each other kind gets the fraction that makes
    val's kind counts follow `val_mix` (capped at 0.5 of that kind). Within a kind, groups are
    stratified by source (and, for other birds, by species) as in `split_groups`. Groups with
    an image from `train_only_sources` always go to train. Without `val_mix`, this is
    `split_groups`.
    """
    train_only = set(train_only_sources)
    groups: dict[str, list[Record]] = defaultdict(list)
    for item in items:
        groups[item["group"]].append(item)
    rank = {"pigeon": 0, "other_bird": 1, "empty": 2}
    kind_of = {
        group: min((image_kind(item["_present"]) for item in members), key=rank.__getitem__)
        for group, members in groups.items()
    }
    forced = {
        group
        for group, members in groups.items()
        if any(i["source"] in train_only for i in members)
    }
    assignment = {group: "train" for group in forced}
    eligible = [item for item in items if item["group"] not in forced]
    if not val_mix:
        assignment.update(split_groups(eligible, val_fraction, seed))
        return assignment, {}
    by_kind: dict[str, list[Record]] = defaultdict(list)
    for item in eligible:
        stratum = item["source"]
        if kind_of[item["group"]] == "other_bird":
            stratum += ":" + ",".join(item["_present"][:1])
        by_kind[kind_of[item["group"]]].append({**item, "_stratum": stratum})
    pigeon_val = val_fraction * len(by_kind["pigeon"])
    fractions = {"pigeon": val_fraction}
    for kind in ("other_bird", "empty"):
        wanted = pigeon_val * val_mix.get(kind, 0.0) / val_mix["pigeon"]
        available = len(by_kind[kind])
        fractions[kind] = min(0.5, wanted / available) if available else 0.0
    for kind, kind_items in by_kind.items():
        assignment.update(split_groups(kind_items, fractions[kind], seed, "_stratum"))
    return assignment, {"val_mix_target": val_mix, "kind_val_fractions": fractions}


def split_composition(entries: Sequence[tuple[Record, str, list[Record], list[str]]]) -> Record:
    """Per split: image counts by kind and their fractions (C4 empty/confuser definitions)."""
    result: Record = {}
    for split in sorted({entry[1] for entry in entries}):
        kinds = Counter(image_kind(present) for _, s, _, present in entries if s == split)
        total = sum(kinds.values())
        result[split] = {
            "images": total,
            "pigeon_images": kinds["pigeon"],
            "confuser_images": kinds["other_bird"],
            "empty_images": kinds["empty"],
            "confuser_fraction": round(kinds["other_bird"] / total, 4) if total else 0.0,
            "empty_fraction": round(kinds["empty"] / total, 4) if total else 0.0,
        }
    return result


# --- build ----------------------------------------------------------------------------------


def manifest_record(item: Record, split: str, file_name: str, classes_present: list[str]) -> Record:
    return {
        "id": item["id"],
        "split": split,
        "file": f"images/{split}/{file_name}",
        "source": item["source"],
        "source_id": item["source_id"],
        "group": item["group"],
        "license": item["license"],
        "attribution": item["attribution"],
        "sha256": item["sha256"],
        "phash": item["phash"],
        "width": item["width"],
        "height": item["height"],
        "style": item.get("style", "original"),
        "derived_from": item.get("derived_from"),
        "label_origin": item["label_origin"],
        "classes_present": sorted(set(classes_present)),
        **({"style_params": item["style_params"]} if item.get("style_params") else {}),
    }


def write_dataset(
    output: Path,
    spec: Record,
    entries: Sequence[tuple[Record, str, list[Record], list[str]]],
    names: Sequence[str],
    extra: Record,
) -> Path:
    """`entries`: (item, split, boxes in taxonomy names, classes_present)."""
    if output.exists():
        raise DatasetError(f"{output} exists; datasets are immutable, use a new name")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    manifest = []
    counts: dict[str, Record] = {}
    splits = TRAIN_SPLITS if spec["kind"] == "train" else TEST_SPLITS
    for split in splits:
        (staging / "images" / split).mkdir(parents=True)
        (staging / "labels" / split).mkdir(parents=True)
        counts[split] = {"images": 0, "boxes": {name: 0 for name in names}, "backgrounds": 0}
    for item, split, boxes, classes_present in sorted(entries, key=lambda entry: entry[0]["id"]):
        source_path = Path(item["path"])
        file_name = item["id"] + source_path.suffix.lower().replace(".jpeg", ".jpg")
        destination = staging / "images" / split / file_name
        if exif_orientation(source_path) != 1:
            # Boxes and width/height are in EXIF-oriented pixels; store the oriented pixels.
            from PIL import Image, ImageOps

            file_name = item["id"] + ".jpg"
            destination = staging / "images" / split / file_name
            with Image.open(source_path) as opened:
                ImageOps.exif_transpose(opened).convert("RGB").save(destination, quality=95)
            item = {**item, "sha256": file_sha256(destination)}
        else:
            shutil.copyfile(source_path, destination)
        lines = yolo_lines(boxes, names, item["width"], item["height"])
        (staging / "labels" / split / f"{item['id']}.txt").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8"
        )
        manifest.append(manifest_record(item, split, file_name, classes_present))
        counts[split]["images"] += 1
        counts[split]["backgrounds"] += 0 if lines else 1
        for line in lines:
            counts[split]["boxes"][names[int(line.split()[0])]] += 1
    with open(staging / "manifest.jsonl", "w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    data: Record = {"names": list(names), "nc": len(names)}
    if spec["kind"] != "train":
        data["test"] = "images/test"
    else:
        data.update(
            train="images/train",
            val="images/val",
            test=TEST_DATASET_RELATIVE.format(test=spec["test_dataset"]),
        )
    (staging / "data.yaml").write_text(
        "# C1 dataset. No `path:` key: Ultralytics then resolves paths from this file's folder.\n"
        + yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    taxonomy = load_taxonomy()
    summary: Record = {
        "name": spec["name"],
        "taxonomy": spec["taxonomy"],
        "taxonomy_version": taxonomy["version"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder_commit": git_commit(),
        "sources": sorted({record["source"] for record in manifest}),
        "counts": counts,
        "dedupe": {
            "phash_max_distance": spec.get("phash_max_distance"),
            "mirror_aware": True,
            "test_dataset": spec.get("test_dataset", spec["name"]),
        },
        "licenses": dict(sorted(Counter(record["license"] for record in manifest).items())),
        "label_origins": dict(
            sorted(Counter(record["label_origin"] for record in manifest).items())
        ),
        "archive_sha256": None,
        "notes": spec.get("notes", ""),
        **extra,
    }
    if spec["kind"] == "train":
        summary["dedupe"]["test_exclusion_distance"] = spec["test_exclusion_distance"]
        summary["dedupe"]["phash256_confirm_distance"] = spec.get("phash256_confirm_distance")
    if any(record["style"] != "original" for record in manifest):
        summary["styles"] = {
            split: dict(
                sorted(Counter(r["style"] for r in manifest if r["split"] == split).items())
            )
            for split in splits
        }
    write_summary(staging, summary)
    validate_dataset(staging)
    staging.rename(output)
    return output


def write_summary(dataset: Path, summary: Record) -> None:
    (dataset / "DATASET.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def build_test(spec: Record, root: Path) -> Path:
    taxonomy = load_taxonomy()
    names = taxonomy_names(taxonomy, spec["taxonomy"])
    items: dict[str, Record] = {}
    for source in spec["sources"]:
        items.update({item["id"]: item for item in load_source_items(root, source["name"])})
    selection = read_provenance(REPOSITORY / spec["selection"])
    chosen = []
    for choice in selection:
        if choice["id"] not in items:
            raise DatasetError(f"selected image {choice['id']} is not in the listed sources")
        chosen.append((items[choice["id"]], choice))
    unique, duplicates = drop_exact_duplicates([item for item, _ in chosen])
    if duplicates:
        raise DatasetError(f"selection has byte-identical images: {[d['id'] for d in duplicates]}")
    groups, _ = assign_groups(unique, spec["phash_max_distance"])
    entries = []
    for item, choice in chosen:
        item["group"] = groups[item["id"]]
        # Images without source labels (iNaturalist) get their (empty) label from the review.
        item["label_origin"] = choice.get("label_origin") or item.get("label_origin") or "human"
        kind = choice["kind"]
        if kind not in SELECTION_KINDS:
            raise DatasetError(f"{item['id']}: unknown selection kind {kind!r}")
        if kind == "pigeon":
            class_map = {**{"bird": "pigeon"}, **spec.get("class_map", {})}
            boxes = [box for box in mapped_boxes(item, class_map) if box["class"] in names]
            if not boxes:
                raise DatasetError(f"{item['id']}: selected as pigeon but has no pigeon boxes")
            classes_present = ["pigeon", *choice.get("classes_present", [])]
        else:
            boxes = []
            classes_present = list(choice.get("classes_present", []))
            if kind == "empty" and classes_present:
                raise DatasetError(f"{item['id']}: an empty scene cannot list birds")
            if kind == "other_bird" and (not classes_present or "pigeon" in classes_present):
                raise DatasetError(f"{item['id']}: other_bird images list non-pigeon birds only")
        entries.append((item, "test", boxes, classes_present))
    extra = {
        "composition": test_composition(entries),
        "review": spec.get("review", ""),
    }
    return write_dataset(root / spec["name"], spec, entries, names, extra)


def test_composition(entries: Sequence[tuple[Record, str, list[Record], list[str]]]) -> Record:
    from tools.contact_sheet import is_small

    images = len(entries)
    empty = sum(1 for _, _, _, present in entries if not present)
    confuser = sum(1 for _, _, _, present in entries if present and "pigeon" not in present)
    boxes = [(box, item) for item, _, item_boxes, _ in entries for box in item_boxes]
    small = sum(1 for box, item in boxes if is_small(box["xyxy"], item["width"], item["height"]))
    return {
        "images": images,
        "empty_images": empty,
        "confuser_images": confuser,
        "pigeon_images": images - empty - confuser,
        "pigeon_boxes": len(boxes),
        "small_pigeon_boxes_384x640": small,
        "empty_fraction": round(empty / images, 4) if images else 0.0,
        "confuser_fraction": round(confuser / images, 4) if images else 0.0,
        "small_box_fraction": round(small / len(boxes), 4) if boxes else 0.0,
    }


def load_dataset_manifest(dataset: Path) -> list[Record]:
    return read_provenance(dataset / "manifest.jsonl")


def raw_scene_ids(root: Path, records: Sequence[Record]) -> dict[str, str]:
    """Map dataset image ids to the scene ids of their raw source images."""
    scenes = {}
    for source in sorted({record["source"] for record in records}):
        provenance = root / "raw" / source / "provenance.jsonl"
        for raw in read_provenance(provenance):
            if raw.get("scene_id"):
                scenes[raw["id"]] = raw["scene_id"]
    return scenes


def load_image_review(path: Path) -> dict[str, Record]:
    return {record["id"]: record for record in read_provenance(path)}


def source_candidates(
    root: Path, source: Record, names: Sequence[str], stats: Counter[str]
) -> list[Record]:
    """Images of one source with their boxes (`_boxes`, taxonomy names) and `_present`.

    Source options: `labels: auto` (reviewed tools/auto_label.py boxes only), `image_review`
    (per-image verdicts: `pigeon` / `other_bird` relabel every box and set classes_present,
    `keep` keeps the source labels, `drop` removes the image; unlisted images take
    `review_default`, and are dropped if it is unset), `class_map`, `classes_present_map`,
    `require_boxes`.
    """
    from tools.auto_label import reviewed_labels

    items = load_source_items(root, source["name"])
    stats["fetched"] = len(items)
    auto = None
    if source.get("labels") == "auto":
        auto = reviewed_labels(
            root / "raw" / source["name"] / "provenance.jsonl", REPOSITORY / REVIEW_DIRECTORY
        )
        if auto is None:
            raise DatasetError(f"{source['name']}: auto labels have not passed review")
    review = (
        load_image_review(REPOSITORY / source["image_review"]) if "image_review" in source else None
    )
    class_map = source.get("class_map", {})
    present_map = source.get("classes_present_map", {})
    candidates = []
    for item in items:
        if auto is not None:
            label = auto.get(item["id"])
            if label is None:
                stats["dropped_by_auto_label_policy_or_review"] += 1
                continue
            item["boxes"] = [{"class": box["class"], "xyxy": box["xyxy"]} for box in label["boxes"]]
            item["label_origin"] = "pseudo_reviewed"
        if review is not None:
            verdict = review.get(item["id"], {}).get("verdict", source.get("review_default"))
            if verdict in (None, "drop"):
                stats["dropped_by_review"] += 1
                continue
            if verdict in ("pigeon", "other_bird"):
                item["boxes"] = [{**box, "class": verdict} for box in item.get("boxes") or []]
                item["classes_present"] = [verdict]
        boxes = [box for box in mapped_boxes(item, class_map) if box["class"] in names]
        if source.get("require_boxes") and not boxes:
            stats["dropped_without_boxes"] += 1
            continue
        if item["label_origin"] not in PUBLISHABLE_LABEL_ORIGINS:
            stats["dropped_unpublishable_labels"] += 1
            continue
        present = {present_map.get(name, name) for name in item.get("classes_present") or []}
        present.update(box["class"] for box in boxes)
        item["_boxes"], item["_present"] = boxes, sorted(present)
        item["_all_boxes"] = item.get("boxes") or []
        candidates.append(item)
    return candidates


def stable_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def apply_caps(items: Sequence[Record], sources: Sequence[Record], stats: Record) -> list[Record]:
    """Per source `max_images`: keep the preferred images. `prefer: small_birds` ranks images by
    their number of small boxes (at the 384x640 deploy letterbox), then box count; otherwise
    a stable pseudo-random order."""
    from tools.contact_sheet import is_small

    options = {source["name"]: source for source in sources}
    by_source: dict[str, list[Record]] = defaultdict(list)
    for item in items:
        by_source[item["source"]].append(item)
    kept: list[Record] = []
    for name, source_items in sorted(by_source.items()):
        option = options[name]
        limit = option.get("max_images")
        if limit is None or len(source_items) <= limit:
            kept.extend(source_items)
            continue
        if option.get("prefer") == "small_birds":

            def rank(item: Record) -> tuple[int, int, str]:
                boxes = item["_all_boxes"]
                small = sum(is_small(b["xyxy"], item["width"], item["height"]) for b in boxes)
                return (-small, -len(boxes), stable_key(item["id"]))

            ordered = sorted(source_items, key=rank)
        else:
            ordered = sorted(source_items, key=lambda item: stable_key(item["id"]))
        kept.extend(ordered[:limit])
        stats[name]["dropped_over_cap"] = len(source_items) - limit
    return kept


def fold_rare_classes(
    entries: Sequence[tuple[Record, str, list[Record], list[str]]], taxonomy: Record
) -> tuple[list[tuple[Record, str, list[Record], list[str]]], list[str]]:
    """Species taxonomy: classes with fewer train boxes than `min_train_instances` become
    `other_bird` boxes (classes_present keeps the real species)."""
    minimum = int(taxonomy["taxonomies"]["species"]["min_train_instances"])
    counts = Counter(
        box["class"] for _, split, boxes, _ in entries if split == "train" for box in boxes
    )
    names = taxonomy_names(taxonomy, "species")
    folded = sorted(
        name for name in names if name not in ("pigeon", "other_bird") and counts[name] < minimum
    )
    if not folded:
        return list(entries), []
    result = []
    for item, split, boxes, present in entries:
        new_boxes = [
            {**box, "class": "other_bird"} if box["class"] in folded else box for box in boxes
        ]
        result.append((item, split, new_boxes, present))
    return result, folded


def build_train(spec: Record, root: Path) -> Path:
    taxonomy = load_taxonomy()
    names = taxonomy_names(taxonomy, spec["taxonomy"])
    test_dir = root / spec["test_dataset"]
    test_records = load_dataset_manifest(test_dir)
    test_fine = {record["id"]: all_hashes(test_dir / record["file"]) for record in test_records}
    test_hashes = {key: (value["phash"], value["phash_mirror"]) for key, value in test_fine.items()}
    test_sha = {record["sha256"] for record in test_records}
    candidates: list[Record] = []
    report: Record = {"per_source": {}}
    for source in spec["sources"]:
        stats: Counter[str] = Counter()
        candidates.extend(source_candidates(root, source, names, stats))
        report["per_source"][source["name"]] = stats
    unique, duplicates = drop_exact_duplicates(candidates)
    for item in duplicates:
        report["per_source"][item["source"]]["dropped_exact_duplicate"] += 1
    # Group train candidates together with the test images, so a group containing a test
    # image (shared scene id, SHA-256 or phash link) is removed from training as a whole.
    test_scene = raw_scene_ids(root, test_records)
    test_items = [
        {
            "id": f"test::{record['id']}",
            "sha256": record["sha256"],
            "phash": test_hashes[record["id"]][0],
            "phash_mirror": test_hashes[record["id"]][1],
            "phash256": test_fine[record["id"]]["phash256"],
            "phash256_mirror": test_fine[record["id"]]["phash256_mirror"],
            "scene_id": test_scene.get(record["id"]),
        }
        for record in test_records
    ]
    keys = ("id", "sha256", "phash", "phash_mirror", "phash256", "phash256_mirror", "scene_id")
    pool = [{key: item.get(key) for key in keys} for item in unique]
    groups, _ = assign_groups(
        [*pool, *test_items], spec["phash_max_distance"], spec.get("phash256_confirm_distance")
    )
    test_groups = {groups[item["id"]] for item in test_items}
    nearest = nearest_distances(
        {item["id"]: [item["phash"], item["phash_mirror"]] for item in unique},
        {key: list(value) for key, value in test_hashes.items()},
    )
    kept = []
    for item in unique:
        item["group"] = groups[item["id"]]
        stats = report["per_source"][item["source"]]
        if item["sha256"] in test_sha or item["group"] in test_groups:
            stats["dropped_test_group"] += 1
        elif nearest[item["id"]][0] <= spec["test_exclusion_distance"]:
            stats["dropped_near_test"] += 1
        else:
            kept.append(item)
    kept = apply_caps(kept, spec["sources"], report["per_source"])
    train_only = [source["name"] for source in spec["sources"] if source.get("train_only")]
    assignment, split_report = split_by_kind(
        kept, spec["val_fraction"], spec["seed"], spec.get("val_mix"), train_only
    )
    entries = [(item, assignment[item["group"]], item["_boxes"], item["_present"]) for item in kept]
    for item in kept:
        report["per_source"][item["source"]][f"kept_{assignment[item['group']]}"] += 1
    min_distance = min((nearest[item["id"]][0] for item in kept), default=None)
    folded: list[str] = []
    if spec["taxonomy"] == "species":
        entries, folded = fold_rare_classes(entries, taxonomy)
    style_report: Record = {}
    if spec.get("style_copies"):
        copies, style_report = style_copies(root, spec, entries)
        entries = [*entries, *copies]
    extra = {
        "composition": split_composition(
            [e for e in entries if e[0].get("style", "original") == "original"]
        ),
        "split": split_report,
        "folded_classes": folded,
        "style_copies": style_report,
        "build_report": {
            "per_source": {name: dict(stats) for name, stats in report["per_source"].items()},
            "min_phash_distance_to_test": min_distance,
            "val_fraction": spec["val_fraction"],
            "seed": spec["seed"],
        },
        "label_conventions": spec.get("label_conventions", ""),
    }
    output = write_dataset(root / spec["name"], spec, entries, names, extra)
    shutil.rmtree(root / "_style_cache" / spec["name"], ignore_errors=True)
    return output


# --- style copies ---------------------------------------------------------------------------


def render_style_copy(job: Record) -> Record:
    """Worker: write one style copy and return its boxes, size, hashes and parameters."""
    from PIL import Image, ImageOps

    from tools.style_augment import JPEG_QUALITY, apply_style

    with Image.open(job["source"]) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    scale = min(1.0, STYLE_MAX_SIDE / max(image.size))
    boxes = job["boxes"]
    if scale < 1.0:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
        boxes = [{**box, "xyxy": [value * scale for value in box["xyxy"]]} for box in boxes]
    canvas = None
    if job.get("canvas"):
        with Image.open(job["canvas"]) as opened:
            canvas = ImageOps.exif_transpose(opened).convert("RGB")
    styled, new_boxes, params = apply_style(
        image, boxes, job["style"], job["seed"], job["original_id"], canvas
    )
    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    styled.save(output, quality=JPEG_QUALITY)
    phash, phash_mirror = perceptual_hashes(output)
    return {
        "boxes": new_boxes,
        "params": params,
        "width": styled.width,
        "height": styled.height,
        "sha256": file_sha256(output),
        "phash": phash,
        "phash_mirror": phash_mirror,
    }


def composite_license(image: str, background: str) -> str:
    """Licence of an image pasted onto a background. Canvases are never ShareAlike, so an SA
    image keeps its licence; a public-domain image takes the background's attribution licence."""
    if image in ("CC0-1.0", "PDM"):
        return background
    return image


def canvas_pool(
    entries: Sequence[tuple[Record, str, list[Record], list[str]]], split: str
) -> list[Record]:
    """Bird-free scenes of one split whose licence lets any original keep its own licence."""
    return sorted(
        (
            item
            for item, entry_split, _, present in entries
            if entry_split == split
            and not present
            and item.get("style", "original") == "original"
            and "-SA-" not in item["license"]
        ),
        key=lambda item: item["id"],
    )


def make_style_entries(
    root: Path,
    name: str,
    originals: Sequence[tuple[Record, str, list[Record], list[str]]],
    styles_for: dict[str, list[str]],
    canvases: Sequence[Record],
    seed: int,
) -> list[tuple[Record, str, list[Record], list[str]]]:
    """Style copies of `originals`; `styles_for` maps an original id to its styles."""
    from concurrent.futures import ProcessPoolExecutor

    from tools.style_augment import style_rng

    work = root / "_style_cache" / name
    if work.exists():
        shutil.rmtree(work)
    jobs: list[tuple[Record, Record, str, list[str], Record | None]] = []
    for item, split, boxes, present in originals:
        for style in styles_for.get(item["id"], []):
            canvas = None
            if style == "wide" and canvases:
                rng = style_rng(seed, item["id"], "canvas")
                choices = [c for c in canvases if c["id"] != item["id"]]
                canvas = choices[int(rng.integers(0, len(choices)))] if choices else None
            job = {
                "source": item["path"],
                "boxes": boxes,
                "style": style,
                "seed": seed,
                "original_id": item["id"],
                "canvas": canvas["path"] if canvas else None,
                "output": str(work / f"{item['id']}-{style}.jpg"),
            }
            jobs.append((job, item, split, present, canvas))
    with ProcessPoolExecutor(max_workers=STYLE_WORKERS) as pool:
        results = list(pool.map(render_style_copy, [job for job, *_ in jobs], chunksize=8))
    copies = []
    for (job, item, split, present, canvas), result in zip(jobs, results, strict=True):
        attribution = item["attribution"]
        licence = item["license"]
        if canvas is not None:
            attribution += f"; background: {canvas['attribution']}"
            licence = composite_license(licence, canvas["license"])
        copy = {
            **{key: item[key] for key in ("source", "source_id", "group", "label_origin")},
            "license": licence,
            "id": f"{item['id']}-{job['style']}",
            "path": job["output"],
            "attribution": attribution,
            "sha256": result["sha256"],
            "phash": result["phash"],
            "width": result["width"],
            "height": result["height"],
            "style": job["style"],
            "derived_from": item["id"],
            "style_params": {
                **result["params"],
                **({"canvas": canvas["id"]} if canvas is not None else {}),
            },
        }
        copies.append((copy, split, result["boxes"], present))
    return copies


def style_copies(
    root: Path, spec: Record, entries: Sequence[tuple[Record, str, list[Record], list[str]]]
) -> tuple[list[tuple[Record, str, list[Record], list[str]]], Record]:
    """One style copy per train image (style drawn with `style_copies.weights`); val stays
    original. Copies share their original's group."""
    from tools.style_augment import choose_style

    options = spec["style_copies"]
    seed = int(options.get("seed", 0))
    train = [entry for entry in entries if entry[1] == "train"]
    styles_for = {
        item["id"]: [choose_style(options["weights"], seed, item["id"])] for item, *_ in train
    }
    copies = make_style_entries(
        root, spec["name"], train, styles_for, canvas_pool(entries, "train"), seed
    )
    report = {
        "seed": seed,
        "weights": options["weights"],
        "copies": dict(sorted(Counter(copy["style"] for copy, *_ in copies).items())),
        "split": "train only (val stays original)",
        "max_side_px": STYLE_MAX_SIDE,
    }
    return copies, report


def yolo_label_boxes(label: Path, names: Sequence[str], width: int, height: int) -> list[Record]:
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index, cx, cy, w, h = line.split()
        x, y, bw, bh = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        boxes.append(
            {"class": names[int(index)], "xyxy": [x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2]}
        )
    return boxes


def build_test_styles(spec: Record, root: Path) -> Path:
    """Every image of a frozen test dataset in every listed style (a new test dataset)."""
    taxonomy = load_taxonomy()
    names = taxonomy_names(taxonomy, spec["taxonomy"])
    source = root / spec["from_dataset"]
    originals = []
    for record in load_dataset_manifest(source):
        item = {**record, "path": str(source / record["file"])}
        label = source / "labels" / "test" / f"{Path(record['file']).stem}.txt"
        boxes = yolo_label_boxes(label, names, record["width"], record["height"])
        originals.append((item, "test", boxes, list(record["classes_present"])))
    seed = int(spec.get("seed", 0))
    styles_for = {item["id"]: list(spec["styles"]) for item, *_ in originals}
    copies = make_style_entries(
        root, spec["name"], originals, styles_for, canvas_pool(originals, "test"), seed
    )
    composition = {
        style: test_composition([entry for entry in copies if entry[0]["style"] == style])
        for style in spec["styles"]
    }
    extra = {"from_dataset": spec["from_dataset"], "seed": seed, "composition": composition}
    output = write_dataset(root / spec["name"], spec, copies, names, extra)
    shutil.rmtree(root / "_style_cache" / spec["name"], ignore_errors=True)
    return output


# --- validation -----------------------------------------------------------------------------


def validate_dataset(dataset: Path, check_files: bool = True) -> list[str]:
    """Check C1 invariants. Raises DatasetError listing every problem; returns notes."""
    problems: list[str] = []
    notes: list[str] = []
    summary = json.loads((dataset / "DATASET.json").read_text(encoding="utf-8"))
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    taxonomy = load_taxonomy()
    names = taxonomy_names(taxonomy, summary["taxonomy"])
    if data.get("names") != names:
        problems.append(f"data.yaml names {data.get('names')} != taxonomy {names}")
    if summary.get("taxonomy_version") != taxonomy["version"]:
        problems.append("DATASET.json taxonomy_version does not match training/classes.yaml")
    for key in ("name", "taxonomy", "created_at", "builder_commit", "sources", "counts", "dedupe"):
        if key not in summary:
            problems.append(f"DATASET.json lacks {key}")
    if summary.get("name") != dataset.name.removesuffix(".building"):
        problems.append(f"DATASET.json name {summary.get('name')} != directory name")
    is_test = "test" in summary.get("counts", {})
    splits = TEST_SPLITS if is_test else TRAIN_SPLITS
    if set(summary.get("counts", {})) != set(splits):
        problems.append(f"splits must be exactly {splits}")
    if not is_test and not str(data.get("test", "")).startswith("../"):
        problems.append("train data.yaml must point test: at the frozen test dataset")
    records = load_dataset_manifest(dataset)
    known_birds = bird_names(taxonomy)
    seen_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, Counter[str]] = {split: Counter() for split in splits}
    for record in records:
        where = record.get("id", "?")
        missing = [field for field in MANIFEST_FIELDS if field not in record]
        if missing:
            problems.append(f"{where}: manifest lacks {missing}")
            continue
        if record["id"] in seen_ids:
            problems.append(f"{where}: duplicate id")
        seen_ids.add(record["id"])
        if record["split"] not in splits:
            problems.append(f"{where}: split {record['split']} not allowed here")
            continue
        if record["label_origin"] not in PUBLISHABLE_LABEL_ORIGINS:
            problems.append(f"{where}: label_origin {record['label_origin']} cannot be published")
        if not license_allowed(record["license"]):
            problems.append(f"{where}: licence {record['license']!r} not allowed")
        if not str(record["attribution"]).strip():
            problems.append(f"{where}: attribution missing")
        if record["style"] not in STYLES:
            problems.append(f"{where}: unknown style {record['style']}")
        if (record["style"] == "original") != (record["derived_from"] is None):
            problems.append(f"{where}: style copies (only) need derived_from")
        unknown = set(record["classes_present"]) - known_birds
        if unknown:
            problems.append(f"{where}: classes_present has unknown names {sorted(unknown)}")
        group_splits[record["group"]].add(record["split"])
        label = dataset / "labels" / record["split"] / f"{Path(record['file']).stem}.txt"
        if not record["file"].startswith(f"images/{record['split']}/"):
            problems.append(f"{where}: file is not under images/{record['split']}/")
        if not label.exists():
            problems.append(f"{where}: label file missing")
            continue
        boxes_by_class: Counter[str] = Counter()
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            try:
                index = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except (ValueError, IndexError):
                problems.append(f"{where}: bad label line {line!r}")
                continue
            if len(values) != 4 or not 0 <= index < len(names):
                problems.append(f"{where}: bad label line {line!r}")
                continue
            cx, cy, width, height = values
            if not (0 <= cx - width / 2 + 1e-6 and cx + width / 2 <= 1 + 1e-6 and width > 0):
                problems.append(f"{where}: box outside the image {line!r}")
            if not (0 <= cy - height / 2 + 1e-6 and cy + height / 2 <= 1 + 1e-6 and height > 0):
                problems.append(f"{where}: box outside the image {line!r}")
            boxes_by_class[names[index]] += 1
        non_pigeon_present = any(name != "pigeon" for name in record["classes_present"])
        for name in boxes_by_class:
            # Species folded into other_bird keep their real name in classes_present.
            folded_ok = name == "other_bird" and non_pigeon_present
            if name not in record["classes_present"] and not folded_ok:
                problems.append(f"{where}: boxed class {name} missing from classes_present")
        split_counts = counts[record["split"]]
        split_counts["images"] += 1
        split_counts["backgrounds"] += 0 if boxes_by_class else 1
        for name, value in boxes_by_class.items():
            split_counts[f"box:{name}"] += value
        if check_files:
            image = dataset / record["file"]
            if not image.exists():
                problems.append(f"{where}: image missing")
            elif file_sha256(image) != record["sha256"]:
                problems.append(f"{where}: sha256 mismatch")
    crossing = [group for group, split_set in group_splits.items() if len(split_set) > 1]
    if crossing:
        problems.append(f"{len(crossing)} groups cross splits, e.g. {crossing[:3]}")
    for split in splits:
        expected = summary.get("counts", {}).get(split, {})
        actual = counts[split]
        if expected.get("images") != actual["images"]:
            problems.append(f"{split}: DATASET.json says {expected.get('images')} images")
        if expected.get("backgrounds") != actual["backgrounds"]:
            problems.append(f"{split}: DATASET.json backgrounds do not match labels")
        for name in names:
            if expected.get("boxes", {}).get(name, 0) != actual[f"box:{name}"]:
                problems.append(f"{split}: DATASET.json {name} box count does not match labels")
        on_disk = {path.name for path in (dataset / "images" / split).glob("*")}
        listed = {Path(record["file"]).name for record in records if record["split"] == split}
        if on_disk != listed:
            problems.append(f"{split}: {len(on_disk ^ listed)} images differ from the manifest")
    if problems:
        raise DatasetError("C1 validation failed:\n- " + "\n- ".join(problems))
    notes.append(f"{dataset.name}: {len(records)} images pass C1 validation")
    return notes


# --- archive and freeze ---------------------------------------------------------------------


def dataset_files(dataset: Path) -> list[Path]:
    return sorted(path for path in dataset.rglob("*") if path.is_file())


def write_archive(dataset: Path, output: Path) -> str:
    """Deterministic `<name>.tar.zst` (sorted entries, fixed owner and mtime). Returns sha256."""
    import zstandard

    temporary = output.with_name(output.name + ".part")
    with open(temporary, "wb") as raw:
        compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL, threads=4)
        with compressor.stream_writer(raw) as stream:
            with tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                entries = [dataset, *sorted(dataset.rglob("*"))]
                for path in entries:
                    name = Path(dataset.name, path.relative_to(dataset)).as_posix()
                    info = tarfile.TarInfo(name)
                    info.mtime = ARCHIVE_MTIME
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    if path.is_dir():
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                        tar.addfile(info)
                    else:
                        info.mode = 0o644
                        info.size = path.stat().st_size
                        with open(path, "rb") as handle:
                            tar.addfile(info, handle)
    temporary.replace(output)
    digest = file_sha256(output)
    output.with_name(output.name + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    return digest


def archive_dataset(dataset: Path) -> str:
    """Archive the dataset as built (archive_sha256 null inside), then record the archive hash
    in the local DATASET.json and in a `<name>.tar.zst.sha256` sidecar."""
    validate_dataset(dataset)
    summary_path = dataset / "DATASET.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("archive_sha256"):
        raise DatasetError(f"{dataset.name} is already archived ({summary['archive_sha256']})")
    digest = write_archive(dataset, dataset.parent / f"{dataset.name}.tar.zst")
    summary["archive_sha256"] = digest
    write_summary(dataset, summary)
    return digest


def freeze_lines(dataset: Path, archive_digest: str, archived_summary_sha256: str) -> list[str]:
    lines = []
    for path in dataset_files(dataset):
        relative = path.relative_to(dataset).as_posix()
        digest = archived_summary_sha256 if relative == "DATASET.json" else file_sha256(path)
        lines.append(f"{digest}  {dataset.name}/{relative}")
    lines.append(f"{archive_digest}  {dataset.name}.tar.zst")
    return lines


def archived_member_sha256(archive: Path, member: str) -> str:
    import zstandard

    with open(archive, "rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                for info in tar:
                    if info.name == member:
                        handle = tar.extractfile(info)
                        if handle is None:
                            break
                        return hashlib.sha256(handle.read()).hexdigest()
    raise DatasetError(f"{member} not found in {archive}")


def freeze(dataset: Path, output: Path) -> None:
    summary = json.loads((dataset / "DATASET.json").read_text(encoding="utf-8"))
    archive = dataset.parent / f"{dataset.name}.tar.zst"
    if not summary.get("archive_sha256") or file_sha256(archive) != summary["archive_sha256"]:
        raise DatasetError("archive the dataset first; archive hash must match DATASET.json")
    archived_summary = archived_member_sha256(archive, f"{dataset.name}/DATASET.json")
    header = [
        f"# Frozen {dataset.name}: SHA-256 of every file as archived, then of the archive.",
        "# DATASET.json is hashed as archived (archive_sha256: null); the unpacked Mac copy",
        "# additionally records archive_sha256. Any change to these files needs a new name.",
    ]
    lines = freeze_lines(dataset, summary["archive_sha256"], archived_summary)
    output.write_text("\n".join([*header, *lines]) + "\n", encoding="utf-8")


def verify_freeze(dataset: Path, freeze_file: Path) -> list[str]:
    """Return mismatches between a freeze file and the dataset on disk (DATASET.json skipped)."""
    problems = []
    for line in freeze_file.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split("  ", 1)
        path = dataset.parent / relative
        if relative.endswith("/DATASET.json"):
            continue
        if not path.exists():
            problems.append(f"missing {relative}")
        elif file_sha256(path) != digest:
            problems.append(f"changed {relative}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build a dataset from a spec")
    build.add_argument("spec", type=Path)
    validate = commands.add_parser("validate", help="check C1 invariants")
    validate.add_argument("dataset", type=Path)
    archive = commands.add_parser("archive", help="write <name>.tar.zst and its sha256 sidecar")
    archive.add_argument("dataset", type=Path)
    freeze_parser = commands.add_parser("freeze", help="write a SHA-256 list of every file")
    freeze_parser.add_argument("dataset", type=Path)
    freeze_parser.add_argument("output", type=Path)
    verify = commands.add_parser("verify-freeze", help="compare a dataset with its freeze file")
    verify.add_argument("dataset", type=Path)
    verify.add_argument("freeze_file", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command == "build":
            spec = load_spec(arguments.spec)
            builder = {"test": build_test, "train": build_train, "test_styles": build_test_styles}[
                spec["kind"]
            ]
            output = builder(spec, data_root())
            summary = json.loads((output / "DATASET.json").read_text(encoding="utf-8"))
            print(json.dumps({key: summary.get(key) for key in ("name", "counts")}, indent=1))
            if "composition" in summary:
                print(json.dumps(summary["composition"], indent=1))
            if "build_report" in summary:
                print(json.dumps(summary["build_report"], indent=1))
        elif arguments.command == "validate":
            print("\n".join(validate_dataset(arguments.dataset)))
        elif arguments.command == "archive":
            print(f"{archive_dataset(arguments.dataset)}  {arguments.dataset.name}.tar.zst")
        elif arguments.command == "freeze":
            freeze(arguments.dataset, arguments.output)
            print(f"wrote {arguments.output}")
        else:
            problems = verify_freeze(arguments.dataset, arguments.freeze_file)
            if problems:
                raise DatasetError("freeze check failed:\n- " + "\n- ".join(problems))
            print(f"{arguments.dataset.name} matches {arguments.freeze_file}")
    except DatasetError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
