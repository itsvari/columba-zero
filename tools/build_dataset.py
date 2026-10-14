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
    if spec.get("kind") not in ("test", "train"):
        raise DatasetError(f"{path}: kind must be test or train")
    spec["sources"] = [
        source if isinstance(source, dict) else {"name": source} for source in spec["sources"]
    ]
    return cast(Record, spec)


def git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "tools", "data_sources", "training"],
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
    """Assign whole groups to train or val. Per stratum (the source of the group's first image),
    groups are shuffled with `seed` and moved to val until it holds `val_fraction` of the images."""
    members: dict[str, list[Record]] = defaultdict(list)
    for item in items:
        members[item["group"]].append(item)
    strata: dict[str, list[str]] = defaultdict(list)
    for group, records in sorted(members.items()):
        first = min(records, key=lambda record: record["id"])
        strata[str(first[stratify_key])].append(group)
    assignment: dict[str, str] = {}
    for stratum, groups in sorted(strata.items()):
        rng = random.Random(f"{seed}:{stratum}")
        rng.shuffle(groups)
        total = sum(len(members[group]) for group in groups)
        in_val = 0
        for group in groups:
            if in_val < val_fraction * total:
                assignment[group] = "val"
                in_val += len(members[group])
            else:
                assignment[group] = "train"
    return assignment


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
        "style": "original",
        "derived_from": None,
        "label_origin": item["label_origin"],
        "classes_present": sorted(set(classes_present)),
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
    splits = TEST_SPLITS if spec["kind"] == "test" else TRAIN_SPLITS
    for split in splits:
        (staging / "images" / split).mkdir(parents=True)
        (staging / "labels" / split).mkdir(parents=True)
        counts[split] = {"images": 0, "boxes": {name: 0 for name in names}, "backgrounds": 0}
    for item, split, boxes, classes_present in sorted(entries, key=lambda entry: entry[0]["id"]):
        source_path = Path(item["path"])
        if exif_orientation(source_path) != 1:
            raise DatasetError(f"{item['id']}: EXIF orientation is not 1; re-encode it first")
        file_name = item["id"] + source_path.suffix.lower().replace(".jpeg", ".jpg")
        shutil.copyfile(source_path, staging / "images" / split / file_name)
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
    if spec["kind"] == "test":
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
            "phash_max_distance": spec["phash_max_distance"],
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


def build_train(spec: Record, root: Path) -> Path:
    taxonomy = load_taxonomy()
    names = taxonomy_names(taxonomy, spec["taxonomy"])
    test_dir = root / spec["test_dataset"]
    test_records = load_dataset_manifest(test_dir)
    test_hashes = {
        record["id"]: perceptual_hashes(test_dir / record["file"]) for record in test_records
    }
    test_sha = {record["sha256"] for record in test_records}
    candidates: list[Record] = []
    report: Record = {"per_source": {}}
    for source in spec["sources"]:
        items = load_source_items(root, source["name"])
        stats: Counter[str] = Counter(fetched=len(items))
        class_map = source.get("class_map", {})
        for item in items:
            boxes = mapped_boxes(item, class_map)
            boxes = [box for box in boxes if box["class"] in names]
            if source.get("require_boxes") and not boxes:
                stats["dropped_without_boxes"] += 1
                continue
            if item["label_origin"] not in PUBLISHABLE_LABEL_ORIGINS:
                stats["dropped_unpublishable_labels"] += 1
                continue
            present = set(item.get("classes_present") or [])
            present = {source.get("classes_present_map", {}).get(name, name) for name in present}
            present.update(box["class"] for box in boxes)
            item["_boxes"], item["_present"] = boxes, sorted(present)
            candidates.append(item)
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
            "scene_id": test_scene.get(record["id"]),
        }
        for record in test_records
    ]
    pool = [
        {key: item.get(key) for key in ("id", "sha256", "phash", "phash_mirror", "scene_id")}
        for item in unique
    ]
    groups, _ = assign_groups([*pool, *test_items], spec["phash_max_distance"])
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
    assignment = split_groups(kept, spec["val_fraction"], spec["seed"])
    entries = [(item, assignment[item["group"]], item["_boxes"], item["_present"]) for item in kept]
    for item in kept:
        report["per_source"][item["source"]][f"kept_{assignment[item['group']]}"] += 1
    min_distance = min((nearest[item["id"]][0] for item in kept), default=None)
    extra = {
        "build_report": {
            "per_source": {name: dict(stats) for name, stats in report["per_source"].items()},
            "min_phash_distance_to_test": min_distance,
            "val_fraction": spec["val_fraction"],
            "seed": spec["seed"],
        },
        "label_conventions": spec.get("label_conventions", ""),
    }
    return write_dataset(root / spec["name"], spec, entries, names, extra)


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
        for name in boxes_by_class:
            if name not in record["classes_present"]:
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
            builder = build_test if spec["kind"] == "test" else build_train
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
