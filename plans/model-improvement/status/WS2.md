# WS2 — Data

- State: in progress (phase 1 done and merged; phase 2 in progress)  <!-- not started | in progress | blocked | done -->
- Branch: `ws2/data`
- Agent: Claude Opus 5.5 via T3 delegate
- Last updated: 2026-10-14

## Done (phase 1)

1. Source definitions in `data_sources/` (no images): `roboflow-nea.yaml`,
   `roboflow-yolo1199.yaml` (with `scene_rules` for capture sessions found on contact
   sheets), `coco-birds.yaml`, `coco-negatives.yaml` (val2017), `coco-negatives-train.yaml`
   (train2017, phase 2), `inaturalist.yaml` (rock pigeon + every C2 species + `other_bird`,
   research grade, Singapore place 6734, CC0/CC BY/CC BY-SA only).
2. `tools/fetch_sources.py`: Roboflow REST export API (key read from
   `~/.config/columba/roboflow.env`, never printed or logged; export links are not
   logged), COCO 2017 by category rules with per-image licence filter, iNaturalist API v1
   at ≤ 1 request/s (photo downloads paced the same way), `large` photos. Writes
   `raw/<source>/provenance.jsonl` (licence, attribution, source id/URL, scene id,
   pixel boxes) and `licence_report.json`.
3. `tools/dedupe_images.py`: SHA-256 + 64-bit phash of the image **and of its mirror**
   (Roboflow sets contain flipped copies), band-indexed near-duplicate search, union-find
   groups that also join shared scene ids (Roboflow `<stem>_jpg.rf.<hash>` copies,
   capture sessions, iNaturalist observations). Group id = hash of the smallest member id.
4. `tools/build_dataset.py`: `build` (test from a reviewed selection, train split by group),
   `validate` (C1 invariants), `archive` (deterministic tar.zst + sidecar), `freeze`,
   `verify-freeze`. Train builds drop any image whose group contains a test image or whose
   phash/mirror is within `test_exclusion_distance` of a test image.
5. `tools/contact_sheet.py`: numbered review sheets from a C1 dataset or a provenance file
   (red = pigeon, yellow = small pigeon at 384×640, blue = other classes).
6. `tools/dataset_audit.py` (extended, CLI unchanged): when `DATASET.json` is present it
   runs the C1 validation, prints licence counts, accepts test-only datasets, and checks
   that no train/val image is byte-identical to or within `phash_max_distance` of a test image.
7. `tests/test_data_tools.py`: 14 tests (licence filter, COCO selection, Roboflow stems,
   rate limiter, banded search = brute force, grouping incl. mirrors and scenes, split by
   group, test and train builds with leakage checks, archive reproducibility and freeze,
   C1 rejection cases, audit). Synthetic images, no network.
8. **`columba-test-v2` built, reviewed, archived, frozen** (`data_sources/columba-test-v2.sha256`),
   uploaded and hash-verified on the cluster.
9. **`columba-train-v1clean` built, audited, archived**, uploaded and hash-verified.

Checks on the Mac (venv python): `unittest discover` 80 tests OK; ruff check and format
check clean on `app tests tools cluster training`; mypy clean (33 files).

## Results

All numbers **measured** on the Mac (M2 Pro) unless stated.

### columba-test-v2 (frozen)

| Item | Value |
|---|---|
| Path (Mac) | `~/columba-data/columba-test-v2/` |
| Cluster | `~/columba/data/columba-test-v2.tar.zst` (+ `.sha256`), `sha256sum -c` OK |
| Archive SHA-256 | `d1cf5ba4b430da5d762eccedee8c4cdce5ed48af842cd3eb4418719e7b20d21b` (65.5 MB) |
| Freeze file | `data_sources/columba-test-v2.sha256` (643 files + archive) |
| Taxonomy | `single` (`[pigeon]`), `classes_present` lists every bird (species names) |
| Images / pigeon boxes / backgrounds | 319 / 353 / 198 |
| Sources | roboflow-yolo1199 119, coco-negatives 125, inaturalist 58, coco-birds 17 |
| Licences | CC-BY-4.0 160, CC-BY-2.0 92, CC-BY-SA-2.0 50, CC0-1.0 13, CC-BY-SA-4.0 4 |

Composition against PLAN 5.1 targets:

| Target | Result | Met? |
|---|---|---|
| ≥ 300 images | 319 | yes |
| ≥ 30% empty scenes | 125 / 319 = **39.2%** (COCO val2017 street scenes, checked: no bird) | yes |
| ≥ 15% other birds / confusers | 73 / 319 = **22.9%** (iNaturalist 58: spotted dove 13, zebra dove 11, pink-necked green pigeon 7, Javan myna 7, house crow 5, tree sparrow 5, bulbul 3, starling 2, koel 2, other 3; COCO birds checked as non-pigeon 15) | yes |
| ≥ 30% of pigeon boxes small (short side < 32 px at 384×640) | 212 / 353 = **60.1%** | yes, **but see caveat** |

Caveat on small boxes: 196 of the 212 small boxes come from 19 images of **one capture
session** (an industrial depot, phone photos and video frames: `WhatsApp-*` in
roboflow-yolo1199; one group). Without it, 16 / 157 = 10% of boxes are small. The small-object
recall on this set is therefore largely a measure of that one site. Human-labelled
sources have few other genuinely small pigeons; nothing was rescaled to manufacture
small boxes. Zoomed-out copies belong in `columba-test-v2-styles` (phase 2).

How it was made: every image was looked at on contact sheets (by Claude Opus 5.5, an
image-capable agent; decision per image in `data_sources/columba-test-v2.selection.jsonl`).
Pigeon positives come from **human-labelled sources only**: roboflow-yolo1199 human boxes
(119 images) and two COCO val2017 images whose bird boxes I confirmed are all feral
pigeons. Left out: images with missing, misplaced or multi-bird boxes, unboxed
background/edge pigeons, birds whose species I could not confirm, Roboflow mosaics and
noise-augmented copies, and all NEA images (stretched to 416×416, heavily augmented).
`label_origin` is `source` for source boxes and `human` for the reviewed iNaturalist
images (image-level "no pigeon" decision, no boxes).

### columba-train-v1clean

| Item | Value |
|---|---|
| Path (Mac) | `~/columba-data/columba-train-v1clean/` |
| Cluster | `~/columba/data/columba-train-v1clean.tar.zst` (+ `.sha256`), `sha256sum -c` OK |
| Archive SHA-256 | `b18b97d077faff2909870b22588448401540be80b0ebd561174e0d95ab04a680` (165.8 MB) |
| train | 868 images, 1639 pigeon boxes, 0 backgrounds |
| val | 156 images, 280 pigeon boxes, 0 backgrounds |
| `data.yaml` | `train: images/train`, `val: images/val`, `test: ../columba-test-v2/images/test`, no `path:` (Ultralytics `check_det_dataset` resolves all three, checked) |
| Licences | CC-BY-4.0 1004, CC-BY-2.0 14, CC-BY-SA-2.0 6 |
| Closest train/val image to any test image | phash distance 14 (> exclusion distance 12 > grouping distance 10) |

Build report (`DATASET.json` `build_report`):

| Source | Fetched | Dropped | Kept train / val |
|---|---|---|---|
| roboflow-nea | 450 | 260 without pigeon box (v1 rule), 1 test group, 4 near test | 157 / 28 |
| roboflow-yolo1199 | 1199 | 1 without box, 82 byte-identical copies, 285 in test groups, 12 near test | 694 / 125 |
| coco-birds | 38 (licence-allowed) | 17 in test groups (all 17 COCO test birds), 1 near test | 17 / 3 |

Convention kept for round 0: all COCO `bird` boxes are labelled `pigeon` (v1 notebook
section 4) although most are other species; recorded in `DATASET.json`
`label_conventions`. Splits are new (v1's split is not reproducible and leaked: 92 of 176
NEA originals and 118 of 789 yolo1199 originals had copies in more than one source split).

### Licence filter (COCO per-image licences)

- **v1 COCO birds: 125 val2017 images → 38 kept** (CC-BY-2.0 24, CC-BY-SA-2.0 14). Lost
  87: CC-BY-NC-SA 35, CC-BY-NC-ND 27, CC-BY-NC 18, CC-BY-ND 7.
- COCO val2017 street scenes: 282 kept of 1,133 candidates (NC/ND 850, "no known copyright
  restrictions" 1 dropped).
- iNaturalist: 330 photos (30 per taxon), all CC BY 4.0 (251), CC0 (63) or CC BY-SA 4.0
  (16), counted from `provenance.jsonl`. The API query already asks for these licences, and
  each photo's `license_code` is checked again. (`licence_report.json` only covers the
  last of two fetch runs.)
- Roboflow sets: both state CC BY 4.0 (Roboflow project API `license` field, 2026-10-14).

### Dedupe findings

- Roboflow `rf.<hash>` copies of one stem are often *different* images (generic names like
  `001` reused); grouping by stem is conservative. Cross-source copies exist (14 groups
  join NEA and yolo1199 images).
- Grouping uses phash distance ≤ 10 with mirror matching; test exclusion uses ≤ 12.
  Chosen from pair contact sheets (`outputs/ws2/sheets/`, scratch): pairs ≤ 10 were
  copies or crops of one photo; 12–14 starts to join different images of one scene type.
- Capture sessions that phash misses (video frames, one loft, one depot) are joined by
  `scene_rules` in `data_sources/roboflow-yolo1199.yaml`.

### Old-test proxy for WS8 parity

The original 63-image v1 test split cannot be rebuilt exactly. Proxies (Roboflow exports,
their own `test` splits, YOLO format):

- `~/columba-data/raw/roboflow-yolo1199/export/test/images` (63 images, the likely origin
  of the old 63) and `.../export/test/labels`
- `~/columba-data/raw/roboflow-nea/export/test/images` (44 images; class 0 = Fox, 1 = pigeon
  in that export's `data.yaml`)

### Contact sheets

- Test set, all 319 images: `~/columba-data/contact_sheets/columba-test-v2/test-v2-000.jpg` … `-007.jpg`
- v1clean val split: `~/columba-data/contact_sheets/columba-train-v1clean/v1clean-val-000.jpg` … `-003.jpg`
- Review sheets used for selection: `~/columba-data/contact_sheets/review/` (`test-v2-final/`,
  `coco-birds/`, `coco-neg/`, `inat/`, `fam-*`, `yolo-*`)
- Committed small copies: `data_sources/contact_sheets/` (test-v2 page 1, the depot
  small-box sheet, v1clean val page 1; 0.5 MB)

### Disk

`~/columba-data/` is 1.5 GB (measured), well under the 40 GB phase-1 limit.

## CLI for other workstreams

```
python -m tools.fetch_sources data_sources/<source>.yaml [--limit N] [--limit-per-taxon N] [--taxon NAME]
python -m tools.dedupe_images raw/<a>/provenance.jsonl ... [--max-distance 10] [--output groups.jsonl] [--clusters clusters.json]
python -m tools.build_dataset build data_sources/builds/<name>.yaml
python -m tools.build_dataset validate|archive <dataset dir>
python -m tools.build_dataset freeze|verify-freeze <dataset dir> data_sources/<name>.sha256
python -m tools.contact_sheet <dataset dir | provenance.jsonl> --output-dir DIR [--split S] [--ids FILE]
python -m tools.dataset_audit <dataset>/data.yaml      # also: python tools/dataset_audit.py
```

`$COLUMBA_DATA_ROOT` overrides `~/columba-data`.

## Phase 2 published (details in the final phase 2 results below as they land)

| Dataset | Published | Archive SHA-256 | Cluster |
|---|---|---|---|
| `columba-test-v2-styles` | 2026-10-14 | `a05b6055960dcea45dabea9ee051f44cccf0e321f7d557f708095480edb4ccab` | `~/columba/data/columba-test-v2-styles.tar.zst` |
| `columba-train-v2` | 2026-10-14 | `90e8d09e7ad455c521ac04bcf72f7c0237e79fbf904fa3d03529e76ca1d00e9b` | `~/columba/data/columba-train-v2.tar.zst` |
| `columba-train-v2s` | 2026-10-14 | `d0604b39476ac7b8df3ea2957a39c5ab20be86b667ec23d71aa3321b025b3230` | `~/columba/data/columba-train-v2s.tar.zst` |

## In progress

Phase 2 (started 2026-10-14): iNaturalist and COCO train2017 fetches, auto-labelling, style copies, `columba-train-v2`, `-v2s`, `columba-test-v2-styles`, species variants.

## Blockers and requests

For the coordinator (C1 clarifications; no blocker):

1. **`archive_sha256`** cannot be inside its own archive. Implemented: the archived
   `DATASET.json` has `archive_sha256: null`; the hash is in the sidecar
   `<name>.tar.zst.sha256` and in the Mac copy's `DATASET.json` (written after archiving).
   Consumers (WS3 `run.json`) should read the sidecar. Please record this in C1.
2. **`data.yaml` has no `path:` key.** Ultralytics then resolves paths from the yaml's own
   folder (a relative `path: .` would resolve against the current directory). `test:` is
   `../columba-test-v2/images/test` relative to the yaml folder. Please make C1 say
   "relative to the folder containing data.yaml".
3. Test `data.yaml` has only `test:` (no `train`/`val`), so Ultralytics `val` on it needs
   a wrapper; WS1 uses its own loop, so nothing breaks.
4. `classes_present` follows C4 v1.1: every bird in the image, by species-taxonomy name,
   `other_bird` when unknown; `[]` = empty scene. For v1clean COCO images it holds
   `pigeon` (the v1 relabel) plus `other_bird`.
5. `label_origin: human` is used for images whose only label is the reviewer's image-level
   "no pigeon" decision (iNaturalist other-bird test images). Say if you prefer another value.
6. DATASET.json carries extra keys beyond C1: `licenses`, `label_origins`, `notes`,
   `review`/`composition` (test) or `build_report`/`label_conventions` (train), and
   `dedupe.mirror_aware`, `dedupe.test_exclusion_distance`.
7. COCO licence ids 7 ("No known copyright restrictions") and 8 ("US Government Work") are
   excluded because they are not in the allowed list; only 1 candidate image was affected.

## Phase 2 plan

- `tools/auto_label.py`: YOLO11x COCO `bird` boxes on iNaturalist (Singapore first, more
  photos per taxon, other places if Singapore runs short), species from the taxon,
  `pseudo_unreviewed` → 200-image contact-sheet review per source → `pseudo_reviewed`.
  Large batches on the arm lane via WS3.
- Training negatives from `coco-negatives-train` (train2017), plus NEA fox-only images if
  they look like urban scenes.
- More small pigeons for training (and a future `columba-test-v3`): iNaturalist flock photos
  after review; zoom-out style copies.
- `tools/style_augment.py` (`noir_night`, `noir_day`, `wide` with box remapping),
  `columba-test-v2-styles` from test images only.
- `columba-train-v2` and `-v2s` in `single` and `species` taxonomies, all excluding
  `columba-test-v2` near-duplicates and capture sessions.
