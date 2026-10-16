# WS2 — Data

- State: done (phases 1 and 2 merged; phase 2 at 4cf06ca, coordinator 2026-10-16)  <!-- not started | in progress | blocked | done -->
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

`~/columba-data/` after phase 2: 14 GB (measured). Species datasets hard-link their images
to the single-taxonomy copies (identical bytes). The local copies of the large train archives
were deleted after `sha256sum -c` passed on the cluster; the sidecars stay, and
`build_dataset archive` rebuilds byte-identical archives (archives are deterministic).

## CLI for other workstreams

```
python -m tools.fetch_sources data_sources/<source>.yaml [--limit N] [--limit-per-taxon N] [--taxon NAME]
python -m tools.dedupe_images raw/<a>/provenance.jsonl ... [--max-distance 10] [--output groups.jsonl] [--clusters clusters.json]
python -m tools.build_dataset build data_sources/builds/<name>.yaml
python -m tools.build_dataset validate|archive <dataset dir>
python -m tools.build_dataset freeze|verify-freeze <dataset dir> data_sources/<name>.sha256
python -m tools.contact_sheet <dataset dir | provenance.jsonl> --output-dir DIR [--split S] [--ids FILE]
python -m tools.dataset_audit <dataset>/data.yaml      # also: python tools/dataset_audit.py
python -m tools.auto_label detect|relabel|sample|summarize ...   # pseudo labels and review gate
python -m tools.style_augment demo <dataset dir> --output-dir DIR   # before/after sheet
```

`$COLUMBA_DATA_ROOT` overrides `~/columba-data`.

## Phase 2 results

All numbers **measured** on the Mac (M2 Pro) on 2026-10-14 unless stated. Every archive
passed `sha256sum -c` on the cluster (`~/columba/data/`, 16 GB in total).

### Published datasets

| Dataset | Archive SHA-256 | Cluster path |
|---|---|---|
| `columba-train-v2` | `90e8d09e7ad455c521ac04bcf72f7c0237e79fbf904fa3d03529e76ca1d00e9b` | `~/columba/data/columba-train-v2.tar.zst` |
| `columba-train-v2s` | `d0604b39476ac7b8df3ea2957a39c5ab20be86b667ec23d71aa3321b025b3230` | `~/columba/data/columba-train-v2s.tar.zst` |
| `columba-test-v2-styles` (frozen, `data_sources/columba-test-v2-styles.sha256`) | `a05b6055960dcea45dabea9ee051f44cccf0e321f7d557f708095480edb4ccab` | `~/columba/data/columba-test-v2-styles.tar.zst` |
| `columba-train-v2-species` | `399ddf99360d59783f2390dd22b88d3fcdb3bb47b848e598d1c8fafb6e5ef2ce` | `~/columba/data/columba-train-v2-species.tar.zst` |
| `columba-train-v2s-species` | `a8e3f5e532307a578870d156fa8fbca3bfce43a9a3b7603922da9dfb10c94ae4` | `~/columba/data/columba-train-v2s-species.tar.zst` |

Each has a `.sha256` sidecar. All five pass `tools/dataset_audit.py` (C1 validation, licences
complete, no train/val image within phash 10 of a test image; closest is 14).

### columba-train-v2 (`single`) and columba-train-v2-species

Same images, groups and split (checked); only the labels differ.

| Split | Images | Pigeon images | Other-bird images | Empty images | Pigeon boxes (small at 384×640) | Backgrounds (`single`) |
|---|---|---|---|---|---|---|
| train | 7,958 | 2,936 (36.9%) | 3,116 (39.2%) | 1,906 (24.0%) | 5,687 (1,283 = 22.6%) | 5,022 |
| val | 1,089 | 490 (45.0%) | 218 (20.0%) | 381 (35.0%) | 1,143 (285 = 24.9%) | 599 |

Val follows the test-v2 kind mix (target 45/20/35, `DATASET.json` `composition` and `split`).
Pigeon groups go to val with fraction 0.15; NEA is train-only (stretched 416×416 images).

Species boxes (`columba-train-v2-species`), train / val: pigeon 5,687 / 1,143; spotted_dove
276 / 19; zebra_dove 277 / 19; pink_necked_green_pigeon 256 / 18; javan_myna 286 / 20;
house_crow 286 / 20; asian_glossy_starling 225 / 16; eurasian_tree_sparrow 244 / 17;
yellow_vented_bulbul 274 / 19; asian_koel 257 / 18; other_bird 1,010 / 58. Backgrounds
1,906 / 381. **No class folded** (`min_train_instances` 150; smallest is asian_glossy_starling, 225).

Train images by source: inat-species 2,381; inat-pigeon 2,076; coco-negatives-train 1,906;
roboflow-yolo1199 697; inat-other 382; coco-birds-train 357; roboflow-nea 149; coco-birds 10.
Licences: CC-BY-4.0 5,280; CC-BY-2.0 1,696; CC-BY-SA-2.0 955; CC0 561; CC-BY-SA-4.0 525;
PDM 30. `label_origin`: pseudo_reviewed 5,398, source 3,649.

### columba-train-v2s and columba-train-v2s-species

v2 plus one camera-style copy per train image (style drawn per image, weights noir_night 0.4,
noir_day 0.3, wide 0.3, seed 0, long side ≤ 1280 px). Val is v2's val, unchanged (no style val
subset). Copies share their original's group and `classes_present`; `derived_from` names the
original; a copy on a background canvas records it in `style_params.canvas`, its attribution
lists both, and a CC0/PDM original on a CC-BY canvas becomes CC-BY.

| Split | Originals | noir_night | noir_day | wide | Total | Pigeon boxes |
|---|---|---|---|---|---|---|
| train | 7,958 | 3,164 | 2,276 | 2,467 | 15,865 | 11,341 |
| val | 1,089 | 0 | 0 | 0 | 1,089 | 1,143 |

51 copies were dropped because their 64-bit phash landed within 12 of a test image (all were
coincidental: 256-bit distance ≥ 88, mostly dark noir_night images); 2 pigeon boxes left
the frame in `wide` copies. Species boxes in v2s-species are v2-species train counts ×≈2.

### columba-test-v2-styles (new frozen test set; test-v2 unchanged)

Every test-v2 image in each style: 957 images, 1,059 pigeon boxes, 594 backgrounds. Per style
the composition equals test-v2 (319 images, 39.2% empty, 22.9% other birds); small pigeon
boxes: noir_night and noir_day 60.1%, wide 69.7% (zoom-out). `wide` canvases are bird-free
test-v2 street scenes (never ShareAlike). Freeze file: `data_sources/columba-test-v2-styles.sha256`.

### Sources and reviews

| Source | Fetched | Used after review/policy | Label review (box error rate, measured) |
|---|---|---|---|
| inat-pigeon (Columba livia; SG 47, MY 119, ID 31, TH 240, rest of Asia 2,563) | 3,000 | 2,537 auto-labelled kept | 200 images, 493 boxes: 18 wrong, 2 missed → **4.0%** (pass) |
| inat-species (9 C2 species × 450; SG first, then MY/ID/TH/Asia) | 4,050 | 2,660 | 216 images, 216 boxes: 4 wrong, 0 missed → **1.9%** (pass) |
| inat-other (Aves minus C2 taxa, Singapore) | 600 | 423 | 200 images, 200 boxes: 4 wrong, 2 missed → **3.0%** (pass) |
| coco-birds-train (train2017 `bird`, licence-filtered) | 858 | 16 feral pigeon, 368 other_bird, 474 dropped | every image reviewed (human COCO boxes; verdict per image) |
| coco-birds (val2017, v1 source) | 38 | 2 pigeon, 27 other_bird, 9 dropped | every image reviewed |
| roboflow-nea | 450 | 154 | every image reviewed: 260 fox-only mosaics and 36 unusable pigeon images dropped |
| coco-negatives-train (train2017 street scenes, no `bird`/`kite`) | 2,300 | 2,298 | YOLO11x bird check on all (2 flagged, dropped) + 94-image visual sample (0 birds) |

Error rate = (wrong + missed) / (boxes + missed). Review files: `data_sources/review/*.review.jsonl`,
summaries `*.summary.json`, per-image verdicts `*.images.jsonl`. Reviewer: Claude Opus 5.5
(image-capable agent) on contact sheets; samples are seeded random draws of kept images.
Auto-label policy (YOLO11x COCO `bird`, imgsz 1024, MPS): keep boxes ≥ 0.25; drop the image if
no box, if an unmatched detection 0.15–0.25 exists, or (non-pigeon taxa) if more than one bird.
Labelling speed: 900 images in 133 s, peak RSS 2.4 GB (measured, Mac, shared with other workers).
Licences of all fetched iNaturalist photos: CC-BY-4.0, CC0 or CC-BY-SA-4.0 only.

### Dedupe change in phase 2

With ~14k images, 64-bit phash ≤ 10 linked unrelated iNaturalist close-ups (a bird on sky or
blurred foliage) and chained 848 images, including test images, into one group. v2 builds
now confirm each 64-bit link with a mirror-aware 256-bit phash ≤ 32 (true copies measured
≤ 30, false links ≥ 42; checked on a contact sheet). Test exclusion stays 64-bit ≤ 12 (the
conservative direction). test-v2 and v1clean were built before this and are unchanged (their
grouping is stricter, not looser).

### Contact sheets (phase 2)

- Style before/after: `~/columba-data/contact_sheets/columba-test-v2-styles/before-after-000.jpg`,
  `~/columba-data/contact_sheets/columba-train-v2s/before-after-000.jpg`; small copies committed in
  `data_sources/contact_sheets/test-v2-styles-before-after.jpg`, `train-v2s-before-after.jpg`
- Auto-label reviews: `~/columba-data/contact_sheets/review/auto-inat-{pigeon,species,other}/`
  (committed example `data_sources/contact_sheets/review-inat-pigeon-000.jpg`)
- COCO train birds: `~/columba-data/contact_sheets/review/coco-birds-train/` and
  `coco-train-pigeons/`; NEA: `review/nea/`; negatives: `review/coco-neg-train/`;
  dedupe calibration: `review/calib/`

## In progress

Nothing.

## Blockers and requests

Phase 1 items 1–7 below were resolved in CONTRACTS.md v1.4 (PDM now allowed and used in v2).

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

### Phase 2 decisions for the coordinator

1. **Background share in `single` train sets is 63%** (24.0% empty scenes + 39.2% other-bird
   images, which are unboxed in `single`). The empty share is inside the 20–30% target; the
   other-bird images are the species images the 5.4 gate wants as unlabelled negatives. If WS5
   sees recall suffer, subsample `inat-species` in a v3 rather than changing v2.
2. **Rock pigeon images are mostly not from Singapore**: only 47 licence-allowed research-grade
   observations exist there; 2,563 of 3,000 come from wider Asia.
3. **Val species counts are small** (16–20 boxes per C2 species), so per-species val AP is noisy.
4. Non-pigeon iNaturalist images with more than one bird were dropped (species could differ),
   so species classes have no flock examples; pigeon flocks were kept.
5. COCO `other_bird` boxes are not split by species even where a bird might be a C2 species.

## Phase 2 plan (as executed; leftovers)

Done as listed below. Not done: a style val subset (not needed; val stays original),
arm-lane labelling (the Mac was enough).

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
