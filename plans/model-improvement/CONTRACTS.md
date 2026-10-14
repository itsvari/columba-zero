# Shared contracts

Version: **v1.4**. Changes go through the coordinator and bump the version.

Changelog:

- v1.4 (2026-10-14, coordinator): C1 clarifications from WS2: archive hash in a
  sidecar, `data.yaml` paths relative to its folder, extra DATASET.json keys,
  image-level `human` labels, public-domain licences allowed.
- v1.3 (2026-10-14, coordinator): C5 gains WS3's recipe fields
  `seed_gpu_constraints`, `test_dataset`, `allow_missing_evaluation`.
- v1.2 (2026-10-14, coordinator): C7 accepts WS4's optional extensions.
- v1.1 (2026-10-14, coordinator): C4 gains exact evaluator semantics from WS1
  (AP method, matching, ties, threshold sentinel, small recall, undefined rates,
  subsets). C3 meta gains optional `ncnn_compute`.

- v1 (2026-10-14, WS0): C2 species list checked against the Singapore checklist,
  with taxon ids; `green_pigeon` renamed `pink_necked_green_pigeon`. C5 records
  the verified distillation arguments. C8 records the verified NCNN layouts:
  YOLO26 exports are *not* end-to-end by default. C1, C3, C5 and C6 gain notes
  on staging, `max_det`, model paths and `results.csv` columns.
- v0: draft.

Paths:

- `$COLUMBA_DATA_ROOT`: dataset root. Mac default `~/columba-data/`;
  cluster `~/columba/data/` (archives) staged to `/tmp/cz-$SLURM_JOB_ID/data/`.
- `$COLUMBA_RUNS_ROOT`: cluster `~/columba/runs/`; Mac copy
  `experiments/runs/` (git-ignored).

## C1 — Dataset

A dataset is an immutable directory `$COLUMBA_DATA_ROOT/<name>/`. Any change
means a new name (`columba-train-v2` → `columba-train-v3`).

```
<name>/
  DATASET.json        summary (below)
  data.yaml           Ultralytics format; names must equal the taxonomy's class list
  manifest.jsonl      one line per image (below)
  images/{train,val}/ ...         training datasets
  labels/{train,val}/ ...         YOLO txt; empty file = background image
  images/test/, labels/test/      test datasets only
```

Training datasets contain only `train` and `val`. Their `data.yaml` has
`test:` pointing at the frozen test dataset (`../columba-test-v2/images/test`,
relative to `path:`), so the test dataset is always staged next to the training
dataset. Test datasets contain only `test`.

Clarifications (v1.4):

- `data.yaml` has no `path:` key; Ultralytics resolves `train`/`val`/`test`
  relative to the folder containing `data.yaml`.
- The archived `DATASET.json` has `archive_sha256: null` (a file can't contain
  its own archive's hash). The hash lives in the sidecar
  `<name>.tar.zst.sha256` (`sha256sum -c` format) and in the unpacked Mac copy's
  `DATASET.json`.
- `DATASET.json` may carry extra keys (`licenses`, `composition`,
  `build_report`, ...); consumers ignore unknown keys.
- `label_origin: human` also covers image-level decisions made by a reviewer
  (e.g. "contains no pigeon" on an unboxed confuser image).
- Allowed licences also include public-domain equivalents ("No known copyright
  restrictions", "US Government Work"), recorded as `PDM`.

`manifest.jsonl` line:

```json
{
  "id": "inat-123456789-0",
  "split": "train",
  "file": "images/train/inat-123456789-0.jpg",
  "source": "inaturalist",
  "source_id": "123456789",
  "group": "g-000123",
  "license": "CC-BY-4.0",
  "attribution": "observer name, URL",
  "sha256": "...",
  "phash": "e3c1...",
  "width": 1024,
  "height": 768,
  "style": "original",
  "derived_from": null,
  "label_origin": "pseudo_reviewed",
  "classes_present": ["pigeon"]
}
```

- `style`: `original` | `noir_night` | `noir_day` | `wide`
- `label_origin`: `human` | `source` | `pseudo_reviewed` | `pseudo_unreviewed`
  (`pseudo_unreviewed` is never allowed in a published dataset)
- `derived_from`: the `id` of the original image, for style copies
- `group`: images in the same group always share a split

`DATASET.json`:

```json
{
  "name": "columba-train-v2s",
  "taxonomy": "single",
  "taxonomy_version": 1,
  "created_at": "2026-10-20T12:00:00Z",
  "builder_commit": "<git sha>",
  "sources": ["roboflow-yolo1199", "roboflow-nea", "inaturalist", "coco-negatives"],
  "counts": {"train": {"images": 0, "boxes": {"pigeon": 0}, "backgrounds": 0}, "val": {}},
  "dedupe": {"phash_max_distance": 6, "test_dataset": "columba-test-v2"},
  "archive_sha256": "<sha256 of <name>.tar.zst>"
}
```

## C2 — Taxonomy (`training/classes.yaml`)

```yaml
version: 1
trigger_classes: [pigeon]          # only these can confirm a trigger
taxonomies:
  single:
    names: [pigeon]                # other birds appear as unlabelled background
  species:
    names:                         # index 0 must be pigeon
      - pigeon                     # Columba livia, feral/rock pigeon
      - spotted_dove               # confuser (native)
      - zebra_dove                 # confuser (native)
      - pink_necked_green_pigeon   # confuser (native)
      - javan_myna
      - house_crow
      - asian_glossy_starling
      - eurasian_tree_sparrow
      - yellow_vented_bulbul
      - asian_koel
      - other_bird                 # catch-all, always last
    min_train_instances: 150       # rarer species fold into other_bird
classes:                           # one entry per species name, same order
  pigeon: {common_name: ..., scientific_name: Columba livia, inat_taxon_id: 3017,
           sg_status: ..., role: trigger}   # role: trigger | confuser | other
```

Verified against the Bird Society of Singapore Records Committee checklist
(AviList taxonomy, 432 species), <https://singaporebirds.com/checklist>, data
from <https://singaporebirds.com/api/species>, accessed 2026-10-14. Every
named species is listed as a "very common" resident (Rock Dove, House Crow and
Javan Myna as "very common introduced resident"). `green_pigeon` became
`pink_necked_green_pigeon` because Thick-billed Green Pigeon is also resident.
Pied Imperial Pigeon (common introduced resident) and Red Collared Dove
(uncommon) are not named and fall into `other_bird`.

| name | scientific name | iNaturalist taxon |
|---|---|---|
| pigeon | *Columba livia* | 3017 |
| spotted_dove | *Spilopelia chinensis* | 1455918 |
| zebra_dove | *Geopelia striata* | 3607 |
| pink_necked_green_pigeon | *Treron vernans* | 3393 |
| javan_myna | *Acridotheres javanicus* | 145095 |
| house_crow | *Corvus splendens* | 8031 |
| asian_glossy_starling | *Aplonis panayensis* | 14965 |
| eurasian_tree_sparrow | *Passer montanus* | 13851 |
| yellow_vented_bulbul | *Pycnonotus goiavier* | 14623 |
| asian_koel | *Eudynamys scolopaceus* | 204510 |
| other_bird | Aves | 3 |

iNaturalist's place id for Singapore is 6734. `tests/test_taxonomy.py` checks
the file's invariants.

## C3 — Predictions

COCO results format, written by any model or framework:

`<run>/predictions_<split>.json`: a list of

```json
{"image_id": "inat-123456789-0", "category_id": 0, "bbox": [x, y, w, h], "score": 0.83}
```

- `image_id` is the C1 manifest `id`. `category_id` is the taxonomy index.
- `bbox` is in **original image pixels**, `[left, top, width, height]`.
- Confidence floor 0.001, at most 300 boxes per image (Ultralytics `max_det`).
  Duplicate-box removal (NMS) as the deployed runtime would do it (IoU 0.7
  for `decode: yolo11`, none for `decode: yolo26`; see C8).

Sidecar `<run>/predictions_<split>.meta.json`:

```json
{"model_id": "...", "framework": "ultralytics", "format": "pt|ncnn-fp32|ncnn-fp16|ncnn-int8|onnx",
 "input_hw": [384, 640], "conf_floor": 0.001, "nms_iou": 0.7, "agnostic_nms": false,
 "host": "xcng0", "dataset": "columba-test-v2", "created_at": "..."}
```

Optional `ncnn_compute`: `runtime_default` (ncnn's fp16 packed/storage/arith
defaults) or `fp32`. `format` describes the package's weight quantisation, not
compute precision.

## C4 — Metrics

Written by `tools/evaluate_predictions.py` as `<run>/metrics_<split>.json`:

```json
{
  "dataset": "columba-test-v2",
  "predictions": "predictions_test.json",
  "input_hw": [384, 640],
  "pigeon": {
    "ap50": 0.0, "ap50_95": 0.0,
    "recall_at_p90": 0.0,
    "small_recall_at_p90": 0.0,
    "precision_at_threshold": 0.0, "recall_at_threshold": 0.0
  },
  "operating_threshold": {"value": 0.42, "source": "metrics_val.json"},
  "empty_false_alarm_rate": 0.0,
  "confuser_false_alarm_rate": 0.0,
  "per_class_ap50": {"spotted_dove": 0.0},
  "by_style": {"original": {"pigeon_recall_at_p90": 0.0}},
  "by_source": {"inaturalist": {"pigeon_recall_at_p90": 0.0}},
  "counts": {"images": 0, "empty_images": 0, "confuser_images": 0, "pigeon_boxes": 0, "small_pigeon_boxes": 0}
}
```

Definitions:

- Matching: one-to-one, IoU ≥ 0.5, per class, highest score first (COCO style).
- `recall_at_p90`: the highest pigeon recall at any threshold where precision ≥ 0.90; 0 if never reached.
- **Small** box: short side < 32 px after letterboxing to `input_hw`.
- `empty_false_alarm_rate`: share of images with no birds that get ≥ 1 pigeon prediction at the operating threshold.
- `confuser_false_alarm_rate`: same, over images containing only non-pigeon birds.
- `operating_threshold` is always taken from the validation split (`--select-threshold`). Never chosen on test.

Exact semantics (v1.1, implemented by `tools/evaluate_predictions.py`):

- AP uses pycocotools' 101-point interpolated precision, IoUs 0.50 through 0.95
  in steps of 0.05, area=all and maxDets=300. A class with no ground-truth boxes
  has null AP; a class with ground truths and no predictions has AP=0.
- For operating metrics, predictions are matched per image/class in descending
  score order to the highest-IoU unmatched ground truth with IoU >= 0.5.
  Equal-score predictions use prediction-file order within an image. A
  threshold includes all predictions with score >= threshold, including ties.
- Validation selects the lowest observed pigeon score at which precision is
  >= 0.90. With no eligible score, it writes `operating_threshold.value =
  1.0000000000000002`, `source = selected`, `attainable = false`, `split = val`.
  This sentinel produces no detections when reused on test. Precision with no
  detections is 0. Recall with no ground truths is 0.
- `small_recall_at_p90` is the highest recall of small pigeon ground truths at
  any threshold where precision over all pigeon predictions is >= 0.90, using
  the same matches as overall recall. Small means `min(box_w, box_h) *
  min(input_h / image_h, input_w / image_w) < 32` px. With no small pigeon
  boxes, small recall is 0.
- Empty/confuser membership uses manifest `classes_present`, including
  unlabelled other birds in single-taxonomy datasets, so the manifest must list
  every bird species (or `other_bird`) present even when it is not boxed. A
  false-alarm rate with no eligible images is null. All non-pigeon bird
  classes count as confusers for this metric.
- `by_style`/`by_source` recompute each subset's PR curve and AP and use the
  shared validation operating threshold for precision, recall and false alarms.

## C5 — Recipe

`training/recipes/<recipe_id>.yaml`:

```yaml
id: yolo26n-v2s-640-distill
framework: ultralytics        # ultralytics | dfine | rfdetr | nanodet
base_model: yolo26n.pt
dataset: columba-train-v2s
taxonomy: single
seeds: [0, 1, 2]
lane: nv                      # nv | arm
gpu_constraint: null          # null | t4 | titanrtx | titanv
est_gpu_hours: 3
teacher: null                 # or "<recipe_id>/seed<k>"
train:                        # passed through to YOLO.train()
  epochs: 300
  patience: 50
  imgsz: 640
  nbs: 64
  cos_lr: true
  close_mosaic: 15
  cache: ram
deploy_sizes: [[352, 608], [384, 640], [448, 800]]
notes: ""
```

Optional fields (v1.3, validated by `cluster/recipe.py`; see
`training/recipes/schema.md`):

```yaml
seed_gpu_constraints: {0: t4, 1: titanrtx, 2: titanv}  # per-seed override of gpu_constraint
test_dataset: columba-test-v2       # default
allow_missing_evaluation: false     # true only for smoke-* recipes
```

`run_job.py` sets `batch` from the GPU's memory class; `seed`, `project`,
`name`, `data` and `distill_model` are set by the runner, not the recipe.
`train.imgsz` must be an integer (training is square; the distillation set-up
also assumes it).

Runner notes (verified with Ultralytics 8.4.175 on the Mac, WS0):

- Pass `base_model` and teacher weights as **absolute paths**. A bare name such
  as `yolo26s.pt` is downloaded into the current directory, not `weights_dir`.
- Ultralytics reads settings from `$YOLO_CONFIG_DIR/Ultralytics/settings.json`
  (note the `Ultralytics/` subdirectory). Without one, `datasets_dir` defaults to
  the parent of the git root plus `/datasets`. Set `datasets_dir`, `weights_dir`
  and `runs_dir` per job, and `sync: false`.

### Knowledge distillation (verified, Ultralytics 8.4.175)

Two `train` arguments (`ultralytics/cfg/default.yaml`):

| Argument | Default | Meaning |
|---|---|---|
| `distill_model` | `None` | Path to the teacher `.pt`. Set → the trainer wraps the student in `ultralytics.nn.distill_model.DistillationModel`. `None` → no distillation. |
| `dis` | `6.0` | Weight of the distillation loss term. May be set in the recipe's `train:` block. |

The runner resolves the recipe's `teacher: "<recipe_id>/seed<k>"` to the
absolute path of `$COLUMBA_RUNS_ROOT/<recipe_id>/seed<k>/weights/best.pt`
(staged to the node) and passes it as `distill_model`.

What it does (read from the installed source, then exercised):

- **Feature distillation**, not box or logit distillation. Forward hooks
  capture the Detect head's three input feature maps (P3/P4/P5 neck outputs:
  layers 16, 19, 22 in both YOLO26n and YOLO11n) and the head output. Each student
  map passes through a learnt projector (1×1 conv, ReLU, 1×1 conv) to the
  teacher's channel count. The loss per level is an MSE weighted by the
  teacher's per-anchor confidence (sigmoid, max over classes, averaged over its
  one-to-many and one-to-one heads where present), summed over the 3 levels,
  times `dis`, times batch size.
- The teacher is frozen (eval mode, no gradients) and sees the **student's**
  augmented batch at the student's `imgsz`. A teacher trained at 960 therefore
  runs at 640 during distillation.
- `compile` is forced off when distilling.
- Checkpoints (`last.pt`, `best.pt`) contain only the student: they load as a
  plain `DetectionModel` and export as usual. `train_args` keeps `distill_model`
  and `dis`. On resume, the teacher is rebuilt from the `distill_model` path,
  so that path must exist again on the resuming node.
- `results.csv` gains `train/dis_loss` and `val/dis_loss` columns.
  `val/dis_loss` is always 0 by design.

Constraints:

- Teacher and student need the same number of Detect input levels with the
  same strides (8/16/32). All YOLO11 and YOLO26 detect models qualify, and the
  families can be mixed (YOLO11 teacher → YOLO26 student was run).
- Channel widths may differ (the projector maps them). Class counts may differ
  (teacher scores are reduced by a max over classes). Run with a COCO-80
  teacher and a 1-class student. A teacher trained on the same dataset and
  taxonomy is still the sensible choice.
- Cost: one teacher forward per batch. Not yet measured on a GPU (WS3/WS5
  should record epoch time with and without a teacher).

Evidence (Mac M2 Pro, CPU, coco8, `imgsz=320`; scripts in
`outputs/ws0/scripts/`, not committed):

- **Measured:** 2-epoch runs with teachers `yolo11n.pt` and `yolo26s.pt`
  (student `yolo26n.pt`), and with `dis=1.0, single_cls=True`, completed. They
  logged `train/dis_loss` ≈ 11.6 / 9.1 / 1.9 in epoch 1; the no-teacher
  baseline has no such column.
- **Measured:** same seed and teacher, `dis=0` vs `dis=6` (`nbs=2` so the
  optimiser steps every batch): final student weights differ (max abs
  difference 0.19). Back-propagating only the distillation term gives
  non-zero gradients in the student (Σ|g| ≈ 3.1e4) and the projector, and zero in
  the teacher.

## C6 — Run artifacts and leaderboard

`$COLUMBA_RUNS_ROOT/<recipe_id>/seed<k>/`:

```
run.json              recipe sha256, git commit, dataset name + archive sha256, host, GPU name,
                      compute capability, torch/ultralytics/CUDA versions, start/end, epochs done, exit status
weights/best.pt, weights/last.pt
results.csv
predictions_val.json (+ .meta.json), predictions_test.json (+ .meta.json)
metrics_val.json, metrics_test.json
DONE                  written last; its presence means everything above is complete
```

`results.csv` columns depend on the model and options: YOLO26 logs
`train/l1_loss` where YOLO11 logs `train/dfl_loss`, and distillation adds
`dis_loss` columns. Read columns by name.

`experiments/leaderboard.csv` (committed; one row per run, failures included):

```
recipe_id,seed,status,dataset,taxonomy,model,train_imgsz,host,gpu,epochs_done,
val_recall_at_p90,test_recall_at_p90,test_small_recall_at_p90,test_ap50,test_ap50_95,
test_empty_far,test_confuser_far,operating_threshold,git_commit,run_path
```

## C7 — Latency benchmark

`experiments/latency/<host>-<model_id>-<format>-<h>x<w>.json`:

```json
{
  "model_id": "yolo26n-v2s-640-distill/seed0",
  "format": "ncnn-int8",
  "input_hw": [384, 640],
  "host": "xcng0",
  "host_kind": "mac-a76compat | xcng-n1 | pi5",
  "threads": 4, "pinned_cores": "0-3",
  "ncnn_version": "1.0.20260526", "ncnn_build_flags": "...",
  "warmup": 20, "iterations": 300,
  "inference_ms": {"p50": 0.0, "p95": 0.0, "mean": 0.0, "std": 0.0},
  "preprocess_ms": {"p50": 0.0}, "postprocess_ms": {"p50": 0.0},
  "end_to_end_ms": {"p50": 0.0, "p95": 0.0},
  "estimated_pi5": {"end_to_end_p95_ms": 0.0, "calibration_id": "cal-v0-published"},
  "label": "measured | estimated"
}
```

Optional C7 fields (v1.2, written by `tools/bench_ncnn.py`): `samples_ms`
(paired end-to-end samples, used by `tools/reaction_budget.py`), camera frame
size, class count, confidence, model file hashes, load averages, software
versions, raw CPU feature report, Slurm allocation, timing boundaries, and
int8 calibration provenance (`latency-only` vs deployment calibration).

Calibration records live in `experiments/latency/calibration.json`, with
`id`, `factor`, `method`, `source` and `date`.

## C8 — Deploy model package

`models/<model_id>/` (git-ignored; hashes in `deploy/model_manifest.json`):

```
model.ncnn.param
model.ncnn.bin
metadata.yaml        Ultralytics keys (task, imgsz, names, stride, end2end, ...)
                     plus a `columba:` block:
                       model_id: ...
                       decode: yolo11 | yolo26 | dfine | rfdetr | nanodet
                       input_hw: [384, 640]
                       letterbox: true
                       pad_value: 114
                       color_order: bgr | rgb
                       normalize: {mean: [0,0,0], std: [255,255,255]}
                       output_layout: "1x(4+nc)xN cxcywh-px sigmoid-scores"
                       nms: {iou: 0.7, agnostic: false}   # decode: yolo11 only
                       max_det: 300
                       quantization: fp32 | fp16 | int8
                       taxonomy: single | species
                       trigger_classes: [pigeon]
```

`decode` values:

- `yolo11`: anchor-grid output, per-class NMS (IoU 0.7). Used by YOLO11 **and by
  standard YOLO26 exports** (see below).
- `yolo26`: the same output tensor, taken from YOLO26's one-to-one head. Decode
  by top-`max_det` score with **no NMS**. Ultralytics' NCNN exporter cannot
  produce this; it needs the head swap described below.

Ultralytics' Detect-family `end2end` flag and `columba.decode` are separate:
the metadata's `end2end` is `false` for every verified NCNN export.
`columba.decode` is authoritative.

#### Verified NCNN layout (Ultralytics 8.4.175, ncnn 1.0.20260526, Mac M2 Pro)

Exports of `yolo11n.pt` and `weights/yolo26n.pt` with
`format="ncnn", imgsz=[384, 640]` were inspected and run with the `ncnn` Python
module on `ultralytics/assets/bus.jpg`. Both have **the same layout**:

| Item | Value |
|---|---|
| Input blob | `in0`, `[1, 3, 384, 640]` float32: RGB, 0–1 (`/255`), letterboxed to the full `input_hw` with **centred** padding, pad value 114. |
| Output blob | one blob, `out0`. ncnn `Mat` dims=2, `h = 4 + nc`, `w = N` → array `(4 + nc, N)` (`(84, 5040)` for COCO). |
| N | `(H/8)(W/8) + (H/16)(W/16) + (H/32)(W/32)`; 5040 at 384×640. Level order is stride 8, then 16, then 32. |
| Rows 0–3 | box centre x, centre y, width, height, in **letterboxed input pixels** (not normalised). |
| Rows 4… | per-class scores, already sigmoid-ed (0–1). No objectness row. |
| Duplicates | Not removed. NMS is required (`decode: yolo11`). |
| `metadata.yaml` | `imgsz: [384, 640]` (`[h, w]`), `stride: 32`, `task: detect`, `head: Detect`, `batch: 1`, `channels: 3`, `end2end: false`, `args: {batch: 1, quantize: null}`, `names: {0: ...}`. |

Parity, **measured**: after NMS at IoU 0.7, raw ncnn boxes matched Ultralytics
PyTorch predictions on the same letterboxed image to within 0.5 px, with
confidences within 0.005 (YOLO26n: 5 boxes; YOLO11n: 4 boxes).

Findings:

1. **YOLO26 is not end-to-end in these exports.** In 8.4.175 the default
   `nms=None` selects YOLO26's one-to-many head (with NMS) for predict, val and
   export alike; only `nms=False` selects the one-to-one (NMS-free) head. The NCNN
   exporter also forces `end2end` off ("format without top-k support").
   `fuse()` then drops the one-to-one head from the export.
2. **One-to-one head export works with a head swap** (**measured**, exploratory):
   load the `.pt`, set `head.cv2, head.cv3 = head.one2one_cv2, head.one2one_cv3`,
   set both `one2one_*` to `None`, save, and export as usual. Output layout is
   identical to the table above. Top-k without NMS matched PyTorch `nms=False`
   predictions to within 0.5 px. Whether one-to-one beats one-to-many + NMS on
   our data is untested (WS5/WS7).
3. Ultralytics `predict` on a rectangular NCNN export works. Ultralytics `val` on
   a non-PyTorch export does not: "non-PyTorch val requires square images".
   WS1 must produce C3 predictions through `predict` or its own loop.
4. `half=`/`int8=` export arguments are deprecated in favour of `quantize`
   (`16`/`8`); the old names still map onto it with a warning.

`app/detector.py` still accepts the existing v1-2 package, which lacks a
`columba:` block (treated as `decode: yolo11`, square input).

## C9 — Runtime records

The existing JSONL fields stay. Additions:

- On `detections[]` items: `class_id`, `track_id`.
- New `frame` fields: `stage_ms` (`{"wait":..,"preprocess":..,"inference":..,"postprocess":..,"track":..}`), `camera` (name).
- New event `trigger`: `{track_id, class, first_seen_captured_at, confirmed_at, frames_seen, k, n}`.
- New event `camera_switch`: `{from, to, lux, reason}`.
