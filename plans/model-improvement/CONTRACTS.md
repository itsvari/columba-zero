# Shared contracts

Version: **v0 (draft)**. WS0 checks the items marked *(verify)* and marks this
file `v1`. After that, changes go through the coordinator and bump the version.

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
`test:` pointing at the frozen test dataset (`../columba-test-v2/images/test`).
Test datasets contain only `test`.

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
      - green_pigeon               # pink-necked green pigeon, confuser (native)
      - javan_myna
      - house_crow
      - asian_glossy_starling
      - eurasian_tree_sparrow
      - yellow_vented_bulbul
      - asian_koel
      - other_bird                 # catch-all
    min_train_instances: 150       # rarer species fold into other_bird
```

The species list is a proposal *(verify)* against an authoritative Singapore
checklist. Each entry also records its iNaturalist taxon id.

## C3 — Predictions

COCO results format, written by any model or framework:

`<run>/predictions_<split>.json`: a list of

```json
{"image_id": "inat-123456789-0", "category_id": 0, "bbox": [x, y, w, h], "score": 0.83}
```

- `image_id` is the C1 manifest `id`. `category_id` is the taxonomy index.
- `bbox` is in **original image pixels**, `[left, top, width, height]`.
- Confidence floor 0.001. Duplicate-box removal (NMS) as the deployed runtime
  would do it (IoU 0.7, or none for end-to-end decoders).

Sidecar `<run>/predictions_<split>.meta.json`:

```json
{"model_id": "...", "framework": "ultralytics", "format": "pt|ncnn-fp32|ncnn-fp16|ncnn-int8|onnx",
 "input_hw": [384, 640], "conf_floor": 0.001, "nms_iou": 0.7, "agnostic_nms": false,
 "host": "xcng0", "dataset": "columba-test-v2", "created_at": "..."}
```

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

`run_job.py` sets `batch` from the GPU's memory class; `seed`, `project`,
`name` and `data` are set by the runner, not the recipe. The distillation
arguments' exact names and meaning *(verify in WS0)* are recorded here.

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
                       output_layout: "<verified layout, e.g. 1x(4+nc)xN xywh | 1x300x6 xyxy,score,cls>"
                       nms: {iou: 0.7, agnostic: false}   # omitted for end-to-end decoders
                       quantization: fp32 | fp16 | int8
                       taxonomy: single | species
                       trigger_classes: [pigeon]
```

Export output layouts for YOLO11 and YOLO26 under Ultralytics 8.4.175
*(verify in WS0)*. `app/detector.py` still accepts the existing v1-2 package,
which lacks a `columba:` block (treated as `decode: yolo11`, square input).

## C9 — Runtime records

The existing JSONL fields stay. Additions:

- On `detections[]` items: `class_id`, `track_id`.
- New `frame` fields: `stage_ms` (`{"wait":..,"preprocess":..,"inference":..,"postprocess":..,"track":..}`), `camera` (name).
- New event `trigger`: `{track_id, class, first_seen_captured_at, confirmed_at, frames_seen, k, n}`.
- New event `camera_switch`: `{from, to, lux, reason}`.
