# WS1 — Evaluation harness

- State: done
- Scope: tasks 1–4 this round; task 5 deferred
- Branch: `ws1/evaluator`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-14

## Done

1. `tools/predict_to_coco.py` writes C3 JSON and `.meta.json` sidecars from
   PyTorch checkpoints and raw NCNN inference. C1 manifest IDs are preserved;
   plain YOLO datasets use unique image stems and record that fallback.
   Rectangular input, C8 metadata, legacy exports and the v1-2 wrapper directory
   containing `best_ncnn_model/` work. `yolo11` uses class-aware NMS;
   `yolo26` uses class-score top-k without NMS. PyTorch head selection is checked
   so a checkpoint cannot silently fall back to an unsupported decode mode.
2. `tools/evaluate_predictions.py` writes C4 metrics using pycocotools AP,
   IoU-0.5 matching, pigeon PR metrics, small pigeon recall, manifest-based empty
   and confuser false alarms, per-class AP, and style/source breakdowns.
   Manifest groups are read and counted. Metric definitions and edge cases are
   in both tools' `--help` and the emitted metrics' `definitions` field.
3. Lowest eligible observed confidence is selected on validation only.
   `--threshold-from` reuses selected validation metrics, including an
   unattainable-precision no-detection sentinel. Selection on train/test fails.
4. Hand-built tests cover perfect predictions, a known COCO PR/AP curve,
   duplicates, tied scores, rectangular size/coordinate mapping, small-box
   boundary and precision denominator, single-taxonomy unlabelled confusers,
   threshold selection/reuse, breakdowns, species class matching, absent
   predictions/ground truths, invalid predictions, stem fallback, C8/legacy
   metadata, mocked PyTorch and NCNN inference, and scratch settings isolation.

## In progress

Nothing in this round. Task 5 awaits WS2's published `columba-test-v2` and a
coordinator follow-up. No `experiments/leaderboard.csv` was created.

## Blockers and requests

No implementation blocker. The full WS1 acceptance criterion remains open for
baseline re-scoring and the leaderboard row, explicitly excluded this round.
The coordinator owns CONTRACTS.md; please add these exact C4 clarifications:

- "AP uses pycocotools' 101-point interpolated precision, IoUs 0.50 through 0.95
  in steps of 0.05, area=all and maxDets=300. A class with no ground-truth boxes
  has null AP; a class with ground truths and no predictions has AP=0."
- "For operating metrics, predictions are matched per image/class in descending
  score order to the highest-IoU unmatched ground truth with IoU >= 0.5.
  Equal-score predictions use prediction-file order within an image. A
  threshold includes all predictions with score >= threshold, including ties."
- "Validation selects the lowest observed pigeon score at which precision is
  >= 0.90. With no eligible score, write operating_threshold.value=
  1.0000000000000002, source=selected, attainable=false and split=val. This
  finite sentinel produces no detections when reused on test. Precision with
  no detections is 0. Recall with no ground truths is 0."
- "small_recall_at_p90 is the highest recall of small pigeon ground truths at
  any threshold where precision over all pigeon predictions is >= 0.90.
  Use the same matches as overall recall; matched large pigeons contribute to
  precision. Small means min(box_width, box_height) multiplied by
  min(input_height/image_height, input_width/image_width) is strictly < 32 px.
  With no small pigeon boxes, small recall is 0."
- "Empty/confuser image membership uses manifest classes_present, including
  unlabelled other birds in single-taxonomy datasets. A false-alarm rate with
  no eligible images is null. All non-pigeon bird classes count as confusers
  for this metric, regardless of their taxonomy role."
- "by_style/by_source recompute each subset's PR curve and AP and use the
  shared validation operating threshold for precision, recall and false alarms.
  Pigeon recall is also available as pigeon_recall_at_p90 in each subset."

C3 addition requested: "The optional ncnn_compute field records runtime_default
or fp32. Format describes package weight quantisation, not compute precision."
NCNN retains the runtime's default fp16 options unless `--ncnn-fp32` is given.
That flag disables fp16 packed/storage/arithmetic without changing the weights.
It is useful for PyTorch parity; it is not the default deployment compute mode.

Supported NCNN input is C8's verified RGB /255, pad=114, centred letterbox,
`in0`/`out0` YOLO layout. Other preprocessing/layouts fail explicitly. Raw NCNN
requires CPU and batch=1; PyTorch supports the requested device and batching.
The NMS-free C8 path is tested with a mocked NCNN model, not a live one-to-one
export. Plain YOLO fallback requires OpenCV from the existing vision extra;
C1 evaluation itself needs only the existing core and eval dependencies.

## Results

All numbers below are **measured on Mac M2 Pro**, host `vexcalibur`, shared with
other workers. Scratch is relative to this worktree:
`/Users/vari/.t3/worktrees/columba-zero/t3-4c6c30b5/outputs/ws1/`.
No Pi performance or frozen-test accuracy is claimed.

- `evaluation-tests.log`: 22 new tests pass in 0.768 s.
- `unittest.log`: all 66 tests pass in 1.488 s. Ruff lint and format checks
  pass on `app tests tools cluster training`; strict mypy passes for 28 files.
  Logs: `ruff-check.log`, `ruff-format.log`, `mypy.log`.
- `smoke/`: an annotated historical pigeon tile and blank image copied from
  the main checkout's `outputs/validation/style-cleanup/images/`, with a loose
  hand-marked pigeon box. Identical val/test copies deliberately exercise
  threshold handoff. This scratch fixture is not an accuracy benchmark.
- `pt/`, `ncnn/`: square 640×640 C3 predictions and selected validation C4
  metrics at the default floor 0.001. The NCNN test metrics reuse the NCNN
  validation threshold. Measured thresholds are 0.8583347797 for PyTorch and
  0.8583984375 for legacy NCNN. Both report AP50=1, AP50-95=0.9 and recall_at_p90=1
  on this plumbing fixture; these numbers must not enter the leaderboard.
- `pt-rect/`, `ncnn-rect/`: the same pipeline at 384×640, with a PyTorch batch
  of two images and a fresh NCNN C8 export made from a scratch copy of `best.pt`.
  Both produce original-pixel boxes and matching smoke metrics. The original
  weights/exports were read only. Export script/log: `export_rectangular.py`
  and `export-rect-smoke.log`; export lives in `rect-model/baseline_ncnn_model/`.
- `parity.json`, `parity-summary.log`, `compare_smoke.py`: reproducible comparisons.
  At confidence 0.5 both square and rectangular runs have one matched pigeon
  and no blank-image detections. Square max coordinate difference=0.5094 px,
  confidence difference=0.00006366; rectangular=0.3436 px and 0.0002265.
  NCNN default fp16 compute changes marginal scores/NMS at floor 0.001:
  square totals are 32 PyTorch vs 31 NCNN detections, 30 matched; rectangular
  totals are 36 vs 36, 33 matched at IoU >= 0.8 and score difference <= 0.005.
- `ncnn-fp32-compute/`, `ncnn-rect-fp32-compute/`: `--ncnn-fp32` restores all
  32 square and all 36 rectangular matches at floor 0.001. Maximum coordinate
  difference is 0.0001221 px square and 0.0001526 px rectangular; maximum score
  difference is 0.000000656 square and 0.000000447 rectangular. Thus the loose
  low-score default-compute parity is a precision effect, not box mapping.
- Initial scratch settings omitted sync=false on first creation; caught by
  inspecting the actual settings, fixed, covered by a regression test, and
  `.pt` inference rerun. Final scratch settings have sync=false and all data,
  weight and run paths under `outputs/ws1/ultralytics/`.

### CLI and importable API

The coordinator's fixed CLI is implemented, with one added NCNN compute option:

```sh
python tools/predict_to_coco.py --model PATH --dataset DIR --split {train,val,test} --out FILE \
  [--imgsz H [W]] [--conf 0.001] [--iou 0.7] [--max-det 300] [--device cpu|mps|0] \
  [--model-id ID] [--decode yolo11|yolo26] [--batch N] [--host NAME] [--ncnn-fp32]
python tools/evaluate_predictions.py --dataset DIR --split {train,val,test} \
  --predictions FILE --out FILE [--input-hw H W] [--classes training/classes.yaml] \
  [--select-threshold | --threshold-from METRICS_VAL_JSON | --threshold X]
```

For WS3:

```python
from tools.evaluate_predictions import evaluate

metrics = evaluate(
    dataset, split, predictions,
    input_hw=None,          # tuple[int, int], otherwise C3 sidecar is required
    classes=classes_path,  # defaults to repo training/classes.yaml; None skips taxonomy checking
    select_threshold=False,
    threshold_from=None,   # path to selected validation metrics
    threshold=None,        # explicit float; otherwise recorded default 0.5
)
# returns dict[str, JsonValue]; the caller writes metrics JSON
```

### Real-model commands

Use the shared venv python. These were actually run; scripts and logs above
retain the preparation and comparison:

```sh
python tools/predict_to_coco.py \
  --model /Users/vari/Developer/columba-zero/runs/detect/pigeon_detector/pigeon_baseline_v1-2/weights/best.pt \
  --dataset outputs/ws1/smoke --split val --out outputs/ws1/pt/predictions_val.json \
  --imgsz 640 --device cpu --host mac-m2-pro
python tools/predict_to_coco.py \
  --model /Users/vari/Developer/columba-zero/models/pigeon_baseline_v1-2 \
  --dataset outputs/ws1/smoke --split val --out outputs/ws1/ncnn/predictions_val.json \
  --device cpu --host mac-m2-pro
python tools/evaluate_predictions.py --dataset outputs/ws1/smoke --split val \
  --predictions outputs/ws1/ncnn/predictions_val.json \
  --out outputs/ws1/ncnn/metrics_val.json --select-threshold
python tools/predict_to_coco.py \
  --model /Users/vari/Developer/columba-zero/models/pigeon_baseline_v1-2 \
  --dataset outputs/ws1/smoke --split test --out outputs/ws1/ncnn/predictions_test.json \
  --host mac-m2-pro
python tools/evaluate_predictions.py --dataset outputs/ws1/smoke --split test \
  --predictions outputs/ws1/ncnn/predictions_test.json --out outputs/ws1/ncnn/metrics_test.json \
  --threshold-from outputs/ws1/ncnn/metrics_val.json
python tools/predict_to_coco.py \
  --model /Users/vari/Developer/columba-zero/runs/detect/pigeon_detector/pigeon_baseline_v1-2/weights/best.pt \
  --dataset outputs/ws1/smoke --split val --out outputs/ws1/pt-rect/predictions_val.json \
  --imgsz 384 640 --batch 2 --host mac-m2-pro
python tools/predict_to_coco.py --model outputs/ws1/rect-model/baseline_ncnn_model \
  --dataset outputs/ws1/smoke --split val --out outputs/ws1/ncnn-rect/predictions_val.json \
  --host mac-m2-pro
python tools/predict_to_coco.py \
  --model /Users/vari/Developer/columba-zero/models/pigeon_baseline_v1-2 \
  --dataset outputs/ws1/smoke --split val --out outputs/ws1/ncnn-fp32-compute/predictions_val.json \
  --host mac-m2-pro --ncnn-fp32
python tools/predict_to_coco.py --model outputs/ws1/rect-model/baseline_ncnn_model \
  --dataset outputs/ws1/smoke --split val \
  --out outputs/ws1/ncnn-rect-fp32-compute/predictions_val.json --host mac-m2-pro --ncnn-fp32
```
