# WS0 — Foundations and contracts

- State: done  <!-- not started | in progress | blocked | done -->
- Branch: `ws0/foundations`
- Agent: Claude Opus 5.5 via T3 delegate
- Last updated: 2026-10-14

## Done

1. Ultralytics 8.4.175 loads `weights/yolo26n.pt`, trains it (1 epoch on
   `coco8`, CPU) and exports it and `yolo11n.pt` to NCNN at `imgsz=[384, 640]`.
   Layouts verified with real ncnn inference and recorded in CONTRACTS.md C8.
2. Distillation (`distill_model`, `dis`) exists and works in 8.4.175. Arguments,
   mechanism, constraints and evidence recorded in CONTRACTS.md C5.
3. `training/classes.yaml` written (C2), species list checked against the Bird
   Society of Singapore checklist, iNaturalist taxon ids added.
   `tests/test_taxonomy.py` checks its invariants.
4. `pyproject.toml`: optional groups `eval`, `data`, `cluster` (empty), `native`;
   `types-requests` added to `dev`; mypy `files` gains `cluster` and `training`
   (placeholder `__init__.py` in each); mypy ignores missing stubs for `ncnn`,
   `pycocotools`, `imagehash`, `columba_native`. The new groups resolve to wheels
   for Python 3.14 (`pip install --dry-run`); none were installed, and the
   main venv is unchanged.
5. `.gitignore`: `experiments/runs/`, `native/build/`, `/data/`, `/columba-data/`,
   `*.tar.zst`. `experiments/README.md` describes the folder (C6, C7).
6. CONTRACTS.md reviewed and marked **v1** (changelog at the top).

Checks (section 0 rule 4, venv python): 44 unit tests OK; ruff check and
format check clean on `app tests tools cluster training`; mypy strict clean
(25 files).

## In progress

Nothing.

## Blockers and requests

For the coordinator:

- **Decision: YOLO26 decode path.** Standard Ultralytics 8.4.175 NCNN exports of
  YOLO26 are *not* end-to-end. They export the one-to-many head and need NMS,
  exactly like YOLO11 (C8, finding 1). PLAN.md WS8 task 1 still says
  "`yolo26` (end-to-end output, no NMS)". Options: (a) treat all YOLO NCNN
  exports as `decode: yolo11` (simplest; WS8 still implements `yolo26` top-k
  for (b)); (b) have WS7 export YOLO26's one-to-one head via the head swap in
  C8 finding 2 and compare its accuracy against one-to-many + NMS. I suggest
  (a) as the default, with (b) as a WS7 experiment. Please update the PLAN.md
  WS8 wording.
- **WS1:** Ultralytics `val` refuses rectangular non-PyTorch exports. Use
  `predict` or a custom loop for NCNN C3 predictions (C8, finding 3).
- **WS3:** use absolute model paths and per-job
  `$YOLO_CONFIG_DIR/Ultralytics/settings.json` (C5 runner notes). Distillation
  resume needs the teacher path to exist on the resuming node. Measure GPU
  epoch time with and without a teacher.
- **WS7:** `tools/export_ncnn.py` uses the deprecated `half=` argument; move to
  `quantize=16`.
- **WS2:** iNaturalist place id for Singapore is 6734. `other_bird` is Aves
  (taxon 3) minus the named taxa.
- C2 rename: `green_pigeon` → `pink_necked_green_pigeon`.
- Housekeeping note: my first scratch settings file was in the wrong place.
  Ultralytics reads `$YOLO_CONFIG_DIR/Ultralytics/settings.json`, so the user's
  global settings applied: `coco8` downloaded to `/Users/vari/Developer/datasets/`
  (since moved into `outputs/ws0/datasets/`, and the directory I created
  removed), and Ultralytics analytics `sync` was on during these runs. Nothing
  was written to `weights/` or `runs/` in either checkout. A teacher passed as a bare
  name (`yolo26s.pt`) downloaded to the worktree root; moved to scratch.

## Results

All on **Mac M2 Pro**, Ultralytics 8.4.175, torch 2.14.1, ncnn 1.0.20260526,
Python 3.14. Scratch artifacts in `outputs/ws0/` (git-ignored): scripts in
`outputs/ws0/scripts/`, runs in `outputs/ws0/runs/`, exports in
`outputs/ws0/weights/*_ncnn_model/`, logs `outputs/ws0/task2_distill*.log`.
These are smoke checks on `coco8` (4 train / 4 val images); none of the numbers
says anything about pigeon accuracy.

| Check | Result | Label |
|---|---|---|
| Train `yolo26n.pt` 1 epoch, coco8, imgsz 320, CPU | completed; val mAP50 0.796 | measured |
| NCNN export `yolo26n` / `yolo11n` at [384, 640] | both succeed; input `in0` [1,3,384,640]; output `out0` (84, 5040); `end2end: false` | measured |
| ncnn vs PyTorch, bus.jpg | boxes within 0.5 px, confidences within 0.005 after NMS | measured |
| YOLO26n one-to-one head via head swap → NCNN | top-k without NMS matches PyTorch `nms=False` within 0.5 px | measured |
| Distillation 2 epochs (teachers yolo11n, yolo26s; student yolo26n) | completes; `train/dis_loss` 11.6 / 9.1 in epoch 1 | measured |
| Distillation, 80-class teacher → 1-class student (`single_cls`, `dis=1`) | completes; `dis_loss` 1.9 | measured |
| `dis=0` vs `dis=6`, same seed and teacher | student weights differ (max abs 0.19) | measured |
| Gradient of the distillation term only | student Σ\|g\| 3.1e4, projector 1.0e3, teacher 0 | measured |
| Checkpoint after distillation | `best.pt` is a plain `DetectionModel`, teacher stripped | measured |
