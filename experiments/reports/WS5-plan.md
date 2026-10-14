# WS5 plan: YOLO training and distillation

Last updated 2026-10-14. Recipes live in `training/recipes/`. Commands run
from the repo root with the shared venv. Always pass `--exclude-nodes xgpe6`.

## Baseline (section 1)

The section-1 comparison baseline is round 0: 3-seed mean and std of test
`pigeon_recall_at_p90` on `columba-test-v2` (PLAN section 1 decision). v1-2's
rows stay on the leaderboard as `incumbent-contaminated` diagnostics.

Two round-0 recipes exist:

- `baseline-repro`: v1-2 args unchanged (patience 10). Seeds 0 and 2 stopped
  at epochs 11 and 53. On T4, seed 0's best checkpoint is epoch 1, picked
  during an early validation collapse. Measured test recall@p90: 0.006
  (seed 0) and 0.218 (seed 2).
- `baseline-repro-p50`: the same args with patience 50. It trains the v1-2
  recipe to convergence.

**Proposal:** use `baseline-repro-p50` as the section-1 baseline. A baseline
truncated by patience 10 would make almost any run look like a win. The
coordinator decides.

Caveat: `columba-train-v1clean` val has no empty or confuser images. So any
threshold selected on v1clean val ignores false alarms, and the empty and
confuser FAR of v1clean-trained runs (round 0, the round-1 v1clean column)
only show what that val-chosen threshold does on test. v2/v2s val contains
negatives. Compare FAR only between runs whose val splits have the same
composition.

## Rounds and recipes

| Round | Recipes | Seeds | Runs | Est. nv GPU-h |
|---|---|---|---:|---:|
| 0 | `baseline-repro`, `baseline-repro-p50` (YOLO11n, 640, v1clean; seeds pinned to T4 / Titan RTX / Titan V) | 3 + 3 | 6 | 3 |
| Teachers | `yolo11x-v2s-960-teacher`, `yolo26x-v2s-960-teacher` (Titan RTX, 960, 300 ep, patience 50) | 1 | 2 | 120 |
| 1 | `{yolo11n,yolo26n,yolo11s,yolo26s}-{v1clean,v2,v2s}-640-r1` (150 ep, patience 50, cos_lr, close_mosaic 15) | 1 | 12 | 35 |
| 2 | top 3 model × data from round 1 × {plain, distilled from best teacher}, 300 ep; plus an imgsz-800 variant per model that passes the 448×800 latency gate | 3 | 18 (+) | 150–270 |
| 3 | round-2 winner with `species` taxonomy | 3 | 3 | 15–30 |

Total **estimated** ≈ 320–460 nv GPU-h, above PLAN's 250. Most of that is
teachers and round 2, and both depend on v2s's size. Round-1 timings will
narrow it.

### Assumptions behind the estimates

- **Measured** (round 0, YOLO11n 640 on 868 train / 156 val images,
  training + validation per epoch, excluding epoch 1's ~35 s start-up):
  - T4 (batch 48): ≈ 7.0 s/epoch.
  - Titan V (batch 32): ≈ 5.3 s/epoch.

  Each job adds ≈ 3–5 min for bootstrap, staging and evaluation.
- **Estimated:** s models take ≈ 2× nano time per epoch. Round 1 is assumed
  to run all 150 epochs (patience 50 rarely fires that early).
- **Assumed** dataset sizes (WS2 has not published them): v2 ≈ 5× v1clean,
  v2s ≈ 10×. Time scales linearly with images.
- **Teachers (estimated):** x models at 960 have ≈ 70× the nano-640 FLOPs.
  On Titan RTX with v2s, that is ≈ 12 min/epoch, or ≈ 60 h for 300 epochs.
  - This is close to the 3-day job limit. Resubmitting resumes from
    `last.pt` (verified by WS3).
  - Usable Titan RTX nodes today are xgpe2 and xgpe4: xgpe3 is down and
    xgpe6 is excluded. Expect queueing.
  - WS3's batch heuristic gives batch **3** for x models at 960 on 24 GB
    (accumulate 21). That is slow and leaves BatchNorm on tiny batches. Batch
    8 probably fits (not measured). This is a request to WS3/coordinator.
- **Round 2:** distillation adds one teacher forward pass per batch,
  assumed ≈ 1.5–2× step time (not yet measured on a GPU).

## Selection rules

- **Round 1 → 2:** rank the 12 runs by val `pigeon_recall_at_p90` and take
  the top 3 model × data combinations. Break ties (within 0.01) by the
  lower estimated Pi 5 frame p95 (WS4 gate, fp16, 384×640).
  - v1clean val has no negatives, while v2/v2s val does. Their val numbers
    are on different val sets, so treat cross-dataset ranking as
    approximate. Before ranking, also check each finalist's test FAR.
- **Round 2 → winner:** best 3-seed mean val `pigeon_recall_at_p90`, then:
  - Section 1 promotion: 3-seed mean test `pigeon_recall_at_p90` beats the
    baseline mean by more than the larger of the two seed std devs.
  - Empty FAR ≤ 0.02 and confuser FAR ≤ 0.05 at the val-selected threshold.
  - Estimated Pi 5 p95 ≤ 100 ms (target 70 ms), reaction p95 ≤ 500 ms, and
    NCNN parity (WS7).
- **Latency:** nano models at 384×640 are estimated at ≈ 33–37 ms and
  448×800 at ≈ 49–52 ms (WS4). s-size latency is not measured yet. Before
  round 2, run one bundled arm benchmark (or ask the coordinator to bundle
  it) of COCO `yolo11s`/`yolo26s` at 384×640 and 448×800 through WS4's gate.
- **Round 3 (section 5.4 species gate):** ship species only if all hold:
  - 3-seed mean `pigeon_recall_at_p90` ≥ single mean − 0.01.
  - False-alarm rates no worse.
  - Estimated Pi 5 per-frame p95 ≤ 5% slower.

  Otherwise ship single-class, trained with the species images as unlabelled
  negatives. The decision goes in `status/WS5.md`.
