# WS5 plan: YOLO training and distillation

Last updated 2026-10-14. Recipes live in `training/recipes/`. Commands run
from the repo root with the shared venv. Bad nodes are excluded by default
via `cluster/excluded_nodes.txt` (xgpe6, xgpe2).

## Baseline (section 1)

**Decided (coordinator):** the section-1 comparison baseline is
`baseline-repro-p50`, 3-seed mean and std of test `pigeon_recall_at_p90` on
`columba-test-v2`. Every comparison also reports test AP50 and
precision/recall at the val-selected threshold. If neither model reaches
P ≥ 0.90 on test, rank by recall at the val threshold, subject to the FAR
gates (PLAN section 1). v1-2's rows stay on the leaderboard as
`incumbent-contaminated` diagnostics.

Round 0, measured. Cluster nv GPUs, YOLO11n 640, `columba-train-v1clean`.
P/R are at the val-selected threshold.

| Run | GPU | Epochs | val R@P90 | test R@P90 | test AP50 | test P / R | empty FAR | confuser FAR |
|---|---|---:|---:|---:|---:|---|---:|---:|
| baseline-repro-p50 s0 | T4 | 151 | 0.529 | 0.283 | 0.571 | 0.855 / 0.334 | 0.000 | 0.219 |
| baseline-repro-p50 s1 | Titan RTX | – | – | – | – | – | – | – |
| baseline-repro-p50 s2 | Titan V | 137 | 0.489 | 0.215 | 0.633 | 0.818 / 0.343 | 0.000 | 0.356 |
| baseline-repro s0 | T4 | 11 | 0.071 | 0.006 | 0.131 | 0.467 / 0.020 | 0.024 | 0.068 |
| baseline-repro s2 | Titan V | 53 | 0.468 | 0.218 | 0.564 | 0.833 / 0.269 | 0.008 | 0.205 |

- `baseline-repro-p50` s1 is job 945774, pending on a Titan RTX. Its first
  attempt, 945409, failed on xgpe2 with no visible GPU.
- `baseline-repro` (patience 10, v1-2's own args) is kept as a recorded
  round-0 reproduction with **2 seeds only**. Its seed 1 (945341) was
  cancelled by the coordinator. Patience 10 stopped T4 seed 0 at epoch 11,
  with its best checkpoint at epoch 1.
- Hardware effect: the 2 p50 seeds differ by 0.07 in test R@P90 and 0.06 in
  AP50, in opposite directions. With one seed per GPU type, seed and GPU
  effects can't be separated. Speed: T4 7.0 s/epoch at batch 48, Titan V
  5.3 s/epoch at batch 32.

Caveat: `columba-train-v1clean` val has no empty or confuser images. So any
threshold selected on v1clean val ignores false alarms, and the empty and
confuser FAR of v1clean-trained runs (round 0, the round-1 v1clean column)
only show what that val-chosen threshold does on test. v2/v2s val contains
negatives. Compare FAR only between runs whose val splits have the same
composition.

Second caveat (measured): v1clean-trained models get confuser false
positives on test-v2, so their test precision sits near 0.9. Test recall@p90
then swings between ≈ 0 and ≈ 0.3 on small differences.
`yolo26n-v1clean-640-r1` has val recall@p90 0.664 but test recall@p90 0.006:
its test precision peaks at ≈ 0.87, while test AP50 is 0.643 and recall at
the val threshold is 0.496. The section-1 rule should therefore report test
AP50 and recall at the val threshold next to the 3-seed recall@p90 mean.
Expect a large std for the v1clean baseline.

## Round 1 on v1clean (measured, 1 seed, T4)

150-epoch budget, patience 50, 640. P/R are at the val-selected threshold.

| Recipe | Batch | s/epoch | Epochs | val R@P90 | test R@P90 | test AP50 | test P / R | empty FAR | confuser FAR |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| yolo11n-v1clean-640-r1 | 48 | 7.4 | 117 | 0.507 | 0.238 | 0.591 | 0.824 / 0.278 | 0.008 | 0.219 |
| yolo26n-v1clean-640-r1 | 48 | 8.3 | 136 | 0.664 | 0.006 | 0.643 | 0.866 / 0.496 | 0.000 | 0.301 |
| yolo11s-v1clean-640-r1 | 24 | 12.0 | 126 | 0.475 | 0.365 | 0.646 | 0.924 / 0.309 | 0.000 | 0.110 |
| yolo26s-v1clean-640-r1 | 24 | 14.6 | 114 | 0.525 | 0.190 | 0.618 | 0.839 / 0.326 | 0.000 | 0.274 |

Observations:

- Only yolo11s reaches P ≥ 0.90 on test at its val threshold.
- Val ranking (26n > 26s > 11n > 11s) disagrees with test AP50 and test FAR,
  because v1clean val has no negatives.
- Confuser FAR is 0.11–0.30 for every model, far above the 0.05 gate. This
  is a data problem: v1clean labels all COCO birds as pigeon and has no
  negatives. v2/v2s should fix it.
- None of these v1clean runs is a round-2 candidate by itself. They give the
  model-size baseline for the v2/v2s comparison.

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
  - Only xgpe4 is a working Titan RTX: xgpe3 is down, and xgpe2 and xgpe6
    show no GPU. Teacher placement and batch are below.
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
- **Latency** (WS4, estimated Pi 5 frame p95, fp16):

  | Model | 352×608 | 384×640 | 448×800 |
  |---|---:|---:|---:|
  | nano | 29–33 ms | 33–37 ms | 49–52 ms (passes) |
  | yolo11s / yolo26s | ≈ 72 ms | 82–84 ms (fails 70 ms target, passes 100 ms limit) | 116–122 ms (fails the limit) |

  YOLO26's one-to-one head saves < 1 ms.
- **Round 2 consequences:**
  - The tie-break uses estimated latency at the intended deploy size.
  - An s model enters round 2 only if, on the same dataset, its val metric
    beats the best nano by more than single-seed noise. Single-seed noise
    is ≥ 0.04: the p50 seeds differ by 0.04 val R@P90 and 0.07 test R@P90.
    It is then trained and deployed at ≤ 384×640 only. Otherwise its slot
    goes to the next nano combination.
  - Every nano round-2 combination gets a `train imgsz 800` variant (deploy
    448×800), 3 seeds, plain. That adds 3 runs per nano combination.
    Distilling it is decided after round 2's 640 results.
- **Round 3 (section 5.4 species gate):** ship species only if all hold:
  - 3-seed mean `pigeon_recall_at_p90` ≥ single mean − 0.01.
  - False-alarm rates no worse.
  - Estimated Pi 5 per-frame p95 ≤ 5% slower.

  Otherwise ship single-class, trained with the species images as unlabelled
  negatives. The decision goes in `status/WS5.md`.
