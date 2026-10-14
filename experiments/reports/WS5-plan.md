# WS5 plan: YOLO training and distillation

Last updated 2026-10-14. Recipes live in `training/recipes/`. Commands run
from the repo root with the shared venv. Bad nodes are excluded by default
via `cluster/excluded_nodes.txt` (xgpe6, xgpe2). From 15 Oct the budget is
8 GPUs, 24 queued, all GPU types (CONTRACTS v1.7, `cluster/gpu_policy.yaml`).

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
| baseline-repro-p50 s1 | Titan V (moved from Titan RTX) | 124 | 0.400 | 0.238 | 0.609 | 0.888 / 0.246 | 0.000 | 0.151 |
| **p50 mean ± sd (3 seeds)** | | | **0.473 ± 0.066** | **0.246 ± 0.035** | **0.604 ± 0.031** | 0.853 / 0.308 | 0.000 | 0.242 |
| baseline-repro-p50 s2 | Titan V | 137 | 0.489 | 0.215 | 0.633 | 0.818 / 0.343 | 0.000 | 0.356 |
| baseline-repro s0 | T4 | 11 | 0.071 | 0.006 | 0.131 | 0.467 / 0.020 | 0.024 | 0.068 |
| baseline-repro s2 | Titan V | 53 | 0.468 | 0.218 | 0.564 | 0.833 / 0.269 | 0.008 | 0.205 |

- `baseline-repro-p50` s1 ran as job 945876 on a Titan V. Earlier attempts
  failed or never started: 945409 hit xgpe2 with no GPU visible, and 945774
  was cancelled while still pending for the only working Titan RTX. The
  section-1 baseline is therefore 2 Titan V seeds plus 1 T4 seed.
- **Promotion bar (section 1):** test R@P90 > 0.246 + max(0.035, the
  candidate's sd). Each comparison also reports test AP50 (baseline
  0.604 ± 0.031) and P/R at the val threshold.
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

## Round 1 on v2/v2s (in progress)

Measured, 1 seed (P/R at the val threshold are in the leaderboard and
metrics files):

| Recipe | GPU | Epochs | val R@P90 | test R@P90 | test small R@P90 | test AP50 | empty FAR | confuser FAR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolo26n-v2-640-r1 | Titan V | 144 | 0.773 | 0.705 | 0.472 | 0.818 | 0.000 | 0.055 |
| yolo11n-v2-640-r1 | T4 | 137 | 0.770 | 0.691 | 0.438 | 0.788 | 0.000 | 0.041 |
| yolo11s-v2-640-r1 | Titan V → H100 NVL (resumed) | 136 | 0.758 | 0.734 | 0.534 | 0.808 | 0.000 | 0.014 |

yolo11s-v2 has the best test numbers, but val R@P90 is the ranking metric
and it does not beat the nanos on val by the 0.04 margin.

The first result, measured: `yolo11n-v2-640-r1` (T4, 137 epochs)
- val R@P90 0.770 (v2 val includes negatives).
- test R@P90 **0.691**, test small R@P90 0.438, test AP50 0.788.
- P/R at val threshold 0.889 / 0.703, empty FAR 0.000, confuser FAR
  **0.041**.

That is a test R@P90 gain of about 0.45 over the baseline, and the first
run under the 0.05 confuser gate. Data matters more than the model change.

## Rounds and recipes

| Round | Recipes | Seeds | Runs | Est. nv GPU-h |
|---|---|---|---:|---:|
| 0 | `baseline-repro`, `baseline-repro-p50` (YOLO11n, 640, v1clean; seeds pinned to T4 / Titan RTX / Titan V) | 3 + 3 | 6 | 3 |
| Teachers | `yolo11x-v2s-960-teacher-b16`, `yolo26x-v2s-960-teacher-b12` (H100 family, gpu-long; 960, 300 ep, patience 50, no RAM cache) | 1 | 2 | 40–70 |
| 1 | `{yolo11n,yolo26n,yolo11s,yolo26s}-{v1clean,v2,v2s}-640-r1` (150 ep, patience 50, cos_lr, close_mosaic 15) | 1 | 12 | 50 |
| 2 | top 3 model × data from round 1 × {plain, distilled from best teacher}, 300 ep; plus a nano `imgsz 800` variant per nano combination | 3 | 18 + 9 | 150–250 |
| 3 | round-2 winner with `species` taxonomy | 3 | 3 | 15–30 |

Total **estimated** ≈ 300–450 GPU-h across GPU types (from 15 Oct: 8-GPU budget, all GPU types), above PLAN's 250. Most of that is
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
- **Actual** dataset sizes (WS2, published): v2 has 7,958 train images
  (9.2× v1clean) and v2s 15,865 (18.3×). Both share a 1,089-image val
  (7× v1clean's).
- **Measured** on v2: `yolo11n-v2-640-r1` on T4, batch 48, takes
  ≈ 59 s/epoch (8× v1clean), so ≈ 2.7 h per 150-epoch run.
- **Measured** round-1 speeds on Titan V (batch 32 nano, 16 s), s/epoch:
  - yolo26n-v2 47, yolo11n-v2s 83, yolo26n-v2s 91.
  - yolo11s-v2 62, yolo26s-v2 75, yolo11s-v2s 118.5.
- **Teachers, first attempt** (cancelled 15 Oct to move to H100s): measured
  946 s/epoch for yolo11x on Titan V at batch 4 and 972 s/epoch for yolo26x
  on Titan RTX at batch 6. At that pace 300 epochs take about 80 h.
- **Teachers, now:** fresh runs under new IDs `-b16` / `-b12`, pinned to the
  H100 family on gpu-long. Both were placed on **h100-96** (a full H100 NVL).
  - Batches come from probe 945775 (Titan V, 960, AMP, v1clean): yolo11x
    ≈ 2.1 GiB/sample, yolo26x ≈ 2.6 GiB/sample.
  - Batch 16 and batch 12 peak at ≈ 34 and ≈ 31 GiB, so they fit the
    46.4 GB h100-47 MIG slice as well as h100-96. If memory runs out,
    Ultralytics retries at half the batch.
  - Why fresh rather than resumed: a changed `batch_override` changes the
    recipe hash, and `run_job.validate_existing_run` rejects that for an
    existing run. Only 10 and 4 epochs were lost.
  - Not on `xgpj0`/H200: those are only in the 3-hour `gpu` partition, so a
    multi-day teacher would re-stage 4.9 GB of v2s on every slice.
  - RAM cache stays off: at 960, v2s needs up to ≈ 44 GB of the 64 GB job
    limit.
  - **Estimated** at 4–5× Titan V speed: ≈ 200–250 s/epoch, so ≈ 17–21 h for
    300 epochs, less if patience 50 fires. Not yet measured; the data
    loader (8 CPUs) may cap it.

- **Optimizer:** `optimizer: auto` picks AdamW when there are fewer than
  10,000 iterations (v1clean) and MuSGD (lr 0.01) above that (v2/v2s at
  150+ epochs). Read cross-dataset comparisons with that in mind. Resume
  verified on 15 Oct: yolo11s-v2 continued from epoch 33 on an H100 NVL
  with batch 16 → 64. nbs 64 keeps the effective batch at 64.

- **Teachers measured (15 Oct):**
  - yolo11x-b16 on a full H100 NVL (h100-96): ≈ 201 s/epoch → ≈ 17 h for
    300 epochs.
  - yolo26x-b12 on an H100 MIG 3g.47gb (h100-47, moved by the
    coordinator): ≈ 367 s/epoch → ≈ 31 h.

### ETAs (estimated, 15 Oct, from the turn-4 snapshot)

- **Round 1:** the 5 runs on Titan V need ≈ 0.7–3.7 h more each (measured
  pace, 150-epoch cap).
  - yolo11s-v2 resumes from epoch 32 on h100-47 (3-hour `gpu` partition).
  - yolo26s-v2s has not started: ≈ 6 h on Titan V, or 2 h-slices on an
    H100.
  - Round 1 should be all DONE in ≈ 5–8 h, assuming the auto-submitter
    places yolo26s-v2s within ≈ 2 h.
- **Teachers:** ≈ 17–21 h after they start. They are queued behind ~14
  h100-96 jobs, so expect DONE about 1–1.5 days from now.
- **Round 2 with 8 GPUs** (v2s, 300 epochs, 3 seeds):
  - Nano runs take ≈ 7.5 h on Titan V (measured 90 s/epoch) or ≈ 3–4 h on
    H100/A100-class GPUs (estimated).
  - Start the **plain** arms (9 runs + 9 nano `imgsz 800` runs) as soon as
    round 1 is ranked, without waiting for the teachers. That is about
    3 waves of 8, ≈ 1 day.
  - Add the 9 **distilled** arms when the better teacher is DONE.
    Distillation runs the teacher at 640 on the student's batch, adding
    ≈ 1.5–2× step time (not yet measured on a GPU).
  - Round 2 complete ≈ 2–2.5 days from now; round 3 (species, 3 runs) adds
    ≈ 0.5 day.

## Selection rules

- **Round 1 → 2: automatic.** Confirmed rules in `training/rounds/round2.yaml`,
  applied by WS3's transition job; no agent reasoning in between.
  - **Inputs:** the 8 v2/v2s round-1 runs. v1clean is excluded (its val has
    no negatives). Any input not DONE → HOLD.
  - **Ranking:** take the top 3 model × data combinations by val
    `pigeon_recall_at_p90`. Test metrics are never used.
  - **Ties:** recalls within 0.01 of the current leader go to the lower
    estimated Pi 5 fp16 p95 at 384×640.
    - All four models use the **COCO** latency records (yolo11n 38.7,
      yolo26n 36.9, yolo11s 81.8, yolo26s 83.5 ms), so the class count is
      the same and only the architecture differs. The draft's one-class
      v1-2 record for yolo11n would have flipped the nano order.
    - Exact latency ties occur only for the same model on v2 vs v2s. They
      prefer **v2s** via `tie_order`: style copies cover the NoIR and wide
      cameras, which the original-style val cannot measure.
  - **s models** enter only with val R@P90 more than 0.04 above the best
    nano on the same dataset, and deploy at ≤ 384×640 only.
  - **Selected nanos** also get a 3-seed `imgsz 800` arm (deploy 448×800).
  - **Training:** 3 seeds, 300 epochs, patience 50, cos_lr, nbs 64,
    save_period 5, cache ram.
    - Optimizer pinned to what `auto` resolved to on v2/v2s in round 1:
      MuSGD, lr0 0.01, momentum 0.9, warmup_bias_lr 0.
    - No `batch_override`: the runner's memory class picks the batch.
    - GPU placement follows `cluster/gpu_policy.yaml`. The 800 arms count as
      heavy. Generated recipes keep the round-1 `est_gpu_hours`
      (nano v2 3.7 → light; v2s ≥ 7.3 → heavy).
- **Distillation: automatic.** `training/rounds/round2-distill.yaml` runs
  after the plain arrays and both teachers finish.
  - Arms: the same 3 combinations at 640 only. The 800 arms are not
    distilled automatically.
  - Teacher: the higher teacher val R@P90. An exact tie prefers yolo11x.
  - `dis` 6.0 (the Ultralytics default). Training block identical to the
    plain arms.
- **Dry run** (local, WS3's own functions on 3 real runs plus labelled
  mocks; `outputs/ws5/scripts/dryrun_round2.py`): with current values and
  v2s equal to v2, the selection is yolo26n-v2s, yolo26n-v2, yolo11n-v2s →
  18 plain + 9 distilled = 27 runs. An s model is selected only in the
  +0.06 scenario. An unfinished input HOLDs.
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
