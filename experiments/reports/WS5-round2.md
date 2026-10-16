# WS5 round 2: interim report

Status on 16 Oct 2026. The summary is in `WS5-final.md`.
- **Plain arms:** complete, 18 of 18 runs DONE.
- **Distillation:**
  - 640: generated mechanically by transition 949889 (yolo26x teacher).
    3 of 3 seeds DONE; compared with plain below.
  - 800: added by coordinator decision on 15 Oct evening. 3 of 3 seeds
    DONE. It passes the section-1 rule against plain-800 (+0.031 test
    R@P90).

Promotion is judged at the deploy size, on native exports (PLAN section 1,
15 Oct decision; WS7). The training-size numbers here are context.

All metrics below are **measured** by WS1's evaluator on the cluster:
- val = the `columba-train-v2`/`-v2s` val split (identical for both,
  1,089 images, includes negatives);
- test = frozen `columba-test-v2` (319 images);
- each model is evaluated square at its training size (C6).

The operating threshold is chosen on val and reused on test. Latency is
**estimated** (WS4 C7 records on xcng0, scaled by `cal-v0-published`), not
measured on a Pi.

## Arms (3-seed mean ± sd)

| Arm | Seeds | val R@P90 | test R@P90 | test small R@P90 | test AP50 | test P @ val thr | test R @ val thr | empty FAR | confuser FAR | confuser FAs per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo26n-v2s-640 | 3/3 | 0.788 ± 0.010 | 0.743 ± 0.003 | 0.534 ± 0.010 | 0.835 ± 0.006 | 0.912 ± 0.008 | 0.732 ± 0.006 | 0.003 ± 0.005 | 0.041 ± 0.014 | s0 **4**, s1 2, s2 3 |
| yolo26n-v2s-800 | 3/3 | 0.794 ± 0.006 | **0.780 ± 0.024** | 0.537 ± 0.051 | **0.860 ± 0.017** | 0.924 ± 0.013 | 0.748 ± 0.006 | 0.000 ± 0.000 | 0.032 ± 0.016 | s0 3, s1 1, s2 3 |
| yolo26n-v2-640 | 3/3 | 0.786 ± 0.005 | 0.740 ± 0.015 | 0.521 ± 0.017 | 0.805 ± 0.011 | 0.915 ± 0.016 | 0.720 ± 0.016 | 0.000 ± 0.000 | 0.037 ± 0.029 | s0 2, s1 1, s2 **5** |
| yolo26n-v2-800 | 3/3 | **0.796 ± 0.013** | 0.779 ± 0.022 | 0.537 ± 0.042 | 0.839 ± 0.012 | **0.931 ± 0.011** | **0.754 ± 0.011** | 0.003 ± 0.005 | **0.023 ± 0.016** | s0 1, s1 1, s2 3 |
| yolo11n-v2s-640 | 3/3 | 0.780 ± 0.027 | 0.736 ± 0.008 | 0.517 ± 0.011 | 0.828 ± 0.021 | 0.927 ± 0.008 | 0.703 ± 0.025 | 0.000 ± 0.000 | 0.037 ± 0.016 | s0 **4**, s1 2, s2 2 |
| yolo11n-v2s-800 | 3/3 | 0.783 ± 0.008 | 0.757 ± 0.003 | 0.494 ± 0.004 | 0.850 ± 0.018 | 0.925 ± 0.008 | 0.728 ± 0.012 | 0.005 ± 0.009 | 0.050 ± 0.008 | s0 **4**, s1 3, s2 **4** |
| *baseline-repro-p50* | 3/3 | 0.473 ± 0.066 | 0.246 ± 0.035 | 0.000 ± 0.000 | 0.604 ± 0.031 | 0.853 ± 0.035 | 0.308 ± 0.053 | 0.000 ± 0.000 | 0.242 ± 0.105 | s0 16, s1 11, s2 26 |

Round-2 IDs end in `-round2-plain` (generated recipes in
`training/recipes/generated/round2-plain/`).


For reference, the teachers (1 seed, 960 px) score val R@P90 0.845 / 0.822
and test R@P90 0.861 / 0.870 (yolo26x / yolo11x). This shows how much
accuracy headroom distillation could transfer.

## Section-1 comparison against `baseline-repro-p50`

Rule: the 3-seed mean test R@P90 must exceed the baseline mean (0.246) by
more than the larger of the two seed sds.
- **Every plain arm passes by a wide margin.** The smallest margin is
  yolo11n-v2s-640 (0.736 ± 0.008), 0.49 above the baseline, against a bar of
  0.035.
- **Test R@P90 is a valid ranking metric here:** precision at the val
  threshold is ≥ 0.90 for every arm (0.909–0.931), unlike the
  v1clean-trained runs.
- The comparison also reports test AP50, which is 0.80–0.86 against
  0.604 ± 0.031.
- Small-pigeon recall@P90 rises from 0.000 to about 0.50–0.54. It is the
  weakest metric, and 196 of the 212 small test boxes come from one depot
  session (PLAN section 1).

## False-alarm resolution (the 0.05 gate)

The confuser subset has 73 images:
- 1 false alarm = 0.0137, so the ≤ 0.05 gate allows **≤ 3** false alarms
  per seed.
- A single extra false alarm moves a seed across the gate, so seed-level
  FAR is coarse: one image either way.

The empty subset has 125 images. 1 false alarm = 0.008, so ≤ 0.02 allows
≤ 2.

Seeds that **breach** the confuser gate (≥ 4 false alarms, FAR ≥ 0.055):

- yolo26n-v2s-640 s0 (4)
- yolo26n-v2-640 s2 (5)
- yolo11n-v2s-640 s0 (4)
- yolo11n-v2s-800 s0 (4) and s2 (4): **2 of 3 seeds** (s1 has 3). The arm
  mean is 0.050, exactly at the gate.

**No 800-px yolo26n seed breaches.** yolo26n-v2-800 has the lowest mean
confuser FAR (0.023: 1, 1, 3 false alarms) and yolo26n-v2s-800 is at 0.032
(3, 1, 3). Every seed of every arm passes the empty-scene gate (≤ 2 of 125;
the worst is yolo11n-v2s-800 s2 with 2).

## Latency per arm

WS4 C7 records, fp16 NCNN, COCO heads, estimated Pi 5 end-to-end p95 at the
arm's deploy size:

| Arm | Deploy input | Est. Pi p95 | 70 ms target / 100 ms limit |
|---|---|---:|---|
| yolo26n-*-640 | 384×640 | 36.9 ms | pass / pass |
| yolo26n-*-800 | 448×800 | 51.6 ms | pass / pass |
| yolo11n-*-640 | 384×640 | 38.7 ms | pass / pass |
| yolo11n-*-800 | 448×800 | 50.1 ms | pass / pass |

The 800 arms spend about 13–15 ms of the headroom to gain about 0.04 test
R@P90 and AP50 and fewer confuser false alarms (yolo26n). That is the trade
PLAN section 4 asks for. Single-class heads are slightly faster than these
80-class COCO records (v1-2 at 384×640: 32.8 ms).

Note: test metrics here are square at the training size, not the 16:9
deploy input. WS7 evaluates the exported candidates at their deploy size.

## Observations (interim, no decision yet)

- **Input size matters more than data style on test-v2.** At 800 px, v2 and
  v2s are within noise on test-v2: 0.779 vs 0.780 test R@P90, val 0.796 vs
  0.794. Test-v2 is original-style only, so it cannot show the style copies'
  benefit for the NoIR and wide cameras. Evaluating the finalists on
  `columba-test-v2-styles` (`by_style`) is the way to separate them. I
  recommend it for the final report.
- yolo26n is ahead of yolo11n at both sizes on test R@P90 and FAR. The
  difference is within about one sd at 640.
  - At 800, yolo11n-v2s has the lowest small-pigeon recall (0.494) and is
    the only 800 arm with seeds over the confuser gate.
- **WS7 (provisional, deploy-size 16:9 fp16 NCNN):** yolo26n-v2s-800 gets
  test R@P90 0.686 and test-v2-styles 0.52, against 0.36 on styles for
  yolo26n-v2-800. So the style copies help a lot where test-v2 can't see it.
  Deploy-size recall is lower than square-800 recall (0.78). WS7 owns that
  scoring.
- The empty-image share in training (63% backgrounds) shows no sign of
  hurting recall. Val recall at P90 ≈ 0.79 with empty FAR ≈ 0, so no
  subsampling is needed for now (WS2's suggestion).

## Distillation

Both arms use teacher `yolo26x-v2s-960-teacher-b12/seed0`: val/test R@P90
0.845/0.861, the better val of the two teachers. `dis` is 6.0. Training is
otherwise identical to the plain arm of the same size: MuSGD, 300 epochs,
patience 50, effective batch 64.

| Arm | Seeds | val R@P90 | test R@P90 | test small R@P90 | test AP50 | test P / R @ val thr | empty FAR | confuser FAR (FAs) |
|---|---|---|---|---|---|---|---|---|
| yolo26n-v2s-640-round2-distill | 3/3 (950745 h100-47, 950746 H100 NVL, 950747 A100-80) | 0.807 ± 0.009 | 0.733 ± 0.013 | 0.517 ± 0.026 | 0.845 ± 0.009 | 0.890 / 0.743 | 0.000 | 0.050 ± 0.016 (3, 3, **5**) |
| *plain yolo26n-v2s-640* | 3/3 | 0.788 ± 0.010 | 0.743 ± 0.003 | 0.534 ± 0.010 | 0.835 ± 0.006 | 0.912 / 0.732 | 0.003 | 0.041 ± 0.014 (**4**, 2, 3) |
| yolo26n-v2s-800-round2-distill | 3/3 (951222 s0/s2, 951220 s1) | **0.812 ± 0.003** | **0.811 ± 0.011** | **0.603 ± 0.022** | **0.893 ± 0.008** | 0.920 / 0.782 | 0.000 | 0.037 ± 0.008 (2, 3, 3) |
| *plain yolo26n-v2s-800* | 3/3 | 0.794 ± 0.006 | 0.780 ± 0.024 | 0.537 ± 0.051 | 0.860 ± 0.017 | 0.924 / 0.748 | 0.000 | 0.032 ± 0.016 (3, 1, 3) |

### Distill-640 vs plain-640 (3 seeds each, measured, training size)

| Metric | Distill | Plain | Δ |
|---|---|---|---:|
| val R@P90 | 0.807 ± 0.009 (0.815, 0.809, 0.797) | 0.788 ± 0.010 | **+0.019** |
| test R@P90 | 0.733 ± 0.013 (0.748, 0.722, 0.728) | 0.743 ± 0.003 | −0.010 |
| test P at val threshold | **0.890 ± 0.004** (all seeds < 0.90) | 0.912 ± 0.008 | −0.022 |
| test R at val threshold | 0.743 ± 0.016 | 0.732 ± 0.006 | +0.011 |
| test AP50 | 0.845 ± 0.009 | 0.835 ± 0.006 | +0.010 |
| test small R@P90 | 0.517 ± 0.026 | 0.534 ± 0.010 | −0.017 |
| test empty FAR | 0.000 | 0.003 | −0.003 |
| test confuser FAR (FAs) | 0.050 (3, 3, 5) | 0.041 (4, 2, 3) | +0.009 |

**Reading:**
- Distillation at 640 raises val recall by about 2 sd, but the gain does
  not reach test.
- Test R@P90 is −0.010, within the plain sd at the section-1 rule's
  granularity: Δ must exceed max(sd) = 0.013, so no improvement.
- Test AP50 and recall at the val threshold are about +0.01, but
  precision at the val threshold drops below 0.90 on every seed. The
  distilled model's val-chosen threshold transfers worse to test.
- Confuser false alarms are equal or one image worse, and s2 breaches
  the gate (5).

**Verdict:** at 640, distillation is not a measurable improvement over
plain on test. Val ranking alone would have picked it (val 0.807 vs 0.788),
which is exactly why promotion uses test and WS7's deploy-size scoring.
By the val seed rule, the distill-640 seed WS7 would export is s0 (val
0.815; val FARs 0.000 / 0.014).

Training cost (measured): 61 s/epoch on H100 NVL, 99 s on the h100-47 MIG
slice, 107 s on A100-80 (xgpj0, 3-hour slices), all at batch 64; 106–157
epochs. Plain-640 is ≈ 2× faster per epoch on comparable GPUs.

### Distill-800 vs plain-800 (3 seeds each, measured, training size)

| Metric | Distill-800 | Plain-800 | Δ |
|---|---|---|---:|
| val R@P90 | 0.812 ± 0.003 (0.813, 0.815, 0.808) | 0.794 ± 0.006 | +0.018 |
| test R@P90 | **0.811 ± 0.011** (0.819, 0.799, 0.816) | 0.780 ± 0.024 (0.779, 0.805, 0.756) | **+0.031** |
| test small R@P90 (of 147 boxes) | 0.603 ± 0.022 | 0.537 ± 0.051 | +0.066 |
| test AP50 | 0.893 ± 0.008 | 0.860 ± 0.017 | +0.033 |
| test AP50-95 | 0.529 ± 0.002 | 0.511 ± 0.008 | +0.018 |
| test P / R at val threshold | 0.920 / 0.782 | 0.924 / 0.748 | −0.004 / +0.034 |
| test empty FAs (of 125), per seed | 0, 0, 0 | 0, 0, 0 | 0 |
| test confuser FAs (of 73), per seed | 2, 3, 3 (mean 0.037) | 3, 1, 3 (mean 0.032) | +1 image in total |
| val empty / confuser FAR (mean) | 0.003 / 0.029 | 0.003 / 0.038 | |

**Section-1 rule, applied exactly:**
- **Recall:** the mean test R@P90 gain must beat the larger seed sd.
  Δ = +0.031 > max(0.011, 0.024) = 0.024: **pass**.
- **False alarms:** every distill seed is within both gates. Empty FAR 0
  (≤ 0.02) and confuser ≤ 3 of 73: **pass**.
- **Distillation beats plain at 800**, unlike at 640. Every accuracy metric
  improves: test R@P90, small recall (+0.066), AP50, AP50-95, and recall at
  the val threshold.
- Test precision at the val threshold stays ≥ 0.91 on every seed, so the
  640-size calibration problem does not recur.

**WS7 candidate (val only, PLAN section 1 seed rule):**
`yolo26n-v2s-800-round2-distill/seed1`.
- Val R@P90 0.8145 is the highest. Val FARs (empty 0.003, confuser 0.041)
  are within 0.02 / 0.05.
- Val operating threshold 0.585.
- Test, for reference only: R@P90 0.799, 0 / 3 false alarms.

**Cross-GPU resumes** (from `run.json.attempts`; batch fixed at 64 by
`batch_override`, so no batch change on resume):
- s1 started on an A100-40 (xgpg7, 951220) and resumed on the h100-47 MIG
  slice (xgpi13): the retargeted legacy pinned array.
- s0 resumed from h100-47 (xgpi13) to a full H100 NVL (xgpi15).
- s2 moved between h100-47 slices (xgpi12 → xgpi11).
- Epochs: 130 / 164 / 175, longer than plain-800's 106–147 before patience
  50 fired.

**800 distill arm** (`training/recipes/yolo26n-v2s-800-round2-distill.yaml`,
commit fd026c4):
- **Setup:** the generated 640 distill recipe with `imgsz: 800`. Its train
  block is identical to `yolo26n-v2s-800-round2-plain` except `dis: 6.0`.
  `batch_override: 64` matches plain-800's batch 64 × accumulate 1; without
  it the runner would pick 40 × 2 = effective 80 on h100-47.
- **Memory:** estimated ≈ 20.5 GB; **measured 20.3 GB** peak on s0 (xgpi13,
  h100-47, batch 64), at **151 s/epoch** (measured). Teacher staged as
  expected.
- **Placement:** the heavy policy's a100-80 slot was taken by distill-640
  s2, so the submission was split. Seed 1 → h100-96 (951220); seeds 0 and 2
  → h100-47 (951222, throttle 1 because 950745 holds the other h100-47
  slot).
- **ETA (estimated):**
  - Speed: ≈ 95 s/epoch on H100 NVL and ≈ 155 s/epoch on the h100-47 MIG
    slice (640 distill measured 61 s and 99 s, × 1.56 for the area).
  - Epochs: plain-800 stopped at 106–147.
  - Seed 1 ≈ 3–4 h after it starts. Seeds 0 and 2 ≈ 5–6 h each and run
    one after the other, so ≈ 10–12 h. A second h100-47 slot frees when
    distill-640 s0 finishes, but the array throttle is 1, so the coordinator
    may raise it with `throttle-arrays`.

## Round 3 (species gate): started early

**Coordinator decision, 15 Oct:** round 3 starts now instead of after
distillation. Reasons: 6 of 8 GPU slots were idle, and the single-taxonomy
comparison arm already has 3 completed seeds.

- **Recipe:** `training/recipes/yolo26n-v2s-species-800-r3.yaml`. It is
  identical to `yolo26n-v2s-800-round2-plain` except:
  - dataset `columba-train-v2s-species` (the same images), taxonomy
    `species`;
  - `batch_override: 16` (see the note on effective batch below).

  v2s rather than v2: the two tie on val within 0.01, and v2s is the
  distillation combination.
- **Array:** **950409**, 3 seeds, nv pool pinned to Titan V
  (`--constraint titanv`), throttle 3. This keeps the heavy slots for
  distillation.
  All 3 tasks run at ≈ 160 s/epoch, batch 16 (measured), and should be DONE
  in ≈ 5–7 h.
- **Effective batch:**
  - The single arm trained at batch 64 × accumulate 1 on H100/A100.
  - On Titan V the runner would pick batch 20 × accumulate 3 (effective
    60), so the recipe sets `batch_override: 16` × accumulate 4. That
    makes the effective batch and weight-decay scaling identical: 64.
  - The per-step BatchNorm batch (16 vs 64) and the GPU type are the
    remaining differences.
- **Test evaluation of a species model:**
  - Previously, WS1's predictor and evaluator rejected an 11-class model on
    the single-taxonomy `columba-test-v2`.
  - The runner now renames the **staged** test `data.yaml` names to the
    species list (`cluster.staging.align_test_taxonomy`). Pigeon stays at
    index 0 and the labels are unchanged.
  - Pigeon metrics and both FARs use only class-0 predictions plus the
    manifest, so they are directly comparable.
  - Checked with real yolo26n-v2 predictions plus 73 synthetic
    high-confidence non-pigeon predictions on confuser images: pigeon
    metrics and FARs were identical to the single-taxonomy evaluation.
- **Latency side of the gate** (≤ 5% slower): **passes, +1.9%**.
  - Method: xcng0 job 950426, WS4 `bench_ncnn.py gate`, fp16 NCNN at
    448×800, 4 pinned cores, 20 warm-up + 200 timed iterations, WS4's
    camera frame. Each head ran twice, interleaved.
  - The heads are sliced from the COCO yolo26n weights, so the backbones
    are identical: nc=1 keeps `bird`; nc=11 keeps bird plus 10 other
    classes. Export shapes were checked: (5, 7350) and (15, 7350).

  | Head | xcng0 inference p50 (measured) | xcng0 end-to-end p95 (measured) | Est. Pi 5 p95 |
  |---|---:|---:|---:|
  | nc=1 (single) | 26.5–26.6 ms | 31.0–31.5 ms | 47.3–48.0 ms (mean 47.7) |
  | nc=11 (species) | 26.9 ms | 31.8 ms | 48.6 ms (mean 48.6) |

  - The difference is +0.9 ms, or +1.9% at p95 (+1.2% at end-to-end p50),
    well within 5%.
  - C7 files: `experiments/latency/ws5-species/`.
- **Result:** see `experiments/reports/WS5-round3.md`. Species passes the
  recall and latency clauses but fails the strict test false-alarm clause
  by 1–2 images, so it is not shipped at training size (provisional; WS7
  decides at deploy size). No species teacher was trained.
