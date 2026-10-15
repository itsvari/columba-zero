# WS5 round 2: interim report (plain arms)

Status on 15 Oct 2026, about 16:45. 16 of 18 plain runs are DONE; the 2
pending seeds are marked below. Distillation (transition 949889,
`round2-distill.yaml`, top_k 1 → yolo26n-v2s-640 with the yolo26x teacher,
3 seeds) waits on those 2 seeds. This is **not** WS5-final.

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
| yolo26n-v2-640 | 2/3 (s0 pending, job 949457_0) | 0.783 ± 0.001 | 0.732 ± 0.006 | 0.511 ± 0.008 | 0.798 ± 0.001 | 0.909 ± 0.018 | 0.721 ± 0.022 | 0.000 ± 0.000 | 0.041 ± 0.039 | s0 pending, s1 1, s2 **5** |
| yolo26n-v2-800 | 3/3 | **0.796 ± 0.013** | 0.779 ± 0.022 | 0.537 ± 0.042 | 0.839 ± 0.012 | **0.931 ± 0.011** | **0.754 ± 0.011** | 0.003 ± 0.005 | **0.023 ± 0.016** | s0 1, s1 1, s2 3 |
| yolo11n-v2s-640 | 3/3 | 0.780 ± 0.027 | 0.736 ± 0.008 | 0.517 ± 0.011 | 0.828 ± 0.021 | 0.927 ± 0.008 | 0.703 ± 0.025 | 0.000 ± 0.000 | 0.037 ± 0.016 | s0 **4**, s1 2, s2 2 |
| yolo11n-v2s-800 | 2/3 (s1 pending, job 947419_4) | 0.787 ± 0.009 | 0.756 ± 0.004 | 0.497 ± 0.000 | 0.844 ± 0.020 | 0.920 ± 0.002 | 0.735 ± 0.002 | 0.008 ± 0.011 | 0.055 ± 0.000 | s0 **4**, s1 pending, s2 **4** |
| *baseline-repro-p50* | 3/3 | 0.473 ± 0.066 | 0.246 ± 0.035 | 0.000 ± 0.000 | 0.604 ± 0.031 | 0.853 ± 0.035 | 0.308 ± 0.053 | 0.000 ± 0.000 | 0.242 ± 0.105 | s0 16, s1 11, s2 26 |

Round-2 IDs end in `-round2-plain` (generated recipes in
`training/recipes/generated/round2-plain/`).

Arms with 2 seeds show n=2 statistics; their sd is not comparable with the
3-seed arms.

For reference, the teachers (1 seed, 960 px) score val R@P90 0.845 / 0.822
and test R@P90 0.861 / 0.870 (yolo26x / yolo11x). This shows how much
accuracy headroom distillation could transfer.

## Section-1 comparison against `baseline-repro-p50`

Rule: the 3-seed mean test R@P90 must exceed the baseline mean (0.246) by
more than the larger of the two seed sds.
- **Every arm passes by a wide margin.** The smallest margin is
  yolo26n-v2-640 (0.732, n=2), about 0.49 above the baseline, against a bar
  of about 0.04.
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
- yolo11n-v2s-800 s0 (4) and s2 (4): **2 of 2 finished seeds**, mean 0.055
  (s1 pending).

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
- yolo26n is ahead of yolo11n at both sizes on test R@P90, AP50 and FAR.
  The gap is within about one sd at 640.
- The empty-image share in training (63% backgrounds) shows no sign of
  hurting recall. Val recall at P90 ≈ 0.79 with empty FAR ≈ 0, so no
  subsampling is needed for now (WS2's suggestion).

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
- The accuracy and false-alarm parts of the gate are decided when the 3
  species seeds finish (next check-in). No species teacher will be
  trained.
