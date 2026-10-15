# WS7 — Export, quantisation and selection

- State: in progress
- Branch: `ws7/export`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-15

## Done

- Rectangular C8 exports and YOLO26 head swap; 36 fp32/fp16 packages for nine
  checkpoints completed on Mac. Sources are unchanged.
- Validation-only, deterministic 200-image NCNN KL calibration, with centred
  deployment letterbox and SHA-256 exclusion of test images.
- Native C3/C4 runner, explicit precision accuracy gate, compare_exports.py
  parity comparator, C7 timing, reaction simulation and three-seed selection.
- Additional-candidate CLI, styles scoring, immutable species test-taxonomy
  views using coordinator main 469c867, and provisional manifest preparation.
- Interim checks: 294 tests pass, seven extension-only skips. Ruff lint,
  formatting and strict mypy pass. Final checks will be repeated after packaging.

## In progress

Contracts v1.7 read. Building rectangular C8 export, validation-only NCNN
calibration and repeatable selection/evaluation commands. Initial matrix uses
all three seeds of yolo26n-v2s-800, yolo26n-v2-800 and yolo26n-v2s-640,
round2-plain. Both YOLO26 heads and fp16/int8 are included. Accuracy uses
columba_native package-default compute on xcng, with fresh val thresholds.
Final selection waits for distilled and species candidates. Packaging is a
provisional dry run only. WS5 jobs are read-only. No WS7 GPU job remains active.

ARM matrix job 950463 is running on xcng0, allocated/pinned cores 0-3.
The coordinator addition does not restart it. Mac CPU references are computed
with four threads. First reference passes all 1,408 ARM CPU val/test image
comparisons, maximum associated box delta below 0.001 px. Complete references
are published atomically only into cases that have not started.

Diagnostic T4 job 950509 was submitted by WS7 and cancelled by WS7 after
10 min 59 s when x86 GPU reference parity failed. Eleven of 20 sampled
preprocessed input hashes differed from ARM. GPU outputs remain diagnostics;
accepted references are CPU. No WS5 job was cancelled or modified.

## Blockers and requests

- No access blocker. Final promotion waits for distilled/species candidates.
- C7 request for the coordinator: document optional `omp_wait_policy`. On this
  NCNN/OpenMP build, PASSIVE more than doubles inference time. Selection
  requires a fresh ACTIVE C7, matching the intended deployed profile. Runtime
  launch should explicitly set `OMP_WAIT_POLICY=ACTIVE`; WS7 does not own the
  runtime launcher. Existing PASSIVE C7s remain diagnostics.
- Predeclared WS7 policies, for review before final promotion: at most 0.01
  loss in each val/test R@P90, AP50 and operating recall versus same-head
  rectangular PyTorch; each seed passes FAR/time/parity gates; otherwise
  identical v2/v2s arms within 0.01 val recall and 5% timing use styles recall
  to break the tie. Original test-v2 still supplies every promotion gate.

## Results

Interim, first yolo26n-v2s-800 seed, **measured** xcng0, 448x800, standard NMS
head: test R@P90 PyTorch/fp32 0.694051, fp16 0.688385, int8 0.331445;
AP50 export-minus-PyTorch deltas 0 / -0.001442 / -0.335211. Strict test parity
failed images 0 / 2 / 62. Standard head confuser FAR is 0.082192, one-to-one
FP32 is 0.054795, both above 0.05 at their own val thresholds. These
deploy-size numbers supersede the square training-size scores for gating.

**Measured** same ARM allocation/package: bridge FP16 extract p50 PASSIVE
53.678 ms vs ACTIVE 22.766 ms. Native ACTIVE extract p50 22.341 ms,
end-to-end p95 23.764 ms; **estimated** Pi p95 36.291 ms, factor 1.527110713.
Native ACTIVE/PASSIVE predictions are bit-identical on 319 test images in
each precision. Scratch evidence: `outputs/ws7/wait-policy/`,
`outputs/ws7/cpu-{gpu,mac}-parity-*.json`, `outputs/ws7/results/`.
