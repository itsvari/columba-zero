# WS7 — Export, quantisation and selection

- State: provisional scope complete; selection HOLD
- Branch: `ws7/export`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-15
- Contracts: v1.7; coordinator main 469c867 merged

## Done when

Steps 1–4 and the step-5 diagnostic dry run are complete. All requested plain
finalists have three-seed rectangular measurements, both heads, fp16/int8,
and an additional fp32 reference export. No configuration passes every gate,
so there is no admissible provisional pick. The packaged diagnostic is
explicitly ineligible. Final promotion waits for distilled/species results.

## Implementation and evidence

- `tools/export_ncnn.py`: rectangular `--imgsz H W`, C8 metadata, both YOLO26
  heads, source checkpoint hashes and unchanged source weights.
- `tools/quantize_ncnn.py`: NCNN KL calibration on the ARM lane, exactly 200
  deterministic val images per export, recorded IDs/SHA-256s and test exclusion.
- `tools/select_model.py`: CPU PyTorch references, deployed native C3/C4,
  independent val thresholds, explicit precision gate, compare_exports parity,
  native C7/WS4 bridge timing, WS4 reaction simulation, styles tie-break,
  pigeon-class-0 species comparison, guarded candidate addition and packaging.
- Nine checkpoints: all seeds of yolo26n-v2s-800, yolo26n-v2-800 and
  yolo26n-v2s-640, round2-plain. Deploy sizes are 448x800 and 384x640.
- All 54 exports scored on val, test-v2 and styles. All 18 PyTorch head/seed
  reference sets scored on the same splits. Native accuracy uses xcng0;
  native styles and ACTIVE timing use xcng0/xcng1. CPU references use Mac
  vexcalibur, plus ARM for the first seed's val/test references. The first
  CPU cross-host comparison passes all 1,408 val/test images, box delta <0.001 px.
- `experiments/reports/WS7-provisional.md`: all candidate metrics, deltas,
  thresholds, rejected gates and package verification.
- `experiments/reports/WS7-selection.json`: all configurations and Pareto
  fronts; `selected` is null and the admissible front is empty.
- `experiments/reports/WS7-candidates/`: portable C4/C7, parity, accuracy and
  provenance evidence. `WS7-platform/` contains controlled platform checks.
- `experiments/reports/WS7-audit.json`: all 54 parity reports reproduced;
  actual model hashes verified; all 18 calibrations have 200 val images and
  zero frozen-test overlap. Raw predictions/models remain in `outputs/ws7/`
  and `~/columba/scratch/ws7/` on the cluster.
- `experiments/reports/WS7-methods.md`: exact add/fetch/select/audit/package
  commands, species handling and package retrieval.

## Labelled results

**Measured**, rectangular standard-head three-seed test R@P90:

| Arm | CPU PyTorch / native fp32 | Native fp16 | Native int8 |
|---|---:|---:|---:|
| v2s-800 | 0.6884 | 0.6856 | 0.3258 |
| v2-800 | 0.6638 | 0.6648 | 0.2984 |
| v2s-640 | 0.6563 | 0.6534 | 0.6412 |

Across both heads, explicit precision accuracy passes fp32 18/18, fp16 15/18,
int8 0/18. Strict parity passes 18/18, 0/18, 0/18 respectively. FP16 has 87
failed image-case comparisons, including 52 count mismatches. These are the
actual deployed arithmetic results, not FP16 storage scored with FP32 compute.

**Measured** native styles R@P90, standard fp16: v2s-800 0.5219 versus v2-800
0.3582. **Estimated** worst-seed Pi frame p95: fp32/fp16/int8 v2s-800
69.44/36.79/33.69 ms; v2s-640 46.75/25.19/23.55 ms, using WS4 x1.527110713.
**Simulated** reaction p95: v2s-800 609.29/345.36/882.96 ms; v2s-640
418.90/300.08/322.59 ms. Still-image operating recall is the per-frame proxy;
independent temporal hits are assumed. No Pi measurement is claimed.

One-to-one fp16 loses 0.0283–0.0293 test R@P90 versus NMS across the three
arms, with <0.08 ms estimated frame-p95 difference. It offers no useful
latency gain here. Confuser FAR rejects seeds in every fp32 configuration;
fp16 parity and int8 accuracy independently prevent promotion.

These rectangular results decide the gates. Square training scores are
context only. Comparison baseline remains WS5 baseline-repro-p50, measured
training-size test R@P90 0.2455 +/- 0.0346, as instructed.

## Package and resources

The C8/hash dry run packages v2s-800 seed1, NMS, fp32 as
`yolo26n-v2s-800-round2-plain-seed1-yolo11-fp32-provisional`.
`prepare_models.py --model-id ID` verifies its actual SHA-256s and byte counts.
The manifest marks it provisional, `eligible: false`, `blocked-by-gates`.
The incumbent manifest fields are unchanged. The package is archived at
`~/columba/scratch/ws7/packages/` for coordinator retrieval.

Both own CPU jobs completed with exit 0: matrix 950463, xcng0, 2:18:44;
supplement 950885, xcng1, 1:29:02. Diagnostic T4 job 950509 was cancelled by
WS7 after 10:59 when GPU references failed CPU parity. Eleven of 20 sampled
preprocessed inputs differed. GPU references are excluded. No WS5 job was
changed or cancelled. No WS7 job or Mac background process remains running.

Final checks: 298 tests OK, seven expected native-extension skips; Ruff lint
and formatting pass across app/tests/tools/cluster/training/native; strict
mypy passes for 88 source files. Logs are in `outputs/ws7/final-tests.log`.

## Blockers and requests

- No access blocker. Keep selection HOLD; rerun when coordinator confirms the
  distilled and species candidates are ready. Species execution is supported
  and tested, but no unfinished species checkpoint was scored.
- C7 request: document optional `omp_wait_policy`. Same-package ARM extract
  p50 was 53.678 ms PASSIVE versus 22.766 ms ACTIVE. Predictions are bit-identical
  on all 319 test images in each precision. Set `OMP_WAIT_POLICY=ACTIVE` in the
  deployed launcher; WS7 does not own it. PASSIVE C7s remain diagnostics.
- Coordinator policy review before final promotion: maximum 0.01 loss in each
  val/test R@P90, AP50 and operating recall versus same-head rectangular PT;
  each seed passes FAR/time/parity; paired matching v2/v2s arms within 0.01 val
  recall and 5% latency use styles recall. Test-v2 remains the primary gate.
- No CONTRACTS.md edit or dependency-version change was made.
