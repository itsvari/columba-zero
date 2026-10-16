# WS7 — Export, quantisation and selection

- State: PTQ study complete; stock EQ unavailable; export work complete
- Branch: `ws7/quant`, created from local main `e6e12be`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-16
- Export contracts: v1.9; original selection based on coordinator main `47a9b8b`

## Quantisation follow-up

Stage 1 is complete for plain-800 seed1 at 448x800, yolo11 decode. The
coordinator's int8 exclusion remains in force. No deployment files or the
`ws7/export` branch changed. Stage 2 is a proposal only; no GPU work started.

- **Measured xcng:** KL/200 reproduces the previous binary byte for byte and
  val/test R@P90 0.529309 / 0.322946. All 102 convolution layers are quantised.
- **Established cause:** KL clips the first RGB input at 0.447998, almost pad
  114/255. In 1,000 val letterboxes this clips 37.31% of all channel values,
  53.38% of non-padding values. Native/calibration letterboxes match pixel for
  pixel. Disabling the redundant resize gives an identical calibration table.
- **Measured intervention:** fixing only the known input range to [0,1]
  recovers val/test R@P90 to 0.746282 / 0.637394. Its **estimated Pi** p95 is
  33.73 ms. Keeping only the first convolution in float gives
  0.743657 / 0.637394 and 32.41 ms estimated.
- KL and ACIQ were measured with 200 and 1,000 val images. Stock EQ crashed
  with SIGSEGV at both counts. Full failure logs and source observations are
  retained; no kernel or pinned NCNN dependency was changed.
- All 33 completed configurations have native C4/C7, fresh val thresholds,
  FARs, six accuracy deltas and layer coverage audited. None of the 31 int8
  or mixed configurations passes the <=0.01 accuracy policy.
- **Best faster configuration by val only:** `kl-200-first-body15-body39`,
  25 float and 77 int8 convolution layers. Val/test R@P90
  **0.777778 / 0.648725**, threshold **0.5546875**, val empty/confuser FAR
  **0 / 0.018349**, test FAR **0 / 0.054795**. **Estimated Pi p95 33.81 ms**.
  It fails accuracy and test confuser FAR; no test-driven alternative was picked.
- Fresh fp16 reference is **measured** at val/test R@P90 0.796150 / 0.691218,
  with **estimated Pi p95 36.49 ms**, consistent with supplied 36.54 ms.
  Optimised fp32 has zero accuracy deltas and passes strict parity.

`tools/quantize_ncnn.py` now exposes methods, count, exact float-layer
exclusions, provenance-checked table reuse, known activation ranges, no-resize
calibration and a timeout. `tools/quantization_study.py` runs calibration,
sensitivity, native evaluation, ACTIVE-policy C7 and reference comparisons.
The report gives exact CLI commands. A guard now rejects the old GPU/batch-16
reference cache; final comparisons all use published CPU/batch-1 predictions.

Artefacts: `experiments/reports/WS7-quant.md`, `WS7-quant-results.json` and
`WS7-quant-evidence/`, plus 37 diagnostic/failure rows in the leaderboard.
Raw predictions and packages remain under `~/columba/scratch/ws7-quant/`
and local `outputs/ws7-quant/`. Source snapshot receipts and every job are
recorded. Jobs 959718, 959719, 959731, 959798 and 959814 completed. Earlier
959716/959717 stopped at staging before calibration. No jobs remain in flight.
All runs used ARM CPUs, at most two project CPU jobs, four cores per job,
nice 1000, node-local staging and ACTIVE waiting. No password was requested,
used or stored.

Final checks: **332 pytest tests and 233 subtests passed, zero skipped;
332 unittest tests passed; Ruff check and format check passed on 96 files;
full mypy passed on 93 source files; git diff --check passed.**

Recommendation: keep fp16. A short, NCNN-matched QAT proposal is in the report,
with a one-seed pilot estimate of at most one H100-equivalent GPU-hour.
The coordinator must obtain the user's go-ahead before any QAT job. Stock
EQ's repaired implementation remains unmeasured. No contract change is needed.

## Export done when

Met. Final selection covers 24 configurations and 72 seed/export cases.
The selected model is durably packaged, its manifest and C8 metadata verify,
the authorised deployment config is set, and post-config full checks pass.
No merge into main was performed.

Selected: `yolo26n-v2s-800-round2-plain/seed1/yolo11/fp16`, 448x800.
The val-derived threshold is 0.62646484375. This is also the plain fallback,
so one package serves both roles without a duplicate manifest entry.

## Selection and gates

Every arm uses the same deploy-size val rule: highest R@P90 among val FAR
passes, then lower confuser FAR, lower empty FAR, higher AP50 and lower seed
index. This rule was confirmed before inspecting deploy-size test results
for switched seeds. Arm ranking uses three-seed val mean; the section-1
improvement rule uses three-seed test mean. Other gates apply to the chosen
export alone. No test-driven seed fallback occurred.

| Arm | WS5 square-val seed | Deploy-val seed | Final result |
|---|---:|---:|---|
| Distill-800 | 1 | 0 | held: test confuser FAR 0.054795 >0.05 |
| Plain-800 | 1 | 1 | selected; all gates pass |
| Species-800 | 2 | 2 | held: relative empty FAR 0.008 >plain's zero |
| Control-b16 | 2 | 0 | held: confuser FAR 0.068493 and fp16 R@P90 loss 0.014164 |

Species seeds 0 and 1 are val-ineligible at confuser FAR 0.050459. The selected
plain seed passes val empty/confuser FAR 0/0.032110 and test FAR 0/0.041096.
Its measured native test R@P90 is 0.691218. Its three-seed test mean is
0.685552 +/-0.007495, versus rectangular native fp16 baseline
0.221907 +/-0.013084. Strict same-graph fp32 parity passes; maximum fp16 loss
across the six accuracy metrics is 0.00000512. Strict fp16 image parity is
diagnostic. Int8 and one-to-one remain excluded; fp32 is a parity companion.

Accuracy and FAR are **measured** on xcng. The selected Pi frame p95 is
**estimated** at 36.543667 ms using WS4 x1.527110713. Reaction p95 is
**simulated** at 345.363965 ms with 30 FPS capture and k2/n3. WS4 already
serialises processing and drops stale captures; the estimated effective
detector rate is 27.446 FPS. No 30-detections-per-second assumption is made.
**WS10 step 3 must confirm reaction p95 on the physical Pi.**

`WS7-final.md` has complete section-1 gate tables for all six deployment
candidates, plus all 18 excluded diagnostic configurations. It records every
seed's FAR, both square and deploy val nominations, precision deltas, styles
scores, species 5.4 gates, the rectangular baseline and geometry caveat.

## Package and configuration

Tracked package:
`runs/detect/pigeon_detector/yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16/weights/best_ncnn_model/`.
Runtime copy: `models/yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16`.
`deploy/model_manifest.json` contains file SHA-256s/byte counts and a durable
repository-relative source. No worktree or ignored-output source remains.
The 4,885,148-byte fp16 binary follows the existing direct-Git convention,
with exact-byte attributes and no Git LFS filter. Existing baseline files
remain unchanged. Clean-clone checksum tests pass with autocrlf false and true.

`app/config.yaml` selects native inference at 448x800 with the val threshold,
confirmation enabled at k2/n3 and an 800x450 lores stream at 30 FPS.
Recording FPS remains 15. All requested keys already existed; other defaults,
schema and launcher remain unchanged.

`prepare_models.py` and C8 `check_model` pass. A fresh native replay processed
20 frames and emitted one confirmed trigger. This is functional verification,
not a live-camera reaction measurement.

## Final validation

All checks passed after the deployment config edit, using the current Mac
native extension. No package versions changed.

- Full pytest: **325 passed, 223 subtests passed, zero skipped**.
- Full unittest discovery: **325 passed**.
- Native CTest: **1 passed**.
- Ruff check on `app tests tools cluster training native`: **pass**.
- Ruff format check on those directories: **94 files already formatted**.
- Full mypy: **91 source files, no issues**.

Two fixtures still assumed the old default deployment. The camera default
test now loads an empty config to verify schema defaults. The warm-up test
injects failure through the backend-neutral factory and asserts that warm-up
ran before capture/output startup. Production behaviour was unchanged.

## Artefacts and resources

- `experiments/reports/WS7-final.md`: final decision, every gate and hand-off.
- `WS7-final-selection.json`: reproducible selector output for all 72 cases.
- `WS7-final-candidates/`: C4/C7, C3 sidecars and provenance for the 18 new cases.
- `WS7-final-audit.json`: all 18 new parity reports reproduced; package bytes,
  native compute, val thresholds and calibrated latency samples checked.
- `WS7-final-policy.json`: coordinator seed rule and WS5 nominations.
- `WS7-final-preparation.json`: completed jobs and source checkpoint hashes.
- `WS7-final-verification.json`: current replay, check counts and log hashes.
- `WS7-methods.md`: exact candidate addition, selection and verification commands.
- Existing `WS7-candidates/`, `WS7-platform/`, `WS7-audit.json`, `WS7-baseline/`
  and `WS7-aspect.{json,md}` retain the earlier evidence. The 15 FPS reaction
  sensitivity is historical; capture is now authorised and configured at 30.

Jobs 958128, 958260 and 958304 completed and were fetched before the pause.
No jobs remain in flight. Finalisation required only the requested successful
BatchMode connection check; no cluster jobs were submitted or changed.
No password was requested, used or stored.

## Remaining work outside WS7

No contract change or selection decision remains open. WS10 step 3 must
measure real Pi reaction. Steps 5–7 remain open for site threshold/k-n,
OMP wait policy and on-Pi geometry. Portrait images account for 38 of 39 lost
full-test P90 matches in the shape diagnostic. Test-v2 has only nine roughly
16:9 images, two positive; `columba-test-site-v1` is the real camera check.
