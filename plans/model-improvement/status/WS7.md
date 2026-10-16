# WS7 — Export, quantisation and selection

- State: complete
- Branch: `ws7/export`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-16
- Contracts: v1.9; based on coordinator main `47a9b8b`

## Done when

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
