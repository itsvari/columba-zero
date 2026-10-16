# WS7 — Export, quantisation and selection

- State: in progress, final deploy-size candidate evaluation
- Branch: `ws7/export`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-16
- Contracts: v1.9; rebased onto coordinator main 47a9b8b

## Done when

The selected provisional model is packaged, its manifest verifies with
`prepare_models.py`, and the report covers all 54 current exports. The
coordinator's 15 October export and shipped-seed rules are implemented.
Final selection still waits for distilled and species candidates.

Selected: `yolo26n-v2s-800-round2-plain/seed1/yolo11/fp16`, 448x800.
Manifest ID: `yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16-provisional`.
The entry is provisional and eligible, with `promotion_status: dry-run-only`.
The incumbent manifest fields remain unchanged. No launcher or capture setting
was changed. The earlier blocked fp32 entry was replaced by this pick.

## Revised selection and gates

Arms rank by three-seed mean val R@P90. Select the highest-val-recall seed
among val FAR passes, then gate that export alone. Never retry another seed
because of test results. All seed FARs remain reported. FP16 parity requires
strict same-graph fp32 parity plus the confirmed precision accuracy policy.
Strict fp16 per-image parity remains diagnostic. Int8 and one-to-one heads
remain in the report but are excluded from deployment. The confirmed styles
tie-break remains within 0.01 val recall and 5% latency.

| Gate | Selected value | Limit | Result |
|---|---:|---:|---|
| Three-seed test improvement | 0.463645 | > larger SD 0.013084 | pass |
| Val empty / confuser FAR | 0 / 0.032110 | <=0.02 / <=0.05 | pass |
| Test empty / confuser FAR | 0 / 0.041096 | <=0.02 / <=0.05 | pass |
| Estimated Pi frame p95 | 36.543667 ms | <=100 ms, target70 | pass |
| Simulated reaction p95 | 345.363965 ms | <=500 ms | pass |
| Maximum precision accuracy loss | 0.000005 | <=0.01 in all six metrics | pass |
| Export parity | strict fp32 pass + accuracy pass | both | pass |

Accuracy/FAR are **measured** via native package-default compute on xcng0.
Latency is **estimated** Pi from xcng0 ACTIVE C7 using WS4 x1.527110713.
Reaction is **simulated** with WS4 at 30 FPS, k2/n3, 20,000 trials, seed42,
independent hits and still-image operating recall as the frame-recall proxy.
The val threshold is 0.62646484375. Selected-seed test R@P90 is 0.691218;
the arm's test mean is 0.685552 +/- 0.007495. Strict fp16 parity fails two
image comparisons; this is diagnostic under the revised rule.

The val-chosen v2-800 seed2 fails test confuser FAR 0.082192. The val-chosen
v2s-640 seed1 fails 0.054795. Neither arm falls back to another seed.

## Rectangular baseline and geometry

New exports were limited to baseline-repro-p50, all three seeds, standard
head, fp32/fp16, 384x640. CPU PyTorch references are **measured** on Mac
vexcalibur; native values on xcng0. Every baseline export independently
selects its threshold on val. Native fp32 strictly matches PT for all three
seeds; all baseline fp16 exports pass the precision policy.

| Format | Test R@P90 seeds 0 / 1 / 2 | Mean +/- SD |
|---|---|---|
| PyTorch / native fp32 | 0.232295 / 0.229462 / 0.206799 | 0.222852 +/- 0.013974 |
| Native fp16 | 0.229462 / 0.229462 / 0.206799 | 0.221907 +/- 0.013084 |

Selection now uses native fp16 rectangular baseline C4, matching the pick's
arithmetic. It no longer uses square leaderboard 0.2455 +/- 0.0346.

PyTorch-only, same checkpoint and Mac CPU implementation: seed1 test R@P90
is 0.801700 at 800x800 versus 0.691218 at 448x800. At respective full-test
P90 cuts, portrait images account for 38 of 39 lost matches; 4:3 accounts for
one; square/wide matches are unchanged. Portrait content scale drops to
0.56. The fixed square-defined small cohort also loses recall, so the effect
is more than changing the WS1 small-object denominator. The geometry
hypothesis is supported for this checkpoint, with padding/sampling changes
also present. The wide bucket has only two positive images, 17 boxes.

`WS7-aspect.md` reports every bucket's R@P90, WS1 small recall/counts and fixed
small-cohort recall. No gate or operating threshold was changed by this
analysis. Existing `app/config.yaml` and CameraConfig default to 640x480;
WS10 should verify the actual deployed 16:9 capture stream before relying on
the scale-match argument. Capture configuration stays unchanged.

Int8 note: full-graph KL quantisation, including the head, may be sensitive
to activation ranges. Both sizes used the same 200 val IDs and RGB /255.
Different 800/640 checkpoints confound a pure shape comparison. No specific
root cause is established; no further investigation was performed.

## Artefacts and resources

- `experiments/reports/WS7-provisional.md`: all candidates, seed FAR context,
  chosen-seed gates, baseline, aspect breakdown and quantisation note.
- `experiments/reports/WS7-selection.json`: reproducible revised decision.
- `experiments/reports/WS7-baseline/`: PT/native fp32/fp16 C4, C3 sidecars,
  parity/accuracy checks and aggregate measurements.
- `experiments/reports/WS7-aspect.{json,md}`: geometry diagnostic and interpretation.
- `experiments/reports/WS7-candidates/`, `WS7-platform/`: existing evidence.
  All 129 C7 records explicitly contain ACTIVE or PASSIVE under contracts1.8.
  Legacy bridge/PASSIVE metadata came from recorded job environments; no
  timing sample was changed.
- `experiments/reports/WS7-audit.json`: 54 parity reports replayed, package
  hashes and all 18 validation-only 200-image calibrations verified again.
- `experiments/reports/WS7-methods.md`: exact add/fetch/select/package and
  geometry replay commands. Future selection defaults to rectangular baseline.
- Raw new C3/C8 evidence: `outputs/ws7/{baseline-native,revision-references,
  revision-exports,aspect,packages}/` and the same folders under cluster
  `~/columba/scratch/ws7/`. No weights or raw prediction arrays were committed.

CPU job 951221 completed exit0 on xcng0 in 2:10, four cores. No GPU was used
this turn. No WS5 job was changed or cancelled. No WS7 job or Mac reference
process remains active. Final checks: 303 tests OK, seven expected native
extension skips; Ruff lint/format and strict mypy pass for 88 source files.

## Remaining decisions

No contract request remains: C7 wait policy is in v1.8. WS10 decides ACTIVE
versus PASSIVE with the complete pipeline on the Pi and verifies the deployed
capture geometry. Distilled/species evaluation and final promotion wait for
the coordinator's follow-up. The current package is provisional only.
