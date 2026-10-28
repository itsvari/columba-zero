# Independent verification and zone handover

Status: paused at the user’s request on 29 October 2026. The current branch is
`feat/independent-verification-and-zone`. These changes have not been merged.
The requested independent Opus review was cancelled during immediate wrap-up;
no independent review of this new implementation is claimed.

## Implemented

- Independent stimulus playlist with distinct approved pigeon assets, continuous
  12-second acquisition, source photograph-area ratios 0.64/2.56, brief and
  sustained challengers, frozen response allowances and paused calibration poses.
- Required-selection versus allowance semantics through preparation/scoring.
  Allowances remain unresolved, with unknown centring time retained. Nested
  qualification evidence is checked again before output.
- Optional per-camera exposure compensation, default unchanged. Unsupported or
  out-of-range requests fail clearly. The camera-only diagnostic found −1.5 EV
  restored detail lost to automatic exposure on the dark field.
- Isolated zone geometry, conservative membership, camera/hash-bound PNG reference
  artefacts and masked ORB registration candidates with held-out screening.
  The 54 focused zone tests pass. These modules are not connected to production
  following; `zone.enabled` remains rejected. No physical error bound is inferred
  from feature residuals.

## Verification and retained results

The tested runtime revision is `b0d891e`; subsequent changes are documentation.
Desktop: 936 tests and 387 subtests passed, six optional skips. Pi: 934 tests and
391 subtests passed, eight optional skips. Desktop strict mypy passes 138 files;
Ruff lint/format passes 141 files. Logs and hashes are listed in the
[evidence manifest](independent-following-20261028-evidence.json).

| Camera-only pilot | Raw JPEGs | Strict optical associations | Retained rejects |
|---|---:|---:|---:|
| Pale field, default exposure | 250 | 246 | 4 |
| Dark field, default exposure | 250 | 178 | 72 |
| Dark field, −1.5 EV | 250 | 247 | 3 |

All pilots had following disabled, simulated pan/tilt configuration, child exit
0, one TERM, no KILL and both servo outputs FULL_OFF. No new servo movement was
performed in this resumed campaign. The dark field improved photo-edge contrast
but automatic exposure washed out interior features; controlled exposure restored
visible detail. This is calibration evidence, not a tracking/performance pass.

The independent Terra audit of the controlled-exposure samples passed 15/16
standing and 3/4 profile photo-corner checks under the declared criterion:
maximum nominal residual plus sqrt(2) times the subjective 3 px per-axis camera
allowance must not exceed 7 px. Failures are retained. These are unpowered
rest-pose observations only. They do not qualify full source-edge uncertainty,
interior extrapolation, comparable-area bounds or moving-camera behaviour.

Annotations were saved before matcher output. Terra did not inspect detector
metadata. Sample selection used optical/frame identity; root did see operational
detector metadata in an early diagnostic print, so root is not claimed blind.
Raw media and detailed annotations remain locally in
`outputs/validation/independent-following-20261028/`; committed hashes preserve
identity without publishing raw recordings or private calibration.

## Resume points

1. Independently review the new source changes and retained pilot evidence.
2. Choose a defensible scoped calibration or per-frame manual-pixel truth method;
   avoid transplanting the old pose’s qualification or globally promoting assets.
3. Complete independently supported source uncertainty and powered-pose coverage,
   then freeze a new confirmation policy and collect fresh following evidence.
4. After that gate, implement reference drawing, runtime registration lifecycle,
   selector/controller integration and simulation. Offline zone foundations alone
   do not constitute the working fixed-zone feature.
5. Keep physical-zone validation deferred: the user currently has no flat textured
   surface and explicitly requested software/offline checks first.

The earlier accepted demonstration and its frozen failed hour assessment remain
unchanged. No new centring, selection, zone or global acceptance pass is claimed.
During wrap-up the final read-only Pi check found no camera owner and registers
9/13 both 16 (FULL_OFF). Owned Firefox test profiles and SSH master were closed.
