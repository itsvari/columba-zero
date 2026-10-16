# WS7 — Export, quantisation and selection

- State: in progress, tunnel restored and final ARM bundles resumed
- Branch: `ws7/export`
- Agent: gpt-6.1-sol via T3 delegate
- Last updated: 2026-10-16
- Contracts: v1.9; rebased onto coordinator main `47a9b8b`

## Done when

Final selection is **not complete**. The selector, final report generator and
durable packaging workflow are ready. The validated plain fallback is tracked,
verified and replayed. The new primary selection and production config remain
pending the required native evidence.

- Final-turn step 1: distill/species checkpoints and all six fp32/fp16 exports
  per arm are prepared at 448x800, yolo11 decode. All three CPU PyTorch seeds
  have fresh val thresholds and test/styles scores. Distill ARM job958128 was
  submitted; species submission failed before any job was submitted. Control
  has not started. Int8 and one-to-one remain excluded.
- Step 2: blocked. No final winner, all-four ranking or completed native seed
  comparison is claimed. `WS7-final.md` records the blocker and existing
  fallback evidence. `--final` selection emits all gates, species 5.4 and
  nomination-versus-deploy seed comparisons when the evidence is available.
- Step 3: plain-800 seed1 fallback is durable and verified. Its source and
  manifest contain no ignored-output dependency. Final primary packaging waits.
- Step 4: production `app/config.yaml` and launcher are unchanged. A scratch
  config tested native 448x800, fresh val threshold, 800x450 lores settings and
  k2/n3 confirmation. It processed 20 replay frames with one confirmed trigger.
- Step 5: full pytest, unittest, Ruff lint/format and mypy pass. Current native
  CTest and native/runtime tests also pass after rebuilding the Mac extension
  from merged source with existing dependencies. No package versions changed.

## Blocker and resources

Species checkout publication failed with `Connection timed out during banner
exchange` and `Connection to 127.0.0.1 port 22022 timed out`. The following
BatchMode SSH check returned `ssh: connect to host 127.0.0.1 port 22022:
Connection refused`. Cluster work stopped under the coordinator's explicit
instruction. No password was requested or used. No subsequent SSH attempt was
made after the failed check.

The only submitted job this turn is `958128`, xcng0, CPU only. Last successful
observation showed distilled seed0 and seed1 fp32/fp16 complete. Its current
state and all-seed/styles completion are unknown after access failed. It was
not cancelled. Species has no submission receipt. No GPU was used; no WS5 or
other worker's job was changed. Project CPU cap stayed at two.

After connectivity is restored: fetch the completed distill bundle, submit the
cached species bundle, add the control, collect evidence and select all arms.
Exact commands and resource limits are in `experiments/reports/WS7-methods.md`.
The distill receipt is `outputs/ws7-final/submissions/` and the cached exports,
checkpoints and references are under `outputs/ws7-final/`.

## Confirmed seed policy

The coordinator confirmed deploy-size val overrides WS5 square-val nominations
before deploy-size test results for switched seeds were inspected. Every arm
uses identical val eligibility and ties: highest R@P90, lower confuser FAR,
lower empty FAR, higher AP50, lower seed index. Test gates apply the chosen
export only; no test-driven fallback. Three-seed arm rules are unchanged.
`WS7-final-policy.json` preserves this decision and the square nominations.

Measured Mac CPU previews, not deployment selection: distilled val R@P90
seeds0/1/2 = 0.810149 / 0.805774 / 0.797025; species = 0.808399 / 0.819773 /
0.813648. Native fp16 selection must still use the ARM evidence.

## Durable fallback and existing gates

`yolo26n-v2s-800-round2-plain/seed1/yolo11/fp16`, 448x800, threshold
0.62646484375 remains the validated fallback. Native accuracy/FAR are
**measured** on xcng0; frame p95 is **estimated** Pi and reaction **simulated**.
Its test R@P90 is 0.691218; three-seed test mean 0.685552 +/-0.007495 beats
rectangular native baseline 0.221907 +/-0.013084. Test empty/confuser FAR is
0 /0.041096. Estimated Pi p95 is 36.543667 ms; simulated 30 FPS k2/n3 reaction
p95 is 345.363965 ms. Strict fp32 parity and fp16 accuracy pass; maximum loss
across six metrics is 0.000005. Strict fp16 image parity is diagnostic.

The unchanged camera default is 15 FPS. A separate simulated sensitivity gives
fallback reaction p95 582.276430 ms at 15 FPS. This is reported, not substituted
for the registered 30 FPS gate. WS10 must reconcile actual FPS with reaction.

Tracked source:
`runs/detect/pigeon_detector/yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16/weights/best_ncnn_model/`.
The 4,885,148-byte fp16 binary follows the incumbent's direct Git convention;
no LFS filter is configured. `deploy/model_manifest.json` now has a named
`fallback_models` entry with hashes, no absolute/worktree/outputs source, and
retains the incumbent default until final selection. `prepare_models.py`
verifies both the default and named fallback. Clean-clone tests cover both
with autocrlf false and true. File hashes are in `WS7-final.md` and the manifest.

## Artefacts and verification

- `experiments/reports/WS7-final.md`: honest blocked final-turn report, existing
  gates, durable package hashes, geometry caveat and resume scope.
- `WS7-final-policy.json`: coordinator seed policy and WS5 nominations.
- `WS7-final-preparation.json`, `WS7-final-references/`: checkpoint SHA-256s,
  CPU preview values, C4 and C3 provenance for prepared arms.
- `WS7-final-verification.json`: measured Mac native replay and checks.
- `WS7-fallback-reaction-15fps.json`: labelled simulation sensitivity.
- Existing `WS7-provisional.md`, `WS7-selection.json`, `WS7-candidates/`,
  `WS7-platform/`, `WS7-audit.json`, `WS7-baseline/` and `WS7-aspect.{json,md}`
  retain the original 54-case matrix, baseline and geometry evidence.

Validation: pytest **322 passed, 228 subtests passed**; unittest **322 passed**;
CTest **1 passed**; Ruff check and format on `app tests tools cluster training
native` pass, **94 files formatted**; mypy **91 source files** passes. Native
extension tests ran, with the newly built current Mac extension. The real
fallback replay passed C8 `check_model`, loaded native compute, processed
20 frames and emitted one confirmed trigger. These are functional checks,
not measured Pi performance.

No contract change is requested. WS10 steps 5–7 remain open: site threshold/
k-n, deployed Pi OMP wait policy and capture/model geometry. Portrait images
account for 38 of 39 lost full-test P90 matches in the prior plain PyTorch
shape diagnostic. Frozen test-v2 has only nine approximately 16:9 images,
two positive; `columba-test-site-v1` is the real check.
