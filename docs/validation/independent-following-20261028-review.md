# Independent Opus review: autonomous V1 work and offline zone foundations

Reviewer: Opus (review-only sub-agent), 29 October 2026. This completes the interrupted
Opus review; its trail was used to avoid repeated work. Every claim below was either
re-checked here or is explicitly attributed to that earlier review.

Constraints followed: no Pi, SSH, SCP, VNC or network-device access; no GUI, browser
(including headless), screenshots or image viewing; no hardware; no source edits; no
subagents. Work was local reads plus small offline Python probes, with scratch files in
`/tmp/opus-review-v1/`. The only file written is this report.

## Revisions and hashes reviewed

| Item | Value |
|---|---|
| Branch / HEAD | `feat/independent-verification-and-zone` / `bd6d8a53fef943cdde10ae908f0c6736c635a2cb` (tree `2455e73e…`), worktree clean |
| Base | `origin/main` `e2d4cd1357ad96f5f3b2b26fb455a8a11c7b9f0c` |
| Code revision | `b0d891eac4b94d87dcb2de3e1fd536775e8cd0ef`. `git diff b0d891e HEAD` is empty for `app/ tools/ tests/ README.md` and source docs. `cc1c300` and `bd6d8a5` change only `PLAN.md`, the decision TSV, the plan doc and the handover doc plus its evidence JSON |
| Reviewed source (sha256 prefix) | `score_monitor_run.py` c739f54e…, `prepare_monitor_truth.py` 38f46b62…, `monitor_stimulus.py` f66f1cde…, `monitor_stimulus.html` 4494e23e… (equals the dark/controlled manifest's `runtime_sha256`), `capture.py` 51fd18c4…, `config.py` 8600b0e2…, `zone_geometry.py` ca6d2000…, `zone_reference.py` f02aa7f0…, `zone_registration.py` 28502416… |
| Pilot app-source hashes | Earlier review: pale/dark pilots 59aeb959 (0181893/b70f5b3 app source); controlled pilot 75c46db3 (b0d891e). Not recomputed here |

## Verification performed

- **Evidence manifest:** all 12 `local_evidence_sha256` entries and all 3 capture
  `evidence.tar.gz` archives in `docs/validation/independent-following-20261028-evidence.json`
  match on disk. Each capture's `final-evidence.json` agrees with the manifest: child
  exit 0, one TERM, no forced kill, both outputs FULL_OFF, hardware not authorised. The
  earlier review checked 43 hashes in total.
- **Focused offline tests at HEAD:** `pytest -o addopts=""` on `test_monitor_scoring`,
  `test_monitor_truth_preparation`, the three `test_zone_*` files and `test_camera` gave
  **226 passed, 37 subtests passed**, exit 0. The browser test was not run because
  browsers are prohibited. The earlier review found that all 6 recorded desktop skips are
  native-detector tests, so the recorded full suite did run the browser test.
- **Recorded counts:** the full-suite, Pi, mypy and Ruff counts (936/387/6, 934/391/8,
  138, 141) match the retained logs; the Pi tail reads `934 passed, 8 skipped, 391
  subtests`. The file times are consistent with the TSV ordering: desktop checks
  15:38:57Z, b0d891e committed 15:39:43Z, Pi deployment 15:40:31Z, Pi tests 15:42:19Z,
  controlled capture 15:42:35Z.
- **Page reproducibility:** offline, HEAD's `monitor_stimulus.py` with `--seed 20261028
  --frame-rate-hz 10 --horizontal-position 0 --vertical-position 1 --display-scale 0.65
  --target-scale 1.25 --optical-code --registration-grid` regenerates
  `calibration-dark.html` and its manifest **byte-identically**. The pale page differs
  only in `background` (plain instead of dark) on the same 22 scenes.
- **Earlier review, accepted after spot checks:** the nested-freeze paths match how
  `read_qualification` reads them (cwd-relative, resolved). `zone.enabled` is still
  rejected (`app/config.py:801-804`), and no production module imports the zone modules.
  The strict optical counts are 250/246/4, 250/178/72 and 250/247/3. The corner
  criterion recomputes to 15/16 and 3/4. The approved boxes equal the complete crops.
  Per-frame exposure is not persisted, which I re-checked: the index `metadata` has no
  exposure or EV fields.

## Code review result

**No correctness defect was found in the new code** within the scope inspected:

- the required/allowance semantics;
- the nested-evidence freeze;
- the ExposureValue path;
- zone geometry.

Specifics:

- **Allowance:** `_exclusion` returns `expected_selection_unresolved`, which is counted
  as unknown time (`tools/score_monitor_run.py:43-50, 628-631`). Selection is
  `unresolved` (`:680-681`). A non-null identity with allowance is rejected in both the
  scorer and preparation (`tools/prepare_monitor_truth.py:436-443`). The policy is
  included in render content through `expected_rule`.
- **Nested freeze:** evidence is hashed before validation and checked twice
  (`prepare_monitor_truth.py:265-298, 379, 1139`). Dependency hashes are recorded.
- **EV:** default `null`; finite-number checks reject booleans and non-finite values
  (`app/config.py:118-121`); the SDK range is checked without clamping
  (`app/capture.py:183-207`).
- **Zone geometry:** conservative membership is correct. A disc within radius of any
  edge is UNCERTAIN, including boundary points at radius 0. Horizon-free support plus
  the vertex-maximum radius is correct for convex projective images
  (`app/zone_geometry.py:134-145, 188-262`).

## Findings

### A. Software and fixture-design blockers before freezing a V1 confirmation policy

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| A1 | High | **The comparable/larger area qualification is required by the docs but not enforced by tooling.** Preparation copies the manifest's expected identity into labels for the similar-size and sustained scenes with no area-ratio gate; nothing in the truth or scoring tools references 1.25 or an area ratio. An unqualified comparable or larger scene would be scored as definite required time. | `tools/prepare_monitor_truth.py:428-443`; `docs/monitor-stimulus.md` independent-fixture section ("Independently qualify projected bird extents … before confirmation") |
| A2 | Medium | **Fixture labels alone cap centring even with perfect truth.** In all three selection scenes, 0–12 s is `allowance` although the incumbent is alone (`tools/monitor_stimulus.py:509-511`). The static scenes label the identical single-bird acquisition `required`. After 6 s settling this puts 18 s of decidable time into the unknown denominator. | Table C; `score_monitor_run.py:756-777` |
| A3 | Medium | **The planned occlusion/loss and empty/confuser claims have no scored window.** With the pinned 6 s per-scene settling (`prepare_monitor_truth.py:311`; `score_monitor_run.py:710-719, 737`), `empty` (4 s) and `confuser` (5 s) lie wholly inside settling. The partial, brief and full occlusion events (3.0–4.5 s) and departure (from 2 s) also fall inside settling. Selection durations for them are zero, and only the final 1–2 s of each motion or occlusion scene is scored. The plan's criteria (plan, Phase 2 table) need longer scenes or a separately frozen event-timing report. | Probe output, Table C |

### B. V1 confirmation and qualification blockers (evidence and policy, not code)

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| B1 | High | **No valid reach truth exists for the new layout.** `reach_bounds` returns `None` when the browser layout differs from the reach reference (`prepare_monitor_truth.py:711-715`). The only reach policy found (`outputs/validation/monitor-20261027/truth-qualification-policy.json`) uses display scale 0.4 with a different target region. The pilot layout uses 0.65 at 0/1, with target region left 160.5, top 717.7, 1284×542 CSS px. Under the existing policy every visible expected target is `reach_unresolved`, so the centring lower bound is **0** for any centre method. Fresh powered reach evidence at the frozen layout needs separate user hardware authorisation. | Offline comparison of the policy and `browser.jsonl` |
| B2 | High | **The comparable case is not qualifiable at the current geometry and allowances.** Only one comparable sample exists (frame 166), with observed camera area ratio 0.853 (nominal 0.64; about 1.33× right-side magnification, as the earlier review found). Worst-case interval bounds: declared ±3 px corners alone give an upper bound of **1.273 > 1.25**. Adding the ±10 source-px edge allowance gives **1.605** when clipped to the crop (inward only) and **2.113** when symmetric. The larger case (observed 3.378) is very likely qualifiable but has not been qualified yet. | `terra-visual/controlled-exposure/camera-checks.json`; crops 200×204 and 121×125 source px; incumbent about 60×65 camera px |
| B3 | Medium | **Pale-field truth is unqualified.** Motion, scale, occlusion, departure and negative scenes keep the plain field (`monitor_stimulus.py:473`); the dark field was introduced because pale boundaries were inconclusive. This leaves 11 s of scored required time unknown. | Table C |
| B4 | Medium | **Corner evidence covers rest poses only, and its sample count overstates independence.** Seven standing checks (frames 151–242) share byte-identical corners across distinct JPEGs, as do three profile checks (196/226/242). That is 10 distinct standing and 2 distinct profile annotations, not 16/4. The geometry is static, so the reuse is plausible, but it is not repeated observation. There is no powered, moving-camera or blur coverage. The −1.5 EV exposure is about 24 ms, so blur risk on moving frames is untested. | `camera-checks.json`; Terra report |
| B5 | Medium | **No confirmation policy is frozen yet.** It needs to pin: page bytes and command; EV −1.5; settling 6 s; gap 1 s; allowances; the scored scope; the truth method and its allowances; blinding; and reach. Recordings lack per-frame AE/EV state (`index.jsonl` metadata), so EV must be bound through the hashed configuration snapshot. | Plan Phase 1 exit |

### C. Centring-cap check: current incomplete qualification versus inherent limits

This is a nominal-timeline calculation from the HEAD independent manifest and the scorer
formula: lower bound = definite ÷ (eligible + unknown), and pass requires a lower bound
of at least 0.80 (`score_monitor_run.py:19, 761-777`). After settling, the playlist
contains 98 s of required-target time, 21 s of allowance time and 1 s of expected
absence. The fractions are the same for any repeat count. Real recordings can only do
worse: latency, unlabelled frames and gaps add unknown time.

| Scenario | Max lower bound | Definite occupancy needed on eligible time |
|---|---:|---:|
| Current labels, all truth and reach qualified | 98/119 = **0.824** | **97.1%** |
| … comparable 12–24 s unknown | 86/119 = **0.723** | impossible |
| … pale-field scenes unknown | 87/119 = 0.731 | impossible |
| … comparable + pale + sustained 15–24 s unknown (closest to today) | 66/119 = 0.555 | impossible |
| … existing reach policy on the new layout | 0 | impossible |
| Relabel 0–12 s as required incumbent, all qualified | 116/119 = 0.975 | 82.1% |
| Relabel, comparable unknown | 104/119 = 0.874 | 91.5% |
| Relabel, pale unknown | 105/119 = 0.882 | 90.7% |
| Relabel, comparable and pale both unknown | 93/119 = 0.782 | impossible |

Conclusions:

- **The 97.1% requirement is inherent to the current fixture labels and the conservative
  scorer**, even with perfect truth. Its cause is the label choice in A2, not truth
  quality. A pass is not impossible, but it would demand near-perfect definite centring,
  including the moving-scene tails.
- **0.723 and 0.731 follow from current incomplete qualification**, not from the method.
  Under the declared allowances, however, the comparable scene is effectively
  unqualifiable at the current geometry (B2). It needs redesign, for example a materially
  smaller ratio, larger on-camera targets or evidence-backed tighter allowances, with the
  bound recomputed before freezing. Alternatively, separate it into its own report, as
  the plan permits.
- **Reach (B1) currently dominates everything** and is independent of the truth method.

### D. Minor wording and provenance

1. `docs/monitor-stimulus.md:322-325`: the documented dark-pilot command uses positions
   0.5/0.5, the default seed and 30 Hz. The captured page used seed 20261028, 10 Hz and
   positions 0/1, so the instruction "Use exactly the same page" is not reproducible from
   the doc as written. `:297` gives 30 Hz pose frames (0/360/450); the pilots ran at 10 Hz
   (0/120/150).
2. `deploy/monitor-validation-decisions.tsv:133-142`: these ten rows all carry the
   commit-time stamp `15:52:28Z` (the `pause` row at `:143` is `16:13:03Z`). They are a retrospective narrative, not contemporaneous
   decisions; the `resume` row post-dates the 15:06Z pilot. The `fixture` row cites
   `0181893` (committed 15:39Z), but the 15:06Z pale page came from an uncommitted
   intermediate generator. HEAD differs from it only in background. The retained
   page/archive hashes preserve its identity; the wording should say so.
3. `README.md:187` says unsupported EV values "fail at camera startup". For a day/night
   pair, the night camera is validated only at switch time. A failure there falls back to
   the previous camera and records an event (`app/switching.py:151-157`).
4. `app/zone_registration.py:379`: "independent held-out matches" overstates the split.
   Held-out points are every fifth correspondence from the same match set. The module
   docstring itself is accurate.
5. Legacy tuning/evaluation manifests now serialise `expected_policy: "required"`, so
   regenerated presentation IDs differ from older recorded ones. The semantics are
   unchanged, as the earlier review found.
6. The handover and `PLAN.md:30` should qualify "15/16 standing, 3/4 profile" with B4's
   distinct-annotation counts. They should also replace "review cancelled" with a
   reference to this report once root records it.

## Recommended smallest defensible next V1 path (not started; no work initiated)

Use the existing **`manual_pixels`** method, scoped to the fresh confirmation run. Do not
build scoped direct-calibration evidence groups first.

`manual_pixels` needs no new trusted mapping or transfer-of-qualification logic. Scoped
calibration would need the new per-group policy reader described in
`docs/monitor-truth-preparation.md:127-145`, plus powered-pose qualification. Manual
annotation directly observes moving-camera frames, and the scorer already accepts
per-axis allowances (`score_monitor_run.py:413-440`).

The cost is labour. At about 5 Hz, a 249 s repeat needs about 1,250 frame labels, of
which roughly 490–580 need expected-target centres. Scoped calibration is the fallback
only if that labour is unacceptable.

Preserve the semantics as follows:

1. **Centre:** annotate the four crop corners of each target. The centre is the
   intersection of the quadrilateral's diagonals, which is the projective image of the
   approved visible-bird box centre because those boxes equal the complete crops. Use
   that, not the corner mean, photograph centre or detector centre.
2. **Allowances:** each axis gets the frozen corner allowance propagated to the centre,
   plus 10 source px scaled by that frame's own corner-derived scale. The policy must
   forbid omitting `centre_uncertainty_px`: the scorer treats omission as zero for
   `manual_pixels`. Before capture, set the corner allowance from an offline
   repeatability check on retained pilot JPEGs (independent repeat annotations, all
   disagreements kept), not from the subjective ±3 px.
3. **Unknown time:** unlocatable corners (blur, occlusion, clipping) become
   `visible: null`, which is unknown, never invisible. Allowance windows stay unknown.
   Unqualified selection scenes stay unknown or inconclusive (closing A1 by policy or a
   gate).
4. **Reach:** collect fresh reach evidence at the exact frozen layout before capture
   (B1). This needs explicit user hardware authorisation.
5. **Fixture decisions before freezing:**
   - relabel 0–12 s (A2);
   - make the comparable scene qualifiable or separate it (B2);
   - put all independent scenes on the dark field or qualify pale truth (B3);
   - give occlusion and negatives a scored window or a frozen event-timing report (A3).

   Then pin the page command and bytes, EV −1.5, settling and gap, the manual protocol,
   the blinding order (annotation receipt hashed before opening detector or selection
   data) and the area-ratio rule. Only after that collect a fresh confirmation capture.

The accepted original hour and its frozen FAIL remain unchanged. This review claims no
centring, selection, zone or global pass.

## Not inspected

- Raw JPEGs and visual annotation correctness (image viewing prohibited). Only numeric
  consistency and duplication were checked.
- The `tools/monitor_stimulus.html` and `tests/monitor_stimulus_browser.js` diffs
  (calibration-pose control, dark-field CSS). The browser test was not run.
- Internals of `zone_reference.py` (PNG artefacts, hash binding) and the
  `zone_registration.py` mask, matching and `_screen_geometry`, beyond an outline and
  the fit path. I relied on the earlier review and the passing zone tests.
- A line-by-line audit of test adequacy.
- Full suite, mypy and Ruff reruns. Only the recorded logs were checked.
- Pi deployed files, calibration and live state (prohibited).
- The EV probe's image content.
- Pilot app-source hashes. Not recomputed; earlier review only.
- Whether the physical camera pose matches the old reach reference.
