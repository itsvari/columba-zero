# Plan independent following verification and a fixed zone

Status: **paused at the user's request on 29 October 2026**.
Save and push the current results; resume only when instructed. The first Opus
review was cancelled during wrap-up; the subsequent
[offline review](validation/independent-following-20261028-review.md) is complete.
Only the review and documentation resumed; implementation remains paused.
The user requested this plan after accepting the measured demonstration and
authorising its merge into `main` on 28 October 2026. The user subsequently
resumed the work and requested autonomous progress, including delegation.

## Scope and starting point

The current demonstration is accepted by the user. Its original hour assessment
remains a failed assessment against its frozen criteria; this plan does not
reopen throughput tuning or require another hour. Independent centring and
expected-selection claims remain unverified.

Complete independent following verification before runtime zone integration.
Isolated zone foundations and offline tests may progress in parallel; they do not
enable the feature. Then implement an opt-in zone
fixed to a physical plane. The zone makes pigeons inside the polygon eligible
for following and excludes pigeons outside it. It is a selection region, not a
guarantee that a bird or camera cannot cross a physical boundary.

Reuse the current tracker, `TargetSelector`, controller, preview and recording
pipeline. Preserve 250 ms observation freshness, the tested home ±100 µs travel,
reduced speed and the single hardware owner. Keep detection-only operation and
ordinary following available when the zone is disabled. Automatic startup,
model retraining and expanded travel are outside this plan.

Relevant starting points:

- [Validation record](../deploy/VALIDATION.md) and
  [source bird labels](../deploy/monitor-source-labels.json).
- [Monitor truth preparation](monitor-truth-preparation.md),
  [photo locations](monitor-bird-locations.md) and
  [optical association](monitor-code-association.md).
- `app/target_selection.py`, `app/pipeline.py`, `app/follow_models.py`,
  `app/pan_tilt_control.py` and `app/preview.py`.
- `ZoneConfig` is currently an `enabled` flag whose activation is rejected.
  There is no implemented zone registration or drawing workflow to enable.

## Phase 1: qualify independent camera truth

### Review findings to close before confirmation

The offline review found no correctness defect in the inspected code paths, but
identified these blockers. They are follow-up work, not completed fixes:

| Finding | Required follow-up |
|---|---|
| A1: scene-area qualification unenforced | Require recorded comparable/larger area evidence before assigning a definite expected selection; otherwise retain unknown time. |
| A2: excessive acquisition allowances | Reconsider the three 0–12 s allowances: the incumbent is alone, unlike the same situation in static scenes. Freeze corrected labels before capture. |
| A3: events hidden by settling | Extend or separately assess occlusion, departure and empty/confuser cases; their events currently occur within the six-second settling window. |
| B1: reach layout mismatch | Obtain reach evidence for the actual 0.65-scale, 0/1 layout after hardware work resumes. The old 0.4-scale policy cannot support it. |
| B2: comparable-size interval crosses 1.25 | Redesign the scene or assess it separately. Do not tighten uncertainty to manufacture qualification. |
| B3/B4: incomplete visual scope | Qualify pale scenes and actual moving-camera views; reused rest-pose annotations are not independent repeats. |
| B5: no frozen confirmation policy | Pin the exact page, exposure configuration, truth/uncertainty method, reach, timing windows, scored scope and annotation blinding before capture. |

Prefer the existing per-frame `manual_pixels` route before building a new scoped
calibration subsystem. The review proposes independently annotated crop corners,
their diagonal intersection for the approved crop-box centre, and explicit
per-axis uncertainty including source-edge allowance. Establish repeat-annotation
uncertainty offline and validate the centre/curvature assumptions; do not omit
uncertainty or infer it from feature-fit residuals. Keep unlocatable frames unknown.
Manual centres still need valid reach evidence. No such confirmation run has begun.

### Resume sequence

Entry: the user resumes work. Inspect the actual camera pose and configuration;
the previous distance and mappings do not automatically qualify a moved base.

1. Preserve the accepted baseline, raw evidence and frozen source snapshots.
   Create a new campaign directory. Regenerate stimuli with the repaired replay
   clock; freeze new page, tool, model, configuration and policy hashes.
2. Reuse the five approved visible-bird source boxes. Review camera annotations
   without detector boxes or selected IDs. Sample each asset, both axes,
   direction reversals, image positions, scales and the two-bird scenes.
3. Validate local photograph mappings against independent observed features.
   Include source-label and camera-annotation uncertainty. A previous 7 px
   allowance applies only where supported by the new checks. Keep rejections.
4. Use optical codes for frame identity. Model-predicted marker positions supply
   sampling proposals, not geometric truth. Reject disagreement, blended scene
   transitions and ambiguous matches. A global planar fit is not assumed valid
   on the curved monitor.
5. Measure reach from camera response across the existing envelope, including
   return hysteresis and cross-axis effects. Do not exclude frames merely because
   the controller reached a limit. Keep uncertain reach explicit.
6. Resolve comparable-size scene ambiguity before confirmation. Use independent
   projected bird extents and their uncertainty at sampled camera poses. If the
   uncertainty crosses the switching ratio, redesign or separate the scene;
   do not choose the expected winner from the detector's output.

Prefer clearly separated live scenes over a fragile threshold-boundary test.
Use similar challengers whose upper area-ratio bound is below 1.25 and sustained
larger challengers whose lower bound is above 1.25 throughout the eligible
challenge. Account for the selector's area smoothing and 0.5-second hold.
Exact 1.25 boundary behaviour belongs in deterministic software tests; visible
bird area is an independent proxy, not the detector's smoothed box area itself.

For centring, use the mapped approved visible-bird box centre, not photograph
centre or detector centre. Partial occlusion and clipping can invalidate that
centre: retain unknown centring time and assess identity retention separately.

Exit: a versioned truth policy, blinded annotation receipts, per-asset/pose
qualification and uncertainty, independent expected-identity timelines, and a
declared reach policy. Freeze them before opening confirmation results. If the
truth remains ambiguous, report inconclusive and improve the fixture first.

## Phase 2: verify centring and target selection

1. Run meaningful offline tests of truth joins, uncertainty, missing frames,
   wrong selection and conservative scoring. Check the runtime's exact switching
   and retention semantics separately from the independent camera labels.
2. Collect a fresh unannotated confirmation recording with the physical camera
   and existing bounded following profile. Calibration-only photo frames are
   disabled. Keep tuning evidence separate from confirmation evidence.
3. Cover static offsets, moving/stopping/reversing birds, similarly sized birds,
   brief and sustained larger challengers, partial/full occlusion, departure,
   reacquisition and empty/confuser scenes. Use longer static holds where needed
   for the unchanged slow controller to settle.
4. Join selected track IDs to independent photograph identities only after truth
   is frozen. This correspondence may use observed boxes; expected identity,
   centre, visibility and reach must not come from the selected track.
5. Report conservative centring bounds, switch/loss events and timing, rejected
   associations, unknown time and recording gaps. Keep every attempted case.
   A failure that triggers a software fix requires a new confirmation capture.

Freeze these criteria before confirmation:

| Claim | Planned criterion |
|---|---|
| Centring | Centre lies in the central 20% of image width and height for a conservative lower bound of at least 80% after the declared six-second settling interval; retain unknown time and the existing one-second maximum sample gap. |
| Comparable birds | No switch away from the incumbent in independently qualified comparable-size scenes after acquisition. |
| Brief larger challenger | No switch before the configured 0.5-second hold is satisfied. |
| Sustained larger challenger | Select the independently labelled challenger after the smoothing/hold response allowance declared before capture. |
| Occlusion/loss | Check identity retention, stale holds and reacquisition against the configured 0.75-second retention; score full-loss cases separately. |
| Evidence integrity | Bind raw frames, optical records, labels, code and policy by hash; ambiguity prevents a pass rather than silently shrinking the denominator. |

Declare acquisition, smoothing-response and recovery timing allowances during
fixture preparation, before confirmation results. Label cases that never exercise
the intended runtime path, such as low-confidence maintenance, as unexercised.
Report the accepted hot throughput and the 300 ms p95 latency goal; do not revive
the superseded 10 FPS floor or require a new hour without a separate decision.

Exit: reproducible independent reports and review for centring and each selection
case, with pass/fail/inconclusive states. Begin zone integration after these core
claims pass; a blocked or failing case stays visible.

## Phase 3: implement the surface-fixed zone

### Define the user workflow and stored reference

At powered home, let the user pause following, freeze a reference image, outline
the registration surface and draw/edit a polygon. Validate and save the reference
before activation. Show zone state, eligibility and registration quality in the
preview; allow explicit disable and reference replacement.

Store a versioned reference image and hash, polygon, registration-surface mask,
camera identity/orientation/resolution, relevant calibration hashes and quality
policy. Validate external files and coordinates at the configuration boundary.
Reject mismatched or incomplete references before enabling zone following.
Replace the current blanket `zone.enabled` rejection only when those validation
and registration paths exist. Keep the default disabled and require the existing
explicit hardware-following opt-in for bounded zone tests.

### Separate registration, membership and selection

Use immutable results with explicit valid/invalid registration and
inside/outside/uncertain membership. Keep reference coordinates distinct from
current image coordinates and preserve capture timestamps and uncertainty.

- Register static surface features to the reference with robust fitting. Mask
  detected/tracked birds in both views with an uncertainty margin. Check held-out
  residuals, spatial support, coverage, transform plausibility and staleness.
- Use optical flow only between accepted reference registrations; regularly
  re-register to bound drift. Flow failure or an expired transform is invalid.
- Transform each bird's bottom-centre box point into reference coordinates.
  This is the defined image proxy; it does not prove physical foot contact for
  flying birds, elevated perches or different planes.
- Expand the membership test by mapping uncertainty. Freeze entry/exit margins
  and dwell times in offline tests before physical validation. Boundary-uncertain
  birds cannot become new targets. Stop corrections for an incumbent whose
  eligibility is uncertain rather than trusting stale polygon membership.
- Filter eligible tracks before the existing largest-bird and switching policy.
  Confirmed exit releases the incumbent; ordinary missing-track retention must
  not authorise continued corrections to an ineligible bird.
- Invalid registration suspends zone following, clears pending challenges and
  holds the current pose through the existing controller. It must not cause an
  automatic return-home movement. Define and test explicit recovery/resume rules
  that cannot override a user's pause or home request.

Expose these states through `/api/state`, JSONL and the browser overlay. Show an
invalid/stale zone as unavailable, not as a current trusted projection. Compose
with `FrameProcessor` and `TargetSelector`; do not add a second tracker,
controller or hardware writer. Keep expensive registration outside the timed
controller and publish only timestamped results.

### Test software behaviour before movement

Cover polygon validation, coordinate transforms, uncertainty near boundaries,
entry/exit hysteresis, a larger bird outside the zone, an incumbent leaving,
occlusion, stale/failed registration, recovery and camera/configuration mismatch.
Exercise reference drawing/saving/reloading and pause/home controls in the browser.
Verify that disabling the zone preserves ordinary following behaviour.

Exit: reviewed configuration/domain changes, offline registration and membership
tests, a working browser workflow and simulation evidence. Physical zone following
remains opt-in until the next phase.

## Phase 4: validate the zone on a physical plane

Use a genuinely flat, textured reference surface with independent check points.
The curved monitor does not establish a globally planar zone. Printed/cut-out
bird fixtures can provide controlled targets without requiring live birds;
verify a positive detection first. Moving fixture texture must not become false
background registration support, and ground-truth labels must not feed runtime
registration or selection.

Validate in this order: camera-only registration, simulation, then bounded
physical following. Check the polygon against independent points across the
surface and the tested camera travel. Cover entering/leaving, distractors,
boundary jitter, camera movement, weak texture, occlusion, leaving the reference
view and recovery. Repeat from more than one pose and sweep direction.

Pin registration-error, containment-uncertainty and recovery criteria before the
physical recording. Report eligible and wrong-zone selections, invalid-registration
duration, recovery and independent polygon alignment. Capture pause/home/shutdown
and both-output-disable receipts. Compare paired hot zone-off/zone-on runs with
matching output modes to quantify cost; retain actual throughput and timing.

Exit: reviewed software checks, independent zone-registration/eligibility results,
bounded hardware evidence and an updated operating guide. Any additional endurance
run is a separately planned follow-up, not part of starting this feature.

## Planned delivery units

| Unit | Deliverable | Status |
|---|---|---|
| V1 | Qualified camera truth, redesigned ambiguous scenes and frozen confirmation policy | Paused; fixture and tooling implemented, pilots retained, qualification incomplete |
| V2 | Independent centring and selection confirmation reports | Not started |
| Z1 | Zone reference, registration and membership domain logic with offline tests | Offline geometry, reference files and registration candidates implemented; independent qualification pending |
| Z2 | Browser workflow, selector/controller integration and simulation | Not started |
| Z3 | Independent planar-zone hardware validation and operating guide | Deferred; no flat textured surface currently available; software/offline checks first |

Use separate reviewable changes for these units. Preserve failed attempts and
freeze new tools/policies before new captures. Keep calibration and raw media
local; commit reusable tools, tests and reports with their evidence hashes.
