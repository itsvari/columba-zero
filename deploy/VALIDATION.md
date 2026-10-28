# Validation status

## Resumed campaign, 28 October 2026

The user resumed the work and explicitly permits movement checks without
continuous physical supervision. The later explicit reply, “Include the hour
this session”, supersedes the wrap-up deferral below.
Keep the tested ±100 µs travel, 3 nominal °/s maximum speed and 250 ms freshness.

Update after the resumed review: `1f518f2` regressed normal low-confidence motion
at 10–11 FPS. `dc444c5` narrows protection to size outliers; six regressions fail
before and pass afterwards, with 82 nearby tests passing. The latest desktop gate,
including the offline assessment and photograph-frame browser checks, passes
863 tests and 370 subtests, with six optional skips. Strict mypy passes across
132 files, and the canonical Ruff gate passes across 135 files. Pi at
runtime revision `dc444c5` passes 847 tests and 375 subtests, with eight optional
skips, plus native CTest and model checks. The runner
now keeps the actor outside the inhibitor's parent-death chain, suppresses later
cleanup interruptions, and passes 24 tests including the real desktop wrappers.
Both policy copies pin truth hashes and explicit gate scopes. The assessment
tool takes its expected policy hash from the frozen run plan, avoiding a circular
tool/policy checksum; its 13 focused tests pass.

The Pi rebooted to flags `0x0` with both channels FULL_OFF. The first fresh repeat
(`matrix-20261028T085100Z-0cae671a`) stopped because no positive target appeared
within the source-confirmation deadline. The application exited 0 after one TERM,
without KILL; both outputs read FULL_OFF and final flags were 0. It issued home
but no tracking corrections. The archive SHA-256 was verified. This is a guarded
failed test setup, not a tracking or centring pass. The actual camera view shows
the monitor edge and room rather than the rendered stimulus. Raising the test
browser and a separate follow-disabled camera recording did not resolve visibility.
The colour diagnostic proved that the physical monitor follows the test browser.
The user then confirmed the Pi/base had moved, invalidating the old home and reach
mapping. No hour run has started. Evidence is in
`outputs/validation/monitor-20261028/`.

Slow servo ramps established new home commands of pan 80.1 and tilt 89.1 nominal
degrees, or 1390 and 1490 µs. Both outputs read FULL_OFF after each ramp. The new
profile uses pan 1290–1490 µs and tilt 1390–1590 µs. Eight subsequent slow sweep
phases reached both endpoints and returned home, with completion and FULL_OFF
receipts for each phase. The separate follow-disabled camera capture exited 0
after one TERM, without KILL. Camera footage and command logs remain separate.
The user did not observe this sweep. There is no new operator confirmation of
smoothness or cable clearance. The footage establishes camera response and the
receipts establish commanded travel and output disable, not cable clearance.

The selected presentation is display scale 0.65, target scale 1.25, horizontal
position 0 and vertical position 1. Its scene list exactly matches the original
tuning playlist. The target-scale-2 diagnostic variants are excluded because
six stationary scenes clip their photographs. The final home has all four markers
visible and reliable detection. Terra independently annotated photo corners
before matching in two home frames and eight powered sweep holds. All sampled
standing-pigeon fits pass the unchanged 7 px limit, including the declared manual
uncertainty. Independent shorebird checks and a follow-disabled profile calibration
with visible photograph frames also pass on their sampled images. In the latter,
eight camera fits across four powered pan holds have worst error 5.941 px including
the declared ±2 px corner uncertainty. Frames are excluded from following captures.

The 201-second following calibration at
`output-matrix/matrix-20261028T100018Z-d237b91d/both` completed. It processed
2,248 frames and retained 1,025 unannotated JPEGs without recording, rendering or
log writer drops. The 3,905 latest-frame capture drops are separate from recording
loss. One TERM, no KILL, child exit 0 and both FULL_OFF were recorded. The maximum
controller interval was 55.906 ms. After the 120-second warm-up, the remaining
84.931 seconds processed 900 frames: 10.597 FPS. Hot p95 first-use sensor age was
192.9 ms, and uncensored tracking/stale age was 250.6 ms, below 300 ms. Full-run
move age was 229.7 ms at p95 and remains freshness-censored. Temperature peaked
at 87.0°C in the app summary and 88.1°C in independent telemetry. Median measured
ARM clock was 1.50 GHz, while the cpufreq request stayed at 2.4 GHz. Final flags
`0xe0008` show thermal history/current soft-temperature
limit with no current or historical under-voltage bits. The 205.132-second MP4
independently decoded without error, versus 205.131130 seconds expected. Evidence
and decode proof are under `outputs/validation/monitor-20261028/`.

The strict optical decoder associated 864 of 1,025 frames across all 22 scenes.
The 161 rejects remain in the report. Initial global-marker proposals failed;
an all-grid proposal also failed, before right-column fixed-grid geometry supplied
successful sampling proposals without
changing CRC or ambiguity thresholds. These are model-predicted marker positions,
not confirmed physical marker centroids. Independent masking checks show that
the reference photograph and its uncertainty envelope contribute no SIFT pixels.
The geometry, visual checks and rejected candidates remain separately auditable.
The typed builder is a later reproduction whose positions match the original
successful proposals exactly.

The new calibration is not a confirmatory centring recording. In the framed
profile samples, independent visible-bird area ratios have point estimates
1.200–1.332 but broad uncertainty intervals spanning 1.25. Consequently the
similar-size scene cannot establish the expected target choice in this camera
view. No runtime threshold or truth target was changed to manufacture a pass.
Both policy copies qualify the current home, bounded travel and visibility only
for the hour's timing, recording and stability gates. Independent centring and
selection remain pending, and the old pose's qualification policy is excluded
from the current frozen inputs. The actor refuses launch for a pending setup,
changed frozen inputs or a framed following page. The hour is authorised but has
not started; its numerical engineering criteria are unchanged.
The hour's default-off photograph-frame runtime differs from the earlier bounded
following runtime. Scene geometry is unchanged and browser regressions pass; the
new runtime has not yet driven a hardware following capture.

The pre-hour Opus review found a shutdown-edge coverage defect in the assessor.
The final processed frame can arrive just after the controller closes. Coverage
now accepts an end gap within the existing 75 ms controller-interval budget; the
13 ms regression passes, while a 100 ms gap stays inconclusive. The assessor now
also reports measured ARM clock, current thermal-limit fraction and final-window
temperature median, separately from the requested clock and existing latch bits.
Both policy copies were re-frozen before capture. Numeric gates did not change.
The baseline already has historical thermal flags `0xe0000` and clear current and
historical under-voltage bits. A below-10-FPS hot window remains a real failure.
The monitor remains the reported 34-inch, 1500R display. The earlier approximately
38 cm distance predates the move; current distance is unconfirmed. Its
350–400 cd/m² brightness remains a quoted specification, not a measurement.

The original 21-observation fixture still retains one identity through the
partial occlusion. The initial broad hold at `1f518f2`, its earlier test counts
and the parent-wrapper runner design are superseded by the corrections above.
The initial desktop browser failures came from a removed temporary executable;
the pinned replacement and both gate logs remain in
`outputs/validation/monitor-20261028/`.

E1–E6 are implemented. Original scripts remain in
`pre-resume/endurance-runner/` beneath the earlier evidence directory. Independent
review identified T1, R1 and P1; the corrections were applied locally and checked
with regressions and the real inhibitor wrappers. A fresh short recording is
still needed for confirmatory centring after expected-selection truth is resolved.
The hour will assess timing, recording and stability separately. Criteria are frozen in
`deploy/monitor-endurance-policy.json`.

## Monitor campaign, 27 October 2026, in progress

The user withdrew the 15 FPS goal and accepted the earlier hot 10–12 FPS range
near 83–85°C for the steady-state demonstration, provided control timing and
stability pass. Report actual hot throughput, temperature and throttling state.
The short approximately 12.5 FPS result was measured during warm-up. Other
acceptance goals remain. Wider movement is authorised; each envelope still needs
measured checks. Following remains opt-in.

The user subsequently relaxed the demonstration latency goal to 300 ms at p95,
while retaining smooth control. The earlier 200 ms failures below remain failures
against that original goal. Keep the 250 ms observation-timeout policy and report
uncensored observation age and stale time alongside move-age statistics. The
official active cooler sits beneath the pan-tilt HAT, which the user reports
slightly restricts airflow.

### Deployment and software

The power-supply replacement interrupted recent filesystem writes. Recent Git
refs, objects and source files became empty. The first transferred bundle also
failed its checksum. Recovered the deployment from a fresh bundle whose SHA-256
matches the development copy, retained the interrupted source and Git directory,
and verified the clean repository with `git fsck`. Ignored calibration, models,
environment and older evidence remain. The older `columba-zero` checkout was not
changed. Fan trip temperatures survived reboot at 45, 50, 55 and 60°C.

The recovered Pi baseline passed 584 tests and 360 subtests with two optional
skips, Ruff lint and formatting, native CTest, and both model-package checks.
The development gate after the startup and stimulus changes passes 585 tests and
362 subtests with six native skips, Ruff lint and formatting, and strict mypy
across 112 files. Real Chromium checks cover both playlists at full and reduced
display sizes. The full Pi typing gate has not been reclassified as passing.
The updated Pi at `3f72dbb` passes 589 tests and 368 subtests with two optional
skips, Ruff lint and formatting, and `git fsck`.

The later pre-endurance gate at `d55db25` passes **834 desktop tests and 369
subtests**, with six optional skips, and **832 Pi tests and 375 subtests**, with
eight optional skips. Real Chromium checks were enabled on the desktop. Ruff
lint and formatting pass on both machines, strict desktop mypy passes across
130 source files, and Pi native CTest, both model-package checks and `git fsck`
pass. The desktop test log retains eight deprecation warnings. The full Pi mypy
gate is still not claimed. Revision, deployment bundle checksums and outputs are
in the campaign's `desktop-pre-endurance-final-*` and
`pi-pre-endurance-final-gate.txt` evidence files. The centre guard only
qualifies fully visible source crops; all four regenerated centring reports match
the final tool hashes and retain exactly the same numerical bounds.
The desktop Ruff commands were `python -m ruff check app tests tools cluster
training native` and `python -m ruff format --check app tests tools cluster
training native`, using `.venv/bin/python`. Generated files under `runs/` and
historical notebooks were outside that gate's scope.

### Display and short physical runs

The physical display is a 34-inch ultrawide with a 1500R curve. The user increased
camera distance from approximately 30 to 38 cm. The quoted 350–400 cd/m² brightness
is a panel specification; actual luminance was not measured. A real camera frame
confirms an upright pigeon at 0.92 confidence initially. Reduced, repositioned
presentations include all four markers; known positives remain confirmed.

The initial home ±50 µs run made no tracking moves. Enabling the saved home
changed camera aim substantially from the unpowered setup, leaving the target
outside the useful view. The user observed a smooth home movement with cable
clearance. The run stopped cleanly and both channels read FULL_OFF. Repositioned
the presentation within the powered-home view before the next run.

The subsequent 30-second run allowed home ±100 µs, retaining a 3 nominal °/s speed
limit and the shipped model and thresholds. It exercised centred and horizontal
offset scenes, then accepted a home-and-pause request. Results:

- 373 processed frames in 29.785 s, or 12.523 FPS, with zero log drops.
- 160 hardware tracking moves, all with sensor-derived command timing. Their
  driver-return age was 198.0 ms at p95, 159.848 ms mean and 206.269 ms maximum.
- 577 logged hardware ticks. Observed interval p95 was 51.2 ms. The 754.573 ms
  maximum includes a startup stall; this is not a clean 20 Hz acceptance result.
- Commands visited pan 100.800 to 107.382 and tilt 93.672 to 97.200 nominal
  degrees. These are commanded coordinates, not measured physical angles. The
  whole ±100 µs envelope was not exercised.
- 145 recorded frames, zero recording drops, and a playable exported MP4 reported
  at 29.586 s against 29.584668 s expected. Annotated JPEGs and metadata remain;
  these images cannot supply independent annotations blinded to detector output.
- Clean shutdown and FULL_OFF readback on both channels.

A new-process diagnostic measured OpenCV import at 834.954 ms and SciPy optimiser
import at 225.548 ms. These explain the first-use tracking stalls. Revision
`6fcf7e4` loads them before camera capture and starts the mount only after tracker
construction succeeds. Its regression fails against the previous startup order
and passes after the fix.

A live repeat at `3f72dbb` retained the same ±100 µs profile, reduced speed and
horizontal scenes. It processed 376 frames in 28.838 s. Its 572 hardware tick
samples have a 50.268 ms mean interval, 50.9 ms p95 and 52.533 ms maximum. It
recorded 158 timed hardware tracking moves, with sensor command age 193.9 ms at
p95 and 206.850 ms maximum. Home-and-pause and shutdown completed, with FULL_OFF
on both channels. The recording retained 144 frames without drops and exported
28.634 s against 28.632597 s expected. This short repeat supports the import-stall
fix; it does not establish centring or sustained acceptance. The earlier MP4 was
also independently decoded with FFmpeg without errors.

The 20-case thread sweep is diagnostic only. Camera position changed during it,
and its original fixed case order confounded case with thermal drift. No setting
was adopted from it. The reusable probe now warms up first, reverses case order
on alternate repetitions, forces simulation with outputs off, and retains source,
model, configuration, temperature, frequency and throttling evidence.

### Independent evidence and expanded-axis sweep

The preview can now retain unannotated JPEGs while serving the annotated browser
view. Recording metadata identifies whether annotations are included; the offline
centring scorer requires explicit unannotated provenance. A separate read-only
process records Pi temperature, frequency, throttling flags and process resources.
It includes export time, so analyse capture and control intervals separately.

A camera-only run at `044d49e` retained 1,123 unannotated images and exported
225.453 s against 225.452605 s expected. FFmpeg independently decoded that MP4
and the earlier startup-fixed MP4 without errors. The camera-only run has following
disabled and does not establish hardware-following throughput or timing.

During that recording, a separate sole hardware owner swept each axis from home
to both ends of the ±100 µs envelope and returned home between excursions. Each
ramp lasted six seconds with a two-second hold, with peak nominal speed 2.25°/s.
The user reported that both axes moved smoothly and cables stayed clear. Both
outputs read FULL_OFF after shutdown.

Independent image-feature registration measured pan excursions of approximately
+95 and −117 camera pixels, and tilt excursions of −116 and +113 pixels against
their respective preceding views. After positive tilt, the home hold remained
approximately 34 pixels above its starting view. The final home returned within
approximately two vertical pixels of the initial tilt-home view. This is evidence
of lag or hysteresis, without isolating mechanical backlash, load, quantisation
or capture timing. Pixel shifts do not measure physical shaft angles. Keep this
envelope while checking closed-loop settling; wider travel is not needed yet.

The analysis retained 326 registered frames with a median 160 homography inliers.
The median inlier residual was 0.15 pixels; that is a feature-fit residual, not
absolute pose accuracy on the curved display. Detailed results and input hashes
are in `outputs/validation/monitor-20261027/axis-analysis-terra/report.json`.

### Full mixed-scene timing checks

The complete 201-second tuning playlist ran with physical following, unannotated
recording, a real browser decoding the preview, and separate thermal telemetry.
The source was `044d49e`, with the same model, threshold, 3 nominal °/s limit,
±100 µs profile and 250 ms observation timeout. The two runs below differ only
in configured output rate; they are single runs, not a controlled speed comparison.

| Output rate | Processed FPS over run | Hot FPS after 120 s | Sensor command age p95 | Tick interval maximum |
|---|---:|---:|---:|---:|
| 5 FPS | 11.507 | 11.186 | 221.3 ms | 55.503 ms |
| 2 FPS | 11.389 | 11.149 | 221.8 ms | 54.374 ms |

Both runs failed the 200 ms timing goal. The hot windows had median temperature
84.8°C, peaks 87.0°C and 87.55°C respectively, and nonzero active throttle flags.
Their hot-window command-age p95 values were 222.3 and 234.25 ms. There were zero
log drops or controller faults, and both channels read FULL_OFF after shutdown.
These short windows do not establish an hour of thermal or memory stability.

The 5 FPS recording exported successfully at 258.424 s. The 2 FPS export failed
its duration check, reporting 216.906 s against 216.933198 s expected. All 434
unannotated source images remain. Browser pause, resume and home-and-pause worked
in that run after the playlist ended, producing paused, home and paused states.
The earlier attempt to click these controls in the 5 FPS run occurred after its
bounded deadline had stopped the service; it provides no control-success evidence.

The optical decoder now searches a declared ±2-camera-pixel translation grid
without expected-payload repair, retaining contrast, cell-margin and CRC checks
and rejecting competing payloads. An agent visually reviewed the full marker
crops and positions in two contact sheets for eleven preselected automatic
centroid sets. The source coordinates were kept unchanged; the user did not
confirm these markers, and no per-marker click measurements were made. Three of
those sample images passed optical association and held-out curved-screen
checks within the predeclared three-pixel residual limit. Five remained unvalidated
for coverage and three had no unique optical match. These sampled checks do not
establish campaign-wide truth or centring acceptance. Per-image reports are in
`outputs/validation/monitor-20261027/clean-reviewed-association/association.json`.

The passive OpenMP-wait repeat also failed the timing goal at 221.6 ms p95 over
864 sensor-timed tracking moves. Its hot window ran at 11.170 FPS, with median
84.8°C and active throttle flags. Tick interval maximum was 55.033 ms. There were
no controller faults or log drops, and both channels read disabled after shutdown.
No throughput improvement or setting advantage is established by these single runs.

The MP4 duration failure reproduced on desktop FFmpeg 9.0.2. All source frame
presentation times were correct; B-frame reordering caused the MP4 duration to
reflect a different decode-time extent. Revision `d7db6c8` disables B-frames for
this variable-rate export while preserving source times, final dwell and the
10 ms tolerance. The retained 2 FPS run now exports at 216.934 s against
216.933198 s expected on both desktop FFmpeg and Pi FFmpeg 7.1.5. All 435 decoded
presentation timestamps are unchanged in the desktop comparison. A real-FFmpeg
regression failed before the fix and passes after it.

The desktop gate at `d7db6c8` passes 673 tests and 369 subtests with six native
skips, Ruff lint and formatting, and strict mypy across 122 files. The Pi passes
all 41 focused recording tests and re-exports the original failed capture.
The full Pi gate passes 674 tests and 375 subtests with five optional skips,
Ruff lint and formatting, native CTest in `native/build/pi5`, both model-package
checks and `git fsck`. The first CTest invocation pointed at the parent build
directory and found no tests; the corrected invocation executed the detection
test successfully. Full Pi mypy remains subject to the previously documented
environment differences.

### Stricter observation freshness, not adopted

A further complete tuning sequence used a 190 ms observation timeout instead of
250 ms, with 2 FPS outputs and passive OpenMP waiting. The controller holds when
observations are older than that limit. This changes the control policy; it is
not an inference speed improvement. Because age is recorded only on move ticks,
and moves require observations younger than the timeout, this caps the reported
move-age distribution by design. It does not pass the original latency goal.

The run issued 1,086 sensor-timed tracking moves. Command age was 188.2 ms at p95
and 191.984 ms maximum, with a maximum tick interval of 55.310 ms. Hot throughput
after 120 seconds was 11.344 FPS, with median 84.8°C and peak 87.55°C. There were
no controller faults or log drops. Controls responded after the playlist, and
both channels read FULL_OFF after shutdown. The outer 250-second timeout sent
TERM and export began; the agent then sent another TERM at 11:32:32.197Z. A second
signal during export cancels it by design. Consequently,
the run correctly reports a recording error and retains all 498 source images.
Do not report this run as a clean end-to-end recording pass.
An offline Pi recovery export subsequently succeeded at 248.649 s against
248.648281 s expected; this recovery does not erase the original run error.

The cost is more stale holds: 1,107 ticks were stale and 2,618 were tracking,
compared with 174 stale and 3,535 tracking in the 250 ms passive-wait run. These
counts are diagnostic controller states, not independently labelled visible time.
Across tracking and stale ticks, real observation-age p95 was 243.0 ms at the
250 ms baseline and 233.9 ms at 190 ms. There were 889 tracking-to-stale transitions
at 190 ms versus eight at 250 ms; 829 stale spells lasted one tick. Stale ticks
reset commanded speed to zero. Detector-based diagnostics also showed worse
centring and lower commanded speed, without establishing independent truth.
The 190 ms policy is rejected. The user confirmed that 200 ms is a p95 latency
goal and smooth control must be preserved. Reduce actual delay and establish
independent centring before starting the one-hour acceptance campaign.

Direct local photograph-feature matching also located both pigeon photographs
and the confuser in sampled camera frames without detector or marker inputs.
This remains exploratory. The source crops cut through bird silhouettes, and
photograph centres can differ from visible-bird bounding-box centres. Define and
review the actual visible target centre before constructing acceptance labels;
do not substitute the photograph rectangle centre. The original full bird's
unseen extent cannot be recovered from a clipped source crop.

### Source-local truth and hardware measurement updates

The user confirmed visible-bird bounding boxes on all five source photographs.
`deploy/monitor-source-labels.json` binds them to source hashes and crop geometry.
The offline local-photo matcher is independent of model detections, controller
offsets and the global monitor homography. The scorer now propagates explicit
centre uncertainty and uses the worst-case occupancy lower bound, including
unknown time, for its scoped centring verdict. Point estimates remain separate.

The original 2-pixel local-mapping check passed only one of 15 independently
annotated camera rectangles. Changing to an affine model did not resolve it.
Those failures remain recorded. A 7-pixel camera allowance was then declared from
the 3.798-pixel calibration residual plus the annotator's per-axis 2-pixel visual
allowance, rounded upwards. On a fresh sample, eight mappings passed the resulting
worst-case check, with a maximum 6.481 pixels including annotation allowance; one
additional checked image was rejected by feature matching. Five other images had
ambiguous photo boundaries and supplied no check. This is limited sample evidence,
not a statistical error guarantee. No profile-pigeon photograph rectangle has an independent
corner check. Separate agent annotations of visible-bird extrema check four
accepted profile fits within the same 7-pixel centre budget, including a
subjective 2-pixel allowance per annotated edge; two other profile fits remain
rejected. This small sample includes adjacent frames and does not supply a
photograph-corner check or a statistical guarantee. The source-edge allowance and camera allowance both reduce the
scoring band; neither changes the 20% centring goal. Reports are under
`outputs/validation/monitor-20261027/photo-camera-heldout-valid/`.

Read-only, simulated-mount profiling ruled out a held Python GIL and BLAS thread
oversubscription as major causes. Native extraction dominates. Actual measured
CPU clock fell to 1.5 GHz during thermal limiting despite the nominal 2.4 GHz
frequency readout. Five balanced repetitions did not support changing full-FOV
capture from 30 to 50 FPS: median uncensored simulated tick-age p95 was 210.5 ms
versus 215.1 ms. Two NCNN threads after thermal equilibration also failed to
improve timing, at 227.2 ms median p95. These are diagnostic simulated runs, not
physical acceptance or speedups; the two-thread comparison has sequential thermal
carryover. Retain four threads, 30 FPS, passive waiting and the original control
policy. Evidence is in `outputs/validation/monitor-20261027/latency-diagnosis/`.

The identity-only optical associator decoded 482/538 passive-run images and
423/498 images from the rejected 190 ms experiment. It uses verified references
only to propose code sampling positions, with independent agreement of both CRC
bands and one exported render match. It makes no geometric registration claim.
The truth preparer combines those identities with local photograph fits and the
user-confirmed visible-bird boxes. It fills a missing code only between nearby
optical anchors when every intervening browser frame has identical target
content and selection rules. Such labels retain an explicit equivalent-render
interval. Missing context, visibility, registration and reach remain unknown;
unknown time is pessimistically off-centre in the acceptance lower bound.

For the retained passive run, the original asset qualification yields a
**79.727%** conservative lower bound and remains inconclusive. Qualification was
developed after initial scoring on the same recording had begun: the first score
was 70.844% at 13:28Z, the profile policy file was created at 13:42Z, and the
preserved 79.727% report was written at 13:45Z. A separately hash-bound
qualification added the four directly checked profile-pigeon centres
described above, retaining two matcher rejections. This converts only 0.967274
seconds of registration-unknown time into qualified centring time. The resulting
lower bound is **80.957%**, above the fixed 80% goal, with 12.955325 seconds still
unknown and no reach exclusions. Its upper bound is 98.686%. The point estimate
on scored time is separate from this conservative statistic. This is a narrow
exploratory scoped pass for the recording and declared empirical uncertainty;
the mapping allowance is not a statistical guarantee, and overall hardware
acceptance remains inconclusive. Opus reproduced the annotations and score
byte-for-byte. The pass has only 0.753 seconds of definitely-centred margin;
scaling centre uncertainty by 1.5 gives a 79.6% lower bound. Qualification and
analysis choices evolved on this recording, so confirmatory acceptance requires
a fresh recording under frozen tools and policy. The six-second settling window
predates scoring; 123.9 seconds are settling exclusions, and 24.5% of the recording
is scored. This passive recording used nominal 2 FPS retained JPEGs, unlike the
later 5 FPS output-mode matrix. Sample-and-hold geometry does not prove
continuous-time occupancy.

Selection analysis across all 22 scenes includes the first six seconds rather
than hiding transition behaviour in settling exclusions. The 1.08× similar-size
challenge and the brief 1.6× challenger produced no switch. The sustained 1.6×
challenger produced the sole switch, approximately 524 ms after confirmation;
exact area ties were not physically tested. Internal smoothed-area reconstruction
is reported separately from independent image identity and is not bird-size truth.
Across the recording, 241.635 seconds were independently correct, 0.963 seconds
disagreed with an immediate expected-identity rule during hold/departure
transitions, and 25.580 seconds remained unresolved. The 66.611-second paused
tail is confuser/no-expected content, not extra centred-positive footage.

The 0.3-second full occlusion retained ID 18. The 0.6-second partial occlusion
reset ID 17 to 18. The earlier attribution to processing-age expiry is incorrect:
ID 18 was assigned at frame 5040, before its confirmed selection at frame 5042.
At assignment the sensor gap was about 633 ms and processing age about 743 ms,
both within retention. An offline tracker replay reproduced the association
failure: a low-confidence partial box drove a shrinking Kalman size prediction,
which reached one pixel wide and failed the IoU match. Changing the pruning clock
alone would not preserve ID 17. The 699.722 ms gap and 836.218 ms processing age
describe the later frame 5042, not initial ID creation. Partial-occlusion identity
retention remains a functional limitation; no behaviour change was made during
wrap-up. Longer full occlusion lost and reacquired the target;
departure returned to commanded home with no subsequent acquisition during empty
or confuser content. Physical home arrival is not measured by these logs.
Source hashes, all scene intervals, timing brackets and unresolved evidence are in
`outputs/validation/monitor-20261027/selection-loss-reacquisition/analysis.json`.

The 190 ms experimental recording remains inconclusive, with a 75.610% lower
bound and its original unsuccessful-run-end integrity issue. Neither the
threshold nor that integrity failure has been relaxed. Preparation, annotation,
score and qualification snapshots are in
`outputs/validation/monitor-20261027/truth-preparation/`; earlier failed and
inconclusive reports remain alongside them.

### Four output modes and hot operation

At deployed revision `7e0076e`, each output mode completed one 201-second tuning
playlist with physical following, the same ±100 µs profile, 3 nominal °/s speed
limit, four NCNN threads, 30 FPS camera request, passive OpenMP waiting and the
original 250 ms freshness setting. Preview and clean recording ran at 5 FPS.
These are sequential single runs; thermal carryover prevents a causal ranking.

| Output mode | Command-age p95 | Tracking/stale observation-age p95 | Hot processing FPS |
| --- | ---: | ---: | ---: |
| Tracking only | 211.1 ms | 226.8 ms | 11.685 |
| Preview only | 224.5 ms | 242.5 ms | 11.018 |
| Recording only | 230.8 ms | 249.4 ms | 10.720 |
| Preview and recording | 234.0 ms | 246.1 ms | 10.669 |

Command age covers sensor-timed move commands and is bounded by the 250 ms
freshness eligibility rule. A 300 ms move-age check therefore cannot independently
establish the latency goal. The separate observation-age distribution includes
tracking and stale ticks; its hot p95 reached 259.9 ms, within the revised goal.
Opus also estimated first-use age at 181–193 ms p95 using a drop-in-age heuristic;
this is a diagnostic rather than an existing formal per-observation metric.
Report all three measures and define the uncensored/first-use gates before a
future run. Hot command-age p95 was 216.1–239.9 ms. Hot processing throughput
uses inference-frame timestamps after the first captured frame plus 120 seconds,
with `(frame_count - 1) / (last_timestamp - first_timestamp)`.

During the timed controller windows, sidecar CPU temperatures peaked at 86.45,
87.0, 87.0 and 88.1°C respectively. Current low throttle bits were nonzero in
179/201, 204/204, 201/201 and 204/204 samples; peak process RSS was
289.4–309.0 MiB. Export samples are excluded. Temperature and throttle flags are
reported observations, not a declaration that long-term thermal acceptance has
passed. The cooler/HAT arrangement was unchanged.

Each run has one untimed initial `acquiring` snapshot before its first frame,
with no driver action. This accounts for the summary's single missing/invalid
interval. All subsequent timed intervals were positive, with 50.99–51.10 ms p95,
53.88–55.97 ms maximum and none over 75 ms. There were no log or render drops.
The bounded latest-frame capture buffer intentionally overwrote camera frames
that processing could not consume; those counts are separate from lost logs or
recording frames.

All four runs exited cleanly with one supervisor stop, no forced kill and both
channels reading FULL_OFF. Enabled preview modes decoded 800×450 images in the
real browser at the start, middle and end. Both recordings exported successfully
and independently decoded through local FFmpeg with exit 0. Source JPEGs remain.
Evidence, archive hashes and the rerunnable offline analysis are in
`outputs/validation/monitor-20261027/output-matrix/matrix-20261027T130524Z-3a8fa141/`.

### Physical controls and injected faults

The corrected control/fault pilot held its commanded pose while paused, accepted
home to 100.8/97.2 nominal degrees, and resumed tracking. An unread MJPEG socket
remained connected for ten seconds while a normal browser continued receiving
images. This establishes continued service with an unread client; it does not
measure the socket timeout. Making only the disposable recording directory
unwritable produced the expected JPEG permission error. Recording entered its
documented `error` state while preview encoding and tracking continued for a
further five seconds. The app retained exit status 1 for that error, and both
channels read FULL_OFF after the single supervisor stop. The earlier pilot's
actor incorrectly waited for `failed` rather than `error`; its assertion failure
is retained, and the corrected typed actor rejects that wrong state in mypy.
Evidence is in `control-fault-actor-v2.jsonl` and the extracted control-fault
archive under the campaign evidence directory.

A separate real-camera/mount fixture stalled producer read 300 for 1.2 seconds
before taking the next real frame, then raised an explicit software capture
error before read 750. Sensor timestamps were unchanged. The original camera
error and exit status 1 remain recorded. The app exported 127 retained clean
JPEGs into a 26.296-second partial recording against 26.295415 seconds expected,
and both servo outputs read FULL_OFF. The child exited without any supervisor
TERM or forced kill. This is software fault injection with a real CSI camera,
not a physical cable-disconnection test. Trace and provenance are in
`outputs/validation/monitor-20261027/capture-fault-fixture/`.

Offline trace inspection found ten stale ticks, all holding unchanged commanded
positions. Return-home movement began on the first tick after the 750 ms retention
allowance, at an observation age of 769.421 ms. Fresh tracking resumed 192.080 ms
after the injected stall ended. The return did not reach home before reacquisition.
The target was already held inside the deadband before the stall, so this trial
does not demonstrate braking from active following. The partial MP4 also passed
an independent complete FFmpeg decode. A later register receipt after the clean
restart records no capture owner and both outputs disabled.

### Outstanding gates

The user deferred the one-hour run until another time and requested completion
of the current wrap-up. No further hardware tests are authorised for this
wrap-up. The prepared endurance actor has not been executed; the endurance gate
remains pending rather than waived.

The read-only wrap-up receipt found no capture, controller or telemetry process,
and PCA9685 registers 9 and 13 both read 16 (FULL_OFF). The Pi checkout remained
clean at `d55db25`. The task's SSH master and its local tunnels were then closed.
Re-establish the connection in a future session before resuming hardware work.
The receipt is `outputs/validation/monitor-20261027/wrap-up-safety.json`.

Independent Opus review found no software shipping blockers. Before the deferred
endurance actor is executed, fix its end-of-playlist race (E1), export all initial
browser records (E2), and replace the outer inhibitor launcher process rather
than orphaning cleanup on Ctrl-C (E3). The review recommends E4–E6 for telemetry
transport recovery, early child-death detection and uncensored latency reporting.
Pin endurance pass/fail gates before launch. Every matrix sample already has the
historical under-voltage latch set, so establish a fresh baseline when hardware
work resumes; no reboot was performed during wrap-up. These are future tasks,
not completed fixes or accepted new gates.

The report is `outputs/validation/monitor-20261027/opus-pre-endurance-review.md`;
its scratch scripts and re-run evidence are preserved in `opus-review-diagnostics/`
under the same directory. The corrected selection analysis preserves its earlier
version in `selection-loss-reacquisition/pre-review/`. The full-visibility guard
changed zero emitted centres or visibility flags in all four report pairs; the
earlier user-facing explanation that settling masked affected frames is
superseded by `truth-preparation/visibility-guard-audit.json`.

Confirm the exploratory centring result on fresh frozen-policy data and resolve
the partial-occlusion retention decision in a later session. A clean
15-second preview-and-recording restart after the injected
capture failure exited 0, exported successfully, and read FULL_OFF on both
channels. Revisit the hour-long campaign in a later session after the short cases
pass and the user resumes that work. Centring needs
independent camera truth with time-weighted exclusions. Four corner markers alone
do not establish a valid homography on the curved monitor. No centring or
one-hour acceptance is claimed yet. The fixed-zone feature remains deferred.

Local evidence is under `outputs/validation/monitor-20261027/`. Device evidence is
under `outputs/setup/`, `outputs/monitor-hardware-smoke/`,
`outputs/monitor-hardware-100us/`, `outputs/monitor-hardware-startup-fixed/`,
`outputs/monitor-axis-calibration/`, `outputs/monitor-hardware-clean-campaign/`,
`outputs/monitor-hardware-clean-2fps/`, `outputs/monitor-hardware-clean-passive/`,
`outputs/monitor-hardware-fresh-190ms/` and `outputs/monitor-thread-probe/`. Raw media,
room images and calibration remain outside Git.

## Pi bring-up, 27 October 2026, in progress

SSH access to the assembled Pi, IMX708 camera, and pan-tilt mount is working.
The device is an 8 GB Raspberry Pi 5 Model B Rev 1.0, running 64-bit Debian 13.7,
kernel `6.18.50+rpt-rpi-2712`, and Python 3.13.5. The user reports the official
Pi 5 supply powers both the Pi and mount.

### Software and camera checks

Created `/home/columba/columba-demo` separately from the existing
`/home/columba/columba-zero` checkout. The older checkout, its modified model
metadata, and its local calibration remain untouched. The new deployment is
revision `a999b98` with working changes overlaid. Per-file SHA-256 values are
recorded in `outputs/setup/deployment-final.json` on the Pi. Earlier deployment
manifests describe earlier checkpoints.

- Installed CMake, OpenCV development libraries, SciPy 1.15.3, and a separate
  system-site-packages environment. Runtime OpenCV 4.10.0 and NumPy 2.2.4 come
  from OS packages. Both model packages pass `prepare_models.py` checksums.
- Built pinned NCNN and the native extension. CTest passes its native test.
  The Pi Python suite passes 576 tests and 355 subtests, with one optional
  comparison skipped because Ultralytics is absent. The other five native tests
  execute. Mocked evaluation tests now supply package metadata themselves.
  No Torch, CUDA, or pip OpenCV was added to this native runtime.
- Pi Ruff lint and formatting pass. The full Pi mypy run reports 47 errors
  involving absent optional Torch imports, missing OS OpenCV typing, and OS stub
  differences. The desktop gate passes 571 tests and 345 subtests, with six
  native-extension skips, Ruff lint and formatting clean, and strict mypy clean
  across 108 source files. These are separate environments and checks.
- The IMX708 captures through the application. The sensor is physically inverted.
  `camera.rotation_deg: 180` applies both libcamera flips, and the application
  verifies the applied transform before starting capture. A real preview image
  was checked upright. Hardware following rejects calibration made for a
  different capture orientation before opening the driver.
- A real browser decodes advancing 800×450 frames over the SSH tunnel. In
  simulation, Pause reaches `paused`, Home stays paused, and Resume returns to
  acquisition. The indoor scene contains no representative pigeon target.
- A live simulated-following run with preview and recording retained 573 JPEGs
  with no writer-queue drops. The exported H.264 MP4 decodes at 800×450 and lasts
  58.152 s against 58.150863 s expected. Source JPEGs and metadata remain.
- A 60-frame positive replay selected a target in 59 frames and ended cleanly.
  It configured hardware mode and a nonexistent calibration path, but correctly
  ran in simulation without loading that file. Replay logs `as_stored` orientation
  and applies no extra rotation. The repeated historical image is a smoke test,
  not representative field or closed-loop evidence.

### Cooling and live processing

The first four-job NCNN build reached 85.1 degrees C and was stopped. The user
confirmed that the fan on the dedicated four-pin connector was spinning, despite
0 RPM telemetry, and authorised continued work at high ambient temperature.
Active fan trip temperatures were lowered to 45, 50, 55, and 60 degrees C.
The same settings are saved under `[pi5]` in `/boot/firmware/config.txt`, with a
backup at `/boot/firmware/config.txt.columba-before-cooling-20261027`.
The live values were read back. The boot configuration has not been tested by
rebooting. PWM levels, hysteresis, firmware temperature limits, and the 110-degree
critical trip remain unchanged.

The incremental build completed at two jobs in 98.125 s. Its 49 two-second samples
peaked at 79.6 degrees C, with no active throttling sampled. Job count also changed,
so this does not isolate the fan curve's effect. Live camera workloads later
reached 85–86 degrees C, with active thermal limiting and a 1.5 GHz CPU frequency
observed. Fan tachometer telemetry remains 0 RPM and is not evidence that the fan
is stopped. Earlier undervoltage remains recorded, but no active undervoltage was
observed in these checks. Flag interpretation follows the
[official Pi documentation](https://www.raspberrypi.com/documentation/computers/os.html#vcgencmd).

Single live runs using the shipped native model produced these results:

| Workload | Frames | Processing FPS | Capture drops | Peak sampled temperature, degrees C |
|---|---:|---:|---:|---:|
| Detection, four threads, active OpenMP wait | 900 | 10.35 | 1674 | 85.35 |
| Detection, four threads, passive wait | 450 | 12.39 | 605 | 85.90 |
| Detection, two threads, active wait | 600 | 11.06 | 994 | 85.35 |
| Simulated following, preview and recording, passive wait | 600 | 10.26 | 1122 | 85.90 |
| Rotated capture, simulated following and preview, passive wait | 600 | 10.75 | 1042 | 85.35 |
| Rotated capture at 15 FPS, preview at 5 FPS, simulated following | 600 | 11.84 | 141 | 85.90 |

The first five runs request 30 FPS capture and 10 FPS preview. All end cleanly
with no log drops. Scenes, run lengths, rotation, and starting temperatures differ,
so these are diagnostic observations, not controlled comparisons. The 15 FPS
processing target is unmet. The lower-rate configuration was retained as an
experiment, not adopted as the default. `outputs/run-preview.sh` launches the
upright, native, loopback preview with simulated following and passive wait.

### Small movements and controller timing

The user observed and approved the small movement checks. Both channels moved
cleanly with cable clearance in the initial 10-microsecond test. That step was
too small for reliable image displacement, so it was not used to establish
configured-image directions.

The separately approved 50-microsecond check used actual rotation-180 application
frames, returned home after each axis, and disabled outputs. Pan produced a
background displacement of +36.80 pixels horizontally, and tilt produced
−24.55 pixels vertically. These establish pan direction −1 and tilt direction +1
for that capture orientation. PCA9685 FULL_OFF readback was true for both channels.

A separate local rotation-180 profile preserves the old home pulses, records the
measured directions, and restricts each axis to home plus or minus 50 microseconds.
The original profile and pulse-preserving migration candidate remain untouched.
Physical degrees, full travel, field of view, and cable clearance outside that
small range remain unmeasured. The hardware prerequisite report passes for the
new profile and rejects the orientation-mismatched candidate.

A real hardware `FollowSession`, paused before startup, reached the saved home,
held without target corrections, and closed cleanly. With command-completion
telemetry enabled, a ten-second paused run recorded 199 hardware ticks. Its 198
intervals averaged 50.0689 ms, with a 50.1092 ms maximum and a histogram p95 of
50.1 ms. No log records were dropped. Both channels read back FULL_OFF afterward.
This measures scheduler and log cadence without camera, inference, or I2C position
writes during ticks. It does not establish 20 Hz under tracking load.

`controller_state` now records monotonic tick-start and driver-return timestamps.
The summariser separates simulation from hardware and sensor from delivery timing.
It reports coverage and excludes holds, home moves, and replay from tracking-command
ages. The paused run contains zero hardware tracking position moves, so both age
distributions are null. There is no measured tracking-latency or physical-response
claim. Claude Opus 5.5 independently reviewed the orientation and telemetry changes
and found no material issues. Its diagnostic, coverage-wording, and unit-naming
suggestions were also applied.

### Remaining acceptance work

Automatic physical following remains off, and no application service was enabled.
Next, test a visible stationary target, then a walking target, using the reduced
speed and narrow profile while the user observes. Verify correction direction,
centring, overshoot, abrupt holds, stale observations, and shutdown before widening
travel or increasing speed. Record command ages during actual tracking, with
sensor and delivery timing kept separate.

Sustained 15 FPS processing, cadence under tracking load, p95 tracking-command age
below 200 ms, centring, real-pigeon association and switching, wider mechanical
travel, direct LAN overhead, and one-hour stability remain unverified. Follow
[the staged hardware runbook](../docs/pi-hardware-testing.md) for these tests.

Device evidence remains under `/home/columba/columba-demo/outputs/setup/`, with run
logs in `outputs/pi-*`. Local ignored copies are under
`outputs/validation/pi-bringup-20261027/`. These include deployment manifests,
build and test logs, per-run summaries, media checks, direction measurements, and
`paused-controller-cadence-summary.json`. Calibration files and room images remain
outside Git.

## Pi test preparation, 27 October 2026

Added [the SSH hardware test runbook](../docs/pi-hardware-testing.md) and
`tools/pi_preflight.py`. The tool checks the selected environment, dependency
imports, model metadata, camera utility availability, and optional recording and
mount prerequisites. Picamera2, libcamera, and smbus2 are checked for installation
without importing them, because importing Picamera2 can query video devices.
Their runtime importability remains unverified. The tool does not start capture
or inference, open the I2C bus, or issue servo commands. A passing prerequisite
report does not establish hardware compatibility or acceptance.

The actual CLI ran on this Mac and returned status 1 with the expected failures
for the Pi platform, Picamera2, libcamera, native extension, and camera utility.
Python, NumPy, PyYAML, OpenCV, psutil, and shipped model metadata checks passed.
Focused tests cover import failures and timeouts, conditional hardware and
recording dependencies, invalid configuration, I2C permission checks, and actual
child-process checks proving hardware-library initialisers do not execute.
At this preparation checkpoint, the full software gate passed 539 tests, with
6 native-extension skips and 307 subtests. Ruff lint, formatting, and strict
mypy were clean.
Claude Opus 5.5 independently verified these results and found no material
issues. Its small runbook sequencing correction puts the SSH tunnel before the
short camera run, so the browser can connect while frames are available.

At this checkpoint, controller snapshots reported observation age but did not
timestamp completed I2C writes or prove worker cadence. That instrumentation was
still needed before measuring the frame-to-command and 20 Hz targets. Physical
response required observation on the assembled mount. No Pi or mount was used
for this preparation work.

## Pan-tilt software, 26 October 2026

The software portions of demonstration stages 1–4 are implemented on
`feature/pan-tilt-demo`. Following, hardware mode, browser preview, and demonstration
recording remain opt-in. No Pi, CSI camera, or mount was available. The fixed-zone
stage remains deferred until the core demonstration passes hardware acceptance.

- `python tools/verify_demo.py` passes: 577 tests, 7 skipped, and 320 subtests.
  Ruff lint and formatting pass, and strict mypy passes across the application,
  tests, tools, cluster, training, and native Python sources. The seven skips need
  the unbuilt `columba_native` extension. C++ tests were not rerun.
- Driver/calibration tests cover nominal pulse mapping, measured travel bounds,
  partial bus failures, preservation of the first error, atomic profile saving,
  permissions, non-ASCII input, and subprocess terminal restoration after Ctrl+C,
  SIGTERM, and SIGHUP. No test sent commands to physical hardware.
- Replay tests cover shared identities, high/low-confidence association, masked
  sparse optical flow, confirmation, area smoothing, switching, occlusion, source
  clocks, stale observations, acceleration, limits, and home/pause behaviour.
  Four start-from-rest cases reproduced an identity break before the Kalman noise
  correction and pass afterward, at 15/30 FPS with 40/80-pixel boxes.
- Runtime tests prove no physical driver or calibration load during replay,
  including when hardware mode is configured. They also cover detection-only
  isolation, cancellation during camera startup, reaction-test isolation, camera
  resets, simultaneous worker/log failures, and controller telemetry backlogs.
- The real shipped NCNN package ran through the Ultralytics adapter on an existing
  five-frame clip. Following selected a confirmed identity, used simulation with
  a missing calibration path, completed cleanly, and produced a five-frame AVI
  that decoded successfully. A recording-only run produced a playable 800×450 MP4
  of 0.334 s from a 0.333333 s source span. Source JPEGs and metadata remain.
- Browser verification used the real HTTP backend and NCNN detections on a static
  reference image with a simulated mount. The browser decoded 800×450 frames;
  pause, home-and-pause, and resume reached the controller; a 375 CSS-pixel view
  had no horizontal overflow. Stopping the server marked the image disconnected
  and disabled controls. This local check processed 2,651 reference frames and
  does not establish live camera, centring, or LAN behaviour.
- Real socket tests cover state, control requests, connection bounds, slow clients,
  and encoder/storage faults. Real FFmpeg tests cover irregular timestamp gaps,
  a single-frame hold, odd image dimensions, decode counts, duration verification,
  and source retention. Forged rebinding hostnames are rejected. Cancellation tests
  prove child-process reaping during both FFmpeg and ffprobe, with signal-handler
  restoration through the runtime. Manual re-export through `tools/export_demo.py`
  succeeded. A real browser showed the recorder's byte-limit reason while preview
  continued streaming.
- `tools/benchmark_demo.py` ran tracking, preview, recording, and both outputs on
  the same 300-frame, 15 FPS reference replay. All four runs processed 300 frames
  and ended cleanly. Both MP4s decode and measure 20.001 s against an expected
  20.000 s. Recording retained 56 JPEGs, and both outputs retained 52; timestamp
  gaps preserve the source span. An extra final image represents the final hold.
  No browser client was connected during this comparison.
- `python tools/prepare_models.py` verifies the shipped model and fallback. Model
  files, acquisition threshold, native backend default, and rectangular input
  configuration are unchanged.

Environment: macOS 27.0.1 ARM64, project Python 3.14.10, OpenCV 5.0.0, SciPy 1.18.1,
and FFmpeg/ffprobe 9.0.2. Local, ignored artefacts are under
`outputs/validation/pan-tilt-demo/`: `follow-replay/`, `timed-recording/`,
`comparison/benchmark.json`, and `reexport.mp4`. The footage repeats an existing
historical positive tile; it supplies software checks, with no new accuracy or
representative field-footage evidence. The comparison is unpaced desktop replay.

Claude Opus 5.5 independently reviewed the driver, core integration, and complete
software change. Its findings were corrected with regressions. A separate final
review of hostname validation, recording status, and cancellable export found no
material flaws and independently passed the 86 preview/recording tests. Pi export
speed remains unmeasured; the default timeout is 900 s with the `veryfast` preset.

Reproduce the software checks and compare a representative clip:

```sh
python tools/verify_demo.py
python tools/prepare_models.py
python tools/benchmark_demo.py /path/to/clip.mp4 --frames 300 --output outputs/demo-benchmark
```

Add `--backend ultralytics` when the native extension is unavailable. See
[the demonstration guide](../docs/pan-tilt.md) for calibration, preview, recording,
and export recovery.

At this desktop checkpoint, pending physical measurements included Pi 5, HAT,
and camera compatibility, measured angle mapping and direction, safe travel and
cable limits, gains and stopping behaviour,
15 FPS throughput with 20 Hz control, p95 frame-to-command age, centring, switching
under occlusion, LAN overhead, and one-hour memory/thermal stability. No hardware
acceptance target is claimed by these desktop results.

## Shipped model — 16 October 2026

The deployed package is `yolo26n-v2s-800-round2-plain-seed1-yolo11-fp16`
(448×800, fp16, yolo11 decode, threshold 0.62646484375, native backend, k2/n3,
800×450 lores capture at 30 FPS). Selection evidence is in the
[WS7 report](../experiments/reports/WS7-final.md); highlights:

- Test R@P90 at deploy size 0.686 ± 0.008 (3-seed mean, **measured** on xcng
  with `columba_native`), against 0.222 ± 0.013 for the baseline reproduction.
  Shipped seed: test empty FAR 0, confuser FAR 3/73.
- Pi 5 frame p95 36.5 ms (**estimated** from xcng with the published
  calibration factor) and reaction p95 345 ms at 30 FPS, k2/n3 (**simulated**).
- fp16 passes the export gate: its fp32 sibling passes strict parity and fp16
  loses at most 0.01 of val/test R@P90, AP50 and operating recall. Int8 was
  studied and rejected ([WS7-quant](../experiments/reports/WS7-quant.md)).
- On Mac vexcalibur, `prepare_models.py` verified the package bytes and C8
  metadata, and a native replay processed 20 frames with one confirmed trigger.
  Unpaced replay proves loading and pipeline operation, not camera reaction time.

`pigeon_baseline_v1-2` stays in the manifest as the fallback and is verified by
the same command. Reproduce the automated checks from the repository root:

```bash
python -m pytest
python -m ruff check app tests tools cluster training native
python -m ruff format --check app tests tools cluster training native
python -m mypy
python tools/prepare_models.py
```

The sections below record the earlier validation of `pigeon_baseline_v1-2`.

## Baseline v1-2: completed on Windows desktop, 10 October 2026

- 25 automated unittest cases passed: configuration and old CLI rejection, model
  metadata/files/classes, inference filtering/warmup, sequential video completion,
  bounded live buffer and drops, capture errors/timeouts, GUI exit behavior,
  JSONL output, recording failure, startup/runtime cleanup, frame limit, and
  absence of GPIO imports, CLI overrides, SIGTERM handling/restoration, and
  cleanup failures.
- Runtime model copies match SHA-256 values in `model_manifest.json`.
- CPU PT/NCNN comparison: **63 test images, zero comparison failures** at confidence
  0.5, input 640, square preprocessing, NMS IoU 0.7. All counts matched and matched
  boxes met IoU >=0.8 and confidence difference <=0.05. See
  `export_comparison_summary.json`; raw predictions are in ignored
  `outputs/validation/export_comparison.json`.
- Authoritative `best.pt` evaluated on the existing **test** split: 63 images,
  183 labeled instances, zero background-only images. Precision **0.9049**,
  recall **0.8318**, mAP50 **0.9048**, mAP50–95 **0.5728**. These are test metrics,
  separate from the training log. Confidence floor 0.001 enables AP integration;
  precision/recall are curve-selected, not fixed-threshold 0.5 measurements.
  See `pt_test_metrics.json`. Split independence/session provenance was not audited.
- Real NCNN runtime processed a five-frame AVI made from held-out images in
  sequence, exited at EOF, logged all five frames with zero drops, and produced
  an annotated MJPEG AVI that decoded to five frames. GUI was disabled for this
  integration check; window behavior was tested with mocked OpenCV calls.

Environment: Windows 11, Intel Core i7-13700K, Python 3.12.10,
Ultralytics 8.4.175, NCNN 1.0.20260526, Torch 2.14.1+cu132 (CPU inference),
OpenCV 5.0.0.93, NumPy 2.5.3. Desktop timing was measured while two validation
processes ran concurrently; do not treat it as a performance benchmark. The
existing desktop venv contains both OpenCV pip variants; new installs must use
only one GUI-capable OpenCV provider.

The evaluator could not write its optional label cache outside the workspace;
validation completed successfully without it. Original labels were unchanged.

## Baseline v1-2: deployment fix and cleanup checks, 12 October 2026

The manifest originally hashed the Windows CRLF copy of NCNN `metadata.yaml`,
447 bytes with SHA-256
`16c589f246b1531c072a4c248629a694a6aa4f03d1f271a1957a9dcc38155f84`.
Git tracked LF bytes, so model preparation failed on the local LF checkout in
the same way as the reported clean Linux checkout. The canonical entry now
records 427 bytes with SHA-256
`180b7e93ae87e5fa4848da6a598c06001362c3b3c410c8bf9412cae8158bae65`.
`.gitattributes` enforces LF for this metadata and exact bytes for the other
four manifest files. Their checksums are unchanged.

- 33 automated unittest cases passed, including the original 25 runtime cases.
  Added checks cover tracked manifest bytes, clean checkouts with both
  `core.autocrlf=false` and `true`, source and copy corruption, export matching,
  streaming summaries, benchmark resource release, and final log failure.
- Model preparation passed and verified all five source and copied files.
  It now checks recorded sizes as well as SHA-256. A Linux execution was not
  repeated here because the local Docker daemon was unavailable.
- Ruff lint and formatting checks passed. Strict mypy passed on application
  code, tests, and tools. Ultralytics and Pi camera bindings are excluded from
  import analysis because their type information is incomplete; prediction
  protocols describe the detector boundary.
- Real NCNN replay processed a five-frame diagnostic AVI in order, logged zero
  drops, exited at EOF, and recorded an AVI that decoded to five frames. Counts
  were `0, 1, 0, 1, 0` on alternating blank frames and a pigeon tile from existing
  annotated validation imagery. GUI calls remain covered by mocks.
- Real CPU PT/NCNN comparison passed on three smoke images at confidence 0.5,
  input 640, square preprocessing, and NMS IoU 0.7. The positive tile had one
  detection in each model, matched IoU 0.99747, and confidence delta 0.000064.
  These derived images do not constitute independent accuracy validation or
  representative negative footage. The earlier 63-image test results remain
  the recorded dataset evidence and were not rerun.
- All eight supporting tools accepted `--help`. The video benchmark processed
  five frames, the summariser read the real replay log, and `pip check` passed.
  Training and export generation were not rerun, preserving the trained files.

Environment: macOS 27.0.1 ARM64, Python 3.14.8, Ultralytics 8.4.175,
NCNN 1.0.20260526, Torch 2.14.1, OpenCV 5.0.0, and NumPy 2.5.2.
Smoke outputs are in ignored `outputs/validation/style-cleanup/`, including
`positive-result.json`, `positive-comparison.json`, replay JSONL, and recording.
Desktop smoke timing is not a Pi performance result.

Reproduce the automated checks from the repository root with the desktop
development dependencies installed:

```bash
python -m unittest discover -s tests -v
python -m ruff check app tests tools
python -m ruff format --check app tests tools
python -m mypy
python tools/prepare_models.py
```

## Baseline v1-2: review fixes verified, 12 October 2026

- All 41 unittest cases passed. New regression cases reproduced the failures
  before the fixes: an incorrect post-training checkpoint path, an autofocus
  request on a camera without that control, order-dependent export matching,
  decoder failures reported as successful benchmarks, warm-up included in
  measured latency, and output cleanup failures missing from `run_end`.
- Training evaluation now reads the actual trainer checkpoint path, including
  incremented run names. The regression uses a simulated training run; training
  was not repeated and original weights were unchanged.
- Camera configuration tests cover advertised and absent autofocus support.
  These checks use mocks and do not establish physical camera compatibility.
- Export matching uses maximum-cardinality bipartite matching over same-class
  pairs meeting both configured tolerances. Regression checks cover detection
  permutations, reassignment of earlier matches, confidence constraints, and
  custom tolerances.
- Benchmarks warm up before timing and reopen the video to measure all decoded
  frames. Deterministic timing tests exclude a simulated 100-second warm-up.
  Real NCNN benchmarking processed the five-frame diagnostic AVI. A deliberately
  truncated copy exited nonzero at frame 2 of 5 and wrote no benchmark report.
- Recording and display cleanup failures now produce `cleanup_error` records,
  retain any previous failure, and leave summaries with `completed_cleanly: false`.
  Both resources are released once, and log closure is still attempted.
- Real NCNN replay processed and recorded five frames in order, with counts
  `0, 1, 0, 1, 0`, zero drops, and clean EOF completion. The recorded AVI decoded
  to five frames. Real CPU PT and NCNN predictions passed comparison on the
  three existing smoke images.
- Ruff lint and formatting checks and strict mypy passed. The desktop environment
  is the same as the earlier 12 October checks. New smoke outputs are in ignored
  `outputs/validation/review-fixes/`.

These smoke clips reuse existing derived imagery. They do not add independent
accuracy evidence or establish Pi performance. Reproduce automated checks with
the commands above; hardware and field-footage acceptance remains pending below.

## Earlier physical Pi 5 acceptance scope (WS10)

Before the 27 October bring-up, the pending work included Pi OS installation,
the NCNN and native extension builds, and ARM dependency
compatibility; measured latency calibration; 800×450 lores capture geometry,
CSI colours and box alignment; monitor interaction, camera restart/shutdown and
recording overhead; one-hour memory/thermal/throttling behaviour; reaction time
for both cameras by day and night; the OpenMP wait policy; and site footage
(`columba-test-site-v1`), the site fine-tune and the final threshold and k/n.
No Raspberry Pi or camera was connected at that checkpoint. The current results
and remaining acceptance tests are recorded at the top of this document. Follow the
[Pi acceptance checklist](../RASPBERRY_PI_PROTOTYPE_PLAN.md) and append actual
measurements here.
