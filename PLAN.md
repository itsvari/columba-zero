# Columba Zero: pan-tilt integration and demonstration video

## 1. Delivery status and next-session handover

### Monitor campaign update, 27 October 2026

On 28 October the user resumed the campaign, confirmed clear cables and readiness
to observe, and explicitly permitted movement checks without continuous physical
supervision. The previous one-hour deferral is superseded by that instruction.
Keep the tested ±100 µs envelope, reduced speed and independent stop deadlines.
The tracker fix at `1f518f2` prevents low-confidence partial boxes from overwriting
learnt size and velocity. A 21-observation regression fails at frame 5040 before
the fix and passes afterwards; acquisition- and maintenance-confidence walking
cases also pass. Runtime expiry and 250 ms freshness are unchanged.

The endurance actor's E1–E6 fixes pass 16 offline tests, including real signal
cleanup. The desktop gate passes 839 tests and 369 subtests; Pi passes 837 tests
and 375 subtests. Strict desktop mypy, Ruff, Pi CTest and model checks pass.
The first desktop browser gate failed because its temporary executable was gone;
the restored pinned browser gate passes and both logs remain. Focused independent
review, a fresh short confirmatory recording and endurance are still pending.
The frozen criteria are in `deploy/monitor-endurance-policy.json`; the new
duty, cadence and memory limits are engineering budgets, not historical results.

The user withdrew the 15 FPS goal and subsequently accepted the earlier hot
10–12 FPS range near 83–85°C for the steady-state demonstration, provided control
timing and stability pass. Record and report the actual hot throughput, temperature
and throttling state. Continue measuring command age, centring, recovery and
stability. The historical handover below retains its original measurements and
targets; the acceptance section now reflects this decision. The user later relaxed
the demonstration latency goal to **300 ms at p95**, retaining smooth control.

The user authorised wider movement and reported smooth movement with cable
clearance during the completed slow sweep through both ends of both axes. The first
expanded profile permits home ±100 µs at the existing reduced speed. A 30-second
run issued 160 timed hardware tracking moves, with 198.0 ms sensor-to-driver-return
age at p95, then returned home and disabled both channels. This is one short run,
not full acceptance or proof that the entire expanded envelope has been checked.
The initial ±50 µs run only acquired home and made no tracking corrections.

The test monitor is a 34-inch, 1500R curved ultrawide, approximately 38 cm from
the camera. Its quoted 350–400 cd/m² brightness is a specification, not a
measurement. Reduced, repositioned stimuli put all four markers within view.
A global planar homography needs independent validation on this curved surface.
The powered home view differs substantially from the earlier unpowered view;
establish framing at powered home before interpreting following results.

Monitor tooling and its usage are in [docs/monitor-stimulus.md](docs/monitor-stimulus.md).
Measurements and outstanding gates are in [deploy/VALIDATION.md](deploy/VALIDATION.md).
Source revision `6fcf7e4` prepares tracking dependencies before camera and mount
startup. Its live repeat recorded a maximum controller interval of 52.533 ms,
158 timed tracking moves, and 193.9 ms sensor command age at p95. These short-run
figures do not represent sustained timing: three full 201-second tuning runs
recorded 221.3, 221.8 and 221.6 ms p95 command age, failing the 200 ms latency
goal. Their hot throughput was approximately 11.2 FPS at median 84.8°C with active
throttling and peaks up to 87.55°C. A 190 ms observation-timeout experiment capped
move-age statistics while causing frequent stop-start corrections; it is not
adopted. The user initially retained the 200 ms p95 latency goal, then explicitly
relaxed it to 300 ms after reviewing the hot measurements and cooling arrangement.
The official active cooler has slightly restricted airflow beneath the HAT.
Keep the original 250 ms freshness policy and preserve smooth control.
Four full tuning-playlist runs cover tracking alone, preview alone, recording
alone and both outputs. Command-age p95 was 211.1–234.0 ms and uncensored
tracking/stale observation-age p95 was 226.8–249.4 ms, within the revised goal.
Hot processing throughput was 10.67–11.69 FPS after 120 seconds; current thermal
throttling occurred and capture-period CPU temperature peaked at 88.1°C. These
are single sequential runs, not a performance ranking or endurance result.
Pause/home/resume, an unread preview client, recording-write failure and an
injected camera stall/read failure have hardware evidence. Independent centring,
selection and endurance remain open for overall acceptance. The offline scorer
now gives an exploratory 80.957% conservative centring lower bound for the retained
passive recording, with empirical uncertainty and unknown time included. The
similar-size and brief-challenger scenes did not switch; a sustained larger
challenger caused the sole switch. Independent Opus review reproduced the scoped
score but found that the partial-occlusion ID reset was an association failure:
the Kalman box prediction collapsed to one pixel wide before retention expired.
The earlier processing-age explanation is superseded. There are no software
shipping blockers. Centring remains exploratory because the profile qualification
was added after scoring, and the narrow pass needs a fresh confirmatory recording.
The hour-long test has not yet run; the user has now resumed that work.

Before a future endurance run, fix the prepared actor's completion race, initial
browser-record export and inhibitor-wrapper process handling (review E1–E3), then
pin the pass/fail gates. Recommended E4–E6 improve evidence recovery and latency
reporting. Decide whether to fix partial-occlusion identity retention before that
run, and establish a clean throttle-latch baseline when hardware work resumes.
The current 300 ms move-age check is bounded by the 250 ms freshness cutoff;
report uncensored ages and an explicitly defined first-use latency measure too.
The full review and preserved diagnostics are under
`outputs/validation/monitor-20261027/`.

Use this plan to implement the demonstration on the current application architecture,
reusing the useful hardware and controller work from
[PR #2](https://github.com/itsvari/columba-zero/pull/2). This document revises the
supplied demonstration plan with the findings from the Codex, Claude Opus 5.5, and
DeepSeek reviews on 26 October 2026.

### Merged delivery, 27 October 2026

The core demonstration and Pi bring-up changes were committed, pushed, and merged
into `main` at `c9d31487531d5a4477690328f7a4838f8e9a5cab`. PR #2 is merged.
Use current `origin/main` for the next session and branch new code changes from it.
The software portions of stages 1–4 are complete. Following remains opt-in.

The final four commits are:

- `ec3f538`: isolate mocked prediction metadata from installed backends.
- `b3812d0`: configure capture rotation and bind calibration to that orientation.
- `a61b40c`: measure controller command completion and report timing coverage.
- `844ea01`: add Pi prerequisite checks, the hardware runbook, and bring-up evidence.

`feature/pan-tilt-demo` retains that implementation history through `844ea01`.
`integration/pan-tilt-pr2` preserves the earlier imported baseline at `a152b00`.
Its merge retained main's versions of all 14 conflicting files and preserved the
original PR history. These branches are historical references for the completed port.

Initial Pi camera and small-movement checks have passed. Following acceptance
remains pending. Measurements and detailed evidence are in
[deploy/VALIDATION.md](deploy/VALIDATION.md#pi-bring-up-27-october-2026-in-progress).

### Completed software

- Driver and calibration repairs include typed interfaces, nominal angle mapping,
  atomic profile saving, terminal/signal cleanup, and regression tests.
- Following uses shared Kalman identities, high/low-confidence association, masked
  sparse optical flow, confirmation, smoothed area, and target-switch hysteresis.
- The timed controller enforces freshness, acceleration and travel bounds, with a
  single hardware owner, cancellation checks, simulation, logs, and overlays.
- Browser preview provides MJPEG, state, and pause/resume/home controls. Bounded
  JPEG recording exports a timed MP4 after processing stops and retains sources.
- Capture supports 0° and 180° orientation. For 180°, it verifies the applied
  libcamera transform before starting. Hardware control rejects a calibration
  whose image orientation differs from capture before opening the driver.
- Controller events timestamp tick start and driver return. The bounded summary
  separates hardware from simulation and sensor timing from delivery timing.
  It reports missing command timing and explains dropped-sample limitations.
- `tools/pi_preflight.py` checks environment and selected configuration without
  opening the camera or I2C bus or issuing servo commands.
- Following, physical movement, preview, and recording remain opt-in.
  Following requires confirmation. Day/night following and zone activation are
  rejected until their prerequisites are defined or measured.
- Desktop verification, real NCNN replay, browser interaction, and FFmpeg export
  evidence are recorded in [deploy/VALIDATION.md](deploy/VALIDATION.md). Commands
  and configuration are in [the demonstration guide](docs/pan-tilt.md).

The committed desktop gate passes 571 tests and 345 subtests, with six
native-extension skips. Ruff lint, formatting, strict mypy across 108 files, and
both model-package checks pass. Independent Opus reviews found no material
issues after fixes. Stage 6, the fixed-zone feature, remains unimplemented and
deferred until core following passes acceptance.

### Pi state at handover

- SSH access is `columba@columba-pi`. VNC is also available. Keep credentials out
  of the repository and use existing authentication or obtain them privately.
- The assembled device is an 8 GB Pi 5 with an IMX708 camera and PCA9685 pan-tilt
  mount. The user reports that the official Pi 5 supply powers both Pi and mount.
  Historical undervoltage flags were present, with none active at the last check.
  The sensor is physically upside-down, so live configuration uses
  `camera.rotation_deg: 180`.
- The working deployment is `/home/columba/columba-demo`, with its own `.venv`.
  Preserve `/home/columba/columba-zero`, which contains older local changes and
  calibration. The new native runtime uses OS NumPy, OpenCV, and SciPy without
  Torch or Ultralytics. NCNN, the extension, and CTest have run successfully.
- The Pi suite passes 576 tests and 355 subtests. One optional Ultralytics
  comparison skips, while the other native tests execute. Ruff passes. Pi mypy
  has dependency and OS-stub discrepancies recorded in the validation document.
  Do not report the full Pi typing gate as passing.
- Live camera, upright browser preview, simulated controls, and timed MP4 export
  work. Short runs reach about 10–12 FPS, below the 15 FPS target. Performance
  optimisation may require code changes after controlled profiling.
- The fan uses the dedicated four-pin connector. The user observed it spinning
  despite 0 RPM telemetry. Fan trip temperatures are 45, 50, 55, and 60°C live
  and saved in boot configuration. Persistence has not been checked by rebooting.
  Firmware temperature limits are unchanged. Live workloads reached 85–86°C
  with thermal limiting. The user accepts operation around 80°C in Singapore's
  ambient conditions. Do not impose an arbitrary 80°C abort threshold.
  Monitor sustained throttling and throughput against the acceptance targets.
- Supervised 10 µs and 50 µs nudges returned home and disabled outputs. The 50 µs
  test established pan direction −1 and tilt direction +1 in rotation-180 capture.
  A separate profile restricts both axes to the saved home plus or minus 50 µs.
  Full travel, physical angle mapping, and field of view remain unmeasured.
- A paused hardware worker recorded 199 ticks over ten seconds, averaging
  50.0689 ms between ticks. Both channels read back FULL_OFF after shutdown.
  Holds perform no I2C position writes, so this proves neither tracking cadence
  under load nor tracking-command latency. There were zero tracking moves.
- At handover the application is stopped, outputs are disabled, and no auto-start
  service is enabled. Automatic physical following has never been tested.
  Verify the current state when resuming.

The Pi still has revision `a999b98` plus the tested source overlay, not a clean
checkout of the merged commit. Its file hashes are in
`outputs/setup/deployment-final.json`. Logs and measurements are under
`outputs/setup/`, with run outputs in `outputs/pi-*`. Local ignored evidence is
under `outputs/validation/pi-bringup-20261027/` on the development computer.

Preserve these ignored device files when updating source:

- `outputs/setup/calibration-rotated.local.yaml`: the narrow rotation-180 profile.
- `outputs/pi-hardware.yaml`: reduced-speed hardware configuration using that profile.
- `outputs/pi-simulate.yaml` and `outputs/run-preview.sh`: upright simulation and preview.
- Existing calibration candidates, original profiles, recordings, and setup evidence.

### Resume in the next agent session

Testing is intentionally deferred to another session. No additional movement or
endurance run is authorised by this documentation update.

1. Read this handover, the validation record, and
   [the Pi hardware runbook](docs/pi-hardware-testing.md). Use current `main`
   rather than the runbook's older feature-branch reference.
2. Reconnect over SSH and inspect processes, PWM state, source revision, local
   changes, and configuration. Preserve the existing deployment and ignored files.
   Reconcile its tested overlay with current `main` before updating source.
   Do not discard the older checkout or its calibration.
3. Set up the physical monitor controlled by the computer running the agent.
   Implement the test presentation and evidence collection described below.
   Confirm through the Pi preview that the displayed content is actually visible.
4. Start with live camera capture and simulated following. Verify a known positive
   image, upright capture, display framing, focus, and the test sequence.
5. Before the first short physical-following run, confirm the user is beside the
   mount and agrees the bounded test. Retain the narrow profile and reduced speed.
   Earlier approval covered manual nudges, not autonomous following or wider travel.
6. Run the monitor-based campaign below, tune against recorded results, and repeat
   the software gate after code changes. Keep throughput, timing, centring, and
   physical observations separate in the report.
7. Run the one-hour test only after short tracking, limits, shutdown, and restart
   cases pass. Leave the fixed-zone feature and unattended startup deferred.

To inspect the controller and tests retained in history, use:

```bash
git show c89dab40a426a0dc7abd0d7304c76a3527e30242:app/tracking.py
git show c89dab40a426a0dc7abd0d7304c76a3527e30242:tests/test_tracking.py
```

### Completed port decisions

- Kept the PCA9685 driver, calibration profile, manual calibration workflow, mock
  driver, and applicable tests after fixing the confirmed defects.
- Reused proportional movement, speed and travel clamping,
  independently timed worker, latest-observation handling, and replay clock.
- Replaced the PR's nearest-centre association and raw-detection acquisition with
  shared motion identities and confirmed target selection.
- Added the selected-target overlay, identity, deadband, limits, lifecycle events,
  and timing through the existing output path.
- Retained the shipped detector and deployment settings rather than the PR's older
  model, single-class runtime, or synchronous output loop.

### Preserve main's architecture

`app/tracking.py` retains confirmation for detection-only operation. Following uses
the shared identities in `app/motion_tracking.py`. Camera control lives separately
in `app/pan_tilt_control.py`, with its own tests. Preserve that division and do not
restore the PR's competing `app/tracking.py` wholesale.

Integrate target processing in `FrameProcessor.process` in `app/pipeline.py`.
Keep inference, display, and log writing in their existing threads. Carry the
controller snapshot through `RenderJob`, and send events through the existing sink.
The controller worker remains the sole owner of physical driver writes.

Use main's `Detection.left_px`, `top_px`, `right_px`, and `bottom_px` fields, together
with `class_id` and the shared `track_id`. Reuse `FramePacket.delivered_monotonic_s`
and available sensor-derived `delivery_ms`. Do not add the PR's duplicate
`delivered_monotonic` field or call delivery time exposure time.

Add typed, opt-in `follow`, `pan_tilt`, `preview`, and `zone` sections using main's
field-checked configuration loader. Continue rejecting historical `tracking:`
configuration. Preserve the native backend, rectangular input, dependency groups,
strict typing settings, and shipped model configuration.

Disable physical following during `--reaction-test`. Preserve detection-only runs
without a calibration file or hardware dependency. Before enabling following with
day/night switching, define which camera is mounted, how calibration applies to
each camera, and how image orientation is configured. Until that is defined,
reject that combination before opening the driver. Clear retained target state on
every camera change.

### Preserve the confirmed review fixes

These fixes are implemented and covered by regression tests. Preserve them when
changing the software:

1. Respect cancellation before driver construction and before the first position
   command. A stop during camera startup must not initialise or move the mount.
2. Route SIGTERM and SIGHUP during calibration through terminal restoration and
   driver cleanup. Preserve normal Ctrl+C handling and restore prior handlers.
3. Distinguish an empty terminal read from an undecodable byte. A non-ASCII
   keypress must not discard unsaved calibration marks.
4. Normalise pulse-to-angle endpoints so valid envelopes cannot fail validation
   because of floating-point rounding. Include the axis in validation errors.
5. Preserve the first runtime failure. Do not report an already-observed worker
   failure again as a new cleanup failure.
6. Keep `calibration.local.yaml` in `.gitignore`. The completed port restored the
   rule omitted from the imported baseline.
7. Explain the configured-capture direction convention in calibration and its
   instructions. Bind profiles to capture rotation so a changed orientation cannot
   silently invert either axis. A separate preview flip must not change calibration.
8. Meet the repository's strict typing and formatting requirements for the imported
   files and all later changes.

Save calibration profiles atomically so an interrupted write preserves the previous
profile. Log the resolved channels, pulse envelopes, travel limits, home positions,
directions, and effective image orientation for each physical run. Keep local
calibration data out of version control.

Do not add a second I2C writer to force shutdown while the controller worker is
stuck. Keep the existing single-owner rule and report the shutdown failure.

## 2. Recommended approach

Keep the existing pigeon detector and add tracking, target selection, and camera control around its bounding-box output. **The first demonstration should require no retraining or detector architecture changes.**

The repository now provides the native NCNN pipeline, shared motion tracking,
target selection, control, preview, and recording. The following design sections
record the implemented core and deferred zone feature. Short Pi runs achieve
about 10–12 FPS. Sustained throughput and following acceptance remain open.

The agreed test programme uses the actual Pi camera and mount pointed at a monitor
controlled by the computer running the agent session. Static and dynamic displayed
targets replace live subjects for this campaign. Real birds are not a prerequisite
for completing the planned bench tests. Keep the resulting claims specific to
monitor-based operation, as detailed under validation below.

Agreed defaults:

- Raspberry Pi 5 with a CSI camera mounted on a pan-tilt assembly.
- Follow walking and perched pigeons; sustained flight is outside first-version acceptance.
- Interpret "closest" as the largest visible pigeon, using smoothed bounding-box area.
- Prioritise a LAN browser view and playable demonstration recording.
- Add the fixed keep-out zone afterward, on one flat physical surface.
- Run detection and control locally on the Pi. GPUs are needed only for optional training.

## 3. Core implementation

### Hardware and calibration

Use the **Waveshare 2-DOF Pan-Tilt HAT with PCA9685** as the reference hardware. Its I²C interface supports a small driver without depending on old camera or GPIO demonstration software. Validate operation on the actual Pi 5 before integrating tracking. [Waveshare manual](https://files.waveshare.com/upload/b/b9/Pan-Tilt_HAT_user_manual_en.pdf).

The assembly is available. Before expanding the tested travel range, verify
camera-bracket fit, cooler clearance, and cable slack throughout that range.
Retain the actual supply arrangement in each run's record.

Do not make the retired Pimoroni HAT the default purchase; its original camera bracket also excludes Camera Module 3. [Pimoroni product information](https://shop.pimoroni.com/products/pan-tilt-hat).

Provide a calibration routine that records:

- Servo channels, direction, centre position, and pulse-to-angle mapping.
- Mechanical and cable travel limits.
- A home position and approximate camera field of view.
- Conservative starting speed and acceleration limits.

### Tracking and target selection

The processing flow becomes:

`camera → existing NCNN detector → motion tracker → target selector → pan/tilt controller`

Retain the shipped YOLO26n FP16 package and its existing acquisition threshold initially.

Add a standalone CPU tracker with Kalman motion prediction, high/low-confidence association, and OpenCV sparse optical-flow camera-motion compensation. Use NumPy, SciPy, and OpenCV, keeping native inference independent of PyTorch. This follows the useful principles of ByteTrack and moving-camera trackers without requiring an appearance model. [Tracking methods](https://docs.ultralytics.com/modes/track/).

Starting behaviour:

- High-confidence detections create tracks and satisfy the existing 2-of-3 confirmation rule.
- Detections down to confidence **0.15** may maintain an existing track, but cannot create or confirm one.
- The tracker supplies the IDs used by confirmation, selection, logging, and display; avoid two independent trackers assigning conflicting IDs.
- Select the confirmed pigeon with the largest smoothed box area.
- Switch only when another candidate is at least **25% larger for 0.5 seconds**, preventing constant switching between similar birds.
- Retain an unmatched identity for **0.75 seconds**, while stopping camera corrections when observations become stale.
- After losing the target, return slowly to home and resume acquisition.

Keep the maintenance detection floor separate from the acquisition threshold.
When following is enabled, retain detections down to 0.15 for association, but use
the shipped acquisition threshold to create and confirm tracks. Keep detection-only
behaviour unchanged. Test that low-confidence detections cannot create tracks,
confirm them, or emit trigger events.

The PR's largest-box acquisition and nearest-centre association are reference code
only. They do not satisfy the shared identity, confirmation, smoothing, or switching
requirements above.

Box area is an approximation: posture, occlusion, and different bird sizes can change the ranking. The interface should describe the selected target as "nearest-looking," without displaying an invented distance.

### Camera control

Run an independently timed controller at **20 Hz**, consuming the latest target state through a bounded mailbox.

Start with proportional control using the target box centre:

- A central deadband of **±5% of image width and height**.
- Initial maximum speed of **30°/s per axis**, with acceleration limiting.
- No integral term initially; tune proportional gain on hardware before adding complexity.
- Hold position when the last observation is older than **250 ms**, capture fails, or tracking becomes uncertain.
- Enforce calibrated travel limits on every command and report when a limit prevents centring.

Use elapsed time and sensor-derived frame age throughout. Preserve the existing delivery-time timestamp semantics rather than relabelling them as exposure time.

Add a mock pan-tilt driver for desktop development and replay. Replay must use the mock driver by default.

Adapt the PR controller rather than importing its defaults unchanged. Change the
one-second observation timeout to 250 ms, implement the separate 0.75-second
identity-retention policy, and add acceleration limiting. Use the frame's delivery
age plus sensor-to-delivery duration when available. Report when sensor timing is
unavailable, and keep delivery-based timing explicitly labelled.

### Configuration and interfaces

Add opt-in `follow`, `pan_tilt`, `preview`, and `zone` configuration sections. Preserve detection-only operation and continue rejecting the obsolete historical `tracking:` configuration.

Introduce small interfaces for:

- `PanTiltDriver`: position commands, hold, close.
- Track and target snapshots: ID, box, confidence, observation time, selection state.
- Controller status: commanded angles, centring error, limits, and fault state.

Extend JSONL logs with target acquisition, switching, loss, controller state, and timing. Include the selected target and centre deadband in the overlay.

## 4. Browser preview and demonstration recording

This milestone is implemented. PR #2 supplied local overlays but no HTTP service, browser
controls, timestamped JPEG recorder, or timed MP4 export. Retain its AVI output only
as the existing diagnostic mode. Do not treat that mode as demonstration recording.

Start with an **annotated MJPEG browser stream at 800×450, capped at 10 FPS**, served by a small local HTTP service. Reuse processed frames so boxes and images correspond exactly.

Provide:

- `/stream.mjpg` for video.
- `/api/state` for target, controller, and performance status.
- A simple page showing the stream, tracking state, and pause/home controls.

Encoding and network delivery run outside the detection/control loop. Slow clients receive the latest frame; they never create an accumulating queue. Encode each selected frame once and share the JPEG between viewing and recording.

**Recording must preserve demonstration timing.** The current fixed-FPS AVI records processed frames and can alter playback speed.

For demonstration recording:

1. Save the already encoded JPEG frames with monotonic timestamps and matching metadata.
2. After detection stops, use FFmpeg to produce an annotated MP4 with timing derived from those timestamps.
3. Retain the source frames if conversion fails.
4. Bound recording duration and disk use.

This moves final video compression outside the live control workload. The Pi 5 uses software video encoding, so neither streaming nor recording should be treated as free processing. [Official camera documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html).

Benchmark tracking alone, then with preview, recording, and both enabled. If video materially affects control latency, reduce the shared output to **5 FPS**, then **2 FPS**. Recording-only mode disables browser delivery; it still incurs encoding and storage work.

## 5. Stretch goal: a zone fixed to the physical surface

Implement this after unrestricted tracking works.

At the home position, let the user freeze a reference image and draw:

- The flat surface used for registration.
- A polygon defining the keep-out zone within that surface.

Register each current image against the reference surface using OpenCV features and a RANSAC homography, masking pigeon detections out of the background features. Use optical flow between reference registrations.

Transform each pigeon's **bottom-centre box point** into reference coordinates and test polygon membership. Select the largest eligible pigeon inside the zone, with brief entry/exit hysteresis. Release it when it leaves and resume acquisition.

This handles camera movement on a plane without requiring a depth network or assuming commanded servo angles equal measured camera angles.

Explicit limits:

- Registration requires sufficient static texture. Add printed textured markers around the demonstration surface if necessary.
- Reject uncertain registration and suspend zone-based selection until it recovers.
- Re-register against the saved reference to limit accumulated drift.
- The zone applies to the calibrated surface; elevated perches and flying birds do not share its geometry.

Store the reference image, polygon, calibration, and registration confidence. Show the projected zone in the browser so its alignment is directly reviewable.

## 6. Validation, delivery order, and optional training

### Delivery sequence

Stages 1–4 are implemented, checked, and merged. Continue from current `main`.
Finish each new software change's checks before starting the next test phase.

1. **Hardware bring-up: software complete, physical calibration partial.** The
   driver, calibration tool, cancellation, terminal restoration, atomic saving,
   pulse mapping, orientation binding, and cleanup have tests. Small supervised
   moves passed. Wider travel and physical angle measurement remain open.
2. **Shared tracking and selection: implemented.** Replay covers prediction,
   confidence passes, optical flow, confirmation, area smoothing, identity
   persistence, and switching. Monitor-camera sequences must validate their
   behaviour under actual camera movement.
3. **Camera control: implemented, closed-loop acceptance open.** Freshness,
   acceleration, limits, replay isolation, and timing telemetry are checked.
   Only a paused physical worker has run. Measure actual corrections, centring,
   command age, and cadence under load next.
4. **Browser preview and demonstration recording: implemented.** Real Pi preview,
   simulated controls, and MP4 export work. Repeat with physical following and
   controlled monitor content in all four output combinations.
5. **Core demonstration tuning: next.** Build the monitor test presentation and
   evidence analysis, run the staged campaign below, and collect acceptance
   results. Keep hardware following opt-in.
6. **Fixed-zone stretch goal: deferred.** Add reference registration, polygon
   membership, uncertainty handling, and projected-zone display only after core
   following passes the monitor-based acceptance campaign.

The original estimate was 4–6 working days for the core demonstration and 2–3 more
for the zone. That estimate predates the merge review. Re-estimate after hardware
bring-up and the shared tracker are verified. Do not count the merge baseline as
completed hardware acceptance or completed tracking and control.

### Verification before delivery

The earlier review measured 61 passing PR tests and 316 passing main tests with
7 skipped. Those are separate revision results, not evidence that the integration
baseline or the future implementation meets this plan. At branch preparation, the
unchanged imported files had strict typing and formatting failures. Those files
and the implementation now pass both checks.

At branch preparation, `python -m pytest` passed on `feature/pan-tilt-demo`:
316 tests passed, 7 skipped, and 209 subtests passed. No hardware acceptance or
review fix is claimed by that result.

Use the full repository checks after each implementation stage that changes code:

```bash
python -m pytest
python -m ruff check app tests tools cluster training native
python -m ruff format --check app tests tools cluster training native
python -m mypy
```

Add regressions for the confirmed review defects and retain existing native
inference, confirmation, reaction-test, timing, and camera-switching coverage.
Exercise the CLI and actual output artefacts as well as unit tests. Record software
checks separately from hardware measurements in `deploy/VALIDATION.md`.

The user authorised the merge to `main`, and it is complete. PR #2 is merged too.
The merge does not establish hardware acceptance. Complete the campaign below
before considering a change to the opt-in default. Start subsequent code changes
from current `main` and keep measured results in `deploy/VALIDATION.md`.

### Monitor-based test campaign

Use the Pi's real CSI camera to observe a physical monitor controlled by the
computer running the agent session. Drive static images, animated targets, and
recorded clips on that monitor. The Pi must continue to run detection, tracking,
and control locally. Do not substitute `--video` replay for this campaign:
replay forces simulation and does not exercise physical following.

This setup is the agreed replacement for difficult-to-control real subjects.
It can qualify the demonstration with reproducible visual inputs and actual mount
motion. It does not establish outdoor accuracy, natural-light coverage, depth
behaviour, or performance on live birds. Those are limits on the claims, not
additional requirements to obtain live subjects for this campaign.

#### Prepare the display and test tooling

1. Place the monitor within the tested movement range, with the camera initially
   aimed near its centre. Record the camera-to-screen distance, screen dimensions,
   resolution, refresh rate, brightness, and visible screen area. Verify clearance
   before changing the mount's physical range.
2. Present the stimuli full-screen on that physical display. Use browser or desktop
   controls available to the agent session. A hidden tab or headless render is
   insufficient. Check through live Pi preview that scene changes reach the camera.
3. Check focus, exposure, glare, visible refresh banding, and moiré with static and
   moving content. Record any display or camera adjustments. Establish one stable
   baseline before varying backgrounds or motion.
4. Build a small local stimulus page or runner with reproducible scene IDs, random
   seeds, target paths, speeds, scales, occlusions, and background choices. It must
   support pause, reset, and replay of an identical sequence. This tooling is
   planned test work, not an existing application feature.
5. Use actual pigeon photographs, cut-outs, or clips. First verify one known
   positive example through the physical display and live camera. Geometric
   markers alone are not a substitute for a pigeon detection test. Include empty
   scenes and confusers, and keep final evaluation sequences separate from tuning.
6. Log intended target identities, positions, sizes, visibility, and presentation
   events. Add a visible scene/frame identifier outside the target region so the
   captured frames can be matched to the sequence. Keep screen-coordinate truth
   distinct from detector IDs and camera coordinates.
7. Add offline analysis for time-weighted centring and expected selection/loss
   behaviour. Existing `summarize_run.py` supplies timing distributions, not this
   full acceptance report. Keep raw captures, scene definitions, and reports so
   another agent can reproduce each result.

#### Run progressively harder scenes

Start in simulation with actual live capture, then repeat bounded cases with the
physical mount once the user approves the short run. Keep the narrow profile and
reduced speed initially. Expand travel or speed only after checking the new range.

- A stationary pigeon at centre, then offsets on each axis and at reachable corners.
  Check correction sign, settling, overshoot, jitter, deadband, and abrupt holds.
- A pigeon starting from rest, moving slowly, stopping, and reversing. Progress to
  diagonal motion, curved paths, speed changes, and scale changes.
- Two or more pigeons of similar apparent size, followed by a persistent larger
  challenger. Test the 25% area advantage and 0.5 s switching hold, including brief
  challenges that must not switch selection. Screen area is the ranking proxy,
  not a physical-distance measurement.
- Brief occlusion, overlap, full disappearance, reappearance, and target departure.
  Check identity retention, stale holds, loss, return home, and reacquisition.
- Plain, textured, cluttered, low-contrast, and natural-scene backgrounds. Include
  confusers and empty periods. Hold the background still initially, so its image
  motion comes from the real camera. Label animated-background cases separately.
- Targets that become unreachable within calibration or leave the visible display.
  Record limit status and distinguish these from missed detections or wrong control.
- Pause, resume, home-and-pause, Ctrl+C, SIGTERM, and restart. Confirm output disable
  on normal shutdown. A disappearing image tests target loss, not a stuck camera.
  Exercise stale observation and capture-failure handling separately using controlled
  fault injection. Keep physical camera disconnection to power-off procedures.

#### Collect comparable measurements

Repeat the same scripted sequences with tracking alone, preview with a real client,
recording alone, and both outputs. Keep camera, model, calibration, display, and
scene definitions fixed. Record starting temperature and warm-up, and repeat runs
to distinguish variation from an output-mode effect. Record export elapsed time
separately from live processing.

Use camera-visible reference markers or validated screen registration to map
scripted target positions into each captured frame. The mapping changes as the
mount moves. Do not measure centring solely against the detector's own boxes.
Validate the mapping or annotate a sample manually. Declare the settling interval
and unreachable, invisible, and registration-failure exclusions before scoring,
and report excluded duration alongside the centring result.

Use Pi monotonic sensor/delivery-to-driver-return measurements for command age.
Host presentation timestamps are not camera exposure timestamps, and host and Pi
clocks must not be subtracted without establishing their relationship. A displayed
frame identifier associates content with observations but does not itself measure
display latency. Report any separately measured stimulus-to-response latency with
its own method and uncertainty.

Inspect timed and untimed hardware tracking-command counts. Keep sensor and
delivery ages separate, and account for log drops and per-command sampling.
A paused scheduler run, simulated commands, and replay do not prove tracking
latency or cadence under load.

After the short cases pass, repeat a declared mixed-scene playlist for at least
one hour with physical following. Record memory, temperatures, throttling,
controller cadence, queue drops, errors, restarts, and recording limits. Preserve
the per-run configuration, source revision, private calibration snapshot, model
hashes, scene manifest, presentation log, camera recording, and summary. Commit
reviewed reports and reusable test tooling. Keep local calibration and raw media
out of Git.

### Acceptance tests

Use [the Pi hardware test runbook](docs/pi-hardware-testing.md) for SSH setup,
non-actuating environment checks, supervised calibration, and staged measurements.
Controller events include monotonic tick and driver-return timestamps. The run
summary reports cadence and hardware tracking-command age separately from
simulation. Collect representative measurements before claiming acceptance.

Use replay for deterministic software checks and the live monitor setup for
closed-loop bench acceptance. Real subjects are not required for this campaign.
Use the acceptance goals below without claiming that displayed-target results
establish field accuracy.

Initial acceptance targets:

- Record sustained hot processing throughput, temperature and throttling state,
  with control running at 20 Hz. The user accepted the earlier **10–12 FPS** hot
  range near **83–85°C**, conditional on control timing and stability, and withdrew
  the earlier 15 FPS requirement on 27 October 2026. The short approximately
  12.5 FPS result was measured while the device was warming up.
- Sensor-to-driver-return command age below **300 ms at p95** during normal
  operation. Report uncensored observation age and stale time alongside it. The
  user relaxed the earlier 200 ms demonstration goal on 27 October 2026.
- After settling, the target centre stays within the central **20% of image width and height for at least 80%** of visible tracking time, within mechanical reach.
- No repeated switching between similarly sized stationary displayed pigeons.
- Brief occlusion, target departure, camera failure, slow browser clients, and recording errors produce predictable recovery.
- Demonstration recording is playable and retains real elapsed timing.
- A one-hour run shows stable memory, acceptable temperature, and no accumulating queues.
- After the deferred zone feature is implemented, zone tests include boundary
  crossings, camera movement, weak texture, and registration loss.

Treat these as acceptance goals to measure on hardware. Diagnose detector misses, identity errors, and controller errors separately.

### Retrain only if detector misses limit the result

First diagnose monitor presentation, exposure, focus, and tracking persistence
using the controlled sequences. Preserve the shipped acquisition threshold for
the initial baseline, and record any later tuning separately. Servo oscillation
or identity switching is not a reason to retrain the detector. Display artefacts
alone do not justify retraining for field deployment.

The following optional field-data and fine-tuning route is outside the monitor
campaign. It is not a prerequisite for bench acceptance.

If persistent detection failures justify a separate training project:

1. **Recover the shipped model's original PyTorch checkpoint from cluster artefacts.** The checkout contains its NCNN export, but not that checkpoint. NCNN is not the fine-tuning starting point.
2. Collect roughly **600–1,000 diverse site frames**, including camera movement, partial birds, lighting changes, empty scenes, and confusing objects. Auto-label, then manually correct.
3. Split by recording session, keeping validation and test sessions separate. Mix site training images with historical data to reduce overfitting.
4. Run one short fine-tune: same architecture and taxonomy, training size 800, **30 epochs with early stopping**, mixed precision, and a reduced learning rate. Start with batch 8 on the 3080 or batch 32 on one A100/H100.
5. Select thresholds and checkpoints using validation footage. Evaluate the chosen candidate once on untouched test footage and tracking clips.
6. Export the same **448×800 FP16 NCNN** format, verify parity with the existing tooling, and measure the complete Pi pipeline before replacing the shipped package.

If the original checkpoint cannot be recovered, fine-tune the repository's available YOLO11n checkpoint as a fallback candidate and promote it only if it improves site results over the current deployment.

A single 3080 is sufficient for this small-model fine-tune. Use an available A100 or H100 to shorten turnaround; avoid distributed training and broad sweeps. Label quality and site coverage will likely dominate the schedule. Keep architecture changes, depth models, appearance models, and another quantisation study outside the first demonstration.
