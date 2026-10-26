# Columba Zero: pan-tilt integration and demonstration video

## 1. Branch strategy and current status

Use this plan to implement the demonstration on the current application architecture,
reusing the useful hardware and controller work from
[PR #2](https://github.com/itsvari/columba-zero/pull/2). This document revises the
supplied demonstration plan with the findings from the Codex, Claude Opus 5.5, and
DeepSeek reviews on 26 October 2026.

### Prepared branches

- `main` remains at `36eab9a92d6808837285da097156ffbd00de511f`.
- `integration/pan-tilt-pr2` contains merge commit
  `a152b00c1bfecc9ed8821e6a0ac056cc15bbba1e`. Its parents are that main commit and
  PR head `c89dab40a426a0dc7abd0d7304c76a3527e30242`.
- `feature/pan-tilt-demo` starts from the integration merge. Its first change adds
  this plan. Make the future implementation changes on this feature branch.

The integration merge keeps main's versions of all 14 conflicting files. It adds
only the unchanged PR files `app/pantilt.py` and `app/calibrate.py`. The merge
preserves the full PR history so the controller, tests, overlays, and documentation
remain available for the later port.

No runtime port or review fix is implemented at this stage. The new calibration
tool still has the reviewed defects. Do not run it against hardware until the
first implementation stage fixes them. The integration branch is a source baseline,
not a hardware-ready release. Its additional files also have known formatting and
strict typing failures.

To inspect the controller and tests retained in history, use:

```bash
git show c89dab40a426a0dc7abd0d7304c76a3527e30242:app/tracking.py
git show c89dab40a426a0dc7abd0d7304c76a3527e30242:tests/test_tracking.py
```

### Reuse the hardware work and adapt the controller

- Keep the PCA9685 driver, calibration profile, manual calibration workflow, mock
  driver, and relevant tests after fixing the confirmed defects.
- Reuse the controller's proportional movement, speed and travel clamping,
  independently timed worker, latest-observation handling, and replay clock.
- Replace the PR's nearest-centre association and raw-detection target acquisition
  with the motion tracker and confirmed target selector specified below.
- Port the selected-target overlay and JSONL fields into main's existing output
  path. Add the planned target identity, deadband, limits, and lifecycle events.
- Keep the existing detector and deployment settings. The PR's older model,
  single-class runtime, and synchronous output loop do not replace main's versions.

### Preserve main's architecture

Keep `app/tracking.py` and `tests/test_tracking.py` for the existing confirmation
behaviour until the shared motion tracker replaces their identity management.
Place camera control in a separate module, such as `app/pan_tilt_control.py`, with
its own tests. Do not restore the PR's competing `app/tracking.py` wholesale.

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

### Fix the confirmed review defects

Complete these fixes as implementation work, with regression tests:

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
6. Restore `calibration.local.yaml` to `.gitignore`. Main removed that rule, and the
   integration merge intentionally leaves existing main files unchanged.
7. Explain the raw-image direction convention in the calibration tool and its
   instructions. A flipped preview must not cause an inverted tilt calibration.
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

The repository already provides native NCNN inference, basic track confirmation, a local display, and diagnostic recording. Its current tracker uses box overlap, which needs improving for a moving camera. Actual Pi performance remains unverified. [Project README](https://github.com/itsvari/columba-zero), [current tracker](https://github.com/itsvari/columba-zero/blob/main/app/tracking.py), [validation status](https://github.com/itsvari/columba-zero/blob/main/deploy/VALIDATION.md).

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

Before purchasing, verify that the bracket fits the chosen camera, allows its full movement, and clears the Pi's cooler. Include the correct Pi 5 CSI cable, adequate cable slack, and power sized for simultaneous Pi and servo operation.

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

This milestone is new work. PR #2 adds local overlays but no HTTP service, browser
controls, timestamped JPEG recorder, or timed MP4 export. Retain its AVI output only
as the existing diagnostic mode. Do not treat that mode as demonstration recording.

Start with an **annotated MJPEG browser stream at 800×450, capped at 10 FPS**, served by a small local HTTP service. Reuse processed frames so boxes and images correspond exactly.

Provide:

- `/stream.mjpg` for video.
- `/api/state` for target, controller, and performance status.
- A simple page showing the stream, tracking state, and pause/home controls.

Encoding and network delivery run outside the detection/control loop. Slow clients receive the latest frame; they never create an accumulating queue. Encode each selected frame once and share the JPEG between viewing and recording.

**Recording must preserve demonstration timing.** The current fixed-FPS AVI records processed frames and can alter playback speed.

For the new demo recording mode:

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

Implement each stage on `feature/pan-tilt-demo`. Finish its checks before starting
the next stage. The branch preparation and this plan are the only completed work.

1. **Repair hardware bring-up.** Fix the imported driver and calibration tool,
   establish `PanTiltDriver`, add atomic profile saving and calibration logging,
   and restore the ignore rule. Port the applicable PR tests into a separate file.
   Prove cancellation, terminal restoration, pulse mapping, clamping, and cleanup
   with mocks and subprocess tests. Then verify calibrated movement on the Pi.
2. **Implement shared tracking and selection.** Add prediction, low-confidence
   association, camera-motion compensation, confirmation, smoothed area, and
   switching hysteresis. Preserve one identity across confirmation, selection,
   logs, and display. Prove identity persistence and acquisition rules with replay.
3. **Integrate camera control.** Adapt the PR's worker and controller to typed target
   snapshots and main's pipeline. Add acceleration limiting, freshness handling,
   limit reporting, opt-in configuration, and reaction-test isolation. Define or
   reject camera-switching combinations. Prove zero physical writes during replay
   and detection-only runs, then measure single-camera centring on hardware.
4. **Add browser preview and demonstration recording.** Implement the latest-frame
   MJPEG service, state endpoint, pause/home controls, shared JPEG encoding,
   bounded recording, and timed MP4 export. Prove slow clients and recorder faults
   cannot block inference or camera control. Verify exported elapsed timing.
5. **Tune and validate the core demonstration.** Measure tracking alone, preview,
   recording, and both outputs. Collect latency, centring, switching, and one-hour
   evidence against the acceptance targets below. Keep hardware following opt-in.
6. **Implement the fixed-zone stretch goal.** Add reference registration, polygon
   membership, uncertainty handling, and projected-zone display only after the
   unrestricted demonstration passes acceptance.

The original estimate was 4–6 working days for the core demonstration and 2–3 more
for the zone. That estimate predates the merge review. Re-estimate after hardware
bring-up and the shared tracker are verified. Do not count the merge baseline as
completed hardware acceptance or completed tracking and control.

### Verification before delivery

The earlier review measured 61 passing PR tests and 316 passing main tests with
7 skipped. Those are separate revision results, not evidence that the integration
baseline or the future implementation meets this plan. The unchanged imported
files have known strict typing and formatting failures.

At branch preparation, `python -m pytest` also passes on `feature/pan-tilt-demo`:
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

Keep PR #2 open and keep this work off `main` during implementation. Before proposing
the feature for `main`, reconcile any newer main changes and repeat the full checks.
Complete the hardware acceptance below before considering a change to the opt-in
default. Merge to `main` only as a separately authorised delivery step.

### Acceptance tests

Use replay tests for association, camera movement, confidence gaps, target switching, timestamps, and zone geometry. Use hardware tests for centring and servo behaviour; replay cannot establish closed-loop performance.

Initial acceptance targets:

- At least **15 processed frames/s** sustained on the Pi, with control running at 20 Hz.
- Frame-to-command age below **200 ms at p95** during normal operation.
- After settling, the target centre stays within the central **20% of image width and height for at least 80%** of visible tracking time, within mechanical reach.
- No repeated switching between similarly sized stationary pigeons.
- Brief occlusion, target departure, camera failure, slow browser clients, and recording errors produce predictable recovery.
- Demonstration recording is playable and retains real elapsed timing.
- A one-hour run shows stable memory, acceptable temperature, and no accumulating queues.
- Zone tests include boundary crossings, camera movement, weak texture, and registration loss.

Treat these as acceptance goals to measure on hardware. Diagnose detector misses, identity errors, and controller errors separately.

### Retrain only if detector misses limit the result

First adjust confidence thresholds, exposure, focus, and tracking persistence using site footage. Servo oscillation or identity switching is not a reason to retrain the detector.

If persistent detection failures remain:

1. **Recover the shipped model's original PyTorch checkpoint from cluster artefacts.** The checkout contains its NCNN export, but not that checkpoint. NCNN is not the fine-tuning starting point.
2. Collect roughly **600–1,000 diverse site frames**, including camera movement, partial birds, lighting changes, empty scenes, and confusing objects. Auto-label, then manually correct.
3. Split by recording session, keeping validation and test sessions separate. Mix site training images with historical data to reduce overfitting.
4. Run one short fine-tune: same architecture and taxonomy, training size 800, **30 epochs with early stopping**, mixed precision, and a reduced learning rate. Start with batch 8 on the 3080 or batch 32 on one A100/H100.
5. Select thresholds and checkpoints using validation footage. Evaluate the chosen candidate once on untouched test footage and tracking clips.
6. Export the same **448×800 FP16 NCNN** format, verify parity with the existing tooling, and measure the complete Pi pipeline before replacing the shipped package.

If the original checkpoint cannot be recovered, fine-tune the repository's available YOLO11n checkpoint as a fallback candidate and promote it only if it improves site results over the current deployment.

A single 3080 is sufficient for this small-model fine-tune. Use an available A100 or H100 to shorten turnaround; avoid distributed training and broad sweeps. Label quality and site coverage will likely dominate the schedule. Keep architecture changes, depth models, appearance models, and another quantisation study outside the first demonstration.
