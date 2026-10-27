# Validation status

## Monitor campaign, 27 October 2026, in progress

The user withdrew the 15 FPS goal and accepted the earlier hot 10–12 FPS range
near 83–85°C for the steady-state demonstration, provided control timing and
stability pass. Report actual hot throughput, temperature and throttling state.
The short approximately 12.5 FPS result was measured during warm-up. Other
acceptance goals remain. Wider movement is authorised; each envelope still needs
measured checks. Following remains opt-in.

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
and rejecting competing payloads. Three of eleven individually reviewed sample
images from the 5 FPS run passed optical association and held-out curved-screen
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

### Outstanding gates

Validate correction signs, settling, all movement
envelopes actually used, selection and loss, pause/resume, faults and restart.
Then compare the four output modes with fixed stimuli and a real preview client,
and run the hour-long campaign after the short cases pass. Centring needs
independent camera truth with time-weighted exclusions. Four corner markers alone
do not establish a valid homography on the curved monitor. No centring or
one-hour acceptance is claimed yet. The fixed-zone feature remains deferred.

Local evidence is under `outputs/validation/monitor-20261027/`. Device evidence is
under `outputs/setup/`, `outputs/monitor-hardware-smoke/`,
`outputs/monitor-hardware-100us/`, `outputs/monitor-hardware-startup-fixed/`,
`outputs/monitor-axis-calibration/` and `outputs/monitor-thread-probe/`. Raw media,
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
