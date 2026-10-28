# Bring up and test the Pi over SSH

Use this runbook when the Pi, camera, and assembled mount are available. Initial
camera and small-movement checks have passed, but following acceptance remains
incomplete. Record results in
[the validation record](../deploy/VALIDATION.md), against the targets in
[PLAN.md](../PLAN.md#acceptance-tests).

## Prepare access and assembly

Install 64-bit Raspberry Pi OS and enable SSH with your usual public key.
[Raspberry Pi's SSH instructions](https://www.raspberrypi.com/documentation/computers/remote-access.html#ssh)
cover configuration through Imager or an existing installation. The computer
running this checkout must be able to reach the Pi. A working SSH host alias,
username, and project location are enough to start the remote session. Keep the
private key on your computer.

I can install dependencies, build the native backend, configure the application,
run tests, and retrieve logs over SSH. You assemble the hardware and observe the
first movements. Confirm the actual HAT revision, camera, power supply, cooler,
and cable arrangement before movement tests. Follow the
[Waveshare assembly manual](https://files.waveshare.com/upload/b/b9/Pan-Tilt_HAT_user_manual_en.pdf)
for the reference PCA9685 HAT.

Keep servo power disconnected for software and camera bring-up. Connect or
re-seat CSI cables only with power off. Leave application auto-start disabled
throughout initial testing. An SSH or browser disconnection does not constitute
a servo stop. For movement tests, stay beside the mount with access to its power.

## 1. Install and check the environment without movement

1. Check out the agreed revision of `feature/pan-tilt-demo` on the Pi.
2. Follow [the installation and native build instructions](../README.md#pi-setup).
   Keep OS OpenCV, NumPy, Picamera2, and libcamera in the system-site-packages
   environment. Use `python` from the activated environment for project commands.
3. Install `python3-scipy` and `ffmpeg` through apt, and install the project's
   `hardware` and `metrics` extras. Do not use the desktop `follow` extra to replace
   OS OpenCV.
4. Run `python tools/prepare_models.py`. Preserve the shipped input geometry,
   acquisition threshold, package checksums, and native backend for initial tests.
5. Run CTest and the project Python checks from the native build instructions.
   Native-extension tests that skip on this desktop must execute on the Pi after
   `columba_native` is installed.

Create `outputs/pi-detection.yaml` from `app/config.yaml`. Set
`display.enabled: false`, keep following disabled, and enable the browser preview
on `127.0.0.1:8080`. Keep both recording modes disabled initially.
For an upside-down sensor, set `camera.rotation_deg: 180` before verifying axis
directions. All downstream frames then share that orientation.

Run the prerequisite report from the repository root:

```sh
mkdir -p outputs/setup
git rev-parse HEAD > outputs/setup/revision.txt
uname -a > outputs/setup/kernel.txt
cat /etc/os-release > outputs/setup/os-release.txt
python -m pip freeze > outputs/setup/installed.txt
python tools/pi_preflight.py --config outputs/pi-detection.yaml > outputs/setup/preflight.json
```

Exit status 0 means the selected prerequisites passed. Exit status 1 means at
least one check failed. The JSON identifies each result. On a desktop, failure
of the Pi platform and camera dependencies is expected.

The tool imports processing dependencies in isolated, timed child processes.
For Picamera2, libcamera, and smbus2 it checks installation without importing them.
Their runtime importability remains unverified, because
[Picamera2's platform initialisation](https://github.com/raspberrypi/picamera2/blob/main/picamera2/platform.py)
can query video devices during import. It checks model metadata without constructing a detector,
and checks recording executables only when demonstration recording is enabled.
With hardware following selected, it also validates the calibration file and I2C
device permissions. It does not open the camera or bus, test a HAT response, run
inference, or send position commands. A passing report is an environment check,
not hardware acceptance.

## 2. Verify the camera and detector with servo power off

List camera modes and capture a still before running the application:

```sh
rpicam-hello --list-cameras
rpicam-still --nopreview --timeout 2000 --output outputs/setup/camera.jpg
```

[The camera documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html)
describes these commands. Check the intended camera index, colour order, field of
view, focus, image orientation, and box alignment. Use the
[offline monitor stimulus tooling](monitor-stimulus.md) for reproducible displayed
targets and browser-side evidence. Verify its static positive through live Pi
preview before running a playlist. Monitor-based acceptance is the current bench
campaign. Field-accuracy claims need separate site footage and pigeons.
The camera utilities do not read the application's YAML. For an upside-down
sensor, add `--rotation 180` to the still command. Verify control directions using
the application's configured capture image, rather than an independently oriented still.

Open a second terminal on your computer and forward the loopback-only preview:

```sh
ssh -N -L 8080:127.0.0.1:8080 <pi-ssh-alias>
```

If port 8080 is occupied locally, choose another port in the Pi's preview
configuration and at both ends of the tunnel. The server checks the request's
port, so keep the local and remote ports equal. This first session needs no
public preview listener or desktop forwarding.

Leave the tunnel running. In the Pi terminal, start the application:

```sh
python -m app.main --config outputs/pi-detection.yaml --max-frames 300
```

Open `http://127.0.0.1:8080` locally while the run is active. It stops after 300
processed frames. Rerun it if you need more time to inspect the preview.

Repeat the live camera run with a copied configuration that enables following
and keeps `pan_tilt.mode: simulate`. Confirm target acquisition, switching,
pause, home-and-pause, resume, and clean restart. Servo power remains off.
This tests the Pi's live timing and software workload before physical following.

## 3. Calibrate with you beside the mount

Enable the ARM I2C interface using `sudo raspi-config`, as described in
[the configuration documentation](https://www.raspberrypi.com/documentation/computers/configuration.html#enable-or-disable-i2c).
Check permissions as the normal application user. Do not use `sudo` to run
calibration or detection to work around a permission error.

With you ready to observe the assembly, connect servo power and follow
[the calibration procedure](pan-tilt.md#calibrate-the-mount). The calibration tool
immediately commands both channels to 1500 microseconds. Ensure this position is
clear before launching it. Use `ssh -t` for terminal control, or its `--line-input`
mode if immediate keys are unavailable.

Test small movements on one axis at a time. Establish channel mapping, configured-image
direction, safe pulse bounds, cable clearance, and home. Save the local profile
as the application user and retain it outside Git. Physical degrees and field of
view need measurement. Nominal servo angles alone do not establish either.
Use `--image-rotation-deg` to record the same orientation as `camera.rotation_deg`.
Hardware following rejects a profile with a different orientation before it opens
the driver.

Create `outputs/pi-follow.yaml` from the working simulation configuration. Set
`pan_tilt.mode: hardware` and its calibration path. Start with reduced gain,
speed, acceleration, and home speed, keeping home speed no greater than maximum
speed. Record the chosen values. Run the preflight again for this configuration
before starting physical following.

## 4. Tune short physical runs before collecting acceptance data

Start with one stationary target within reach, then a walking target. Verify the
correction direction separately for pan and tilt before increasing gain or speed.
Measure overshoot, settling, jitter, abrupt holds, and motion blur. Record detector
misses separately from identity changes and control errors.

Exercise these cases in supervised short runs:

- A second similarly sized target, then a clearly larger one.
- Brief occlusion, target departure, and reappearance.
- A target beyond each calibrated travel limit.
- Pause, resume, home, Ctrl+C, and SIGTERM, followed by restart.
- Camera startup failure and simulated delayed observations. Disconnect CSI only
  with power off. Existing mock tests cover capture faults and bus-write failures.
- A slow browser client and a small configured recording limit. Use a disposable
  output location for storage-fault tests, rather than filling the Pi's root disk.

Confirm normal shutdown disables PWM. Verify the observed behaviour when control
holds without disabling output. A stuck bus or native inference call may prevent
prompt cleanup. The current application has no independent physical stop or hard
watchdog, so do not infer one from successful signal tests.

## 5. Measure the demonstration and run endurance tests

Keep camera, scene, model, control settings, cooling, and power supply consistent.
Run tracking alone, preview with a real client, recording alone, and both outputs.
Repeat short runs to identify variation. The replay comparison tool provides an
additional diagnostic, but does not run this live four-mode test.

| Measurement | Initial target | Evidence |
|---|---|---|
| Processing throughput | Report hot throughput; the user accepted the earlier 10–12 FPS range near 83–85°C, conditional on timing and stability | Live frame windows, run summary, temperature and throttle flags |
| Controller cadence | 20 Hz | Monotonic worker tick intervals, with log drops and workload recorded |
| Frame-to-command age | p95 below 300 ms, as revised by the user | Observation timestamp through completion of the driver write, with sensor or delivery timing labelled; also report uncensored observation age and stale time |
| Centring after settling | Within ±10% on each axis for at least 80% of visible, reachable tracking time | Unannotated camera recording, independent target truth and time-weighted offsets, with settling and exclusions recorded |
| Selection | No repeated switching among similarly sized stationary birds | Target events and matching footage |
| Preview and recording | Playable, correct elapsed timing, bounded queues | Real browser session, recorder counts, MP4 decode and duration checks |
| Endurance | At least one hour, stable memory and temperature | Health samples, drops, errors, restart, and Pi throttling before and after |

Use `controller_metrics` from `tools/summarize_run.py` for timing evidence.
Controller events carry monotonic tick-start and driver-return timestamps.
Frame snapshots can precede the latest tick, and the JSONL `timestamp` records
writer time, so neither substitutes for these worker timestamps.

The report separates live cadence by mode. Dropped controller samples widen
observed intervals, undercount commands, and can bias age distributions. Inspect
`run_end.log_dropped_records`, which combines frame and controller drops and is
only an upper bound on controller sample loss. Measure cadence
under the intended camera, inference, and output workload. A paused worker alone
does not establish cadence under load.

Command-age distributions include only hardware tracking moves with known timing.
Sensor-based ages include sensor-to-delivery delay. Delivery-based ages start
when the frame reaches the application and omit that delay. Holds, home moves,
simulation, and replay provide no hardware tracking latency evidence. Inspect
the timed and untimed move counts before interpreting percentiles. Ages are
sampled per command, so several commands can use the same observation. These
values measure software driver return, not physical servo response, which needs
video or a separate sensor.

Run the one-hour test in `tmux` after the short cases pass. Set a unique output
directory for each run, sample health more frequently than the default 30 seconds,
and note that the default demonstration recorder stops after ten minutes. Keep
that limit for initial endurance tests, or explicitly budget duration and disk
space before increasing it. Export occurs after control stops, so measure export
time separately.

```sh
vcgencmd get_throttled > outputs/setup/throttled-before.txt
timeout --signal=TERM --preserve-status 3600 python -m app.main --config outputs/pi-follow.yaml
vcgencmd get_throttled > outputs/setup/throttled-after.txt
python tools/summarize_run.py outputs/<run>/detections.jsonl > outputs/<run>/summary.json
```

Preserve the configuration, revision, calibration snapshot, logs, recording,
summary, environment, physical observations, and export elapsed time for each
measurement. Extend the run when startup consumed part of the hour. Commit only
reviewed summaries, keeping local calibration and raw footage out of Git.

## Continue desktop work while waiting

Useful next work needs no hardware:

- Add a deterministic closed-loop scene model with camera movement, delay,
  dropped observations, servo lag, and direction reversal. It can expose software
  failures and test controller bounds, but cannot establish physical tuning.
- Run representative recorded scenes through the shared tracker and label misses,
  identity breaks, and switches. The existing tiled-image replay is a software
  smoke test with limited scene variety.
- Exercise longer browser and recording runs with slow clients and bounded disk
  use, preserving exported media and timing reports.

Keep the fixed-zone feature deferred until unrestricted following passes physical
acceptance. Add unattended startup after restart and shutdown behaviour have been
verified on this Pi.
