# Columba Zero: Raspberry Pi 4 Prototype Plan

## 1. Objective

Build a supervised first prototype that:

1. captures video locally on a Raspberry Pi 4;
2. detects pigeons in a defined area;
3. confirms detections over time before acting;
4. moves a two-axis turret in yaw and pitch toward a permitted target zone;
5. activates a laser-based visual deterrent within a controlled, bounded target zone; and
6. fails safe whenever the camera, software, turret, or safety checks are unhealthy.

The laser is included in the first hardware prototype, but initial validation must be performed with the beam terminated inside an enclosure or on an approved matte beam-dump target. The beam must never be directed at a pigeon, person, vehicle, aircraft, reflective surface, or open sky. Any live-animal or outdoor use requires qualified laser-safety review, site-owner approval, and all applicable agency approvals.

## 2. Current starting point

The existing [pigeon_detection_training.ipynb](./pigeon_detection_training.ipynb) is a Colab-oriented training notebook, not an executable Pi runtime. Its saved training run reports:

- 1,121 training images, 293 validation images, and 100 test images;
- precision `0.804`;
- recall `0.772`;
- mAP50 `0.835`;
- mAP50-95 `0.611`.

These results are only a baseline. The current data merge labels general COCO birds as pigeons, excludes most background-only images, evaluates the validation split rather than the held-out test split, and has no actuator-control code.

## 3. Raspberry Pi 4 target

### Recommended baseline

- Raspberry Pi 4, preferably 4 GB or 8 GB RAM.
- 64-bit Raspberry Pi OS Bookworm or the currently supported 64-bit Raspberry Pi OS image.
- Active cooling and a stable power supply.
- Raspberry Pi camera connected through the supported camera stack.
- Picamera2 for Python capture. See the [Raspberry Pi camera documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html).
- YOLO11n exported to NCNN for CPU inference. See the [Ultralytics NCNN documentation](https://docs.ultralytics.com/integrations/ncnn).

### Performance target

Do not assume real-time performance from the Colab timings. Establish a Pi 4 benchmark first:

- capture resolution: start at 640x480 or 640x360;
- inference resolution: start at 416 or 512 pixels, then test 640;
- target: at least 3 confirmed inference frames per second;
- measure end-to-end latency, not only model inference time;
- record CPU utilisation, RAM usage, temperature, dropped frames, and throttling.

If the Pi 4 cannot meet the required latency, reduce the inference resolution or process every second/third camera frame. A Coral USB accelerator or a different Raspberry Pi AI platform can be considered later, but it should not block the first laser bench prototype.

## 4. System architecture

```text
Picamera2
   -> latest-frame queue
   -> YOLO/NCNN detector
   -> temporal confirmation and target selection
   -> safety gate
   -> pixel-to-world calibration
   -> yaw/pitch turret controller
   -> laser driver + normally-closed shutter / optional air or audio actuator
   -> event logger and watchdog
```

Keep the camera fixed during the first prototype. A fixed camera provides a stable view and avoids detector instability while the turret moves. The turret should carry the laser module and shutter, not the camera. The laser should be aimed at a predefined safe surface adjacent to the target area, never at the detected animal itself.

## 5. Dataset and model work

### 5.1 Correct the label semantics

Do not train general COCO bird annotations as `pigeon`. Use one of these approaches:

- remove the COCO bird boxes from the pigeon training set; or
- use classes `pigeon` and `other_bird`, then only activate the deterrent for `pigeon`; or
- retain suitable COCO images as background-only images with empty labels where appropriate.

The preferred approach is `pigeon`, `other_bird`, and `person` as explicit classes, with additional background-only frames. A separate safety detector can be used later if a single combined model is too slow.

### 5.2 Add deployment-domain data

Collect and label footage from the intended type of hotspot:

- close, medium, and distant pigeons;
- partial occlusion and overlapping pigeons;
- dawn, overcast, bright sunlight, and artificial lighting;
- rain, glare, shadows, foliage, ledges, pavements, and food areas;
- empty scenes;
- people, pets, vehicles, bicycles, and non-target birds.

Split by recording session and location, not by adjacent frames. Keep one fully held-out site or session for final evaluation.

### 5.3 Validate for deployment

Add a real test evaluation, for example:

```python
test_metrics = model.val(data=DATA_YAML, split="test")
```

Tune the operating confidence threshold using the held-out set. Report precision, recall, F1, false positives per hour, and false negatives per event. Do not use accuracy as the primary metric because empty frames will dominate the result.

### 5.4 Export and verify

Export the trained model to NCNN, copy the complete model directory to the Pi, and compare Pi detections against the original `.pt` model on the same fixed test frames. Confirm that bounding boxes and confidence scores are acceptably similar before using the exported model in the runtime.

For long videos or live input, use streaming inference rather than collecting all results in memory. See the [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict).

## 6. Pi runtime software

Replace the notebook-only deployment assumption with a small Python application. Suggested modules:

```text
app/
  main.py          # orchestration and state machine
  config.yaml      # camera, model, GPIO, limits, thresholds
  capture.py       # Picamera2 frame acquisition
  detector.py      # NCNN model loading and inference
  tracking.py      # temporal confirmation and target selection
  safety.py        # ROI, person detection, E-stop, stale-frame checks
  geometry.py      # camera calibration and turret angle conversion
  turret.py        # pitch/yaw servo control
  actuators.py     # laser driver, shutter, and optional air/audio output
  logger.py        # structured events and health metrics
```

Use a bounded latest-frame queue. If inference is slower than capture, discard old frames instead of allowing latency to grow indefinitely.

## 7. Detection and deterrence logic

The detector must not directly open the laser shutter on one frame. Use this sequence:

1. Capture a frame and timestamp it.
2. Run pigeon and safety-object detection.
3. Restrict detections to a configured permitted ROI.
4. Confirm a pigeon in several recent frames, initially 3 of 5.
5. Select a target centroid or the centre of the largest pigeon cluster.
6. Reject the event if a person, vehicle, pet, stale frame, camera fault, or out-of-bounds target is present.
7. Convert the target to a permitted yaw/pitch command.
8. Move the turret at a limited speed.
9. Wait briefly for the turret to settle.
10. Confirm the hardwired laser interlock, then open the shutter for a short bounded pulse or pattern on the approved target surface.
11. Close the shutter before any further turret movement.
12. Return to neutral or continue a bounded sweep.
13. Enter cooldown and record the event.

Use separate activation and release thresholds, plus a maximum activation count per minute. This prevents rapid oscillation and repeated triggering when a pigeon remains in view.

Suggested states:

```text
SAFE -> IDLE -> VERIFYING -> AIMING -> ACTIVE -> COOLDOWN -> IDLE
```

Any emergency-stop input, camera timeout, model exception, servo timeout, over-temperature warning, or invalid angle must transition immediately to `SAFE` and disable the actuators.

## 8. Two-axis pitch/yaw turret

### Mechanical design

- Use a pan/tilt bracket with a yaw servo at the base and a pitch servo on the tilt frame.
- Mount the nozzle and optional speaker close to the pitch axis to reduce servo load.
- Add physical travel stops.
- Keep the camera fixed and define the turret origin relative to the camera.
- Start with a narrow, approved target zone rather than full 360-degree movement.

### Electrical design

- Use a PCA9685 servo driver or GPIO Zero servo abstraction. GPIO Zero documents `Servo` and `AngularServo` for PWM-controlled servos; see the [GPIO Zero output-device documentation](https://gpiozero.readthedocs.io/en/stable/api_output.html).
- Power servos from a separate regulated supply sized for stall current.
- Connect the Pi, servo driver, and actuator supply grounds together.
- Do not power both servos from a Raspberry Pi GPIO pin.
- Drive the pump or valve through a suitable MOSFET/relay module with flyback protection where required.
- Put the laser supply, shutter, and any other actuator supply behind a physical emergency-stop circuit.
- Use a certified laser module and driver selected by a qualified laser-safety reviewer.
- Use a normally-closed beam shutter or equivalent hardware block that closes on power loss.
- Terminate the beam in a matte, non-reflective beam dump during bench testing.
- Do not rely on a GPIO signal or the object detector as the only laser interlock.

### Software interface

Implement a `TurretController` with:

```text
home()
move_to(yaw, pitch)
stop()
safe_position()
set_limits(yaw_min, yaw_max, pitch_min, pitch_max)
```

The controller must clamp every requested angle, limit angular velocity, reject stale commands, and stop on timeout. Servo angle values must be calibrated experimentally because nominal servo angles do not guarantee physical angles.

## 9. Camera-to-turret calibration

For the first prototype, assume pigeons are on a known ground or ledge plane:

1. Calibrate camera distortion with a checkerboard.
2. Mark at least four known points in the target area.
3. Compute a homography from image pixels to target-plane coordinates.
4. Measure the turret position and nozzle height relative to that plane.
5. Convert the selected target point to yaw and pitch using `atan2` geometry.
6. Record the actual servo positions at several calibration points.
7. Fit correction offsets and verify the pointing error across the complete ROI.

If birds are flying or perched at substantially different heights, the system only has a bearing estimate. It must not use that estimate to point a laser at the bird. Disable the laser or direct only a safe, pre-approved pattern at a fixed beam-dump/target surface.

## 10. Safety requirements

The first prototype must include:

- a physical emergency-stop button;
- a physical exclusion zone around the turret;
- mechanical yaw/pitch limits;
- default actuator-off behaviour at boot and shutdown;
- a software watchdog;
- stale-frame and camera-disconnect handling;
- protection against activation when a person or vehicle is detected;
- bounded burst duration and cooldown;
- normally-closed laser shutter and independent hardware interlock;
- beam enclosure or fixed beam dump for all bench tests;
- no beam path intersecting people, animals, roads, windows, reflective surfaces, or sky;
- local-only processing unless recording is explicitly approved;
- weatherproofing and electrical fusing for outdoor testing.

Computer vision must not be the only safety barrier for a laser. The turret must be mechanically constrained so that valid software commands cannot move the beam outside the approved target envelope. The laser must be disabled whenever the shutter state, turret position, camera health, or safety envelope is unknown. Do not conduct outdoor testing without site-owner, qualified laser-safety, and relevant agency approval.

Air and ultrasonic output should be treated as optional secondary actuators. Test whether they have a measurable deterrent effect and monitor for non-target animal impact before combining them with the laser prototype.

## 11. Staged implementation schedule

### Stage 1: Dataset correction and offline evaluation

- Fix COCO label handling.
- Add true negatives and safety-object examples.
- Add site-specific footage.
- Evaluate the held-out test set.
- Select a confidence threshold using false activations per hour.

### Stage 2: Pi 4 camera and model benchmark

- Install Raspberry Pi OS and Picamera2.
- Verify camera capture independently.
- Run NCNN inference on saved frames.
- Benchmark 416/512/640 input sizes.
- Measure sustained FPS, latency, RAM, CPU temperature, and throttling.

### Stage 3: Turret bench prototype

- Assemble the pan/tilt bracket.
- Validate servo power and GPIO isolation.
- Implement homing, angle limits, `move_to()`, and emergency stop.
- Add a normally-closed shutter, beam dump, key switch, and hardwired emergency stop.
- Use a visible LED or disconnected laser driver as the first software output.

### Stage 4: Integrated closed-loop test

- Replay recorded footage through the live runtime.
- Confirm that detections generate correct yaw/pitch commands.
- Test people, empty scenes, shadows, and non-target birds.
- Verify that every fault leaves the actuator off.

### Stage 5: Supervised laser bench testing

- Connect the certified laser module only inside an enclosed bench rig with a beam dump.
- Test target-zone mapping, shutter timing, turret limits, and fault shutdown using inert targets or recorded video.
- Do not test on live birds, people, vehicles, or an open outdoor scene at this stage.
- Measure detection-to-action latency and false activations.

### Stage 6: Approved controlled-site evaluation

- Complete qualified laser-safety and site approval reviews.
- Use a fixed, non-reflective target surface and a mechanically bounded beam path.
- Begin with the laser disabled, then enable it only under direct supervision after all interlocks pass.
- Stop immediately on any unexpected reflection, person/vehicle entry, camera fault, or turret error.
- Compare bird presence and re-entry time against a no-actuator baseline.

### Stage 7: Hardening

- Add systemd startup and restart policy.
- Rotate logs and expose health metrics.
- Add enclosure, cooling, power monitoring, and remote maintenance.
- Repeat testing across weather, lighting, and hotspot layouts.

## 12. Prototype acceptance criteria

Before considering a field trial, the prototype should demonstrate:

- stable Pi 4 operation for at least several hours;
- no unbounded memory growth during continuous operation;
- measured, repeatable yaw/pitch positioning within the approved ROI;
- zero actuator output when the emergency stop is pressed;
- laser shutter closed on boot, shutdown, power loss, camera fault, software exception, and turret timeout;
- beam remains inside the approved, non-reflective target envelope;
- zero activation on safety-test footage containing people or vehicles;
- documented false activations per hour on negative footage;
- documented detection latency and sustained inference performance;
- laser-disabled baseline results followed by approved, supervised laser-deterrence results.
