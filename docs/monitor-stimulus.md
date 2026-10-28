# Run the monitor stimulus campaign

Generate an offline HTML presentation on the computer attached to the physical
test monitor. This tool uses Python's standard library and does not import the
application, open a camera, connect to the Pi, or command the mount.

## Generate the two playlists

Run these commands from the checkout:

```sh
python tools/monitor_stimulus.py --playlist tuning --seed 20261027 --output outputs/monitor/tuning.html
python tools/monitor_stimulus.py --playlist evaluation --seed 20261027 --output outputs/monitor/evaluation.html
```

Each command writes an HTML file and a `.manifest.json` beside it. Open the HTML
directly in a current desktop browser. It embeds the photographs, styles, code,
and manifest, so it needs no server or network connection. You can move the HTML
to another computer without copying the repository.

If the camera cannot see all four markers, reduce the presentation inside the
full-screen page with `--display-scale 0.65`. The accepted range is 0.4 to 1.
The scale is included in the manifest, presentation ID and each frame record.
Target and marker coordinates include this reduction. Verify detection again
after changing it, and freeze the scale before comparing runs.
Use `--horizontal-position` and `--vertical-position` to place the reduced area
within the available padding. Each accepts 0 to 1, from left to right or top to
bottom. Both default to 0.5. This can align the stimulus with the camera's existing
aim without changing servo joints. These fractions also appear in the manifest
and frame records.

Add `--registration-grid` to draw 25 fixed crosses behind the targets. Their
screen positions and scripted visibility are included in each browser frame
record. The photograph rectangles cover any crosses behind them. This grid gives
independent interior features for camera registration. Its visibility field only
describes scripted coverage; inspect actual camera visibility and association
before using a point. Keep this background choice fixed across compared runs.
Use `--target-scale 1.25` to enlarge all photograph rectangles while retaining
their paths and relative area ratios. This is useful when reducing the whole
presentation to keep its markers below the camera overlay. The accepted range is
0.5 to 2. Check scripted clipping and actual detection again after changing it.

For camera calibration, add `--photo-registration-frame` to expose pale photograph
boundaries. It draws a black frame with a white outer border outside each fully
visible photograph. The inner black edge marks the original photograph boundary.
The photograph coordinates, crops and paths stay unchanged. The option has its
own presentation identity and appears in the manifest and frame records.
Partly occluded, fully occluded and clipped photographs receive no frame. The
target-region clip can also cut a frame's outer border, so inspect each boundary
before annotating it.

Use framed pages for calibration with following disabled. Collect the separate
confirmatory following recording without this option, and preserve both pages
and manifests. Camera annotations must still identify the observed photograph
corners independently of the matcher.

Add `--optical-code` to display matching machine-readable identity bands above
and below the target region. This changes the presentation ID and abbreviates
the lower human caption to run, render and state. See the [offline association
guide](monitor-association.md) for marker checks and optical limitations.

A curved monitor does not satisfy the flat-surface assumption of a single
homography. Check mapping error across the target region using independent camera
annotations, or use a validated mapping that accounts for the curve. Four visible
corner markers alone do not establish accurate registration between them.

Use tuning results to choose settings. Freeze those settings before running the
evaluation playlist. Preserve the HTML, manifest, browser JSONL, application
configuration, Pi logs, and camera recording for each run. Reuse the same HTML
when comparing output modes. Identical arguments and source files produce
identical HTML and presentation IDs. The ID includes the runtime template hash.

Each playlist has 22 scenes and lasts 129 seconds at the default 30 Hz. To repeat
the evaluation sequence for just over an hour, generate 28 repeats:

```sh
python tools/monitor_stimulus.py --playlist evaluation --seed 20261027 --repeat-count 28 --output outputs/monitor/evaluation-hour.html
```

Use `--static-duration-s 12` for the reduced-speed hardware profile. This extends
the nine stationary scenes to 12 seconds each, retaining the movement paths and
other durations. The resulting cycle is 201 seconds. Declare the settling window
before scoring; a four-second scene does not demonstrate settling for this profile.

`--frame-rate-hz` accepts integers from 10 to 60. It sets the logical stimulus
rate, not the display refresh rate or camera frame rate. `--repeat-count` accepts
1 to 1000. The presentation pauses at its log capacity even if repeats remain.

## Present and record a sequence

1. Open the tuning HTML on the physical monitor. It starts paused on the centred
   positive so you can check framing before motion begins.
2. Select **Full screen** or press **F**. Keep all four corner markers and the
   scene/frame text visible through the camera. The grey rectangle contains the
   targets. The marker patterns and identifiers stay outside that rectangle.
3. Establish the static positive through live Pi preview, following
   [the hardware runbook](pi-hardware-testing.md#2-verify-the-camera-and-detector-with-servo-power-off).
   Record distance, display resolution, refresh rate, brightness, focus, exposure,
   glare, and visible banding. Software checks do not establish this positive.
4. Select **Play / pause** or press **Space** to start or pause. **Reset** or **R**
   returns to frame zero and stays paused. **Replay** or **P** returns to frame
   zero and starts immediately. Reset and replay create a new run number while
   preserving earlier log records.
5. Select **Export JSONL** or press **E** to download the evidence. Export during
   a pause or after completion for a declared endpoint. Export during playback
   snapshots the log at that moment and playback continues.

The sequence pauses when the page becomes hidden or its viewport changes.
Resume explicitly after checking framing. Browser full-screen changes are
logged. If the browser refuses the full-screen request, use its own full-screen
command. Pausing the presentation freezes its image and logical clock. It does
not pause the Pi controller or simulate a failed camera.

Logs live in browser memory until exported. Reloading or closing the tab loses
them. The buffer holds at most 150,000 records and stops rendering before dropping
evidence. At 30 Hz, one hour needs about 108,000 frame records, plus events.
Repeated resets, resizes, or higher stimulus rates can exhaust the buffer sooner.
Export and reload to begin a new browser session after exhaustion. Keep the tab
in the foreground and prevent display sleep for a physical run.

## Identify the stimuli and intended behaviour

Both playlists include these labelled cases:

- A centred static pigeon and eight axis or corner offsets.
- Rest, movement, stop, reversal, diagonal motion, a curved path, and scale changes.
- An incumbent acquired for two seconds before a challenger appears. Similar-size
  challengers have 1.08 times the incumbent's photograph area. Larger challengers
  have 1.6 times its area, corrected for the photographs' different aspect ratios.
  The brief larger challenger lasts 0.3 seconds. The sustained one stays for six
  seconds. The intended switch includes confirmation, smoothing, and the 0.5-second
  switching hold, rather than an exact screen-time switch deadline.
- Partial occlusion of 60% of the photograph rectangle, full coverage for 0.3
  seconds, and full coverage for 1.5 seconds. The same intended identity returns.
- Departure beyond the target region, an empty scene, and a non-pigeon shorebird.

The manifest gives each scene its own ID, seed, keyframes, target lifetimes,
occlusion intervals, and behaviour annotations. Keyframes interpolate linearly
in logical frame time. Seeds change the scene IDs and the curved path. Repeats
reuse the exact scene definitions. Annotations describe intended behaviour and
transition periods. They are not measured Pi results or exact detector deadlines.
Target identities are scoped to a scene and repeat. Reset and replay do not reset
the Pi's tracker; account for retained tracker state at scene boundaries.

Tuning and evaluation use distinct pigeon photographs, different scene IDs and
path seeds, and separate behaviour annotations. Tuning uses a plain background.
Evaluation uses a stationary dotted background. Both reuse one confuser. These
are separate bench playlists, not independent samples for a field-accuracy claim.

The positive images are visually verified crops of original photographs in the
committed contact sheets. Augmented day, night, and composite columns are excluded:

| Playlist | Contact sheet and tile | Visible subject |
|---|---|---|
| Tuning | `train-v2s-before-after.jpg`, bottom-left | Grey pigeon on paving |
| Tuning | `train-v1clean-val-000.jpg`, bottom row, second tile | Dark pigeon on paving |
| Evaluation | `test-v2-styles-before-after.jpg`, second row, first tile | Pigeon on a branch |
| Evaluation | `test-v2-styles-before-after.jpg`, third row, first tile | Pigeon on the ground |

These paths are relative to `data_sources/contact_sheets/`. The initial review
also checked `review-inat-pigeon-000.jpg` and its existing review labels, which
include rejected feather images. The chosen originals give larger, cleaner crops.
The confuser is the small brown and white shorebird in `coco-186938` from
`test-v2-000.jpg`. Visual inspection and `data_sources/review/coco-birds.images.jsonl`
agree that it is not a pigeon. Crops exclude captions and drawn annotation borders.
Original JPEGs remain unchanged. The manifest records crop rectangles, review
descriptions, and source SHA-256 hashes.

## Interpret the exported JSONL

The first record has `kind: "session"`, the manifest without image payloads,
`browser_time_origin_unix_ms`, and explicit clock and coordinate descriptions.
Subsequent records have `kind: "event"` or `kind: "frame"`. Every record carries
the presentation ID, playlist, seed, run number, repeat index, scene ID, scene
frame, sequence frame, and `browser_monotonic_ms` from `performance.now()`.

Frame records contain the visible identifier, render number, playing state,
viewport size, device pixel ratio, marker rectangles, target-region rectangle,
and each intended target's photograph rectangle, identity, subject, presence,
visibility, visible fraction, and occlusion coverage. Rectangles are screen
content positions in browser **viewport CSS pixels**, with origin at the viewport's
top-left. They include the controls' height and any letterboxing. They are not
camera pixels, physical monitor pixels, or desktop coordinates. The manifest's
1600 × 900 design coordinates are scaled into this viewport for each render.

`visible_fraction` measures the photograph rectangle remaining after lifetime,
target-region clipping, and opaque rectangular coverage. It does not measure
visible bird silhouette area. `box_screen_css_px` describes the intended full
photograph rectangle, including its hidden or clipped portion. A fully occluded
target remains present with zero visible fraction. A departed target has
`visibility: "outside_target_region"`. An absent challenger is labelled `absent`.

Events include readiness, run starts, scene starts and ends, visibility changes,
pause, resume, reset, replay, completion, export, viewport and full-screen changes,
asset errors, log exhaustion, and skipped logical frames. Only frames actually
drawn by the page are logged. Slow animation callbacks skip logical frames and
report the gap. Pauses and geometry changes can draw multiple render numbers
with the same scene frame. The camera-visible render number distinguishes them.

Browser timestamps describe code execution around canvas drawing. They do not
measure display scan-out, Pi exposure, camera delivery, or servo completion.
Do not subtract them from Pi clocks without independently establishing clock
alignment. Use the displayed scene/frame text and render number as human-readable
context. The opt-in bands can provide association evidence as described in the
[offline association guide](monitor-association.md). Inspect each complete marker
and its position; the marker patterns are not rotation-unique. A mapping must be
independently checked for each camera frame. These tools do not calculate
acceptance scores.

The source crops are small, compressed photographs with some original background.
Enlarging them cannot restore detail. The size challenge uses photograph area as
a proxy for detector-box area. Verify the actual detections before interpreting
selection behaviour. Display artefacts, occlusion boundaries, and photographed
posture can affect detection. The tool does not model natural walking, depth,
moving-camera geometry, mechanical reach, capture faults, or field accuracy.

## Verify changes locally

Run the focused Python checks:

```sh
python -m pytest tests/test_monitor_stimulus.py
python -m ruff check tools/monitor_stimulus.py tests/test_monitor_stimulus.py
python -m ruff format --check tools/monitor_stimulus.py tests/test_monitor_stimulus.py
python -m mypy tools/monitor_stimulus.py tests/test_monitor_stimulus.py
```

To include actual browser behaviour, set `COLUMBA_TEST_BROWSER` to a local
Chromium-compatible executable and ensure `node` is on `PATH`:

```sh
COLUMBA_TEST_BROWSER=/usr/bin/chromium python -m pytest tests/test_monitor_stimulus.py
```

The optional browser test uses Node's built-in modules and the browser's DevTools
pipe. It opens local HTML in a disposable profile and drives a controlled browser
clock. It checks real JPEG decoding, canvas output, controls, replay, scene
geometry, challengers, occlusion, clipping, repeated playback, JSONL export, and
bounded logging for both playlists. It does not run the Pi or prove physical
display visibility. No project dependencies are added.

## Independent verification fixture

The opt-in `independent` playlist separates fixture preparation from the accepted
`tuning` and `evaluation` presentations. Their durations, paths and photographic
choices remain unchanged. Generate a fresh local page and manifest:

```sh
python tools/monitor_stimulus.py --playlist independent \
  --output outputs/independent/monitor.html --optical-code
```

This playlist reuses tuning motion, scale, loss and negative cases. Its nine
static offset holds last at least 12 seconds (`--static-duration-s` can extend
them). Its three selection scenes each last 24 seconds. Each starts with the
incumbent alone for 12 seconds; the challenger appears within that same scene,
without moving or replacing the incumbent. Scene boundaries are separate trials
and require fresh acquisition; they do not assert cross-scene identity retention.

Selection identities use distinct approved crops: `tuning-standing-pigeon` for
the incumbent and `tuning-profile-pigeon` for the challenger. This preserves the
truth tools' single-instance-per-asset association. The subthreshold/comparable
challenger's source photograph area is 0.64 times the incumbent's; larger
challengers are 2.56 times its area. Height accounts for the crops' different
aspect ratios. The existing approved nominal source visible-bird boxes cover
both complete crops, so this also normalises those nominal boxes. Their source
edge allowances and camera projection uncertainty still require fresh independent
qualification; this does not normalise detector boxes. These deliberately generous margins
avoid testing the exact 1.25 boundary in a camera fixture. Exact near-equal
and switching-boundary behaviour remains a deterministic software test. They do **not** establish
camera-visible bird area or detector box ratios. Independently qualify projected
bird extents and uncertainty across the actual camera poses before confirmation:
comparable upper ratio bound below 1.25; larger lower bound above 1.25. If either
fails, retain the rejection and redesign; detector output cannot supply truth.

Selection annotations freeze this timeline, with half-open frame intervals:

| Case | 0–12 seconds | 12–15 seconds | 15–24 seconds |
|---|---|---|---|
| Subthreshold/comparable | Acquisition/settling allowance | Incumbent required | Incumbent required |
| Brief larger | Acquisition/settling allowance | Incumbent required | Incumbent required |
| Sustained larger | Acquisition/settling allowance | Smoothing/hold response allowance | Challenger required |

The brief challenger exists only from 12 through 12.3 seconds, shorter than the
configured 0.5-second switch hold. The sustained challenger's three-second response
allowance includes confirmation, smoothing and hold; freeze it before opening
confirmation results. An allowance has `expected_policy: "allowance"` and a null
`selected_identity`; it means no required identity during that interval, **not** an
expected absence of selection. Other annotations default to `"required"`.
Non-selection legacy annotations remain descriptive; they are not a replacement
for a separately frozen camera truth and reach policy. Failure to acquire or
exercise the intended path must remain visible as unexercised or inconclusive.

For preparation, keep hardware following disabled. The independent page exposes
**Calibration pose → Show paused pose**. It seeks to exact static and selection
annotation-start frames in the first repeat, pauses playback and logs a
`calibration_pose` event. At 30 Hz the selection poses are frame 0 (incumbent),
360 (challenge begins), and additionally 450 (sustained response allowance ends).
The two challenge poses have identical static geometry. The brief frame-360 pose
stays frozen until resumed, allowing independent inspection without changing its
0.3-second playback lifetime. These are calibration-only observations: do not
reuse their frozen duration as confirmation evidence. Reset/replay after preparing
truth and capture a fresh continuously playing confirmation run. Photograph
registration frames remain optional calibration aids and must be disabled for
confirmation. Source dimensions and browser CSS coordinates do not qualify the
curved monitor's optical geometry or mechanical reach.

The independent static and selection scenes use a dark `#242424` field. Motion,
scale, occlusion, departure and negative scenes retain the tuning plain field.
Reference crosses on the dark field have a light core with a dark outline;
marker and optical-code identity and geometry are unchanged. This field is part
of both calibration and future confirmation, not a calibration-only overlay.
The pilot assumption is that contrast makes photograph edges measurable without
altering pigeon pixels. Browser checks establish unchanged photograph pixels,
dark fill and light cross visibility; they do not establish camera qualification
or detection performance. Preserve the unsuccessful unframed pale-field capture
and qualify this new presentation independently before any confirmation.

Build the paired dark-field pilot without photograph registration frames:

```sh
python tools/monitor_stimulus.py --playlist independent \
  --output outputs/independent-verification-v1-dark/monitor.html \
  --optical-code --registration-grid --display-scale 0.65 \
  --target-scale 1.25 --horizontal-position 0.5 --vertical-position 0.5
```

Use exactly the same page and display settings for its calibration and future
confirmation. Any changed geometry needs a new frozen manifest and qualification.
