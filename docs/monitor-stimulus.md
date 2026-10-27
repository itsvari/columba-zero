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
alignment. Use the displayed scene/frame text and render number to associate
captured content. Use the four distinct marker patterns, their logged rectangles,
and a validated mapping for each camera frame to transform screen coordinates.
The tooling does not perform registration or calculate acceptance scores.

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
