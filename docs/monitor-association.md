# Offline optical association and registration

Generate the presentation with both `--registration-grid` and `--optical-code`.
The code is opt-in. It adds matching bands above and below the target region;
the scene/frame caption and abbreviated human run/render/state caption remain.
The ordinary presentation retains its existing captions and has no codes.

```sh
python tools/monitor_stimulus.py --playlist tuning --seed 20261027 \
  --display-scale 0.4 --target-scale 1.25 --horizontal-position 0.7 \
  --vertical-position 1 --registration-grid --optical-code \
  --output outputs/monitor/tuning-coded.html
```

Each band has 32 × 6 black/white cells, each 12 design pixels square. The band
rectangles are `(1056, 96, 384, 72)` and `(1056, 740, 384, 72)` in the
1600 × 900 design. At approximately 0.2 camera pixels per design pixel a cell
is about 2.4 camera pixels: focus, contrast, aliasing, curvature and exposure
can make decoding impossible. Inspect both bands and all marker margins through
the camera before recording. No physical capture of this new code has been
validated yet.

The 24-byte, row-major, most-significant-bit-first payload contains the first
16 bits of the presentation SHA-256, a fresh random 64-bit browser-session ID,
16-bit run number, 32-bit render number, and CRC-64/ECMA-182 of those first 16
bytes. The session ID changes on reload. The browser export contains the full
presentation ID and session ID; the tool constructs the code from every frame
record and accepts only one matching record across the supplied export. Duplicate
identities and presentation-prefix collisions are therefore rejected when they
make the match ambiguous. CRC-64 detects accidental corruption; it is not an
authentication mechanism. A collision or an adversarially altered log is not
ruled out. Keep the complete original JSONL and recording together.

Both independently sampled bands must decode, pass their checksums, and agree.
This rejects visible band disagreement and most torn or corrupted codes. It does
**not** prove that every target-region pixel came from the same display render:
rolling exposure, display scan-out, temporal blending and changes outside the
sampled bands can still mix imagery while both codes look valid. Treat the result
as optical identity evidence, not proof of instantaneous exposure or clock
synchronisation. Reject frames with visible tearing, blending, motion or other
ambiguity during review; do not infer timing by subtracting browser and Pi clocks.

## Prepare each image

Preserve `demo/index.jsonl`, all original JPEGs and the exported browser JSONL.
Select images by their actual `filename` in the index. The CLI uses `sequence`
from that index; the JPEG filename number is a spool ordinal, not the source
sequence. Record marker centres for **each** camera image, measured at the
centres of all four complete white squares. Confirm the marker identities from
position and the whole pattern, and inspect their complete margins. The patterns
are not rotation-unique. Do not copy a seed across camera motion without checking
it against that image; the tool neither detects markers nor verifies the manual
assertions. Omit an image or set either confirmation false if a marker is partial,
obscured or ambiguous.

For example, create `markers.json` alongside any other working files:

```json
{
  "frames": [{
    "image": "frame_000000057.jpg",
    "marker_identities_confirmed": true,
    "all_markers_fully_visible": true,
    "marker_centres_camera_px": {
      "top-left": [150, 120], "top-right": [485, 148],
      "bottom-right": [485, 315], "bottom-left": [138, 307]
    }
  }]
}
```

These coordinates illustrate the schema only, not measurements to reuse. Run
with the root environment's NumPy and OpenCV, and choose the maximum residual
in camera pixels before inspecting results:

```sh
python tools/associate_monitor_frames.py demo/index.jsonl browser.jsonl markers.json \
  --output outputs/association --max-error-px 3
```

The output directory must be new. Exit 0 means every requested image passed the
limited registration checks; exit 1 means at least one did not, and exit 2 means
the input or output could not be processed. The report and registration overlays
are retained even for failures. Review all observations and overlays: the tool
cannot prove that an independently matched feature is the intended cross in a
textured or overlaid camera image. The optional dependencies are not added to the
application.

## Report and interpretation

`association.json` hashes the index, browser export, marker annotations, each
JPEG and each uniquely selected browser frame. `image_sequence` is taken from
the index. `association` contains the full selected presentation, session, run,
render and scene/frame identities; `optical_code_hex` records the two agreeing
bands' decoded bytes. `candidate` means no unique optical association was
established. `unvalidated` means association succeeded but independent geometry
checks did not. `validated` means the limited checks passed; the detailed
registration result calls this `validated_within_sampled_hull`.

The marker-derived homography only bounds the search for each grid cross. A
camera-pixel template match, with a minimum correlation and a distinct peak,
provides each observed centre; a predicted location is never an observation.
The thresholds are conservative heuristics tested on synthetic samples, not
calibrated probabilities. The report lists accepted observations and rejected
crosses with their reasons. Scripted photograph coverage excludes a cross before
matching, but the visibility flag cannot see physical glare, preview overlays or
other optical occlusion. Failure to detect a feature is not evidence of its
absence. All 25 logged cross identities and coordinates must match the known
grid. The exact central cross, plus the existing centre and four-quadrant
coverage checks, must be independently observed. The existing registration tool
then checks the marker geometry and held-out residuals against `--max-error-px`.
No cross is used to fit the four-marker homography. A cross beyond the five-pixel
search radius is rejected, which can make a curved or poorly seeded image
unvalidated rather than measuring its full error.

The centred positive photograph covers the exact central cross in many frames,
so those frames are expected to remain unvalidated. An empty scene may provide
better coverage; that does not transfer its mapping to another camera image.

For a validated mapping, `projected_photograph_centres` contains only present
photograph rectangle centres inside the observed cross hull and camera image.
These are projected **photograph rectangle centres**, not observed bird
silhouette centres, detector boxes, target locations, or hardware acceptance.
Points outside the checked hull are not projected. The 1500R curved monitor does
not obey a global homography; even a passing sampled hull does not guarantee
accuracy between crosses. Camera movement requires a fresh per-frame check.
No servo command, detector output or cross-clock subtraction is used as truth.

Existing grid captures made before `--optical-code` have no such association
and must remain candidates. They can still be used with the separate manual
[registration workflow](monitor-registration.md) for feasibility or negative
checks, without claiming that the new code was physically tested.
