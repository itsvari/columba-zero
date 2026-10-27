# Offline monitor registration

`tools/register_monitor_frames.py` fits one planar screen-to-camera homography per
annotated camera image. It consumes existing schema 2 browser frame records and
manual camera annotations. It does not operate the browser, camera or mount,
detect markers, calculate tracking scores, or match frames automatically. NumPy
and OpenCV are needed, as in the existing root `.venv`; no application dependency
is added. Run from this checkout with that environment's `python`.

The 34-inch ultrawide monitor is curved (1500R), and the camera was approximately
38 cm away in this campaign. The quoted 350–400 cd/m² is a display specification,
not a measured brightness. A homography is only a candidate approximation: four
marker centres determine its fit but do not establish its accuracy between them.
The tool calls a candidate `validated_within_sampled_hull` only when independent
camera observations pass the user-supplied error limit and coverage checks below.
That label describes these sampled observations, not a tracking or campaign
acceptance decision, a physical surface model, or a guarantee between samples.

## Prepare annotations

1. Preserve the original images and the exported browser JSONL. Visually read the
   scene, frame, run, repeat and render text in each camera image and select the
   corresponding `kind: "frame"` record. Render numbers can repeat after a new
   session, so keep the presentation and run context too. If the text cannot be
   read or the record is ambiguous, do not claim association. No clock subtraction
   is needed or performed. The tool records the operator-supplied reference and
   never verifies it.
2. Identify the four **complete** markers using their positions and visible
   patterns together. Inspect the whole marker, including its white margin.
   Do not identify them from a rotated 5×5 pattern alone: the existing patterns
   are not guaranteed to be orientation-unique. When a marker is clipped,
   obscured, ambiguous or cannot be checked, leave the relevant confirmation
   false and the tool rejects that candidate. The tool cannot verify a manual
   assertion or infer missing marker identity. Measure the centres in camera image
   pixels with origin at the image's top-left. Use centres of the white 64 × 64
   design-pixel squares, not the edge of a detector box.
3. Independently locate visible points whose screen coordinates are known, such
   as target-region corners or deliberately displayed calibration features, in
   both the selected frame record's viewport CSS pixels and the camera image.
   Do not derive their camera coordinates from the fitted homography, marker
   measurements, detector boxes, or servo commands. At least five distinct
   points are required: one near the target-region centre and one in each of its
   four outer quadrants. Specifically, the centre must be within 20% of each
   region dimension from its centre; each quadrant point must be at least 20%
   from the centre along both axes. All must be inside the target region and
   strictly inside the marker quadrilateral. A real point must be optically
   identifiable; the geometric centre of a blank rectangle is not an observed
   camera point. More independently observed points, including edge and interior
   points across the area of interest, give a stronger check. Each image needs
   its own observations, including after camera movement.
4. Choose a maximum camera-pixel error appropriate to the task and annotation
   uncertainty **before** judging the results. The tool has no universal
   tolerance. A large threshold can make a poor mapping pass this limited check.
   Only request points within the convex hull of the independent screen points;
   there is no extrapolation. If any requested point is outside that hull or
   maps outside the image, the frame emits no transformed points.

Create a JSON file containing `frames`. The image path is absolute or relative to
the annotation file. Copy the *entire selected browser frame record*, rather than
reconstructing its layout from the manifest: the record contains actual viewport
CSS rectangles, including layout and scaling. `image_sequence` is the camera
source sequence copied from `sequence` in the recording's `demo/index.jsonl`.
It must be unique within this file. It is not the JPEG filename's spool ordinal.
For example, `frame_000000057.jpg` in the startup-fixed run
`20261027T095120Z-8c3509d2` has source sequence 340. Other runs differ. The tool
records the supplied sequence and does not join or verify it against the index.
The numerical coordinates below illustrate the shape only; they are not real
measurements or a usable calibration.

```json
{
  "frames": [{
    "image": "frame_000000057.jpg",
    "image_sequence": 340,
    "stimulus_frame": {
      "schema_version": 2,
      "kind": "frame",
      "presentation_id": "copy-from-browser-record",
      "run_number": 2,
      "scene_id": "copy-from-browser-record",
      "scene_frame": 75,
      "sequence_frame": 315,
      "render_number": 317,
      "registration_markers": [
        {"marker_id": "top-left", "box_screen_css_px": {"left_px": 24, "top_px": 24, "width_px": 64, "height_px": 64}},
        {"marker_id": "top-right", "box_screen_css_px": {"left_px": 1512, "top_px": 24, "width_px": 64, "height_px": 64}},
        {"marker_id": "bottom-right", "box_screen_css_px": {"left_px": 1512, "top_px": 812, "width_px": 64, "height_px": 64}},
        {"marker_id": "bottom-left", "box_screen_css_px": {"left_px": 24, "top_px": 812, "width_px": 64, "height_px": 64}}
      ],
      "target_region_screen_css_px": {"left_px": 160, "top_px": 180, "width_px": 1280, "height_px": 540}
    },
    "marker_identities_confirmed": true,
    "all_markers_fully_visible": true,
    "marker_centres_camera_px": {
      "top-left": [100, 100], "top-right": [700, 100],
      "bottom-right": [700, 400], "bottom-left": [100, 400]
    },
    "validation_points_independently_observed": true,
    "validation_points": [
      {"name": "upper-left", "screen_css_px": [240, 220], "camera_px": [174, 162]},
      {"name": "upper-right", "screen_css_px": [1360, 220], "camera_px": [626, 162]},
      {"name": "lower-right", "screen_css_px": [1360, 680], "camera_px": [626, 338]},
      {"name": "lower-left", "screen_css_px": [240, 680], "camera_px": [174, 338]},
      {"name": "centre", "screen_css_px": [800, 450], "camera_px": [400, 250]}
    ],
    "screen_points": [[800, 450]]
  }]
}
```

The marker rectangles and target region are checked against the schema 2 design
geometry and common scale. They are not an optical check of the record selection.
The model requires a convex, consistently ordered, non-degenerate quadrilateral
in both coordinate systems. Mirrored, partial, missing and non-finite inputs are
rejected. If there are fewer than five validation points, available residuals
are still reported but the candidate remains unvalidated. Setting either marker
confirmation or the independent-observation confirmation to false prevents a
validated result.

## Run and inspect

```sh
python tools/register_monitor_frames.py annotations.json --output registration-output --max-error-px 3
```

The output directory must not exist. It receives `registration.json` and one PNG
per readable image. Exit status is 0 only if every frame passes the supplied
checks and all requested points are inside the validated hull; status 1 means at
least one frame is failed or unvalidated; status 2 indicates an input or output
setup error. The numeric threshold above is an example, not a campaign criterion.

The report preserves the supplied image path, image sequence, selected stimulus
reference, model, homography, fit residuals, independent residuals, threshold,
failures, sampled hull and transformed points where available. A near-zero fit
residual merely confirms the four inputs fit the four point pairs. It is not
an independent check. In each overlay, cyan circles and labels show annotated
marker centres, green circles show independently observed camera points, and red
crosses and lines show predictions and discrepancies. A failed frame still gets
an overlay if its image can be read. Inspect annotation placement and visible
content alongside the report. An overlay does not verify marker identity or
frame association.

## Feasibility on the saved campaign

Visual inspection of `live-38cm-aligned.jpg` shows all four markers, but their
small camera size makes exact manual centres and pattern checks uncertain.
`hardware-home-first.jpg` shows only part of the presentation and a single marker;
it cannot support a four-marker fit. These views differ substantially.

In saved `demo/frame_000000057.jpg`, the visible scene text and render 317 match
the corresponding `presentation-hardware-startup-fixed.jsonl` record. A rough
manual trial using the four marker centres and the four visible target-region
corners produced edge residuals of roughly 0.7–2.1 camera pixels. Those are
approximate annotations, not a precision measurement. No independent centre
feature was identified, so the tool correctly reports this candidate as
unvalidated and emits no transformed points. This trial does not establish that
a planar mapping works across the curved display or that other frames have the
same mapping. The aligned and powered-home images were also exercised with
unconfirmed or incomplete markers and were rejected. The unrelated render 317
record supplied to those two negative checks was only a geometry placeholder,
not an asserted scene association.

For repeatable calibration, capture an optically identifiable grid or other
independent interior and edge features at the same display layout and camera
pose, then annotate each frame and choose a justified error threshold. A central
pigeon, detector overlay or controller crosshair is not a substitute for an
independent known screen point.

## Checks

```sh
python -m pytest tests/test_monitor_registration.py
python -m ruff check tools/register_monitor_frames.py tests/test_monitor_registration.py
python -m ruff format --check tools/register_monitor_frames.py tests/test_monitor_registration.py
python -m mypy tools/register_monitor_frames.py tests/test_monitor_registration.py
```
