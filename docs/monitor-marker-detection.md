# Monitor marker candidates

`tools/detect_monitor_markers.py` finds the four marker centres independently in each camera JPEG. It produces candidates for the existing association CLI. It does not perform optical association or establish tracking or hardware acceptance.

```sh
.venv/bin/python tools/detect_monitor_markers.py path/to/demo/index.jsonl \
  --output path/to/marker-candidates.json \
  --diagnostics path/to/marker-diagnostics.json
```

Repeat `--image frame_000000372.jpg` to limit processing to specific index filenames. The output's `frames` array has the association tool's expected shape. Its `marker_identities_confirmed` and `all_markers_fully_visible` flags are **false**, including when the algorithm finds all four patterns. The association tool requires manual confirmations: review each source image and its four marker identities, completeness, quiet zones and centres before setting those flags in a separately reviewed copy. Sampling a few images does not confirm the rest. The diagnostic JSON records every processed image's SHA-256, index SHA-256, candidate corners, centres, contrast, minimum cell classification margin and rejection status. Paths for output must differ from inputs and one another.

The detector thresholds bright contours, fits convex quadrilaterals, rectifies each white marker square, and samples the 25 cells. It requires the exact upright stimulus patterns, local contrast and class margin, and a predominantly bright sampled quiet zone. It rejects small, clipped and non-convex candidates, additional recognisable or unrecognised marker-like squares, missing identities, and arrangements whose four centres do not form an upright convex quadrilateral in top-left, top-right, bottom-right, bottom-left order. It assigns identities from the patterns and the whole arrangement, not previous frames or model detections. The patterns alone are not rotation-unique; identity is only proposed in conjunction with the required global geometry. Any resulting centre is an algorithmic observation, not a human confirmation.

These checks are conservative but cannot prove every pixel of the quiet zone is visible or exclude all misleading distractors, partial occlusion, small-cell aliasing, exposure effects or a coincidentally matching scene. White contour geometry and JPEG resolution can shift centre estimates. A candidate is not a guarantee of detection accuracy; unrecognised or absent markers need review, and no estimate is copied from another frame. The detector requires at least 12 pixels per outer side, and operates best with substantially more. Reject a candidate if orientation, pattern, visibility or position is uncertain. The association CLI's separate optical and held-out grid checks remain necessary and unchanged.
