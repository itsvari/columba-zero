# Prepare retained monitor truth

`tools/prepare_monitor_truth.py` joins optical render identities and independent
per-image photograph fits into conservative annotations for
`tools/score_monitor_run.py`. It preserves unresolved evidence and always reports
an inconclusive global verdict. A scoped numerical centring result cannot establish
that the complete hardware qualification passed.

Run the tool against a frozen qualification policy and a new output directory:

```sh
.venv/bin/python tools/prepare_monitor_truth.py RUN_DIRECTORY BROWSER.jsonl \
  --associations ASSOCIATION.json \
  --locations LOCATIONS.json \
  --source-labels deploy/monitor-source-labels.json \
  --qualification QUALIFICATION.json \
  --output NEW_OUTPUT_DIRECTORY
```

The outputs are `annotations.json`, `score.json`, `preparation.json` and
`qualification-snapshot.json`. The preparation report records input and code
hashes, qualification evidence, per-frame render evidence, local-fit decisions,
reach bounds, selection mapping and rejection reasons. Existing output directories
are refused. Input and code hashes must remain unchanged during preparation.

## Render evidence

Exact associations come from the identity-only association report. The tool binds
its source index, browser export, source sequence, current JPEG, optical payload
and unique browser record. Duplicate filenames or source sequences, different-run
paths, JPEG aliases and mismatching hashes are rejected. Optical identity does
not validate photograph geometry or a target centre.

A frame without a decoded code can receive invariant-content evidence only from
its nearest surrounding optical camera anchors. Their camera-time span must be
at most two seconds. Presentation, optical session, run, repeat and scene must
agree, with ordered render numbers. Every browser render number in the inclusive
interval must exist, and its complete target content and manifest selection rule
must equal the lower anchor's content. Moving targets, changing rules, missing
records and scene boundaries remain unresolved. Browser render counters must
increase strictly within each session. The tool does not compare browser and
camera timestamps.

Equal anchor render numbers produce an exact render label. Different render
numbers produce `render_number: null` and an explicit confirmed content interval,
with both optical anchors and every browser record hash. This establishes
invariant intended content; it does not identify the exposure's exact render.

Expected identity comes from the session manifest annotation whose half-open
scene-frame interval contains the render. Every intended target slot, including
confusers and missed targets, remains in the label. Expected identity never
comes from the target detector or the selected track.

## Visibility, centres and selection

Each photograph fit is checked independently for the current image. The existing
locator gates remain unchanged: match and inlier counts, held-out support and
error, spatial support, centre containment, and a bounded convex photograph
quadrilateral. The tool also reproduces its corners, visible-bird centre and
uncertainty from the reviewed source box and local source-to-camera mapping.
Source, image, asset and locator-code hashes must agree with their evidence.

Rendered-present targets need an independently passing local photograph fit for
`visible: true`. A failed fit yields `visible: null`, no centre and invalid
registration. It does not prove absence. Known render context and the complete
intended inventory remain available, so the scorer applies the declared six-second
settling period and counts subsequent unresolved time as unknown. Controlled
absence or full occlusion can establish `visible: false`; contradictory passing
photograph evidence makes the whole frame label unresolved.

Only fully visible crops of policy-qualified assets receive centres. Partial
occlusion or clipping can change the visible-bird box, so an otherwise passing
photo fit supplies identity evidence while its centre remains unknown. The tool
does not infer the remaining visible silhouette from the original source box.
An unqualified passing fit can
establish photograph identity and visibility while its centre remains unknown.
The centre is the mapped reviewed visible-bird box centre, which can differ from
the photograph's centre. Axis uncertainty comes directly from the local fit's
centre bounds, including the mapped ten-source-pixel edge allowance and
seven-camera-pixel registration allowance. The tool does not add that allowance
twice or reuse a neighbouring frame's centre.

Original qualification covers the held-out standing-pigeon and confuser checks.
A separate policy may add a directly checked asset. Its comparison, independent
visible-bird checks and locator results must all be hash-bound; every checked
observation and matcher rejection must remain present. Recomputed worst-case
centre errors, including annotation edge uncertainty, must fit the same seven-pixel
budget. Such evidence is limited to the observed poses. It does not establish
photograph-corner accuracy or statistical confidence. A new qualification policy
requires a new output directory; earlier reports remain unchanged.

The recorded selected bounding box is used only to map a selected track to one
independently fitted photograph identity. Exactly one photograph must have
intersection-over-union of at least 0.25. Ambiguity remains unresolved, and mapping
to the wrong photograph remains a wrong selection. This box never seeds a centre,
visibility, reachability or expected identity.

## Reachability and scoring limits

Reach uses the policy's fixed home CSS-to-camera mapping and measured usable and
outer pan/tilt pixel-shift intervals. The browser layout must match the home
reference. The reviewed source-bird centre and its source-edge allowance are
mapped through the CSS photograph box and home mapping, then expanded by the
home registration allowance.

Reach means that a permitted pose can put the centre inside the central 20% band
on both image axes. It does not require the exact image centre. A target is
guaranteed reachable only when every point in its home uncertainty rectangle can
reach this band through the usable shift intervals. It is proven unreachable only
when even the outer shift intervals cannot intersect the band. Intermediate cases
remain unknown. The empirical single-sweep rectangular approximation retains its
cross-axis and history limitations; recorded commands and clamps cannot exclude
a frame.

The scorer retains the fixed six-second settling period, one-second maximum
sample gap and conservative occupancy bounds, including all unknown time.
Recording-integrity failures remain visible. Report the actual lower and upper
bounds and scoped verdict alongside these limitations; the preparation tool's
global verdict remains inconclusive.
