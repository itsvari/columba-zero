# Score an independently reviewed monitor recording

`tools/score_monitor_run.py` is an offline, standard-library scorer for the
recordings described in [the stimulus guide](monitor-stimulus.md). It reads
`demo/index.jsonl`, the referenced JPEGs and `detections.jsonl` from one run
directory. It opens no camera, hardware connection, browser or server.

Use `preview.record_unannotated: true` for new camera evidence. The index must
explicitly mark every retained frame with `annotations_included: false`.
Annotated demonstration frames and older frames without this provenance remain
inconclusive for centring, even when their labels are complete.

The centring goal from [the plan](../PLAN.md) is at least 80% occupancy inside
the central 20% of **both** recording dimensions after a declared settling
period. For an 800 × 450 recording, that rectangle is x = 320–480 and
y = 180–270 pixels, including its boundary. These coordinates describe the
goal, not a measured result. Expected targets remain in the denominator when
the detector misses them or selects another target.

## Create and complete a blank review template

From the repository root, set the run directory and declare the policy before
examining scores. The following durations are example policy choices, not
validated settling or sampling limits for the mount:

```sh
run_directory=/path/to/outputs/monitor-hardware-startup-fixed/run-id
python -m tools.score_monitor_run template "$run_directory" \
    --settling-s 1.0 --maximum-sample-gap-s 0.5 \
    --policy-evidence 'review record identifying who declared this policy and when' \
    --output /path/to/review.json
```

Every frame entry initially has `label: null`. The tool supplies actual
filenames, sequences, JPEG dimensions and SHA-256 hashes. It also binds the
document to the complete index and detections log with SHA-256 hashes. It
supplies **no** scene associations, target coordinates or positive judgements.

Review all intended targets in the captured content, including missed targets,
challengers, occluded targets, empty scenes and confusers. Replace each reviewed
frame's null label with the schema below. An omitted entry or a null label stays
unknown; neither counts as invisible, unreachable or successful. The final
frame needs a label even though it contributes no extrapolated duration.

Obtain target centres directly from camera pixels, independently of detector
boxes, controller errors and servo commands. Do not copy the selected box centre
into truth. Use the camera-visible scene/frame text and render number to associate
content with the preserved stimulus manifest and browser export. Review selection
identity separately: a tracker ID alone is not a stimulus identity.

For the 1500R curved display, four corner markers do not validate a planar
homography. Use manual camera-pixel truth or independently validated registration
across the target region for each changing view. Registration evidence must
identify its method, independent validation and residual error. A failed or
unvalidated mapping must have `registration_valid: false`.

## Annotation schema, version 1

All listed fields are required; extra fields and duplicate JSON keys are
rejected. JSON numbers must be finite; booleans do not count as numbers. Strings
must be non-blank. The scorer validates structure and provenance; the operator
remains responsible for the independence and accuracy of the evidence.

The root document has exactly these fields:

| Field | Value |
| --- | --- |
| `schema_version` | Integer `1`. |
| `index_sha256`, `detections_sha256` | Generated hashes; retain unchanged. |
| `policy` | The declaration described below. |
| `frames` | List of frame entries; do not reduce it to detector-selected frames. |

`policy` has exactly `settling_s` (non-negative seconds),
`maximum_sample_gap_s` (positive seconds), `evidence` (reviewer, declaration time,
reason and evidence reference), and `target_inventory` with the literal value
`"all_intended_targets_independent_of_selection"`. Scoring accepts no policy or
window overrides. Preserve the original declaration and freeze it between
comparable runs; changing it after seeing a score invalidates that comparison.

Each frame entry has exactly `filename`, `sequence`, `width_px`, `height_px`,
`jpeg_sha256` and `label`. Preserve the generated values. The filename/sequence
pair must exist in the recording and appear at most once. Dimensions and hash
must match the referenced JPEG. A non-null `label` has exactly these fields:

| Label field | Value |
| --- | --- |
| `presentation_id` | Preserved stimulus presentation ID. |
| `run_number`, `repeat_index` | Non-negative integers from the associated render record. Preserve the browser's numbering. |
| `scene_id` | Actual scene ID, scoped by presentation, run and repeat. |
| `render_number` | Non-negative integer read from captured content and associated with the export. Multiple camera samples may share a render. |
| `inventory_complete` | `true`, explicitly affirming review of every intended target independently of selection. Otherwise leave the whole label null. |
| `expected_identity` | Expected stimulus identity string, or `null` when selection should be empty at this sample. |
| `targets` | List of all intended targets, including invisible ones, using the target schema below. An empty list is permitted for a reviewed empty scene. |
| `selection` | `null` if identity mapping is unresolved or no track is selected; otherwise exactly `{"identity": "reviewed stimulus identity", "evidence": "independent association reference"}`. |
| `evidence` | Reviewer and evidence references establishing scene/render association, the full inventory and the expected-selection rule, including any transition allowance. |

Each target has exactly these fields:

| Target field | Value |
| --- | --- |
| `identity` | Unique stimulus identity within this scene occurrence. The expected identity must be in this inventory. |
| `visible` | Boolean established from the recording and stimulus evidence, independently of whether a detector found it. |
| `reachable` | Boolean established from independently checked movement bounds and scene geometry, not inferred from a clamped command alone. |
| `registration_valid` | Boolean confirming valid camera-coordinate truth. Set true for reviewed manual pixels without requiring a homography; for mapped truth, require independent registration validation. False excludes the sample as unknown geometry. |
| `centre_method` | Exactly `"manual_pixels"` or `"validated_registration"`. |
| `centre_px` | `null`, or exactly `{"x_px": number, "y_px": number}` in the stored JPEG's orientation and pixel coordinates. Required when visible, reachable and registration-valid; x and y must lie inside the image. |
| `evidence` | Reviewer and explicit references supporting centre, visibility, reach, and coordinate validity. Reference validation residuals for mapped truth. |

Unresolved visibility or reach is not false: leave the frame label null. Invisible
or unreachable targets may have null centres. A visibly clipped target whose
centre cannot be independently located inside the image is outside this scorer's
coordinate model; retain it as unknown rather than inventing a centre or calling
it invisible. Partial occlusion still counts as visible when its centre has valid
independent truth. Do not redefine visibility using detector confidence.

When a track is selected, `selection` maps that recorded selection to an
independently reviewed stimulus identity. A wrong selected identity may lie
outside the intended-target inventory. If the mapping is unresolved, null remains
unresolved even when `expected_identity` is null. If no track is selected, a
non-null mapping is invalid. Expected non-null identity with no selection is an
observed selection failure and still contributes eligible centring time.

For loss cases, annotate `expected_identity: null` only once the independently
specified loss/retention allowance has elapsed. During a brief occlusion the
expected identity may remain non-null while `visible` is false. The scorer does
not invent a switch deadline, a loss deadline, or a successful recovery from a
controller state name.

## Score and interpret the report

```sh
python -m tools.score_monitor_run score "$run_directory" \
    --annotations /path/to/review.json --output /path/to/score.json
```

Both commands create their output exclusively and refuse to overwrite existing
files. Use new paths for revisions. Exit status 0 means a valid artefact was
written, including inconclusive reports; status 2 means invalid input or an I/O
failure. Do not treat exit status 0 as hardware acceptance. The Python API exposes
`load_recording`, `make_review_template` and `score_recording` for reuse, using
`app.json_data.JsonValue` at JSON boundaries and frozen records internally.

The fixed window is the first through the last retained JPEG timestamp. It
cannot be cropped by annotation bounds. Each sample describes the half-open
interval until the next recorded sample, using its **left-hand** coordinates
and labels. The final image has zero duration. This is a declared sample-and-hold
estimate, not proof of continuous motion between camera samples. There is no
interpolation and no final synthetic frame hold.

Settling begins at the first labelled recorded sample of each
presentation/run/repeat/scene occurrence and is split exactly at the declared
elapsed-time boundary. Selecting another tracker ID does not restart it. An
occurrence cannot disappear behind another scene and then reappear; use correct
repeat/run identifiers. Render numbers cannot decrease within an occurrence.
Unlabelled scene boundaries make coverage incomplete. This first-observed-scene
convention is conservative about the settling start; it does not establish the
browser's exact transition time.

`coverage.excluded_duration_s_by_reason` gives disjoint elapsed seconds for
settling, sampling gaps, missing annotations, no expected target, invisibility,
unreachability and failed registration. Settling is deducted first where the
scene is labelled. A gap **strictly greater than** the declared maximum excludes
its entire remaining interval. Other exclusions take precedence in this order:
missing annotation, no expected target, invisible, unreachable, failed
registration. Overlapping reasons therefore do not inflate the sum. The
`long_sampling_gaps_total_s` diagnostic also includes any overlap with settling.
Scored duration plus all exclusion durations equals the fixed recording span.

Missing labels, sampling gaps and failed registration contribute to
`unknown_duration_s`. They cannot create passing coverage. The report includes
labelled and scored fractions of the full recording, frame counts, and a
conservative occupancy bound that treats unknown time as off-centre while
retaining independently justified exclusions. These fields expose a high score
computed from a small subset. Check the evidence for every exclusion and inspect
the full inventory; software cannot detect dishonest human labels.

`centring.goal_met_on_scored_time` is only a numeric comparison with 0.8. Its
`verdict` is inconclusive if any frame lacks a label, any unknown duration
remains, telemetry is incomplete, or no eligible time remains. Independently
reviewed invisible, unreachable, empty and settling periods are explicit policy
exclusions rather than unknown coverage. A complete result can pass or fail the
scoped centring goal in either live hardware or simulation mode; the report
always identifies which mode was used.

Selection reports contain per-frame expected identity, actual selected track,
independently observed selected identity, controller selection-state observation,
and correct/incorrect/unresolved comparisons. Post-settling durations are grouped
by comparison; gaps and unlabelled intervals stay unresolved. Point observations
inside a gap do not fill its duration. These comparisons support inspection of
switching, loss and reacquisition; they do not independently qualify recovery,
identity retention during unobserved intervals or no repeated switching.

`global_verdict` is always `"inconclusive"`: geometry and labelled selection
observations cover only part of the campaign. Timing, cadence, recovery, full
movement envelopes, shutdown, output-mode comparisons and endurance remain
separate [validation gates](../deploy/VALIDATION.md). An entirely blank template
therefore produces null occupancy and explicit missing-annotation duration, never
passing hardware acceptance. No hardware annotations or bench scores ship with
this tool.

## Timing, provenance and limits

Only live `csi_camera` runs using one `delivery_monotonic` recording clock and
consistent hardware/simulation metadata are accepted. Replay and processing-clock
recordings, mixed clocks/modes, non-increasing sequences/times, duplicate
filenames/sequences/labels and non-finite JSON numbers are rejected. Each JPEG's
captured-at value and selected-target metadata must also match a logged frame.
Missing or unsuccessful run-end telemetry and missing/non-zero log-drop counts
make coverage incomplete, as do recording drops or absent recording-bound
telemetry. The retained frame count and first/last timestamps must match the
run-end recording summary, preventing a trimmed index from silently narrowing
the denominator. Existing application telemetry fields are read as observations,
never as independent truth.

Duration is measured on the Pi delivery-monotonic recording timeline. It is not
sensor exposure time. Unix captured-at values are used only to join records with
identical values, not to measure intervals or align clocks. Browser timestamps
are never subtracted from Pi timestamps. Render associations establish content
identity, not display scan-out, exposure, delivery or physical-response latency.

Coverage stops at the retained recording boundaries, even if the application or
presentation ran longer. The scorer cannot recover recording omissions outside
those boundaries, detect movement between samples, decode the JPEG image payload,
verify human evidence, validate a curved-surface registration, measure physical
angles, or establish field accuracy. JPEG headers supply dimensions and content
hashes bind labels to their source files. Preserve the original recordings,
manifest, browser export, policy declaration and annotation evidence for review.

## Verify locally

```sh
python -m pytest tests/test_monitor_scoring.py
python -m ruff check tools/score_monitor_run.py tests/test_monitor_scoring.py
python -m ruff format --check tools/score_monitor_run.py tests/test_monitor_scoring.py
python -m mypy --strict tools/score_monitor_run.py tests/test_monitor_scoring.py
```

Tests use explicitly synthetic camera truth and logs, including absent selection,
irregular timing, gaps, loss expectations, registration failure and invalid
provenance. They do not generate annotations for real recordings.
