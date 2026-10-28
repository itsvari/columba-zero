# Associate retained images with optical frame identities

`tools/associate_monitor_codes.py` reads retained JPEGs and browser exports offline.
It reports optical scene, run and render identities without asserting geometric
registration or photograph positions. It makes no detector, control, freshness,
latency or hardware acceptance decision.

Preserve each recording's original `demo/index.jsonl`, JPEGs and browser JSONL.
Provide a reference recording with the same browser layout and a marker proposal
file. The source and reference recordings may differ. Reference marker entries use
the `image` and `marker_centres_camera_px` fields documented in the
[registration workflow](monitor-registration.md). An optional `review_evidence`
value is retained as provenance. Manual-confirmation fields are not interpreted
or copied into current-image results.

Run from the repository root with NumPy and OpenCV installed:

```sh
python tools/associate_monitor_codes.py target/demo/index.jsonl target-browser.jsonl \
  --reference-index reference/demo/index.jsonl \
  --reference-browser reference-browser.jsonl \
  --reference-markers reference-marker-proposals.json \
  --reference-report reference-review-report.json \
  --output outputs/code-association
```

The output directory must be new. `--reference-report` is optional provenance;
its contents never select identities, sampling positions or code bits. Exit 0
means every requested image has an optical identity, exit 1 means some were
rejected, and exit 2 means an input or output could not be processed. An exit 0
result is not a registration or hardware pass.

Each export must contain one fixed layout. Target and reference layouts must
match, including corner-marker geometry, target region and both code rectangles.
The tool uses each image's actual filename and source sequence from its own index.
It rejects duplicate names or sequences, path traversal and symlinked JPEGs.
It does not rename reference frames as target observations.

The tool verifies every reference independently. Both reference bands must pass
the existing cell-size, contrast, margin and CRC-64 checks, agree, and identify one
record in the reference browser export. Failed references remain in the report
with their reasons and never provide sampling proposals. Reference centroids
provide sampling geometry only; successful decoding does not certify those
centroids as geometric truth.

The proposal positions may also be model predictions from independently located
fixed-grid crosses. Keep the observed marker annotations separately. Record the
grid inputs, model, hashes and predicted-position status in `review_evidence`;
do not label model predictions as confirmed marker observations. Check that the
resulting reference exclusion mask covers every photograph, including annotation
uncertainty. The decoder still requires agreeing CRC-valid bands and a unique
export match. These proposals supply optical sampling geometry only.

For each target image, the tool tries exact marker-pattern candidates first.
Those candidates provide a sampling proposal without manual-confirmation flags.
If they yield no optical association, the tool evaluates every verified reference
in marker-file order. SIFT matches the reference camera image to the current
camera image; RANSAC proposes a camera-to-camera homography. Composition with the
reference sampling homography proposes current code-cell positions.

The SIFT reference mask includes code bands, marker neighbourhoods and fixed
monitor-boundary strips. The entire reference target region, plus a margin of four camera
pixels, is excluded and filled with a constant before feature extraction.
Reference photograph pixels therefore cannot provide descriptors. Features in
the current image are detected across the full camera frame. The fixed boundary
strips require 800×450 reference JPEGs. SIFT ratios, inlier limits and mask regions
are retained in the report; inlier residuals describe proposal fitting, not an
independent registration check.

Every proposed target decode uses the existing 0.25-pixel translation search and
unchanged minimum cell size, contrast, cell margin and CRC-64 gates. Both target
bands must decode and agree under a proposal. All individually CRC-valid band
payloads across evaluated proposals must agree, and exactly one browser record
must match. A passing proposal is never chosen by its expected browser bits.
Competing payloads, valid band disagreement and duplicate export identities fail
closed. The tool performs no bit repair.

`association.json` contains `optically_associated` or `rejected` per image. It
retains source sequences, input and JPEG hashes, reference verification, cell
quality, proposal reasons, masks and SIFT inliers. Accepted entries include the
full matching presentation, session, run, render and scene identity and a hash of
that browser record. The report hashes the source and reference indexes, browser
exports, marker file, optional review report, implementation and decoder code.
`association.sha256` hashes the completed report.

Counts include every rejection and the exported scenes with and without recovered
identities. Missing identities do not establish which scenes those images depict.
Keep rejected images in subsequent coverage accounting. Uneven decoding success
can bias a scored subset, so optical coverage alone does not establish a run pass.

These results are separate from
[independent geometric registration](monitor-association.md). No target centres
are projected. A curved display, interpolation, repeated features or extrapolation
can give a poor sampling homography; the CRC and uniqueness gates remain required.
Agreement of two bands also does not prove that every target-region pixel came
from one render during rolling exposure or display scan-out. CRC-64 detects
accidental corruption and does not authenticate a recording or browser export.
