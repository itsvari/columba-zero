# Locate visible birds in monitor photographs

`tools/locate_monitor_birds.py` estimates the camera position of a **reviewed visible-bird bounding-box centre** from local matches between each source photograph crop and each recorded JPEG. It does not use detector output, controller state, commanded motion or the monitor-marker homography. Source cropping may cut through the bird: the result describes the labelled visible fragment and makes no claim about unseen parts. It is an offline candidate measurement, not tracking or hardware acceptance.

Create a JSON source-label file after reviewing each source crop:

```json
{
  "labels": [
    {
      "asset_id": "tuning-standing-pigeon",
      "source_sha256": "SHA-256 of the source photograph file",
      "crop_xywh_px": [25, 777, 200, 204],
      "visible_bbox_xyxy_px": [0, 0, 200, 204],
      "edge_allowance_source_px": 10,
      "registration_allowance_camera_px": 2,
      "confirmed": true,
      "reviewer": "name or review record identifier"
    }
  ]
}
```

The bounds above are an **illustrative placeholder**, not a reviewed label. Coordinates refer to the displayed crop, from its top-left corner, with right and bottom as geometric edges. The photograph path, crop and dimensions come from the source catalogue. Each entry requires an exact source-file hash and crop match, finite in-crop bounds, explicit confirmation and nonempty reviewer provenance. The reviewer supplies the edge allowance for each bbox edge and a positive camera-pixel allowance for registration. Confirmation records the supplied provenance; this program cannot verify a person's review.

```sh
.venv/bin/python tools/locate_monitor_birds.py path/to/demo/index.jsonl \
  --source-labels path/to/reviewed-labels.json --output path/to/new-directory \
  --image frame_000000720.jpg --image frame_000001032.jpg
```

Omit `--image` to process every index entry. The output directory must not already exist. `locations.json` has one result per selected image and source label. Each result is `rejected`, `candidate`, or `camera_checked`; a rejected source is not evidence of absence. The report hashes the index, labels, source files, camera JPEGs, optional camera checks and tool code. No inputs are modified. `candidate` results include the source-to-camera homography, mapped photograph corners, visible-bbox centre and an axis-aligned centre allowance rectangle. The allowance maps the four combinations of ± the declared source-edge allowance around the box centre, then expands by the declared camera registration allowance. It is an explicit geometric allowance, not a measured confidence interval or bound on all matching error.

An optional independent camera check file may be passed via `--camera-checks`:

```json
{
  "checks": [{
    "image": "frame_000000720.jpg",
    "image_sha256": "SHA-256 of this exact camera JPEG",
    "asset_id": "tuning-standing-pigeon",
    "corners_camera_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
    "maximum_error_px": 7,
    "corner_allowance_px": 2,
    "reviewer": "independent camera-corner reviewer"
  }]
}
```

The sample coordinates are placeholders. Supply actual independently observed photograph rectangle corners in top-left, top-right, bottom-right, bottom-left order. These points are checked **after** fitting and never contribute to it. Each check must include the exact camera JPEG SHA-256; a mismatch rejects that image and asset before matching. `corner_allowance_px` is a finite nonnegative per-axis uncertainty of each independently observed corner; it defaults to zero when omitted. The program reports the raw maximum Euclidean corner residual and the maximum worst-case residual `hypot(abs(dx) + allowance, abs(dy) + allowance)`. A check passes only if that worst-case residual is at most `maximum_error_px`, and the check threshold cannot exceed the label’s declared camera registration allowance. A passing check changes the status to `camera_checked`; a failing check rejects the result. `camera_checked` only reports this geometry check and is not a complete validation or acceptance result. Without a check the status stays `candidate`.

SIFT finds image features separately in each source crop and current camera frame. Source observations are extracted once per label and camera observations once per image, then reused for matching. Matches pass a 0.7 ratio and unique camera keypoint filter, then are split before RANSAC into fit and held-out sets. A candidate requires at least eight fit inliers and eight held-out matches, at least 80% of held-out matches within 2 camera pixels, median held-out error under 1 pixel, source inlier hull covering at least 20% of the crop and containing the labelled centre, and a convex, reasonably shaped and fully visible mapped rectangle. These are exploratory thresholds, not calibrated probabilities. Feature matches can be spatially correlated; low texture, repeated features, blur, changing scale, occlusion and image compression may cause rejects or misleading matches. Held-out feature residuals do not establish absolute bird-centre accuracy. Independently reviewed camera corners, stratified real-frame testing and appropriate uncertainty analysis are needed before interpreting a candidate as validation evidence.

The provisional 7 px registration allowance is a budget, not proof of accuracy or a change to earlier 2 px results. Its calibration used the prior sample’s 3.798 px worst corner residual plus a conservative 2√2 px for an independent ±2 px per-axis annotation allowance, rounded up. This is an empirical planning allowance, not a statistical confidence interval. It must be evaluated against a fresh independent hold-out sample using the worst-case check above; the earlier failures at the 2 px criterion remain failures.
