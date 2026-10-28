"use strict";
let testNowMs = performance.now();
let nextAnimation = null;
let downloadedLog = null;
let downloadedName = null;
const checks = [];
Object.defineProperty(performance, "now", {value: () => testNowMs});
window.requestAnimationFrame = callback => { nextAnimation = callback; return 1; };
URL.createObjectURL = blob => { downloadedLog = blob; return "blob:monitor-test"; };
URL.revokeObjectURL = () => {};
HTMLAnchorElement.prototype.click = function () { downloadedName = this.download; };

function check(condition, description) {
  if (!condition) throw new Error(description);
  checks.push(description);
}

function advance(milliseconds) {
  testNowMs += milliseconds;
  const callback = nextAnimation;
  nextAnimation = null;
  callback(testNowMs);
}

function logged() {
  return records.map(line => JSON.parse(line));
}

function latestFrame() {
  return logged().findLast(row => row.kind === "frame");
}

function jumpToScene(name, seconds = 0) {
  const index = manifest.scenes.findIndex(scene => scene.name === name);
  const startFrame = manifest.scenes.slice(0, index).reduce(
    (total, scene) => total + scene.duration_frames, 0);
  positionMs = startFrame * frameDurationMs + seconds * millisecondsPerSecond;
  anchorMs = testNowMs;
  sequenceFrame = Math.floor(positionMs / frameDurationMs);
  render(true);
}

window.monitorTestCompletion = (async () => {
  const maximumLoadingAttempts = 500;
  const loadingIntervalMs = 10;
  for (let attempt = 0; !ready && attempt < maximumLoadingAttempts; attempt += 1) {
    await new Promise(resolve => setTimeout(resolve, loadingIntervalMs));
  }
  check(ready, "all embedded JPEGs decode with their declared dimensions");
  check(!playing && latestFrame().scene_frame === 0, "presentation opens paused at frame zero");
  check(photographs.size === 3, "playlist loads two pigeon photographs and a confuser");
  const initial = latestFrame();
  const initialTarget = initial.targets[0];
  const region = initial.target_region_screen_css_px;
  check(Math.abs(initialTarget.box_screen_css_px.left_px +
    initialTarget.box_screen_css_px.width_px / 2 - region.left_px - region.width_px / 2) < 0.001,
  "centred target is centred in recorded viewport CSS coordinates");
  check(initialTarget.visible_fraction === 1 && initialTarget.subject === "pigeon",
    "known positive is fully visible");
  const layout = geometry();
  const sample = context.getImageData(
    Math.round((initialTarget.box_screen_css_px.left_px - layout.bounds.left) * layout.devicePixelRatio),
    Math.round((initialTarget.box_screen_css_px.top_px - layout.bounds.top) * layout.devicePixelRatio),
    Math.max(1, Math.floor(initialTarget.box_screen_css_px.width_px * layout.devicePixelRatio)),
    Math.max(1, Math.floor(initialTarget.box_screen_css_px.height_px * layout.devicePixelRatio))).data;
  const colours = new Set();
  for (let index = 0; index < sample.length; index += 4) {
    colours.add(`${sample[index]}:${sample[index + 1]}:${sample[index + 2]}`);
  }
  check(colours.size > 100, "canvas contains decoded photographic pixels");
  check(initial.registration_markers.length === 4 && initial.registration_markers.every(marker => {
    const bounds = marker.box_screen_css_px;
    return intersectionArea(bounds, region) === 0;
  }), "all four camera-visible registration markers lie outside target region");

  if (manifest.photo_registration_frame) {
    const interiorInsetPx = 2;
    const sourceLeftPx = Math.ceil(
      (initialTarget.box_screen_css_px.left_px - layout.bounds.left) * layout.devicePixelRatio);
    const sourceTopPx = Math.ceil(
      (initialTarget.box_screen_css_px.top_px - layout.bounds.top) * layout.devicePixelRatio);
    const interiorWidthPx = Math.floor(initialTarget.box_screen_css_px.width_px *
      layout.devicePixelRatio) - 2 * interiorInsetPx;
    const interiorHeightPx = Math.floor(initialTarget.box_screen_css_px.height_px *
      layout.devicePixelRatio) - 2 * interiorInsetPx;
    const framedInterior = context.getImageData(sourceLeftPx + interiorInsetPx,
      sourceTopPx + interiorInsetPx, interiorWidthPx, interiorHeightPx).data;
    const frameSampleLeftPx = Math.round((initialTarget.box_screen_css_px.left_px -
      layout.bounds.left - photographFrameInnerPaddingPx / 2 * layout.scale) *
      layout.devicePixelRatio);
    const frameSampleTopPx = Math.round((initialTarget.box_screen_css_px.top_px -
      layout.bounds.top + initialTarget.box_screen_css_px.height_px / 2) * layout.devicePixelRatio);
    const framedEdge = context.getImageData(frameSampleLeftPx, frameSampleTopPx, 1, 1).data;
    check(initial.photo_registration_frame && initialTarget.photo_registration_frame_drawn,
      "calibration frame is declared in the manifest and camera-facing frame record");
    manifest.photo_registration_frame = false;
    render(true);
    const plainInterior = context.getImageData(sourceLeftPx + interiorInsetPx,
      sourceTopPx + interiorInsetPx, interiorWidthPx, interiorHeightPx).data;
    const plainEdge = context.getImageData(frameSampleLeftPx, frameSampleTopPx, 1, 1).data;
    check(framedInterior.every((value, index) => value === plainInterior[index]),
      "calibration frame leaves the photograph interior unchanged");
    check(framedEdge[0] === 0 && framedEdge[1] === 0 && framedEdge[2] === 0 && plainEdge[0] > 0,
      "calibration frame draws a visible black edge outside the photograph");
    check(JSON.stringify(latestFrame().targets[0].box_screen_css_px) ===
      JSON.stringify(initialTarget.box_screen_css_px),
      "calibration frame leaves photograph coordinates unchanged");
    manifest.photo_registration_frame = true;
    render(true);
  }

  document.getElementById("play").click();
  advance(1100);
  const beforePause = sequenceFrame;
  document.dispatchEvent(new KeyboardEvent("keydown", {key: " "}));
  advance(3000);
  check(!playing && sequenceFrame === beforePause, "pause freezes stimulus time");
  document.getElementById("play").click();
  advance(200);
  check(sequenceFrame > beforePause && sequenceFrame < beforePause + manifest.frame_rate_hz,
    "resume excludes paused duration");
  check(logged().some(row => row.event === "logical_frames_skipped"),
    "late animation callbacks report skipped logical frames");
  document.getElementById("reset").click();
  check(!playing && sequenceFrame === 0 && runNumber === 2, "reset starts a fresh paused run");
  document.getElementById("replay").click();
  check(playing && sequenceFrame === 0 && runNumber === 3, "replay starts a fresh running sequence");
  check(JSON.stringify(latestFrame().targets) === JSON.stringify(initial.targets),
    "replay reproduces the exact target identity and geometry");
  const earlierAnimationBrowserMs = testNowMs - 2;
  const replayAnimation = nextAnimation;
  nextAnimation = null;
  replayAnimation(earlierAnimationBrowserMs);
  check(sequenceFrame === 0 && latestFrame().repeat_index === 0 &&
    latestFrame().targets[0].visibility === "visible",
    "an animation timestamp before replay stays at the first visible frame");
  check(logged().filter(row => row.kind === "frame").every(row =>
    row.sequence_frame >= 0 && row.repeat_index >= 0),
    "replay never exports negative frame or repeat indices");

  jumpToScene("stop-reverse", 2);
  const movingLeft = latestFrame().targets[0].box_screen_css_px.left_px;
  jumpToScene("stop-reverse", 4);
  const movingRight = latestFrame().targets[0].box_screen_css_px.left_px;
  jumpToScene("stop-reverse", 6);
  check(movingLeft < movingRight && latestFrame().targets[0].box_screen_css_px.left_px < movingRight,
    "motion path moves then reverses");
  jumpToScene("scale", 1);
  const smallWidth = latestFrame().targets[0].box_screen_css_px.width_px;
  jumpToScene("scale", 4);
  check(latestFrame().targets[0].box_screen_css_px.width_px > smallWidth,
    "scale scene changes displayed target size");
  jumpToScene("similar-size", 3);
  const similar = latestFrame().targets;
  const area = target => target.box_screen_css_px.width_px * target.box_screen_css_px.height_px;
  check(similar.every(target => target.visible_fraction === 1) &&
    area(similar[1]) / area(similar[0]) < 1.25, "similar challenger stays below area-switch threshold");
  jumpToScene("brief-challenger", 2.1);
  check(latestFrame().targets[1].present, "brief larger challenger appears");
  jumpToScene("brief-challenger", 2.4);
  check(!latestFrame().targets[1].present, "brief challenger leaves before switch hold");
  jumpToScene("sustained-challenger", 4);
  const sustained = latestFrame().targets;
  check(sustained[1].present && area(sustained[1]) / area(sustained[0]) > 1.25,
    "sustained challenger remains sufficiently larger");
  jumpToScene("partial-occlusion", 3.1);
  check(Math.abs(latestFrame().targets[0].visible_fraction - 0.4) < 0.001,
    "partial occlusion exports remaining photograph fraction");
  check(!latestFrame().targets[0].photo_registration_frame_drawn,
    "calibration frame is absent during partial occlusion");
  for (const name of ["brief-full-occlusion", "full-occlusion"]) {
    jumpToScene(name, 3.1);
    check(latestFrame().targets[0].visibility === "occluded" &&
      latestFrame().targets[0].visible_fraction === 0, `${name} hides the complete target`);
    check(!latestFrame().targets[0].photo_registration_frame_drawn,
      `${name} leaves no calibration frame around the hidden target`);
    jumpToScene(name, 5);
    check(latestFrame().targets[0].visible_fraction === 1, `${name} restores the same identity`);
  }
  jumpToScene("departure", 6);
  check(latestFrame().targets[0].visibility === "outside_target_region",
    "departure clips the target before it reaches reference markers");
  jumpToScene("empty", 1);
  check(latestFrame().targets.length === 0, "empty scene has no intended targets");
  jumpToScene("confuser", 1);
  check(latestFrame().targets[0].subject === "other_bird", "confuser is independently labelled other bird");

  Object.defineProperty(document, "hidden", {configurable: true, value: true});
  document.dispatchEvent(new Event("visibilitychange"));
  check(!playing, "hidden page pauses the sequence");
  document.getElementById("play").click();
  check(!playing, "hidden page cannot silently resume");
  Object.defineProperty(document, "hidden", {configurable: true, value: false});
  document.dispatchEvent(new Event("visibilitychange"));
  document.getElementById("play").click();
  canvas.style.flex = "none";
  canvas.style.width = "640px";
  canvas.style.height = "360px";
  window.dispatchEvent(new Event("resize"));
  check(!playing && latestFrame().target_region_screen_css_px.width_px < region.width_px,
    "viewport change pauses and records recalculated screen geometry");
  const fullscreenRequest = document.documentElement.requestFullscreen;
  document.documentElement.requestFullscreen = async () => { throw new Error("test rejection"); };
  await toggleFullscreen();
  document.documentElement.requestFullscreen = fullscreenRequest;
  check(logged().some(row => row.event === "fullscreen_error"), "fullscreen rejection is exported");

  restart(true);
  positionMs = (cycleFrames - 1) * frameDurationMs;
  anchorMs = testNowMs;
  advance(frameDurationMs * 2);
  check(locateFrame(sequenceFrame).repeatIndex === 1 && playing,
    "repeat boundary advances without restarting browser time");
  positionMs = durationMs - frameDurationMs;
  anchorMs = testNowMs;
  advance(frameDurationMs * 2);
  check(!playing && logged().some(row => row.event === "complete"),
    "playlist finishes at its final frame and stops");
  const sceneBalance = new Map();
  for (const row of logged()) {
    if (row.event === "scene_start") {
      const key = `${row.run_number}:${row.repeat_index}:${row.scene_id}`;
      sceneBalance.set(key, (sceneBalance.get(key) || 0) + 1);
    }
    if (row.event === "scene_end") {
      const key = `${row.ended_run_number}:${row.ended_repeat_index}:${row.ended_scene_id}`;
      sceneBalance.set(key, (sceneBalance.get(key) || 0) - 1);
    }
  }
  check([...sceneBalance.values()].every(balance => balance === 0),
    "every started scene ends once with its original run and repeat identity");
  document.getElementById("export").click();
  const exported = (await downloadedLog.text()).trim().split("\n").map(line => JSON.parse(line));
  check(exported.length === records.length && downloadedName.endsWith(".jsonl"),
    "export contains every retained record as valid JSONL");
  check(exported[0].kind === "session" && exported[0].clock_domain.includes("browser") &&
    exported[0].manifest.assets.every(asset => !Object.hasOwn(asset, "data_url")),
    "session log declares browser clock, assets and independently labelled manifest");
  check(exported.every(row => !Object.keys(row).some(key => /exposure|sensor_timestamp/.test(key))),
    "records never claim Pi exposure timestamps");
  check(exported.every((row, index) => index === 0 ||
    row.browser_monotonic_ms >= exported[index - 1].browser_monotonic_ms),
    "browser clock stays monotonic across pause, reset and replay");
  const exportCount = exported.length;
  restart(true);
  manifest.maximum_log_records = records.length + 3;
  document.getElementById("export").click();
  document.getElementById("export").click();
  document.getElementById("export").click();
  check(logFull && !playing && logged().at(-1).event === "log_limit" &&
    records.length <= manifest.maximum_log_records,
    "event-only log growth stops presentation at the hard buffer limit");
  const stoppedRender = renderNumber;
  advance(1000);
  check(renderNumber === stoppedRender, "log exhaustion prevents unlogged stimulus frames");
  return {checks, export_count: exportCount, playlist: manifest.playlist};
})();
