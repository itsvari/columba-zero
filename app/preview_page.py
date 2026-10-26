"""Self-contained browser page that the preview service serves at ``/``.

The page reads ``/api/state``, shows ``/stream.mjpg`` and posts control requests to
``/api/control``. It never talks to the mount itself: the controller worker owns all
physical writes. The markup stays ASCII so any encoding of the response is safe.
"""

from __future__ import annotations

PREVIEW_PAGE: str = r"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Columba preview</title>
<style>
  :root {
    --bg: #0e100e;
    --housing: #151815;
    --panel: #1a1e1a;
    --sunken: #080908;
    --rule: #2a2f29;
    --rule-strong: #3d443b;
    --text: #e0e3d9;
    --muted: #9ea597;
    --faint: #6c7368;
    --ok: #93d26e;
    --info: #00c8ff;
    --warn: #f0b440;
    --bad: #f2695c;
    --label-font: "DIN Alternate", "DIN 2014", "D-DIN", "Bahnschrift",
      "Barlow Semi Condensed", "Roboto Condensed", "Arial Narrow", sans-serif;
    --mono: ui-monospace, "SF Mono", "Cascadia Mono", "JetBrains Mono", Menlo,
      Consolas, monospace;
    --gap: 1rem;
    --radius: 2px;
    color-scheme: dark;
  }

  *, *::before, *::after { box-sizing: border-box; }

  html { background: var(--bg); }

  body {
    margin: 0;
    min-height: 100vh;
    color: var(--text);
    font-family: var(--label-font);
    font-size: 15px;
    line-height: 1.4;
    -webkit-font-smoothing: antialiased;
  }

  body::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    background:
      repeating-linear-gradient(0deg, transparent 0 23px, rgb(255 255 255 / 0.022) 23px 24px),
      repeating-linear-gradient(90deg, transparent 0 23px, rgb(255 255 255 / 0.022) 23px 24px);
  }

  .visually-hidden {
    position: absolute !important;
    width: 1px;
    height: 1px;
    margin: -1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0 0 0 0);
    white-space: nowrap;
    border: 0;
  }

  .skip {
    position: absolute;
    left: 0.75rem;
    top: -3rem;
    z-index: 10;
    padding: 0.5rem 0.75rem;
    color: var(--bg);
    background: var(--info);
    font-weight: 700;
    text-decoration: none;
  }

  .skip:focus { top: 0.75rem; }

  :focus-visible {
    outline: 2px solid var(--info);
    outline-offset: 2px;
  }

  [data-tone="ok"] { --tone: var(--ok); }
  [data-tone="info"] { --tone: var(--info); }
  [data-tone="warn"] { --tone: var(--warn); }
  [data-tone="bad"] { --tone: var(--bad); }
  [data-tone="idle"] { --tone: var(--muted); }

  .bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem 1.5rem;
    padding: 0.7rem clamp(0.75rem, 2.5vw, 1.75rem);
    border-bottom: 1px solid var(--rule);
    background: var(--housing);
  }

  .brand {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin: 0;
    font-size: 1rem;
    font-weight: 400;
  }

  .brand-mark {
    align-self: center;
    position: relative;
    width: 14px;
    height: 14px;
    border: 1.5px solid var(--text);
  }

  .brand-mark::after {
    content: "";
    position: absolute;
    inset: 3.5px;
    background: var(--info);
  }

  .brand-name {
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
  }

  .brand-sub {
    color: var(--muted);
    letter-spacing: 0.06em;
  }

  .bar-status {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    min-height: 1.75rem;
    padding: 0.15rem 0.6rem;
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius);
    color: var(--text);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    white-space: nowrap;
  }

  .badge[data-tone] { border-color: color-mix(in srgb, var(--tone) 55%, transparent); }
  .badge[hidden] { display: none; }

  .lamp {
    flex: none;
    width: 0.55rem;
    height: 0.55rem;
    border-radius: 50%;
    background: var(--tone, var(--muted));
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--tone, var(--muted)) 22%, transparent);
  }

  .chip[data-tone="ok"] > .lamp { animation: breathe 2.4s ease-in-out infinite; }

  @keyframes breathe {
    50% { box-shadow: 0 0 0 4px color-mix(in srgb, var(--tone) 10%, transparent); }
  }

  .layout {
    display: grid;
    gap: var(--gap);
    max-width: 1440px;
    margin: 0 auto;
    padding: var(--gap) clamp(0.75rem, 2.5vw, 1.75rem) 2rem;
  }

  @media (min-width: 1024px) {
    .layout {
      grid-template-columns: minmax(0, 1fr) 23rem;
      align-items: start;
    }

    .view {
      position: sticky;
      top: var(--gap);
    }
  }

  .view {
    display: grid;
    gap: 0.75rem;
    min-width: 0;
  }

  .frame {
    position: relative;
    aspect-ratio: 16 / 9;
    overflow: hidden;
    border: 1px solid var(--rule-strong);
    background:
      radial-gradient(circle at 50% 50%, rgb(255 255 255 / 0.03), transparent 60%),
      var(--sunken);
    transition: border-color 160ms ease;
  }

  .frame[data-tone="warn"], .frame[data-tone="bad"] { border-color: var(--tone); }

  .frame img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
    transition: filter 200ms ease, opacity 200ms ease;
  }

  .frame[data-image="none"] img { opacity: 0; }
  .frame[data-dim="true"] img { filter: grayscale(1) brightness(0.5) contrast(0.9); }

  .tick {
    position: absolute;
    width: 18px;
    height: 18px;
    border: 0 solid var(--muted);
    pointer-events: none;
  }

  .tick-tl { top: 8px; left: 8px; border-top-width: 1.5px; border-left-width: 1.5px; }
  .tick-tr { top: 8px; right: 8px; border-top-width: 1.5px; border-right-width: 1.5px; }
  .tick-bl { bottom: 8px; left: 8px; border-bottom-width: 1.5px; border-left-width: 1.5px; }
  .tick-br { bottom: 8px; right: 8px; border-bottom-width: 1.5px; border-right-width: 1.5px; }

  .chips {
    position: absolute;
    top: 14px;
    left: 14px;
    right: 14px;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 0.5rem;
    pointer-events: none;
  }

  .chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.6rem;
    border: 1px solid color-mix(in srgb, var(--tone, var(--muted)) 60%, transparent);
    background: rgb(8 9 8 / 0.82);
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-variant-numeric: tabular-nums;
  }

  .chip[hidden] { display: none; }

  .chip-age { color: var(--muted); }

  .veil {
    position: absolute;
    inset: 0;
    display: grid;
    place-content: center;
    gap: 0.4rem;
    padding: 3.5rem 1.5rem 1.5rem;
    text-align: center;
    background:
      repeating-linear-gradient(
        135deg,
        color-mix(in srgb, var(--tone, var(--muted)) 7%, transparent) 0 10px,
        transparent 10px 20px
      ),
      rgb(8 9 8 / 0.55);
  }

  .frame[data-image="none"] .veil { background-color: transparent; }
  .veil[hidden] { display: none; }

  .veil-title {
    margin: 0;
    color: var(--tone, var(--text));
    font-size: clamp(1.05rem, 2.4vw, 1.5rem);
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .veil-detail {
    max-width: 34rem;
    margin: 0 auto;
    color: var(--text);
    font-size: 0.95rem;
  }

  .controls {
    display: grid;
    gap: 0.6rem;
    padding: 0.9rem;
    border: 1px solid var(--rule);
    background: var(--panel);
  }

  .controls-head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.25rem 1rem;
  }

  .buttons {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.5rem;
  }

  @media (max-width: 420px) {
    .buttons { grid-template-columns: 1fr; }
  }

  button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.55rem;
    min-height: 2.9rem;
    padding: 0.5rem 0.75rem;
    border: 1px solid var(--rule-strong);
    border-radius: var(--radius);
    color: var(--text);
    background: linear-gradient(#232823, #1c201c);
    font: inherit;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    transition: border-color 120ms ease, background 120ms ease;
  }

  button svg {
    flex: none;
    width: 1rem;
    height: 1rem;
    fill: none;
    stroke: currentColor;
    stroke-width: 2;
  }

  button:hover:enabled {
    border-color: var(--muted);
    background: linear-gradient(#2a302a, #222722);
  }

  button:active:enabled { transform: translateY(1px); }

  button:disabled {
    color: var(--faint);
    background: transparent;
    border-color: var(--rule);
    cursor: not-allowed;
  }

  button[aria-busy="true"] { border-color: var(--info); }

  .note {
    margin: 0;
    color: var(--muted);
    font-size: 0.9rem;
  }

  .feedback {
    min-height: 1.4em;
    margin: 0;
    font-size: 0.92rem;
  }

  .feedback[data-tone] { color: var(--tone); }

  .rail {
    display: grid;
    gap: var(--gap);
    min-width: 0;
  }

  @media (min-width: 640px) and (max-width: 1023px) {
    .rail { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .rail-note { grid-column: 1 / -1; }
  }

  .rail-note {
    margin: 0;
    padding: 0.45rem 0.7rem;
    border-left: 2px solid var(--tone, var(--rule-strong));
    color: var(--muted);
    font-family: var(--mono);
    font-size: 0.78rem;
    font-variant-numeric: tabular-nums;
  }

  .rail-note[data-tone="warn"], .rail-note[data-tone="bad"] { color: var(--tone); }

  .panel {
    min-width: 0;
    padding: 0.9rem;
    border: 1px solid var(--rule);
    background: var(--panel);
  }

  .panel-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.7rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid var(--rule);
  }

  h2 {
    margin: 0;
    color: var(--muted);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
  }

  .state {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    color: var(--tone, var(--text));
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.85rem;
  }

  .headline {
    margin: 0 0 0.6rem;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.02em;
  }

  .headline .mono { font-weight: 400; }

  dl.readouts {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.3rem 1rem;
    margin: 0;
  }

  dl.readouts dt {
    color: var(--muted);
    font-size: 0.88rem;
  }

  dl.readouts dd {
    margin: 0;
    text-align: right;
    font-family: var(--mono);
    font-size: 0.86rem;
    font-variant-numeric: tabular-nums;
    overflow-wrap: anywhere;
  }

  dl.readouts dd[data-empty="true"] { color: var(--faint); }
  dl.readouts dd[data-tone] { color: var(--tone); }

  .rail[data-frozen="true"] dl.readouts dd,
  .rail[data-frozen="true"] .headline,
  .rail[data-frozen="true"] .plot { opacity: 0.55; }

  .mono {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
  }

  .fine {
    margin: 0.7rem 0 0;
    color: var(--muted);
    font-size: 0.82rem;
  }

  .alert {
    margin: 0.7rem 0 0;
    padding: 0.5rem 0.6rem;
    border-left: 2px solid var(--tone, var(--bad));
    color: var(--text);
    background: color-mix(in srgb, var(--tone, var(--bad)) 9%, transparent);
    font-size: 0.88rem;
    overflow-wrap: anywhere;
  }

  .alert[hidden] { display: none; }

  .plot {
    display: block;
    width: 100%;
    max-width: 16rem;
    height: auto;
    margin: 0 auto 0.75rem;
    overflow: visible;
  }

  .plot-frame { fill: var(--sunken); stroke: var(--rule-strong); stroke-width: 1; }
  .plot-cross { stroke: var(--faint); stroke-width: 0.75; stroke-dasharray: 2 3; }
  .plot-deadband { fill: rgb(180 180 180 / 0.08); stroke: rgb(180 180 180 / 0.7); }
  .plot-dot { fill: var(--info); stroke: var(--sunken); stroke-width: 1.5; }
  .plot-dot[data-held="true"] { fill: none; stroke: var(--info); }

  .plot-caption {
    margin: -0.35rem 0 0.75rem;
    color: var(--muted);
    font-size: 0.8rem;
    text-align: center;
  }

  .key {
    display: grid;
    gap: 0.35rem;
    margin: 0;
    padding: 0;
    list-style: none;
    font-size: 0.88rem;
  }

  .key li {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .swatch {
    flex: none;
    width: 1.1rem;
    height: 0.75rem;
    border: 2px solid var(--swatch);
  }

  .swatch-dashed { border-style: dashed; }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation: none !important;
      transition: none !important;
    }
  }

  @media (forced-colors: active) {
    .lamp { forced-color-adjust: none; }
    .frame, .panel, .controls { border-color: CanvasText; }
  }
</style>
</head>
<body>
<a class="skip" href="#controls">Skip to controls</a>

<header class="bar">
  <h1 class="brand">
    <span class="brand-mark" aria-hidden="true"></span>
    <span class="brand-name">Columba</span>
    <span class="brand-sub">Pan-tilt preview</span>
  </h1>
  <div class="bar-status">
    <span class="badge" id="mount-badge" data-tone="idle">
      <span class="lamp" aria-hidden="true"></span><span id="mount-text">Mount unknown</span>
    </span>
    <span class="badge" id="timing-badge" data-tone="info" hidden>Replay</span>
    <span class="badge" id="link-badge" data-tone="idle">
      <span class="lamp" aria-hidden="true"></span><span id="link-text">Connecting</span>
    </span>
  </div>
</header>

<main class="layout">
  <section class="view" aria-labelledby="view-heading">
    <h2 id="view-heading" class="visually-hidden">Camera view</h2>
    <div class="frame" id="frame" data-image="none" data-dim="false" data-tone="idle">
      <img id="stream" alt="Annotated camera view with detection boxes and the centre deadband">
      <span class="tick tick-tl" aria-hidden="true"></span>
      <span class="tick tick-tr" aria-hidden="true"></span>
      <span class="tick tick-bl" aria-hidden="true"></span>
      <span class="tick tick-br" aria-hidden="true"></span>
      <div class="chips">
        <span class="chip" id="feed-chip" data-tone="idle">
          <span class="lamp" aria-hidden="true"></span>
          <span id="feed-label">Connecting</span>
          <span class="chip-age" id="feed-age"></span>
        </span>
        <span class="chip" id="sim-chip" data-tone="info" hidden>Simulated mount</span>
      </div>
      <div class="veil" id="veil">
        <p class="veil-title" id="veil-title">Connecting to preview</p>
        <p class="veil-detail" id="veil-detail">Requesting state from this device.</p>
      </div>
    </div>
    <p class="visually-hidden" id="feed-announcer" role="status" aria-live="polite"></p>

    <section class="controls" id="controls" aria-labelledby="controls-heading" tabindex="-1">
      <div class="controls-head">
        <h2 id="controls-heading">Following controls</h2>
        <span class="state" id="control-state" data-tone="idle"></span>
      </div>
      <div class="buttons" role="group" aria-labelledby="controls-heading"
        aria-describedby="control-note">
        <button type="button" id="pause-button" data-action="pause" disabled>
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3v10M11 3v10"/></svg>
          Pause
        </button>
        <button type="button" id="resume-button" data-action="resume" disabled>
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3l8 5-8 5z"/></svg>
          Resume
        </button>
        <button type="button" id="home-button" data-action="home" disabled>
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <circle cx="8" cy="8" r="4.5"/><path d="M8 1v3M8 12v3M1 8h3M12 8h3"/>
          </svg>
          Home and pause
        </button>
      </div>
      <p class="note" id="control-note">Waiting for controller state.</p>
      <p class="feedback" id="control-feedback" role="status" aria-live="polite"></p>
    </section>
  </section>

  <aside class="rail" id="rail" aria-label="Tracking and system readings" data-frozen="false">
    <p class="rail-note" id="rail-note" data-tone="idle">No readings yet</p>

    <section class="panel" aria-labelledby="target-heading">
      <div class="panel-head">
        <h2 id="target-heading">Target</h2>
        <span class="state" id="target-state" data-tone="idle"></span>
      </div>
      <p class="headline" id="target-headline">No target data</p>
      <dl class="readouts">
        <dt>Confidence</dt><dd id="target-confidence"></dd>
        <dt>Box size</dt><dd id="target-box"></dd>
        <dt>Smoothed area</dt><dd id="target-area"></dd>
        <dt>This frame</dt><dd id="target-matched"></dd>
      </dl>
      <p class="fine">
        Nearest-looking means the confirmed pigeon with the largest smoothed box area.
        The system ranks by area and does not estimate distance.
      </p>
    </section>

    <section class="panel" aria-labelledby="controller-heading">
      <div class="panel-head">
        <h2 id="controller-heading">Controller</h2>
        <span class="state" id="controller-state" data-tone="idle"></span>
      </div>
      <svg class="plot" id="plot" viewBox="-2 -2 164 94" role="img"
        aria-labelledby="plot-title">
        <title id="plot-title">Target offset from image centre: not reported</title>
        <rect class="plot-frame" x="0" y="0" width="160" height="90"/>
        <line class="plot-cross" x1="80" y1="0" x2="80" y2="90"/>
        <line class="plot-cross" x1="0" y1="45" x2="160" y2="45"/>
        <rect class="plot-deadband" id="plot-deadband" x="80" y="45" width="0" height="0"/>
        <circle class="plot-dot" id="plot-dot" cx="80" cy="45" r="3.5" visibility="hidden"/>
      </svg>
      <p class="plot-caption">Offset in the raw image, x right, y down. Grey box: deadband.</p>
      <dl class="readouts">
        <dt>Pan command</dt><dd id="controller-pan"></dd>
        <dt>Tilt command</dt><dd id="controller-tilt"></dd>
        <dt>Offset x / y</dt><dd id="controller-error"></dd>
        <dt>Deadband</dt><dd id="controller-deadband"></dd>
        <dt>Observation age</dt><dd id="controller-age"></dd>
        <dt>Timing source</dt><dd id="controller-timing"></dd>
        <dt>Tracked ID</dt><dd id="controller-track"></dd>
      </dl>
      <p class="alert" id="controller-fault" hidden></p>
    </section>

    <section class="panel" aria-labelledby="performance-heading">
      <div class="panel-head"><h2 id="performance-heading">Performance</h2></div>
      <dl class="readouts">
        <dt>Processing</dt><dd id="performance-processing"></dd>
        <dt>Capture</dt><dd id="performance-capture"></dd>
        <dt>Newest frame age</dt><dd id="performance-age"></dd>
        <dt>Shown frame received</dt><dd id="performance-received"></dd>
        <dt>Last encode</dt><dd id="performance-encode"></dd>
        <dt>Encoded frames</dt><dd id="performance-encoded"></dd>
        <dt>Overwritten frames</dt><dd id="performance-overwritten"></dd>
      </dl>
      <p class="fine">
        Overwritten frames were processed but replaced by a newer one before encoding.
      </p>
    </section>

    <section class="panel" aria-labelledby="recording-heading">
      <div class="panel-head">
        <h2 id="recording-heading">Recording</h2>
        <span class="state" id="recording-state" data-tone="idle"></span>
      </div>
      <dl class="readouts">
        <dt>Recorded span</dt><dd id="recording-span"></dd>
        <dt>Frames written</dt><dd id="recording-written"></dd>
        <dt>Frames dropped</dt><dd id="recording-dropped"></dd>
        <dt>Stored</dt><dd id="recording-bytes"></dd>
      </dl>
      <p class="alert" id="recording-error" hidden></p>
    </section>

    <section class="panel" aria-labelledby="key-heading">
      <div class="panel-head"><h2 id="key-heading">Overlay key</h2></div>
      <ul class="key">
        <li><span class="swatch" style="--swatch: rgb(0 200 255)" aria-hidden="true"></span>
          Followed target</li>
        <li><span class="swatch" style="--swatch: rgb(255 0 0)" aria-hidden="true"></span>
          Confirmed pigeon</li>
        <li><span class="swatch" style="--swatch: rgb(0 255 0)" aria-hidden="true"></span>
          Unconfirmed pigeon</li>
        <li><span class="swatch" style="--swatch: rgb(255 165 0)" aria-hidden="true"></span>
          Other class</li>
        <li><span class="swatch" style="--swatch: rgb(150 150 150)" aria-hidden="true"></span>
          Low confidence: ASSOCIATION (untracked) or MAINTENANCE (tracked)</li>
        <li><span class="swatch swatch-dashed" style="--swatch: rgb(180 180 180)"
          aria-hidden="true"></span>Centre deadband</li>
      </ul>
    </section>
  </aside>
</main>

<script>
(() => {
  "use strict";

  const POLL_INTERVAL_MS = 500;
  const POLL_HIDDEN_INTERVAL_MS = 2000;
  const POLL_BACKOFF_MAX_MS = 4000;
  const STATE_TIMEOUT_MS = 3000;
  const CONTROL_TIMEOUT_MS = 5000;
  const STALE_FRAME_AGE_MS = 2000;
  const STATE_LOST_AFTER_MS = 3000;
  const STREAM_RETRY_INITIAL_MS = 1000;
  const STREAM_RETRY_MAX_MS = 8000;
  const STREAM_SILENCE_RECONNECT_MS = 5000;
  const RENDER_INTERVAL_MS = 250;
  const FEEDBACK_VISIBLE_MS = 8000;
  const MILLISECONDS_PER_SECOND = 1000;
  const SECONDS_PER_MINUTE = 60;
  const BYTES_PER_KIBIBYTE = 1024;
  const BYTES_PER_MEBIBYTE = 1048576;
  const STREAM_BUFFER_MAX_BYTES = 8 * BYTES_PER_MEBIBYTE;
  const PERCENT = 100;
  const PLOT_WIDTH = 160;
  const PLOT_HEIGHT = 90;
  const IMAGE_CENTRE_FRACTION = 0.5;
  const STREAM_PATH = "/stream.mjpg";
  const STATE_PATH = "/api/state";
  const CONTROL_PATH = "/api/control";
  const EMPTY = "\u2014";
  const DEGREE = "\u00b0";
  const TIMES = "\u00d7";
  const PLUS_MINUS = "\u00b1";
  const ELLIPSIS = "\u2026";
  const SQUARED = "\u00b2";

  const TARGET_STATES = {
    acquiring: { label: "Acquiring", tone: "idle" },
    tracking: { label: "Tracking", tone: "ok" },
    waiting: { label: "Waiting", tone: "warn" },
    lost: { label: "Lost", tone: "bad" },
  };

  const CONTROLLER_STATES = {
    acquiring: { label: "Acquiring", tone: "idle" },
    tracking: { label: "Following", tone: "ok" },
    waiting: { label: "Holding", tone: "warn" },
    stale: { label: "Stale input", tone: "warn" },
    uncertain: { label: "Uncertain", tone: "warn" },
    paused: { label: "Paused", tone: "info" },
    returning: { label: "Returning home", tone: "info" },
    home: { label: "At home", tone: "info" },
    fault: { label: "Fault", tone: "bad" },
  };

  const RECORDING_STATES = {
    idle: { label: "Idle", tone: "idle" },
    recording: { label: "Recording", tone: "ok" },
    finished: { label: "Finished", tone: "idle" },
    limited: { label: "Limit reached", tone: "warn" },
    error: { label: "Fault", tone: "bad" },
  };

  const TIMING_SOURCES = {
    sensor: "Sensor timestamp",
    delivery: "Frame delivery",
    replay: "Replay source",
    unavailable: "Unavailable",
  };

  const ACTIONS = {
    pause: {
      label: "Pause",
      accepted: "Pause accepted. The controller holds position until you resume.",
    },
    resume: {
      label: "Resume",
      accepted: "Resume accepted. Following restarts when a confirmed target is visible.",
    },
    home: {
      label: "Home and pause",
      accepted: "Home accepted. The mount returns home and stays paused until you resume.",
    },
  };

  const byId = (id) => document.getElementById(id);

  const ui = {
    mountBadge: byId("mount-badge"),
    mountText: byId("mount-text"),
    timingBadge: byId("timing-badge"),
    linkBadge: byId("link-badge"),
    linkText: byId("link-text"),
    frame: byId("frame"),
    stream: byId("stream"),
    feedChip: byId("feed-chip"),
    feedLabel: byId("feed-label"),
    feedAge: byId("feed-age"),
    simChip: byId("sim-chip"),
    veil: byId("veil"),
    veilTitle: byId("veil-title"),
    veilDetail: byId("veil-detail"),
    feedAnnouncer: byId("feed-announcer"),
    controlState: byId("control-state"),
    buttons: Array.from(document.querySelectorAll("button[data-action]")),
    controlNote: byId("control-note"),
    controlFeedback: byId("control-feedback"),
    rail: byId("rail"),
    railNote: byId("rail-note"),
    targetState: byId("target-state"),
    targetHeadline: byId("target-headline"),
    targetConfidence: byId("target-confidence"),
    targetBox: byId("target-box"),
    targetArea: byId("target-area"),
    targetMatched: byId("target-matched"),
    controllerState: byId("controller-state"),
    plotTitle: byId("plot-title"),
    plotDeadband: byId("plot-deadband"),
    plotDot: byId("plot-dot"),
    controllerPan: byId("controller-pan"),
    controllerTilt: byId("controller-tilt"),
    controllerError: byId("controller-error"),
    controllerDeadband: byId("controller-deadband"),
    controllerAge: byId("controller-age"),
    controllerTiming: byId("controller-timing"),
    controllerTrack: byId("controller-track"),
    controllerFault: byId("controller-fault"),
    performanceProcessing: byId("performance-processing"),
    performanceCapture: byId("performance-capture"),
    performanceAge: byId("performance-age"),
    performanceReceived: byId("performance-received"),
    performanceEncode: byId("performance-encode"),
    performanceEncoded: byId("performance-encoded"),
    performanceOverwritten: byId("performance-overwritten"),
    recordingState: byId("recording-state"),
    recordingSpan: byId("recording-span"),
    recordingWritten: byId("recording-written"),
    recordingDropped: byId("recording-dropped"),
    recordingBytes: byId("recording-bytes"),
    recordingError: byId("recording-error"),
  };

  const session = {
    state: null,
    receivedAtMs: null,
    pollFailures: 0,
    pollError: null,
    pollTimer: null,
    polling: false,
    pendingAction: null,
    feedbackTimer: null,
    streamHasFrame: false,
    streamFailed: false,
    streamError: null,
    streamAbort: null,
    streamOpenedAtMs: null,
    frameReceivedAtMs: null,
    frameUrl: null,
    streamRetryMs: STREAM_RETRY_INITIAL_MS,
    streamRetryTimer: null,
    streamRetryAtMs: null,
    wasConnected: true,
    lastAnnouncement: "",
  };

  const groupedInteger = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });

  const record = (value) =>
    value !== null && typeof value === "object" && !Array.isArray(value) ? value : null;
  const finite = (value) => (typeof value === "number" && Number.isFinite(value) ? value : null);
  const text = (value) => (typeof value === "string" && value !== "" ? value : null);

  function fixed(value, digits, unit) {
    const number = finite(value);
    return number === null ? null : `${number.toFixed(digits)}${unit}`;
  }

  function integer(value) {
    const number = finite(value);
    return number === null ? null : groupedInteger.format(number);
  }

  function duration(milliseconds) {
    const number = finite(milliseconds);
    if (number === null) {
      return null;
    }
    if (number < MILLISECONDS_PER_SECOND) {
      return `${Math.max(0, Math.round(number))} ms`;
    }
    const seconds = number / MILLISECONDS_PER_SECOND;
    if (seconds < SECONDS_PER_MINUTE) {
      return `${seconds.toFixed(1)} s`;
    }
    const minutes = Math.floor(seconds / SECONDS_PER_MINUTE);
    const remainder = Math.floor(seconds % SECONDS_PER_MINUTE);
    return `${minutes} min ${String(remainder).padStart(2, "0")} s`;
  }

  function size(bytes) {
    const number = finite(bytes);
    if (number === null) {
      return null;
    }
    if (number < BYTES_PER_KIBIBYTE) {
      return `${groupedInteger.format(number)} B`;
    }
    if (number < BYTES_PER_MEBIBYTE) {
      return `${(number / BYTES_PER_KIBIBYTE).toFixed(1)} KiB`;
    }
    return `${(number / BYTES_PER_MEBIBYTE).toFixed(1)} MiB`;
  }

  function signedPercent(fraction) {
    const number = finite(fraction);
    if (number === null) {
      return null;
    }
    const percent = number * PERCENT;
    return `${percent >= 0 ? "+" : ""}${percent.toFixed(1)}%`;
  }

  function sentence(fragment) {
    return fragment.charAt(0).toUpperCase() + fragment.slice(1);
  }

  function setReadout(element, content, tone) {
    if (tone) {
      element.dataset.tone = tone;
    } else {
      delete element.dataset.tone;
    }
    if (content === null || content === undefined) {
      element.dataset.empty = "true";
      const dash = document.createElement("span");
      dash.setAttribute("aria-hidden", "true");
      dash.textContent = EMPTY;
      const spoken = document.createElement("span");
      spoken.className = "visually-hidden";
      spoken.textContent = "not reported";
      element.replaceChildren(dash, spoken);
      return;
    }
    delete element.dataset.empty;
    element.textContent = content;
  }

  function setState(element, descriptor) {
    if (descriptor === null) {
      element.replaceChildren();
      element.dataset.tone = "idle";
      return;
    }
    element.dataset.tone = descriptor.tone;
    const lamp = document.createElement("span");
    lamp.className = "lamp";
    lamp.setAttribute("aria-hidden", "true");
    element.replaceChildren(lamp, document.createTextNode(descriptor.label));
  }

  function lookup(table, key) {
    const name = text(key);
    if (name === null) {
      return null;
    }
    return table[name] ?? { label: name.replaceAll("_", " "), tone: "idle" };
  }

  function errorMessage(error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return "the request timed out";
    }
    if (error instanceof Error && error.message) {
      return error.message;
    }
    return "the request failed";
  }

  function describeFailure(response) {
    const body = record(response.body);
    const reason = body === null ? null : text(body.error);
    if (response.status === 409 && reason === null) {
      return "controls are not available in this session (409)";
    }
    return reason === null ? `HTTP ${response.status}` : `${reason} (${response.status})`;
  }

  async function requestJson(path, options, timeoutMs) {
    const abort = new AbortController();
    const timer = window.setTimeout(() => abort.abort(), timeoutMs);
    try {
      const response = await fetch(path, {
        ...options,
        cache: "no-store",
        credentials: "same-origin",
        signal: abort.signal,
      });
      let body = null;
      const type = response.headers.get("Content-Type") ?? "";
      if (type.includes("application/json")) {
        body = await response.json().catch(() => null);
      }
      return { ok: response.ok, status: response.status, body };
    } finally {
      window.clearTimeout(timer);
    }
  }

  function stateAgeMs(now) {
    return session.receivedAtMs === null ? null : now - session.receivedAtMs;
  }

  function isConnected(now) {
    const age = stateAgeMs(now);
    return age !== null && (session.pollFailures === 0 || age <= STATE_LOST_AFTER_MS);
  }

  function frameAgeMs(now) {
    const preview = record(session.state?.preview);
    const reported = finite(preview?.frame_age_ms);
    if (reported === null) {
      return null;
    }
    return reported + (now - session.receivedAtMs);
  }

  function receivedAgeMs(now) {
    return session.frameReceivedAtMs === null ? null : now - session.frameReceivedAtMs;
  }

  function shownFrameAgeMs(now) {
    const received = receivedAgeMs(now);
    const newest = frameAgeMs(now);
    if (received === null) {
      return newest;
    }
    return Math.max(received, newest ?? 0);
  }

  function feedCondition(now) {
    const state = session.state;
    if (state === null) {
      if (session.pollFailures === 0) {
        return {
          kind: "connecting", tone: "idle", label: "Connecting",
          title: "Connecting to preview", detail: "Requesting state from this device.",
        };
      }
      return {
        kind: "unreachable", tone: "bad", label: "Offline",
        title: "Preview server unreachable",
        detail: `Last attempt failed: ${session.pollError}. Retrying.`,
      };
    }
    if (!isConnected(now)) {
      return {
        kind: "disconnected", tone: "bad", label: "Disconnected", dim: true,
        title: "Connection lost",
        detail: `The image and readings stopped updating ${duration(stateAgeMs(now))} ago. `
          + "Retrying.",
      };
    }
    const preview = record(state.preview) ?? {};
    const failure = text(preview.failure);
    if (failure !== null) {
      return {
        kind: "failed", tone: "bad", label: "Preview fault", dim: true,
        title: "Preview failed", detail: failure,
      };
    }
    const age = frameAgeMs(now);
    if (preview.running !== true) {
      return {
        kind: "stopped", tone: "warn", label: "Stopped", dim: true,
        title: "Preview not running",
        detail: age === null
          ? "No frame has been encoded."
          : `The last frame is ${duration(age)} old and is not live.`,
      };
    }
    if (age === null) {
      return {
        kind: "waiting", tone: "idle", label: "No frame",
        title: "Waiting for first frame",
        detail: "Controls and readings work before video arrives.",
      };
    }
    if (session.streamFailed) {
      const retryMs = session.streamRetryAtMs === null
        ? null : Math.max(0, session.streamRetryAtMs - now);
      return {
        kind: "stream-error", tone: "warn", label: "Reconnecting", dim: true,
        title: "Video stream interrupted",
        detail: `${sentence(session.streamError ?? "the stream closed")}. `
          + (retryMs === null ? "Reconnecting" : `Reconnecting in ${duration(retryMs)}`)
          + ". Readings are still live.",
      };
    }
    if (age > STALE_FRAME_AGE_MS) {
      return {
        kind: "stale", tone: "warn", label: "Stale", dim: true,
        title: `No new frame for ${duration(age)}`,
        detail: "This image is not live. Processing may have stalled or slowed.",
      };
    }
    const received = receivedAgeMs(now);
    if (received !== null && received > STALE_FRAME_AGE_MS) {
      return {
        kind: "stale", tone: "warn", label: "Delayed", dim: true,
        title: `No frame received for ${duration(received)}`,
        detail: "The device reports newer frames than this browser has received.",
      };
    }
    if (!session.streamHasFrame) {
      return {
        kind: "opening", tone: "idle", label: "Opening stream",
        title: "Opening video stream", detail: "Frames are being encoded.",
      };
    }
    return { kind: "live", tone: "ok", label: "Live" };
  }

  function renderFeed(now, condition) {
    ui.frame.dataset.tone = condition.kind === "live" ? "idle" : condition.tone;
    ui.frame.dataset.dim = condition.dim ? "true" : "false";
    ui.frame.dataset.image = session.streamHasFrame ? "frame" : "none";
    ui.feedChip.dataset.tone = condition.tone;
    ui.feedLabel.textContent = condition.label;
    const age = shownFrameAgeMs(now);
    ui.feedAge.textContent = age === null ? "" : duration(age);
    ui.veil.hidden = condition.kind === "live";
    ui.veil.dataset.tone = condition.tone;
    ui.veilTitle.textContent = condition.title ?? "";
    ui.veilDetail.textContent = condition.detail ?? "";
    const announcement = condition.title ?? "Video live";
    if (announcement !== session.lastAnnouncement && condition.kind !== "connecting") {
      session.lastAnnouncement = announcement;
      ui.feedAnnouncer.textContent = condition.kind === "stale"
        ? "Video stale, image is not live"
        : announcement;
    }
  }

  function renderHeader(now, controller) {
    const connected = isConnected(now);
    if (session.state === null && session.pollFailures === 0) {
      ui.linkBadge.dataset.tone = "idle";
      ui.linkText.textContent = "Connecting";
    } else if (!connected) {
      ui.linkBadge.dataset.tone = "bad";
      ui.linkText.textContent = "Disconnected";
    } else if (session.pollFailures > 0) {
      ui.linkBadge.dataset.tone = "warn";
      ui.linkText.textContent = "Retrying";
    } else {
      ui.linkBadge.dataset.tone = "ok";
      ui.linkText.textContent = "Connected";
    }
    const mode = text(controller?.mode);
    ui.simChip.hidden = mode !== "simulate";
    if (mode === "simulate") {
      ui.mountBadge.dataset.tone = "info";
      ui.mountText.textContent = "Simulated mount, no servo output";
    } else if (mode === "hardware") {
      ui.mountBadge.dataset.tone = "warn";
      ui.mountText.textContent = "Hardware mount";
    } else if (session.state !== null) {
      ui.mountBadge.dataset.tone = "idle";
      ui.mountText.textContent = "Following off";
    } else {
      ui.mountBadge.dataset.tone = "idle";
      ui.mountText.textContent = "Mount unknown";
    }
    ui.timingBadge.hidden = text(controller?.timing_source) !== "replay";
  }

  function controlAvailability(now, controller) {
    if (session.state === null) {
      return session.pollFailures === 0
        ? { enabled: false, note: "Waiting for controller state." }
        : { enabled: false, note: "Controls disabled: cannot reach the preview server." };
    }
    if (!isConnected(now)) {
      return { enabled: false, note: "Controls disabled: connection to the device lost." };
    }
    if (session.state.controls_enabled !== true) {
      return {
        enabled: false,
        note: "Controls unavailable: no pan-tilt controller is attached to this session.",
      };
    }
    const mode = text(controller?.mode);
    const route = "Pause holds the current position. Home returns the mount and keeps "
      + "following paused until you resume.";
    if (mode === "simulate") {
      return { enabled: true, note: `${route} This mount is simulated: no servo moves.` };
    }
    if (mode === "hardware") {
      return { enabled: true, note: `${route} This mount drives physical servos.` };
    }
    return { enabled: true, note: route };
  }

  function renderControls(now, controller) {
    const availability = controlAvailability(now, controller);
    const controllerState = text(controller?.state);
    for (const button of ui.buttons) {
      const action = button.dataset.action;
      const redundant = action === "pause" && controllerState === "paused";
      button.disabled = !availability.enabled || session.pendingAction !== null || redundant;
      button.setAttribute("aria-busy", session.pendingAction === action ? "true" : "false");
    }
    ui.controlNote.textContent = availability.note;
    setState(ui.controlState, controller === null ? null : lookup(CONTROLLER_STATES,
      controller.state));
  }

  function renderRailNote(now) {
    const age = stateAgeMs(now);
    ui.rail.dataset.frozen = session.state !== null && !isConnected(now) ? "true" : "false";
    if (age === null) {
      ui.railNote.dataset.tone = session.pollFailures === 0 ? "idle" : "bad";
      ui.railNote.textContent = session.pollFailures === 0
        ? "No readings yet"
        : `No readings: ${session.pollError}`;
      return;
    }
    if (!isConnected(now)) {
      ui.railNote.dataset.tone = "bad";
      ui.railNote.textContent = `Readings frozen ${duration(age)} ago: ${session.pollError}`;
      return;
    }
    ui.railNote.dataset.tone = session.pollFailures > 0 ? "warn" : "idle";
    ui.railNote.textContent = `Readings updated ${duration(age)} ago`;
  }

  function renderTarget(target) {
    if (target === null) {
      setState(ui.targetState, null);
      ui.targetHeadline.textContent = "No target data";
      for (const element of [ui.targetConfidence, ui.targetBox, ui.targetArea,
        ui.targetMatched]) {
        setReadout(element, null);
      }
      return;
    }
    setState(ui.targetState, lookup(TARGET_STATES, target.state));
    const track = record(target.track);
    const detection = record(track?.detection);
    if (track === null) {
      ui.targetHeadline.textContent = "No pigeon selected";
    } else {
      const identity = integer(track.track_id);
      ui.targetHeadline.replaceChildren(
        document.createTextNode("Nearest-looking pigeon "),
        Object.assign(document.createElement("span"), {
          className: "mono",
          textContent: identity === null ? "" : `#${identity}`,
        }),
      );
    }
    const confidence = finite(detection?.confidence);
    setReadout(ui.targetConfidence,
      confidence === null ? null : `${(confidence * PERCENT).toFixed(0)}%`);
    const box = Array.isArray(detection?.bbox) ? detection.bbox.map(finite) : [];
    const boxValid = box.length === 4 && box.every((value) => value !== null);
    setReadout(ui.targetBox, boxValid
      ? `${Math.round(box[2] - box[0])} ${TIMES} ${Math.round(box[3] - box[1])} px`
      : null);
    const area = integer(track?.smoothed_area_px2);
    setReadout(ui.targetArea, area === null ? null : `${area} px${SQUARED}`);
    if (track === null || typeof track.matched !== "boolean") {
      setReadout(ui.targetMatched, null);
    } else {
      setReadout(ui.targetMatched, track.matched ? "Detected" : "Not detected",
        track.matched ? null : "warn");
    }
  }

  function renderPlot(controller) {
    const deadband = finite(controller?.deadband);
    if (deadband === null) {
      ui.plotDeadband.setAttribute("width", "0");
      ui.plotDeadband.setAttribute("height", "0");
    } else {
      const bandWidth = deadband * 2 * PLOT_WIDTH;
      const bandHeight = deadband * 2 * PLOT_HEIGHT;
      ui.plotDeadband.setAttribute("x", String(PLOT_WIDTH / 2 - bandWidth / 2));
      ui.plotDeadband.setAttribute("y", String(PLOT_HEIGHT / 2 - bandHeight / 2));
      ui.plotDeadband.setAttribute("width", String(bandWidth));
      ui.plotDeadband.setAttribute("height", String(bandHeight));
    }
    const errorX = finite(controller?.error_x_fraction);
    const errorY = finite(controller?.error_y_fraction);
    if (errorX === null || errorY === null) {
      ui.plotDot.setAttribute("visibility", "hidden");
      ui.plotTitle.textContent = "Target offset from image centre: not reported";
      return;
    }
    const clamp = (value) => Math.min(IMAGE_CENTRE_FRACTION,
      Math.max(-IMAGE_CENTRE_FRACTION, value));
    ui.plotDot.setAttribute("cx", String((IMAGE_CENTRE_FRACTION + clamp(errorX)) * PLOT_WIDTH));
    ui.plotDot.setAttribute("cy", String((IMAGE_CENTRE_FRACTION + clamp(errorY)) * PLOT_HEIGHT));
    ui.plotDot.setAttribute("visibility", "visible");
    ui.plotDot.dataset.held = text(controller?.state) === "tracking" ? "false" : "true";
    ui.plotTitle.textContent = `Target offset from image centre: x ${signedPercent(errorX)}, `
      + `y ${signedPercent(errorY)} of the frame`;
  }

  function renderController(controller) {
    renderPlot(controller);
    if (controller === null) {
      setState(ui.controllerState, { label: "Not running", tone: "idle" });
      for (const element of [ui.controllerPan, ui.controllerTilt, ui.controllerError,
        ui.controllerDeadband, ui.controllerAge, ui.controllerTiming, ui.controllerTrack]) {
        setReadout(element, null);
      }
      ui.controllerFault.hidden = true;
      return;
    }
    setState(ui.controllerState, lookup(CONTROLLER_STATES, controller.state));
    const simulated = text(controller.mode) === "simulate" ? " (sim)" : "";
    const axis = (angle, limited) => {
      const value = fixed(angle, 1, DEGREE);
      if (value === null) {
        return null;
      }
      return `${value}${limited === true ? " at limit" : ""}${simulated}`;
    };
    setReadout(ui.controllerPan, axis(controller.commanded_pan_deg, controller.pan_limited),
      controller.pan_limited === true ? "warn" : null);
    setReadout(ui.controllerTilt, axis(controller.commanded_tilt_deg, controller.tilt_limited),
      controller.tilt_limited === true ? "warn" : null);
    const errorX = signedPercent(controller.error_x_fraction);
    const errorY = signedPercent(controller.error_y_fraction);
    setReadout(ui.controllerError,
      errorX === null || errorY === null ? null : `${errorX} / ${errorY}`);
    const deadband = finite(controller.deadband);
    setReadout(ui.controllerDeadband,
      deadband === null ? null : `${PLUS_MINUS}${(deadband * PERCENT).toFixed(1)}%`);
    setReadout(ui.controllerAge, duration(controller.observation_age_ms));
    const timing = text(controller.timing_source);
    setReadout(ui.controllerTiming, timing === null ? null : TIMING_SOURCES[timing] ?? timing,
      timing === "unavailable" ? "warn" : null);
    const track = integer(controller.target_track_id);
    setReadout(ui.controllerTrack, track === null ? null : `#${track}`);
    const faults = [text(controller.fault), text(controller.close_fault)]
      .filter((fault) => fault !== null);
    ui.controllerFault.hidden = faults.length === 0;
    ui.controllerFault.textContent = faults.join(" ");
  }

  function renderPerformance(now) {
    const metrics = record(session.state?.performance) ?? {};
    setReadout(ui.performanceProcessing, fixed(metrics.processing_fps, 1, " fps"));
    setReadout(ui.performanceCapture, fixed(metrics.capture_fps, 1, " fps"));
    const age = frameAgeMs(now);
    setReadout(ui.performanceAge, duration(age),
      age !== null && age > STALE_FRAME_AGE_MS ? "warn" : null);
    const received = receivedAgeMs(now);
    setReadout(ui.performanceReceived, received === null ? null : `${duration(received)} ago`,
      received !== null && received > STALE_FRAME_AGE_MS ? "warn" : null);
    setReadout(ui.performanceEncode, fixed(metrics.last_encode_ms, 1, " ms"));
    setReadout(ui.performanceEncoded, integer(metrics.encoded_frames));
    setReadout(ui.performanceOverwritten, integer(metrics.overwritten_frames));
  }

  function renderRecording(recording) {
    if (recording === null) {
      setState(ui.recordingState, session.state === null
        ? null : { label: "Off", tone: "idle" });
      for (const element of [ui.recordingSpan, ui.recordingWritten, ui.recordingDropped,
        ui.recordingBytes]) {
        setReadout(element, null);
      }
      ui.recordingError.hidden = true;
      return;
    }
    setState(ui.recordingState, lookup(RECORDING_STATES, recording.state));
    const first = finite(recording.first_monotonic_s);
    const last = finite(recording.last_monotonic_s);
    setReadout(ui.recordingSpan, first === null || last === null
      ? null : duration((last - first) * MILLISECONDS_PER_SECOND));
    setReadout(ui.recordingWritten, integer(recording.frames_written));
    const dropped = finite(recording.frames_dropped);
    setReadout(ui.recordingDropped, integer(dropped),
      dropped !== null && dropped > 0 ? "warn" : null);
    setReadout(ui.recordingBytes, size(recording.bytes_written));
    const alert = recordingAlert(recording);
    ui.recordingError.hidden = alert === null;
    ui.recordingError.dataset.tone = alert?.tone ?? "bad";
    ui.recordingError.textContent = alert?.message ?? "";
  }

  function recordingAlert(recording) {
    const written = finite(recording.frames_written) ?? 0;
    const kept = written > 0
      ? ` The ${integer(written)} frames already written stay on disk.`
      : "";
    const error = text(recording.error);
    if (recording.state === "error" || error !== null) {
      return {
        tone: "bad",
        message: `Recording stopped: ${error ?? "the recorder reported an error"}.${kept}`,
      };
    }
    if (recording.state === "limited") {
      const reason = text(recording.limit_reason) ?? "a configured recording limit was reached";
      return {
        tone: "warn",
        message: `Recording stopped: ${reason}. The run continues.${kept}`,
      };
    }
    return null;
  }

  function render() {
    const now = performance.now();
    const controller = record(session.state?.controller);
    renderHeader(now, controller);
    renderFeed(now, feedCondition(now));
    renderControls(now, controller);
    renderRailNote(now);
    renderTarget(record(session.state?.target));
    renderController(controller);
    renderPerformance(now);
    renderRecording(record(session.state?.recording));
  }

  function nextPollDelayMs() {
    const base = document.hidden ? POLL_HIDDEN_INTERVAL_MS : POLL_INTERVAL_MS;
    if (session.pollFailures === 0) {
      return base;
    }
    return Math.min(POLL_BACKOFF_MAX_MS, base * 2 ** Math.min(session.pollFailures, 4));
  }

  function schedulePoll(delayMs) {
    window.clearTimeout(session.pollTimer);
    session.pollTimer = window.setTimeout(poll, delayMs);
  }

  async function poll() {
    if (session.polling) {
      return;
    }
    session.polling = true;
    window.clearTimeout(session.pollTimer);
    try {
      const response = await requestJson(STATE_PATH, { method: "GET" }, STATE_TIMEOUT_MS);
      if (!response.ok) {
        throw new Error(describeFailure(response));
      }
      const state = record(response.body);
      if (state === null) {
        throw new Error("the state response was not a JSON object");
      }
      session.state = state;
      session.receivedAtMs = performance.now();
      session.pollFailures = 0;
      session.pollError = null;
    } catch (error) {
      session.pollFailures += 1;
      session.pollError = errorMessage(error);
    } finally {
      session.polling = false;
    }
    const connected = isConnected(performance.now());
    if (connected && !session.wasConnected && session.streamFailed && !document.hidden) {
      connectStream();
    }
    session.wasConnected = connected;
    render();
    schedulePoll(nextPollDelayMs());
  }

  function showFeedback(tone, message) {
    window.clearTimeout(session.feedbackTimer);
    ui.controlFeedback.dataset.tone = tone;
    ui.controlFeedback.textContent = message;
    if (tone === "ok") {
      session.feedbackTimer = window.setTimeout(() => {
        ui.controlFeedback.textContent = "";
      }, FEEDBACK_VISIBLE_MS);
    }
  }

  async function sendControl(action) {
    const descriptor = ACTIONS[action];
    if (descriptor === undefined || session.pendingAction !== null) {
      return;
    }
    session.pendingAction = action;
    showFeedback("info", `Sending ${descriptor.label.toLowerCase()}${ELLIPSIS}`);
    render();
    try {
      const response = await requestJson(CONTROL_PATH, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Columba-Control": "1" },
        body: JSON.stringify({ action }),
      }, CONTROL_TIMEOUT_MS);
      if (response.ok && record(response.body)?.ok === true) {
        showFeedback("ok", descriptor.accepted);
      } else {
        showFeedback("bad", `${descriptor.label} rejected: ${describeFailure(response)}.`);
      }
    } catch (error) {
      showFeedback("bad", `${descriptor.label} not confirmed: ${errorMessage(error)}. `
        + "Check the controller state before retrying.");
    } finally {
      session.pendingAction = null;
      render();
      poll();
    }
  }

  function find(haystack, needle, from) {
    let index = haystack.indexOf(needle[0], from);
    while (index !== -1 && index + needle.length <= haystack.length) {
      if (needle.every((byte, offset) => haystack[index + offset] === byte)) {
        return index;
      }
      index = haystack.indexOf(needle[0], index + 1);
    }
    return -1;
  }

  function boundaryFrom(contentType) {
    const match = /boundary="?([^";]+)"?/i.exec(contentType ?? "");
    if (match === null) {
      throw new Error("the stream has no multipart boundary");
    }
    return match[1].trim();
  }

  function createPartReader(boundary, onPart) {
    const encoder = new TextEncoder();
    const headerDecoder = new TextDecoder("windows-1252");
    const delimiter = encoder.encode(`--${boundary}`);
    const headerEnd = encoder.encode("\r\n\r\n");
    let buffer = new Uint8Array(0);
    let phase = "seek";
    let bodyLength = null;

    function seekDelimiter() {
      const at = find(buffer, delimiter, 0);
      if (at < 0) {
        buffer = buffer.subarray(Math.max(0, buffer.length - delimiter.length + 1));
        return false;
      }
      buffer = buffer.subarray(at + delimiter.length);
      phase = "headers";
      return true;
    }

    function readHeaders() {
      const at = find(buffer, headerEnd, 0);
      if (at < 0) {
        return false;
      }
      const headers = headerDecoder.decode(buffer.subarray(0, at));
      const length = /content-length:\s*(\d+)/i.exec(headers);
      bodyLength = length === null ? null : Number(length[1]);
      if (bodyLength !== null && bodyLength > STREAM_BUFFER_MAX_BYTES) {
        throw new Error("a stream frame exceeded the size limit");
      }
      buffer = buffer.subarray(at + headerEnd.length);
      phase = "body";
      return true;
    }

    function readBody() {
      if (bodyLength !== null) {
        if (buffer.length < bodyLength) {
          return false;
        }
        onPart(buffer.slice(0, bodyLength));
        buffer = buffer.subarray(bodyLength);
        phase = "seek";
        return true;
      }
      // without a Content-Length the next delimiter is the only end marker.
      const at = find(buffer, delimiter, 0);
      if (at < 0) {
        return false;
      }
      onPart(buffer.slice(0, at));
      buffer = buffer.subarray(at);
      phase = "seek";
      return true;
    }

    const steps = { seek: seekDelimiter, headers: readHeaders, body: readBody };

    return (chunk) => {
      const merged = new Uint8Array(buffer.length + chunk.length);
      merged.set(buffer);
      merged.set(chunk, buffer.length);
      buffer = merged;
      if (buffer.length > STREAM_BUFFER_MAX_BYTES) {
        throw new Error("the stream sent an oversized or malformed frame");
      }
      let progressing = true;
      while (progressing) {
        progressing = steps[phase]();
      }
    };
  }

  function showFrame(jpeg) {
    const url = URL.createObjectURL(new Blob([jpeg], { type: "image/jpeg" }));
    const previous = session.frameUrl;
    session.frameUrl = url;
    session.frameReceivedAtMs = performance.now();
    session.streamRetryMs = STREAM_RETRY_INITIAL_MS;
    ui.stream.src = url;
    if (previous !== null) {
      URL.revokeObjectURL(previous);
    }
  }

  async function readStream(abort) {
    const response = await fetch(STREAM_PATH, {
      cache: "no-store",
      credentials: "same-origin",
      signal: abort.signal,
    });
    if (!response.ok || response.body === null) {
      throw new Error(`the stream returned HTTP ${response.status}`);
    }
    const push = createPartReader(boundaryFrom(response.headers.get("Content-Type")), showFrame);
    const reader = response.body.getReader();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) {
        throw new Error("the stream closed");
      }
      push(value);
    }
  }

  function stopStream() {
    window.clearTimeout(session.streamRetryTimer);
    session.streamRetryTimer = null;
    session.streamRetryAtMs = null;
    const abort = session.streamAbort;
    session.streamAbort = null;
    abort?.abort();
  }

  async function connectStream() {
    stopStream();
    const abort = new AbortController();
    session.streamAbort = abort;
    session.streamFailed = false;
    session.streamError = null;
    session.streamOpenedAtMs = performance.now();
    try {
      await readStream(abort);
    } catch (error) {
      if (session.streamAbort !== abort) {
        return;
      }
      session.streamAbort = null;
      session.streamFailed = true;
      session.streamError = errorMessage(error);
      scheduleStreamRetry();
      render();
    }
  }

  function scheduleStreamRetry() {
    if (session.streamRetryTimer !== null || document.hidden) {
      return;
    }
    const delayMs = session.streamRetryMs;
    session.streamRetryMs = Math.min(STREAM_RETRY_MAX_MS, session.streamRetryMs * 2);
    session.streamRetryAtMs = performance.now() + delayMs;
    session.streamRetryTimer = window.setTimeout(connectStream, delayMs);
  }

  function reconnectSilentStream(now) {
    if (session.streamAbort === null || session.streamOpenedAtMs === null) {
      return;
    }
    const newest = frameAgeMs(now);
    if (newest === null || newest > STALE_FRAME_AGE_MS) {
      return;
    }
    const lastActivityMs = Math.max(session.streamOpenedAtMs, session.frameReceivedAtMs ?? 0);
    if (now - lastActivityMs > STREAM_SILENCE_RECONNECT_MS) {
      connectStream();
    }
  }

  ui.stream.addEventListener("load", () => {
    session.streamHasFrame = true;
    ui.frame.style.aspectRatio = `${ui.stream.naturalWidth} / ${ui.stream.naturalHeight}`;
  });

  for (const button of ui.buttons) {
    button.addEventListener("click", () => sendControl(button.dataset.action));
  }

  // hidden tabs drop the stream so the device stops sending frames nobody can see.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopStream();
      return;
    }
    connectStream();
    poll();
  });

  window.addEventListener("pagehide", stopStream);

  window.setInterval(() => {
    reconnectSilentStream(performance.now());
    render();
  }, RENDER_INTERVAL_MS);

  render();
  poll();
  // a stream request started during parsing would keep the document loading indefinitely.
  if (document.readyState === "complete") {
    connectStream();
  } else {
    window.addEventListener("load", connectStream, { once: true });
  }
})();
</script>
</body>
</html>
"""
