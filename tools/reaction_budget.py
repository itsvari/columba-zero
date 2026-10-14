"""Simulate sensor delivery, latest-frame processing and k-of-n confirmation.

Origin is the capture timestamp of the first visible frame, as PLAN section 4.
Sensor delay is uniform in 1–2 camera periods by default. Processing samples
are bootstrapped independently. Optional Markov hits preserve marginal recall;
correlation decays over skipped camera frames. No tracking/occlusion model.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.json_data import JsonValue
from tools.bench_ncnn import calibration_factor, number, read_report, summary

MILLISECONDS_PER_SECOND = 1000.0
DEFAULT_TRIALS = 20000
DEFAULT_SEED = 42
BURN_IN_FRAMES = 128
MAXIMUM_PROCESSED_FRAMES = 10000
NORMAL_P95_Z = 1.6448536269514722


@dataclass(frozen=True)
class ReactionSettings:
    camera_fps: float = 30.0
    recall: float = 0.8
    k: int = 2
    n: int = 3
    correlation: float = 0.0
    sensor_min_periods: float = 1.0
    sensor_max_periods: float = 2.0
    trials: int = DEFAULT_TRIALS
    seed: int = DEFAULT_SEED
    frame_policy: str = "latest"

    def validate(self) -> None:
        if not math.isfinite(self.camera_fps) or self.camera_fps <= 0:
            raise ValueError("camera FPS must be finite and positive")
        if not 0 < self.recall <= 1 or not 0 <= self.correlation < 1:
            raise ValueError("recall must be in (0,1]; correlation in [0,1)")
        if not 1 <= self.k <= self.n or self.trials < 1:
            raise ValueError("require 1 <= k <= n and positive trials")
        if not 0 <= self.sensor_min_periods <= self.sensor_max_periods < math.inf:
            raise ValueError("sensor range must be finite, nonnegative and ordered")
        if self.frame_policy not in ("latest", "fifo"):
            raise ValueError("frame policy must be latest or fifo")


def processing_samples(
    report: dict[str, JsonValue], factor: float = 1.0, seed: int = DEFAULT_SEED
) -> tuple[list[float], str]:
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be finite and positive")
    samples = report.get("samples_ms")
    if isinstance(samples, list) and samples:
        values = [number(value, "sample") * factor for value in samples]
        if any(value <= 0 for value in values):
            raise ValueError("processing samples must be positive")
        return values, "empirical paired end-to-end bootstrap"
    timings = report.get("end_to_end_ms")
    if not isinstance(timings, dict):
        raise ValueError("C7 needs samples_ms or end_to_end_ms p50/p95")
    median_ms = number(timings["p50"], "p50")
    p95_ms = number(timings["p95"], "p95")
    if not 0 < median_ms <= p95_ms:
        raise ValueError("require 0 < p50 <= p95")
    sigma = math.log(p95_ms / median_ms) / NORMAL_P95_Z
    draws = np.random.default_rng(seed).lognormal(math.log(median_ms), sigma, DEFAULT_TRIALS)
    return [float(value) * factor for value in draws], "estimated lognormal fitted to C7 p50/p95"


DEFAULT_SETTINGS = ReactionSettings()


def simulate_reaction(
    values_ms: list[float], settings: ReactionSettings = DEFAULT_SETTINGS
) -> dict[str, JsonValue]:
    settings.validate()
    if not values_ms or any(not math.isfinite(value) or value <= 0 for value in values_ms):
        raise ValueError("processing times must be finite and positive")
    processing_ms = np.asarray(values_ms, dtype=np.float64)
    random = np.random.default_rng(settings.seed)
    camera_period_ms = MILLISECONDS_PER_SECOND / settings.camera_fps
    sensor_ms = random.uniform(
        settings.sensor_min_periods * camera_period_ms,
        settings.sensor_max_periods * camera_period_ms,
        settings.trials,
    )
    # pre-roll places an appearance inside the ongoing loop, rather than at a cold start.
    initial_index = -BURN_IN_FRAMES if settings.frame_policy == "latest" else 0
    captured_index = np.full(settings.trials, initial_index, dtype=np.int64)
    started_ms = sensor_ms + captured_index * camera_period_ms
    if settings.frame_policy == "latest":
        started_ms += random.uniform(0, camera_period_ms, settings.trials)
    previous_hit = random.random(settings.trials) < settings.recall
    previous_index = captured_index.copy()
    history = np.zeros((settings.trials, settings.n), dtype=np.bool_)
    visible_frames = np.zeros(settings.trials, dtype=np.int64)
    result_ms = np.full(settings.trials, np.inf)
    first_wait_ms = np.zeros(settings.trials)
    dropped_frames = np.zeros(settings.trials, dtype=np.int64)
    for _ in range(MAXIMUM_PROCESSED_FRAMES + BURN_IN_FRAMES):
        active = np.isinf(result_ms)
        if not np.any(active):
            break
        finished_ms = started_ms + random.choice(processing_ms, settings.trials)
        visible = active & (captured_index >= 0)
        first = visible & (visible_frames == 0)
        first_wait_ms[first] = started_ms[first] - sensor_ms[first]
        gap = np.maximum(1, captured_index - previous_index)
        persistence = np.power(settings.correlation, gap)
        hit_probability = settings.recall + (previous_hit - settings.recall) * persistence
        hit = random.random(settings.trials) < hit_probability
        history[visible, :-1] = history[visible, 1:]
        history[visible, -1] = hit[visible]
        visible_frames[visible] += 1
        confirmed = visible & (history.sum(axis=1) >= settings.k)
        result_ms[confirmed] = finished_ms[confirmed]
        previous_hit[visible] = hit[visible]
        previous_index[visible] = captured_index[visible]
        next_index = captured_index + 1
        if settings.frame_policy == "latest":
            newest = np.floor((finished_ms - sensor_ms) / camera_period_ms).astype(np.int64)
            next_index = np.maximum(next_index, newest)
            dropped_frames[visible] += (next_index - captured_index - 1)[visible]
        started_ms = np.maximum(finished_ms, sensor_ms + next_index * camera_period_ms)
        captured_index = next_index
    finite: NDArray[np.bool_] = np.isfinite(result_ms)
    if not np.all(finite):
        raise ValueError(
            "confirmation did not finish within 10000 processed frames; increase recall"
        )
    return {
        "label": "simulated",
        "reaction_ms": summary([float(value) for value in result_ms]),
        "sensor_ms": summary([float(value) for value in sensor_ms]),
        "first_wait_ms": summary([float(value) for value in first_wait_ms]),
        "mean_processed_visible_frames": float(visible_frames.mean()),
        "mean_dropped_visible_frames": float(dropped_frames.mean()),
        "camera_fps": settings.camera_fps,
        "recall": settings.recall,
        "k": settings.k,
        "n": settings.n,
        "correlation": settings.correlation,
        "trials": settings.trials,
        "seed": settings.seed,
        "frame_policy": settings.frame_policy,
        "sensor_periods": [settings.sensor_min_periods, settings.sensor_max_periods],
        "origin": "capture timestamp of first visible frame; no random pre-capture wait",
        "hit_model": "stationary Bernoulli Markov; lag correlation rho^camera-frame-gap",
        "processing_model": "independent empirical bootstrap; latest-frame pre-roll 128 frames",
    }


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark", type=Path, help="C7 JSON; samples preferred, else lognormal p50/p95"
    )
    parser.add_argument("--calibration", type=Path, help="scale matched host to estimated Pi")
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--recall", type=float, default=0.8)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--correlation", type=float, default=0.0)
    parser.add_argument("--sensor-periods", type=float, nargs=2, default=(1.0, 2.0))
    parser.add_argument("--frame-policy", choices=("latest", "fifo"), default="latest")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        report = read_report(arguments.benchmark)
        factor = (
            calibration_factor(read_report(arguments.calibration), str(report["host_kind"]))
            if arguments.calibration
            else 1.0
        )
        samples_ms, distribution = processing_samples(report, factor, arguments.seed)
        settings = ReactionSettings(
            camera_fps=arguments.camera_fps,
            recall=arguments.recall,
            k=arguments.k,
            n=arguments.n,
            correlation=arguments.correlation,
            sensor_min_periods=arguments.sensor_periods[0],
            sensor_max_periods=arguments.sensor_periods[1],
            trials=arguments.trials,
            seed=arguments.seed,
            frame_policy=arguments.frame_policy,
        )
        result = simulate_reaction(samples_ms, settings)
        result.update(
            {
                "benchmark": str(arguments.benchmark),
                "factor": factor,
                "distribution": distribution,
                "processing_host": "estimated Pi5" if arguments.calibration else report["host"],
            }
        )
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if arguments.output:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(rendered)
        print(rendered, end="")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
