#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit an image-pixel to target-plane homography from surveyed points"
    )
    parser.add_argument("points", help="JSON with image_points and plane_points arrays")
    parser.add_argument("--output", default="calibration.local.yaml")
    args = parser.parse_args()

    import cv2

    raw = json.loads(Path(args.points).read_text(encoding="utf-8"))
    image_points = np.asarray(raw["image_points"], dtype=np.float64)
    plane_points = np.asarray(raw["plane_points"], dtype=np.float64)
    if image_points.shape != plane_points.shape or image_points.ndim != 2:
        raise SystemExit("image_points and plane_points must have matching Nx2 shapes")
    if image_points.shape[0] < 4 or image_points.shape[1] != 2:
        raise SystemExit("at least four 2D point pairs are required")
    homography, mask = cv2.findHomography(image_points, plane_points, method=0)
    if homography is None:
        raise SystemExit("homography fitting failed")
    projected = cv2.perspectiveTransform(image_points[None, :, :], homography)[0]
    errors = np.linalg.norm(projected - plane_points, axis=1)
    output = {
        "geometry": {
            "calibrated": False,
            "homography": homography.tolist(),
            "fit_error_mean_m": float(errors.mean()),
            "fit_error_max_m": float(errors.max()),
            "verification_required": True,
        }
    }
    Path(args.output).write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(output, sort_keys=False))
    print("Calibration remains disabled until physical pointing-error verification is complete.")


if __name__ == "__main__":
    main()
