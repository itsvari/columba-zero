#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLO detection model to NCNN")
    parser.add_argument("model")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    exported = YOLO(args.model).export(format="ncnn", imgsz=args.imgsz, half=args.half)
    print(exported)


if __name__ == "__main__":
    main()
