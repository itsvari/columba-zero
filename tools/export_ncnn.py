#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLO detection model to NCNN")
    parser.add_argument("model")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    arguments = parser.parse_args()

    from ultralytics import YOLO

    exported = YOLO(arguments.model, task="detect").export(
        format="ncnn", imgsz=arguments.imgsz, half=arguments.half
    )
    print(exported)


if __name__ == "__main__":
    main()
