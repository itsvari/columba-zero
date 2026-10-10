"""Copy and verify the original model artifacts using the tracked manifest."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "deploy/model_manifest.json").read_text())
    source = root / manifest["source"]
    target = root / manifest["runtime_directory"]
    for record in manifest["files"]:
        original = source / record["file"]
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise SystemExit(f"source checksum mismatch: {original}")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "best.pt", target / "best.pt")
    shutil.copytree(source / "best_ncnn_model", target / "best_ncnn_model", dirs_exist_ok=True)
    for record in manifest["files"]:
        copied = target / record["file"]
        if hashlib.sha256(copied.read_bytes()).hexdigest() != record["sha256"]:
            raise SystemExit(f"copy checksum mismatch: {copied}")
    print(f"Verified models: {target}")


if __name__ == "__main__":
    main()
