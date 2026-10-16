"""Copy and verify selected, fallback or provisional C8 artefacts from the manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    file: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ModelManifest:
    source: str
    runtime_directory: str
    files: tuple[ModelFile, ...]

    @classmethod
    def load(cls, path: Path, model_id: str | None = None) -> ModelManifest:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if model_id is not None:
                matches = [
                    entry
                    for entry in [
                        manifest,
                        *manifest.get("fallback_models", []),
                        *manifest.get("provisional_models", []),
                    ]
                    if entry.get("model_id") == model_id
                ]
                if len(matches) != 1:
                    raise ValueError(f"expected one model entry for {model_id}")
                manifest = matches[0]
            return cls(
                source=manifest["source"],
                runtime_directory=manifest["runtime_directory"],
                files=tuple(ModelFile(**record) for record in manifest["files"]),
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SystemExit(f"cannot load model manifest {path}: {error}") from error


def verify_files(directory: Path, records: tuple[ModelFile, ...], stage: str) -> None:
    for record in records:
        path = directory / record.file
        try:
            content = path.read_bytes()
        except OSError as error:
            raise SystemExit(f"cannot read {stage} model file {path}: {error}") from error
        if len(content) != record.bytes or hashlib.sha256(content).hexdigest() != record.sha256:
            raise SystemExit(f"{stage} checksum mismatch: {path}")


def prepare_models(repository: Path, model_id: str | None = None) -> Path:
    manifest = ModelManifest.load(repository / "deploy/model_manifest.json", model_id)
    source = repository / manifest.source
    target = repository / manifest.runtime_directory
    verify_files(source, manifest.files, "source")
    target.mkdir(parents=True, exist_ok=True)
    for record in manifest.files:
        destination = target / record.file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / record.file, destination)
    verify_files(target, manifest.files, "copy")
    return target


def main(model_id: str | None = None) -> None:
    print(f"Verified models: {prepare_models(Path(__file__).resolve().parents[1], model_id)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", help="verify a named C8 entry; default verifies the selected model"
    )
    main(parser.parse_args().model_id)
