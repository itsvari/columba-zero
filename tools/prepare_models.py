"""Copy and verify the original model artefacts using the tracked manifest."""

from __future__ import annotations

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
    def load(cls, path: Path) -> ModelManifest:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
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


def prepare_models(repository: Path) -> Path:
    manifest = ModelManifest.load(repository / "deploy/model_manifest.json")
    source = repository / manifest.source
    target = repository / manifest.runtime_directory
    verify_files(source, manifest.files, "source")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "best.pt", target / "best.pt")
    shutil.copytree(source / "best_ncnn_model", target / "best_ncnn_model", dirs_exist_ok=True)
    verify_files(target, manifest.files, "copy")
    return target


def main() -> None:
    print(f"Verified models: {prepare_models(Path(__file__).resolve().parents[1])}")


if __name__ == "__main__":
    main()
