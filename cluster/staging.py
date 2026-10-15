"""Stage immutable C1 archives and verified weights to node-local storage."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

from cluster.recipe import component

ASSET_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_dataset(name: str, archives_root: Path, destination: Path) -> tuple[Path, str]:
    import yaml
    import zstandard

    component(name, "dataset")
    archive = archives_root / f"{name}.tar.zst"
    checksum_path = archive.with_name(archive.name + ".sha256")
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(archive)
    if actual != expected:
        raise ValueError(f"archive checksum mismatch: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    with archive.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as bundle:
            for member in bundle:
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != name:
                    raise ValueError(f"archive member outside dataset: {member.name}")
                if not member.isfile() and not member.isdir():
                    raise ValueError(f"archive links/devices are forbidden: {member.name}")
                bundle.extract(member, destination, filter="data")
    dataset = destination / name
    for required in ("DATASET.json", "manifest.jsonl", "data.yaml"):
        if not (dataset / required).is_file():
            raise ValueError(f"C1 dataset missing {required}: {name}")
    configuration = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise ValueError("data.yaml must be a mapping")
    configuration["path"] = str(dataset.resolve())
    for split in ("train", "val", "test"):
        if split not in configuration:
            continue
        expected_path = (
            f"images/{split}" if (dataset / f"images/{split}").exists() else configuration[split]
        )
        configuration[split] = expected_path
    (dataset / "data.yaml").write_text(yaml.safe_dump(configuration), encoding="utf-8")
    return dataset, actual


def stage_weights(name: str, weights_root: Path, destination: Path) -> Path:
    import fcntl

    component(name, "base_model")
    weights_root.mkdir(parents=True, exist_ok=True)
    cached = weights_root / name
    with (weights_root / f".{name}.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not cached.is_file():
            temporary = cached.with_suffix(".download")
            urllib.request.urlretrieve(f"{ASSET_RELEASE}/{name}", temporary)
            temporary.replace(cached)
    destination.mkdir(parents=True, exist_ok=True)
    staged = destination / name
    shutil.copy2(cached, staged)
    return staged.resolve()


def align_test_taxonomy(dataset: Path, taxonomy: str, classes: Path) -> bool:
    """Give a staged single-taxonomy test set the recipe's taxonomy names.

    Only the node-local staged ``data.yaml`` changes. Valid only when the test set
    is `single` ([pigeon]) and the target taxonomy keeps pigeon at index 0, so the
    label files (pigeon boxes only) mean the same thing; other birds stay unboxed.
    Pigeon metrics and the manifest-based false-alarm rates are therefore
    comparable with single-taxonomy models. Returns whether names changed.
    """
    import yaml

    configuration = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    taxonomies = yaml.safe_load(classes.read_text(encoding="utf-8"))["taxonomies"]
    if taxonomy not in taxonomies:
        raise ValueError(f"unknown taxonomy: {taxonomy}")
    target = list(taxonomies[taxonomy]["names"])
    current = configuration.get("names")
    if isinstance(current, dict):
        current = [current[index] for index in sorted(current)]
    if current == target:
        return False
    if current != ["pigeon"] or target[0] != "pigeon":
        raise ValueError("only a single-taxonomy test set can be aligned to pigeon-first names")
    configuration["names"] = target
    configuration["nc"] = len(target)
    (dataset / "data.yaml").write_text(yaml.safe_dump(configuration), encoding="utf-8")
    return True
