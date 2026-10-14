"""Publish immutable source archives with one atomic directory rename."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO

READY_FILE = ".columba-ready.json"


def publish_archive(root: Path, digest: str, commit: str, source: BinaryIO) -> Path:
    if not re.fullmatch(r"[0-9a-f]{16,64}", digest):
        raise ValueError("invalid checkout digest")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / digest
    with (root / ".publication.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            if not (destination / READY_FILE).is_file():
                raise RuntimeError(
                    f"refusing to rewrite an existing unmarked checkout: {destination}"
                )
            return destination
        staging = Path(tempfile.mkdtemp(prefix=f".{digest}-", dir=root))
        try:
            with tarfile.open(fileobj=source, mode="r|") as archive:
                archive.extractall(staging, filter="data")
            (staging / READY_FILE).write_text(
                json.dumps({"digest": digest, "commit": commit}) + "\n"
            )
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    expected_root = (Path.home() / "columba/checkouts").resolve()
    if arguments.root.resolve() != expected_root:
        raise ValueError("remote checkout publication must stay in ~/columba/checkouts")
    print(publish_archive(arguments.root, arguments.digest, arguments.commit, sys.stdin.buffer))


if __name__ == "__main__":
    main()
