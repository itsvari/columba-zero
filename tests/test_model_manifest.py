from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import prepare_models

REPOSITORY = Path(__file__).resolve().parents[1]
METADATA_FILE = "best_ncnn_model/metadata.yaml"


class ModelManifestTests(unittest.TestCase):
    def test_clean_checkouts_preserve_checksums_with_auto_crlf(self) -> None:
        manifest = json.loads((REPOSITORY / "deploy/model_manifest.json").read_text())
        for automatic_crlf in ("false", "true"):
            with (
                self.subTest(automatic_crlf=automatic_crlf),
                tempfile.TemporaryDirectory() as directory,
            ):
                checkout = Path(directory) / "checkout"
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--quiet",
                        "--shared",
                        "--no-checkout",
                        str(REPOSITORY),
                        str(checkout),
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(checkout),
                        "-c",
                        f"core.autocrlf={automatic_crlf}",
                        "checkout",
                        "HEAD",
                        "--",
                        ".gitattributes",
                        "deploy/model_manifest.json",
                        "tools/prepare_models.py",
                        manifest["source"],
                    ],
                    check=True,
                )
                subprocess.run(
                    [sys.executable, "tools/prepare_models.py"],
                    cwd=checkout,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for record in manifest["files"]:
                    copied = checkout / manifest["runtime_directory"] / record["file"]
                    self.assertEqual(
                        hashlib.sha256(copied.read_bytes()).hexdigest(), record["sha256"]
                    )
                    self.assertEqual(copied.stat().st_size, record["bytes"])

    def test_manifest_matches_tracked_bytes(self) -> None:
        manifest = json.loads((REPOSITORY / "deploy/model_manifest.json").read_text())
        for record in manifest["files"]:
            source = f"{manifest['source']}/{record['file']}"
            content = subprocess.check_output(["git", "show", f"HEAD:{source}"], cwd=REPOSITORY)
            with self.subTest(file=record["file"]):
                self.assertEqual(len(content), record["bytes"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), record["sha256"])
                if record["file"] == METADATA_FILE:
                    self.assertNotIn(b"\r", content)

    def test_preparation_rejects_changed_source_and_copy(self) -> None:
        for corruption in ("source", "copy"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                repository = Path(directory)
                source = repository / "weights"
                export = source / "best_ncnn_model"
                export.mkdir(parents=True)
                (source / "best.pt").write_bytes(b"checkpoint")
                metadata = export / "metadata.yaml"
                metadata.write_bytes(b"task: detect\n")
                records = [
                    {
                        "file": str(path.relative_to(source)),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in (source / "best.pt", metadata)
                ]
                (repository / "deploy").mkdir()
                (repository / "deploy/model_manifest.json").write_text(
                    json.dumps(
                        {
                            "source": "weights",
                            "runtime_directory": "models",
                            "files": records,
                        }
                    )
                )
                if corruption == "source":
                    metadata.write_bytes(b"task: detect\r\n")
                with patch.object(
                    prepare_models, "__file__", str(repository / "tools/prepare_models.py")
                ):
                    if corruption == "source":
                        with self.assertRaisesRegex(SystemExit, "source checksum mismatch"):
                            prepare_models.main()
                        self.assertFalse((repository / "models").exists())
                    else:
                        with (
                            patch("tools.prepare_models.shutil.copy2"),
                            self.assertRaisesRegex(SystemExit, "copy checksum mismatch"),
                        ):
                            (repository / "models").mkdir()
                            (repository / "models/best.pt").write_bytes(b"damaged")
                            prepare_models.main()
