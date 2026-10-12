from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import prepare_models

REPOSITORY = Path(__file__).resolve().parents[1]
METADATA_FILE = "best_ncnn_model/metadata.yaml"


class ModelManifestTests(unittest.TestCase):
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
                    {"file": str(path.relative_to(source)), "bytes": path.stat().st_size,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in (source / "best.pt", metadata)
                ]
                (repository / "deploy").mkdir()
                (repository / "deploy/model_manifest.json").write_text(json.dumps({
                    "source": "weights", "runtime_directory": "models", "files": records,
                }))
                if corruption == "source":
                    metadata.write_bytes(b"task: detect\r\n")
                with patch.object(prepare_models, "__file__", str(repository / "tools/prepare_models.py")):
                    if corruption == "source":
                        with self.assertRaisesRegex(SystemExit, "source checksum mismatch"):
                            prepare_models.main()
                        self.assertFalse((repository / "models").exists())
                    else:
                        with patch.object(prepare_models.shutil, "copy2"), self.assertRaisesRegex(
                            SystemExit, "copy checksum mismatch"
                        ):
                            (repository / "models").mkdir()
                            (repository / "models/best.pt").write_bytes(b"damaged")
                            prepare_models.main()
