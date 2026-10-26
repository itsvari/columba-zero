from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import yaml

CLASSES_FILE = Path(__file__).resolve().parents[1] / "training/classes.yaml"


class TaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.taxonomy: dict[str, Any] = yaml.safe_load(CLASSES_FILE.read_text())

    def test_pigeon_is_first_and_the_only_trigger_class(self) -> None:
        self.assertEqual(self.taxonomy["trigger_classes"], ["pigeon"])
        for name, taxonomy in self.taxonomy["taxonomies"].items():
            with self.subTest(taxonomy=name):
                self.assertEqual(taxonomy["names"][0], "pigeon")
        self.assertEqual(self.taxonomy["taxonomies"]["single"]["names"], ["pigeon"])
        roles = {name: details["role"] for name, details in self.taxonomy["classes"].items()}
        self.assertEqual([name for name, role in roles.items() if role == "trigger"], ["pigeon"])

    def test_species_names_are_unique_end_with_catch_all_and_match_class_details(self) -> None:
        names = self.taxonomy["taxonomies"]["species"]["names"]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[-1], "other_bird")
        self.assertEqual(list(self.taxonomy["classes"]), names)


if __name__ == "__main__":
    unittest.main()
