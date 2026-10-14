from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import yaml

CLASSES_FILE = Path(__file__).resolve().parents[1] / "training/classes.yaml"


class TaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.taxonomy: dict[str, Any] = yaml.safe_load(CLASSES_FILE.read_text())

    def test_pigeon_is_first_and_only_trigger_class(self) -> None:
        self.assertEqual(self.taxonomy["trigger_classes"], ["pigeon"])
        for name, taxonomy in self.taxonomy["taxonomies"].items():
            with self.subTest(taxonomy=name):
                self.assertEqual(taxonomy["names"][0], "pigeon")
        self.assertEqual(self.taxonomy["taxonomies"]["single"]["names"], ["pigeon"])

    def test_species_names_are_unique_and_end_with_catch_all(self) -> None:
        names = self.taxonomy["taxonomies"]["species"]["names"]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[-1], "other_bird")

    def test_every_class_has_details_and_a_taxon_id(self) -> None:
        details = self.taxonomy["classes"]
        names = self.taxonomy["taxonomies"]["species"]["names"]
        self.assertEqual(list(details), names)
        taxon_ids = [details[name]["inat_taxon_id"] for name in names]
        self.assertTrue(all(isinstance(value, int) and value > 0 for value in taxon_ids))
        self.assertEqual(len(taxon_ids), len(set(taxon_ids)))
        self.assertEqual(details["pigeon"]["scientific_name"], "Columba livia")
        roles = {name: details[name]["role"] for name in names}
        self.assertEqual([name for name, role in roles.items() if role == "trigger"], ["pigeon"])


if __name__ == "__main__":
    unittest.main()
