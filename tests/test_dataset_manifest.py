"""
Unit tests for dataset manifest and purpose mapping generation (src.data.dataset_report).
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.data.dataset_scanner import DatasetScanner
from src.data.dataset_report import DatasetReportGenerator


class TestDatasetManifest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.processed_dir = self.root / "processed"
        self.outputs_dir = self.root / "outputs"
        self.reporter = DatasetReportGenerator(
            processed_dir=self.processed_dir,
            outputs_dir=self.outputs_dir
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_purpose_json(self):
        """Purpose mapping must contain all 6 target datasets plus sample_benchmark."""
        purpose_file = self.reporter.save_dataset_purpose_json()
        self.assertTrue(purpose_file.is_file())

        with open(purpose_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        required = [
            "FoodSeg103",
            "UECFoodPixComplete",
            "Fruits360_100x100",
            "Fruits360_multi",
            "weight_dataset",
            "freshness_dataset"
        ]
        for ds in required:
            self.assertIn(ds, data)
            self.assertIsInstance(data[ds], list)
            self.assertGreater(len(data[ds]), 0)

    def test_dataset_manifest_structure(self):
        """Manifest JSON must contain all required specification fields."""
        scanner = DatasetScanner(raw_dir=self.root)
        scans = scanner.scan_all()

        manifest_file = self.reporter.save_dataset_manifest_json(scans)
        self.assertTrue(manifest_file.is_file())

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn("datasets", manifest)
        self.assertGreater(len(manifest["datasets"]), 0)

        first = manifest["datasets"][0]
        expected_keys = [
            "name",
            "path",
            "status",
            "number_of_images",
            "number_of_classes",
            "image_formats",
            "image_dimensions",
            "annotation_type",
            "mask_availability",
            "bbox_availability",
            "official_splits",
            "validation_status",
            "warnings",
            "intended_project_module"
        ]
        for key in expected_keys:
            self.assertIn(key, first)


if __name__ == "__main__":
    unittest.main()
