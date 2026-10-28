"""
Unit tests for dataset discovery and scanner (src.data.dataset_scanner).
"""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.data.dataset_scanner import DatasetScanner, DatasetScanResult


class TestDatasetScanner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_dataset_handled_gracefully(self):
        """Missing datasets should report NOT AVAILABLE without throwing unhandled exceptions."""
        scanner = DatasetScanner(raw_dir=self.root)
        result = scanner.scan_dataset("FoodSeg103", self.root / "FoodSeg103")
        self.assertIsInstance(result, DatasetScanResult)
        self.assertEqual(result.status, "NOT AVAILABLE")
        self.assertFalse(result.is_available)
        self.assertEqual(result.image_count, 0)

    def test_empty_dataset_folder_handled_gracefully(self):
        """Empty folder with only a hidden file should be marked NOT AVAILABLE."""
        empty_dir = self.root / "UECFoodPixComplete"
        empty_dir.mkdir(parents=True)
        (empty_dir / ".gitkeep").touch()

        scanner = DatasetScanner(raw_dir=self.root)
        result = scanner.scan_dataset("UECFoodPixComplete", empty_dir)
        self.assertEqual(result.status, "NOT AVAILABLE")
        self.assertFalse(result.is_available)

    def test_scan_classification_dataset(self):
        """Scanner should correctly detect classes and split structures in classification layout."""
        f360_dir = self.root / "Fruits360_100x100"
        train_apple = f360_dir / "Training" / "Apple_Braeburn"
        test_apple = f360_dir / "Test" / "Apple_Braeburn"
        train_apple.mkdir(parents=True)
        test_apple.mkdir(parents=True)

        # Create dummy images
        img = Image.new("RGB", (100, 100), color="red")
        img.save(train_apple / "r_0_100.jpg")
        img.save(test_apple / "r_1_100.jpg")

        scanner = DatasetScanner(raw_dir=self.root)
        result = scanner.scan_dataset("Fruits360_100x100", f360_dir)

        self.assertEqual(result.status, "AVAILABLE")
        self.assertTrue(result.is_available)
        self.assertEqual(result.image_count, 2)
        self.assertEqual(result.classes_count, 1)
        self.assertIn("Apple_Braeburn", result.class_names)
        self.assertEqual(result.splits_detected.get("Training"), 1)
        self.assertEqual(result.splits_detected.get("Test"), 1)

    def test_fruits360_multi_reports_missing_bbox(self):
        """Fruits-360 multi must explicitly report 'Bounding-box annotations are unavailable.' if unannotated."""
        f360_multi = self.root / "Fruits360_multi"
        f360_multi.mkdir(parents=True)

        img = Image.new("RGB", (300, 300), color="blue")
        img.save(f360_multi / "multi_fruits_1.jpg")

        scanner = DatasetScanner(raw_dir=self.root)
        result = scanner.scan_dataset("Fruits360_multi", f360_multi)

        self.assertEqual(result.status, "AVAILABLE")
        self.assertFalse(result.bbox_available)
        self.assertIn("Bounding-box annotations are unavailable.", result.notes)


if __name__ == "__main__":
    unittest.main()
