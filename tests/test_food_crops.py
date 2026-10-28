"""
NutriVision AI — Food Cropping Utility Unit Tests
Phase 2: YOLO11 Food Detection Pipeline
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from src.detection.crop_utils import crop_bounding_box, save_detection_crops
from src.detection.detector import BoundingBox, FoodDetection


class TestFoodCropping(unittest.TestCase):
    """Tests food region extraction and persistent metadata generation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)
        # Create a sample 200x200 RGB canvas
        self.canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        self.canvas[20:60, 30:80] = [255, 0, 0]    # Red patch (food 1)
        self.canvas[120:180, 110:170] = [0, 255, 0] # Green patch (food 2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_crop(self):
        """Crops rectangular area accurately."""
        bbox = [30, 20, 80, 60]
        crop, coords = crop_bounding_box(self.canvas, bbox)
        self.assertEqual(crop.shape, (40, 50, 3))
        self.assertEqual(coords, (30, 20, 80, 60))
        # Ensure red patch content is preserved
        self.assertEqual(crop[0, 0, 0], 255)

    def test_boundary_clamped_crop(self):
        """Bounding box extending outside canvas clamps safely without error."""
        bbox = [-10, -5, 50, 40]
        crop, coords = crop_bounding_box(self.canvas, bbox)
        self.assertEqual(coords, (0, 0, 50, 40))
        self.assertEqual(crop.shape, (40, 50, 3))

    def test_invalid_inverted_crop(self):
        """Inverted bounding box raises ValueError."""
        with self.assertRaises(ValueError):
            crop_bounding_box(self.canvas, [100, 100, 50, 50])

    def test_save_detection_crops(self):
        """Batch crops all detections and creates sidecar metadata."""
        dets = [
            FoodDetection(class_id=1, class_name="apple", confidence=0.92, bbox=[30, 20, 80, 60]),
            FoodDetection(class_id=2, class_name="banana", confidence=0.88, bbox=[110, 120, 170, 180]),
        ]
        records = save_detection_crops(
            self.canvas, dets, output_dir=self.out_dir, source_image_name="dish_sample"
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["class_name"], "apple")
        self.assertEqual(records[1]["class_name"], "banana")

        # Verify crop files and metadata JSON were written
        f1 = self.out_dir / "dish_sample_food_01_apple.jpg"
        j1 = self.out_dir / "dish_sample_food_01_apple.json"
        self.assertTrue(f1.is_file())
        self.assertTrue(j1.is_file())


if __name__ == "__main__":
    unittest.main()
