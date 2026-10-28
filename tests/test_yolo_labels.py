"""
NutriVision AI — YOLO Detection Label Validation Tests
Phase 2: YOLO11 Food Detection Pipeline
"""

import unittest
from src.detection.mask_to_yolo import YOLOBox


class TestYOLOLabelValidation(unittest.TestCase):
    """Tests validation constraints on YOLO format bounding boxes."""

    def test_valid_label(self):
        """Valid normalized coordinates within (0, 1] pass validation."""
        box = YOLOBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.3)
        valid, err = box.validate()
        self.assertTrue(valid)
        self.assertIsNone(err)
        self.assertEqual(box.to_yolo_line(), "0 0.500000 0.500000 0.200000 0.300000")

    def test_negative_coordinates(self):
        """Negative x_center or y_center fails validation."""
        box = YOLOBox(class_id=1, x_center=-0.1, y_center=0.5, width=0.2, height=0.3)
        valid, err = box.validate()
        self.assertFalse(valid)
        self.assertIn("x_center out of bounds", err)

        box2 = YOLOBox(class_id=1, x_center=0.5, y_center=-0.2, width=0.2, height=0.3)
        valid2, err2 = box2.validate()
        self.assertFalse(valid2)
        self.assertIn("y_center out of bounds", err2)

    def test_coordinate_greater_than_one(self):
        """Coordinate > 1.0 fails validation."""
        box = YOLOBox(class_id=2, x_center=1.1, y_center=0.5, width=0.2, height=0.3)
        valid, err = box.validate()
        self.assertFalse(valid)
        self.assertIn("x_center out of bounds", err)

        box2 = YOLOBox(class_id=2, x_center=0.5, y_center=1.05, width=0.2, height=0.3)
        valid2, err2 = box2.validate()
        self.assertFalse(valid2)
        self.assertIn("y_center out of bounds", err2)

    def test_zero_or_negative_dimension(self):
        """Zero or negative width/height fails validation."""
        box = YOLOBox(class_id=3, x_center=0.5, y_center=0.5, width=0.0, height=0.3)
        valid, err = box.validate()
        self.assertFalse(valid)
        self.assertIn("width out of bounds", err)

        box2 = YOLOBox(class_id=3, x_center=0.5, y_center=0.5, width=0.2, height=-0.1)
        valid2, err2 = box2.validate()
        self.assertFalse(valid2)
        self.assertIn("height out of bounds", err2)

    def test_dimension_greater_than_one(self):
        """Width or height > 1.0 fails validation."""
        box = YOLOBox(class_id=4, x_center=0.5, y_center=0.5, width=1.2, height=0.5)
        valid, err = box.validate()
        self.assertFalse(valid)
        self.assertIn("width out of bounds", err)


if __name__ == "__main__":
    unittest.main()
