"""
NutriVision AI — Mask to YOLO Bounding Box Unit Tests
Phase 2: YOLO11 Food Detection Pipeline
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from src.detection.detection_config import MaskConversionConfig
from src.detection.mask_to_yolo import MaskToYOLOConverter, YOLOBox, MaskConversionResult


class TestMaskToYOLOConverter(unittest.TestCase):
    """Tests segmentation mask to bounding box derivation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.converter = MaskToYOLOConverter(
            MaskConversionConfig(min_area_pixels=4, min_dimension_pixels=2)
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_image_and_mask(self, mask_array: np.ndarray, width: int = 100, height: int = 100):
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        img_path = self.root / "test_img.jpg"
        img.save(img_path)

        if mask_array.ndim == 2:
            mask = Image.fromarray(mask_array.astype(np.uint8), mode="L")
        else:
            mask = Image.fromarray(mask_array.astype(np.uint8))
        mask_path = self.root / "test_mask.png"
        mask.save(mask_path)

        return img_path, mask_path

    def test_empty_mask(self):
        """Empty mask (all zeros / background) yields zero bounding boxes."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        img_p, msk_p = self._create_image_and_mask(arr)

        res = self.converter.convert(img_p, msk_p)
        self.assertTrue(res.is_empty_mask)
        self.assertEqual(res.boxes_generated, 0)
        self.assertEqual(len(res.boxes), 0)

    def test_single_object(self):
        """Single rectangular mask region yields exactly one bounding box."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        # 20x20 box at (20..40, 30..50), class_id = 5
        arr[20:40, 30:50] = 5
        img_p, msk_p = self._create_image_and_mask(arr)

        res = self.converter.convert(img_p, msk_p)
        self.assertFalse(res.is_empty_mask)
        self.assertEqual(res.boxes_generated, 1)

        box = res.boxes[0]
        self.assertEqual(box.class_id, 5)
        # x_min=30, x_max=50 -> center=40/100=0.40, width=20/100=0.20
        # y_min=20, y_max=40 -> center=30/100=0.30, height=20/100=0.20
        self.assertAlmostEqual(box.x_center, 0.40, places=2)
        self.assertAlmostEqual(box.y_center, 0.30, places=2)
        self.assertAlmostEqual(box.width, 0.20, places=2)
        self.assertAlmostEqual(box.height, 0.20, places=2)

    def test_multiple_connected_components(self):
        """Two separated objects of the same class produce two distinct bounding boxes."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        # Component 1 at top-left
        arr[10:20, 10:20] = 3
        # Component 2 at bottom-right
        arr[70:80, 70:80] = 3
        img_p, msk_p = self._create_image_and_mask(arr)

        res = self.converter.convert(img_p, msk_p)
        self.assertEqual(res.boxes_generated, 2)
        for b in res.boxes:
            self.assertEqual(b.class_id, 3)

    def test_multiple_classes(self):
        """Mask containing two different food classes produces boxes for both."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[10:25, 10:25] = 1  # class 1
        arr[50:65, 50:65] = 2  # class 2
        img_p, msk_p = self._create_image_and_mask(arr)

        res = self.converter.convert(img_p, msk_p)
        self.assertEqual(res.boxes_generated, 2)
        classes = {b.class_id for b in res.boxes}
        self.assertEqual(classes, {1, 2})

    def test_boundary_object(self):
        """Object touching or clipping against image boundaries clamps cleanly."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        # Object at boundary: (0..10, 0..10)
        arr[0:10, 0:10] = 7
        img_p, msk_p = self._create_image_and_mask(arr)

        res = self.converter.convert(img_p, msk_p)
        self.assertEqual(res.boxes_generated, 1)
        b = res.boxes[0]
        valid, err = b.validate()
        self.assertTrue(valid, f"Boundary box validation failed: {err}")
        self.assertTrue(0.0 <= b.x_center <= 1.0)
        self.assertTrue(0.0 <= b.y_center <= 1.0)

    def test_invalid_dimensions(self):
        """Mismatched image and mask dimensions returns error without crashing."""
        arr = np.zeros((80, 80), dtype=np.uint8)
        img_p, msk_p = self._create_image_and_mask(arr, width=100, height=100)

        res = self.converter.convert(img_p, msk_p)
        self.assertIsNotNone(res.error_message)
        self.assertIn("Dimension mismatch", res.error_message)

    def test_class_mapping(self):
        """Maps native class IDs to detector class IDs."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[20:40, 20:40] = 42
        img_p, msk_p = self._create_image_and_mask(arr)

        res = self.converter.convert(img_p, msk_p, class_mapping={42: 3})
        self.assertEqual(res.boxes[0].class_id, 3)
        self.assertEqual(res.boxes[0].native_class_id, 42)


if __name__ == "__main__":
    unittest.main()
