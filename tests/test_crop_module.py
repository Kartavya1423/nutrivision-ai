"""
Unit tests for food-only crop module (src.utils.crop).
Validates background removal, padding, immutability, and boundary conditions.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from src.segmentation.segmenter import FoodMask
from src.utils.crop import extract_food_region


class TestCropModule(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 100x100 RGB image with distinctive background (blue)
        self.h, self.w = 100, 100
        self.image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.image[:, :] = [0, 0, 255]  # Blue background

        # Place a 20x20 "food" square in center with red color [255, 0, 0]
        self.food_y1, self.food_y2 = 40, 60
        self.food_x1, self.food_x2 = 40, 60
        self.image[self.food_y1:self.food_y2, self.food_x1:self.food_x2] = [255, 0, 0]

        # Binary mask corresponding to food region
        self.mask = np.zeros((self.h, self.w), dtype=bool)
        self.mask[self.food_y1:self.food_y2, self.food_x1:self.food_x2] = True

    def test_extract_food_region_rgba(self):
        crop_arr, crop_pil = extract_food_region(
            image=self.image,
            mask=self.mask,
            padding=0,
            return_rgba=True,
        )

        # Dimension without padding should match 20x20
        self.assertEqual(crop_arr.shape, (20, 20, 4))
        # All pixels in this tight mask should have alpha 255 and red color
        self.assertTrue(np.all(crop_arr[:, :, 3] == 255))
        self.assertTrue(np.all(crop_arr[:, :, 0] == 255))
        self.assertTrue(np.all(crop_arr[:, :, 1] == 0))
        self.assertTrue(np.all(crop_arr[:, :, 2] == 0))

    def test_extract_food_region_with_padding(self):
        pad = 5
        crop_arr, _ = extract_food_region(
            image=self.image,
            mask=self.mask,
            padding=pad,
            return_rgba=True,
        )

        # Expected size: (20 + 2*5) x (20 + 2*5) = 30x30
        self.assertEqual(crop_arr.shape, (30, 30, 4))
        # Outer border pixels must have alpha = 0 (background removed)
        self.assertEqual(crop_arr[0, 0, 3], 0)
        self.assertEqual(crop_arr[29, 29, 3], 0)
        # Center food pixels must have alpha = 255
        self.assertEqual(crop_arr[15, 15, 3], 255)

    def test_preserves_original_image(self):
        orig_copy = self.image.copy()
        extract_food_region(self.image, self.mask, padding=5)
        # Verify original was NOT mutated in place
        self.assertTrue(np.array_equal(self.image, orig_copy))

    def test_food_mask_object_input(self):
        food_mask_obj = FoodMask(
            mask=self.mask,
            class_name="apple",
            class_id=1,
            confidence=0.95,
        )
        crop_arr, _ = extract_food_region(self.image, food_mask_obj, padding=2)
        self.assertEqual(crop_arr.shape, (24, 24, 4))

    def test_save_artifact_and_visualization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_crop.png"
            extract_food_region(
                self.image,
                self.mask,
                padding=2,
                save_path=save_path,
                save_visualization=True,
            )
            self.assertTrue(save_path.is_file())
            vis_path = Path(tmpdir) / "test_crop_viz.png"
            self.assertTrue(vis_path.is_file())

    def test_empty_mask_handling(self):
        empty_mask = np.zeros((self.h, self.w), dtype=bool)
        crop_arr, crop_pil = extract_food_region(self.image, empty_mask)
        self.assertEqual(crop_arr.shape[0], 1)
        self.assertEqual(crop_arr.shape[1], 1)

    def test_dimension_mismatch_raises_error(self):
        bad_mask = np.zeros((50, 50), dtype=bool)
        with self.assertRaises(ValueError):
            extract_food_region(self.image, bad_mask)


if __name__ == "__main__":
    unittest.main()
