"""
Unit tests for image loading and validation (src.utils.image_loader).
Tests format validation (JPG, JPEG, PNG, WEBP), dimension limits, and corruption checks.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from src.utils.image_loader import (
    ImageValidationError,
    SUPPORTED_EXTENSIONS,
    load_and_validate_image,
)


class TestImageLoader(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_supported_formats(self):
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            file_p = self.tmp_path / f"test{ext}"
            img = Image.new("RGB", (64, 64), color=(100, 150, 200))
            img.save(file_p)

            arr, pil_img = load_and_validate_image(file_p)
            self.assertEqual(arr.shape, (64, 64, 3))
            self.assertEqual(pil_img.size, (64, 64))

    def test_unsupported_format_raises(self):
        file_p = self.tmp_path / "test.bmp"
        img = Image.new("RGB", (64, 64))
        img.save(file_p)

        with self.assertRaises(ImageValidationError):
            load_and_validate_image(file_p)

    def test_corrupted_image_raises(self):
        corrupt_p = self.tmp_path / "corrupt.png"
        with open(corrupt_p, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_CONTENT_BYTES")

        with self.assertRaises(ImageValidationError):
            load_and_validate_image(corrupt_p)

    def test_dimensions_bounds(self):
        # Too small (< 10)
        small_img = Image.new("RGB", (5, 5))
        with self.assertRaises(ImageValidationError):
            load_and_validate_image(small_img, min_dimension=10)

        # Exceeds max resolution
        large_img = Image.new("RGB", (500, 500))
        with self.assertRaises(ImageValidationError):
            load_and_validate_image(large_img, max_dimension=400)


if __name__ == "__main__":
    unittest.main()
