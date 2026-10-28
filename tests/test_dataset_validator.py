"""
Unit tests for image, mask, and duplicate validation (src.data.dataset_validator).
"""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.data.dataset_validator import DatasetValidator


class TestDatasetValidator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.validator = DatasetValidator(processed_dir=self.root / "processed")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_image(self):
        """Valid JPEG image should pass verification."""
        img_path = self.root / "valid.jpg"
        img = Image.new("RGB", (640, 480), color="green")
        img.save(img_path)

        res = self.validator.validate_image(img_path)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.width, 640)
        self.assertEqual(res.height, 480)
        self.assertEqual(res.channels, 3)
        self.assertFalse(res.is_corrupted)

    def test_zero_byte_image(self):
        """Zero-byte file must be detected and marked invalid."""
        zero_path = self.root / "empty.jpg"
        zero_path.touch()

        res = self.validator.validate_image(zero_path)
        self.assertFalse(res.is_valid)
        self.assertTrue(res.is_zero_byte)
        self.assertIn("zero bytes", res.error_message.lower())

    def test_corrupted_image(self):
        """Corrupted image header must be detected and reported."""
        corrupt_path = self.root / "corrupt.png"
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_BYTE_HEADER_DATA_1234567890")

        res = self.validator.validate_image(corrupt_path)
        self.assertFalse(res.is_valid)
        self.assertTrue(res.is_corrupted)

    def test_unsupported_format(self):
        """Files with unsupported extensions must be flagged."""
        txt_path = self.root / "document.pdf"
        txt_path.write_text("dummy text")

        res = self.validator.validate_image(txt_path)
        self.assertFalse(res.is_valid)
        self.assertTrue(res.is_unsupported_format)

    def test_mask_pairing_and_dimension_mismatch(self):
        """Mask pairs with mismatched dimensions must be flagged with a descriptive error."""
        img_path = self.root / "food_001.jpg"
        mask_path = self.root / "food_001.png"

        img = Image.new("RGB", (400, 300), color="white")
        img.save(img_path)

        # Create mask with different resolution (200x200)
        mask = Image.new("L", (200, 200), color=1)
        mask.save(mask_path)

        results = self.validator.validate_segmentation_pairs(
            "FoodSeg103", [img_path], [mask_path]
        )

        self.assertEqual(len(results), 1)
        pair = results[0]
        self.assertTrue(pair.is_paired)
        self.assertTrue(pair.is_mask_readable)
        self.assertFalse(pair.dimensions_match)
        self.assertIn("Dimension mismatch", pair.issue_description)

    def test_exact_and_visual_duplicate_detection(self):
        """Exact copies and visually identical images must be flagged as duplicates."""
        img1 = self.root / "img_orig.jpg"
        img2 = self.root / "img_copy.jpg"

        im = Image.new("RGB", (128, 128), color=(200, 100, 50))
        im.save(img1)
        im.save(img2)

        dups = self.validator.find_duplicates([img1, img2])
        self.assertGreaterEqual(len(dups), 1)
        exact = [d for d in dups if d["duplicate_type"] == "EXACT"]
        self.assertEqual(len(exact), 1)


if __name__ == "__main__":
    unittest.main()
