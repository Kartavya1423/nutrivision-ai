"""
Integration tests for SAM 2 with real model checkpoint on real food images.
Phase 3: SAM 2 Food Instance Segmentation
"""

from pathlib import Path
import unittest
import numpy as np
from PIL import Image

from src.detection.detector import ModelStatus
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.segmentation.segmenter import SegmentationStatusCode

SAM_CHECKPOINT = Path("models/segmentation/sam2.1_t.pt")
TEST_IMAGE = Path("data/raw/FoodSeg103/Images/img_dir/train/00000001.jpg")


class TestSAM2Integration(unittest.TestCase):
    @unittest.skipUnless(SAM_CHECKPOINT.is_file(), "Real SAM 2 checkpoint required")
    def setUp(self):
        self.segmenter = SAM2FoodSegmenter(
            checkpoint_path=SAM_CHECKPOINT,
            device="cpu",
            min_mask_area=50,
            apply_postprocessing=True,
        )

    def test_real_sam2_model_loading(self):
        self.assertTrue(self.segmenter.is_ready())
        self.assertEqual(self.segmenter.status, ModelStatus.READY)
        info = self.segmenter.get_model_info()
        self.assertEqual(info["status"], "READY")
        self.assertEqual(info["device"], "cpu")

    @unittest.skipUnless(TEST_IMAGE.is_file(), "Test image required")
    def test_real_sam2_box_prompt_inference(self):
        # Known food item box in 00000001.jpg
        # Box: [67.0, 33.0, 276.0, 197.0]
        prompt_box = [67.0, 33.0, 276.0, 197.0]

        res = self.segmenter.predict_masks(
            image=TEST_IMAGE,
            bboxes=[prompt_box],
            class_names=["candy_cluster"],
            confidences=[0.95],
            crop_padding=8,
            generate_crops=True,
        )

        self.assertTrue(res.is_success)
        self.assertEqual(res.status_code, SegmentationStatusCode.SEGMENTATION_SUCCESS.value)
        self.assertEqual(res.count, 1)

        mask_item = res.masks[0]
        self.assertEqual(mask_item.class_name, "candy_cluster")
        self.assertGreater(mask_item.area_pixels, 15000)
        self.assertGreater(mask_item.confidence, 0.70)
        self.assertEqual(mask_item.status, SegmentationStatusCode.SEGMENTATION_SUCCESS.value)

        # Check crops
        self.assertIsNotNone(mask_item.crop_rgb)
        self.assertIsNotNone(mask_item.crop_rgba)
        self.assertEqual(mask_item.crop_rgba.shape[2], 4)

        # Combined mask check
        self.assertIsNotNone(res.combined_mask)
        self.assertEqual(res.combined_mask.shape, (384, 512))

    @unittest.skipUnless(TEST_IMAGE.is_file(), "Test image required")
    def test_real_sam2_multi_box_prompt(self):
        # Multiple food items
        boxes = [
            [67.0, 33.0, 276.0, 197.0],
            [260.0, 66.0, 418.0, 191.0],
        ]

        res = self.segmenter.predict_masks(
            image=TEST_IMAGE,
            bboxes=boxes,
            class_names=["item_1", "item_2"],
            generate_crops=True,
        )

        self.assertTrue(res.is_success)
        self.assertEqual(res.count, 2)
        for m in res.masks:
            self.assertGreater(m.area_pixels, 5000)
            self.assertIsNotNone(m.crop_rgba)


if __name__ == "__main__":
    unittest.main()
