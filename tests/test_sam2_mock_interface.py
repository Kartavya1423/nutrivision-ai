"""
Unit tests for SAM 2 interface, failure modes, and lifecycle states.
Phase 3: SAM 2 Food Instance Segmentation

These tests run purely on synthetic data / mock configurations without
requiring GPU hardware or pre-existing model weights.
"""

from pathlib import Path
import unittest
import numpy as np

from src.detection.detector import ModelStatus
from src.segmentation.crop_export import save_segmentation_artifacts
from src.segmentation.pipeline import YOLOToSAMPipeline
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.segmentation.segmenter import (
    FoodMask,
    SegmentationResult,
    SegmentationStatusCode,
)
from src.utils.crop import extract_food_region


class TestSAM2InterfaceAndLifecycle(unittest.TestCase):
    def test_missing_checkpoint_reports_honest_status(self):
        # Point to a guaranteed non-existent checkpoint
        fake_path = Path("models/segmentation/non_existent_sam2_fake.pt")
        seg = SAM2FoodSegmenter(checkpoint_path=fake_path)

        self.assertFalse(seg.is_ready())
        self.assertEqual(seg.status, ModelStatus.MODEL_NOT_LOADED)

        # Attempting inference on missing checkpoint returns structured error envelope
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = seg.predict_masks(dummy_img, bboxes=[[10, 10, 50, 50]])

        self.assertFalse(res.is_success)
        self.assertEqual(res.status, ModelStatus.MODEL_NOT_LOADED)
        self.assertEqual(res.status_code, SegmentationStatusCode.SEGMENTATION_MODEL_NOT_LOADED.value)
        self.assertEqual(res.count, 0)
        self.assertIn("non_existent_sam2_fake.pt", res.error_message)

    def test_invalid_image_handling(self):
        fake_path = Path("models/segmentation/non_existent_sam2_fake.pt")
        seg = SAM2FoodSegmenter(checkpoint_path=fake_path)

        # Non-existent image file
        res = seg.predict_masks("this_image_does_not_exist_9999.xyz")
        self.assertFalse(res.is_success)
        self.assertEqual(res.status, ModelStatus.ERROR)
        self.assertEqual(res.status_code, SegmentationStatusCode.INVALID_IMAGE.value)

    def test_empty_detections_handling(self):
        # Dummy segmenter with mock readiness
        seg = SAM2FoodSegmenter(checkpoint_path=Path("non_existent.pt"))
        seg._status = ModelStatus.READY  # simulate ready

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = seg.predict_masks(dummy_img, bboxes=[])

        self.assertTrue(res.is_success)
        self.assertEqual(res.status_code, SegmentationStatusCode.NO_DETECTIONS.value)
        self.assertEqual(res.count, 0)

    def test_alpha_channel_correctness(self):
        # Create a synthetic image: red square on green background
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [0, 255, 0]        # Green background
        img[30:70, 30:70] = [255, 0, 0]  # Red food item

        # Binary food mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True

        rgba_arr, rgba_pil = extract_food_region(img, mask, padding=5, return_rgba=True)

        # Check dimensions
        self.assertEqual(rgba_arr.ndim, 3)
        self.assertEqual(rgba_arr.shape[2], 4)  # 4 channels: RGBA

        # Inside food: original RGB intact, alpha = 255
        # The crop has padding=5, so food is offset by 5
        food_pixel = rgba_arr[10, 10]
        self.assertEqual(food_pixel[0], 255)  # R
        self.assertEqual(food_pixel[1], 0)    # G
        self.assertEqual(food_pixel[2], 0)    # B
        self.assertEqual(food_pixel[3], 255)  # Alpha = 255 (opaque food)

        # Outside food (padding region): alpha = 0 (transparent)
        pad_pixel = rgba_arr[1, 1]
        self.assertEqual(pad_pixel[3], 0)  # Alpha = 0 (transparent background)

    def test_save_segmentation_artifacts(self, tmp_path_factory=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_out = Path(tmp_dir)

            img = np.zeros((60, 60, 3), dtype=np.uint8)
            img[15:45, 15:45] = [200, 100, 50]

            mask = np.zeros((60, 60), dtype=bool)
            mask[15:45, 15:45] = True

            rgba_crop, _ = extract_food_region(img, mask, padding=2, return_rgba=True)
            rgb_crop, _ = extract_food_region(img, mask, padding=2, return_rgba=False)

            f_mask = FoodMask(
                mask=mask,
                class_name="carrot",
                class_id=3,
                confidence=0.94,
                prompt_bbox=[15.0, 15.0, 45.0, 45.0],
                crop_rgb=rgb_crop,
                crop_rgba=rgba_crop,
            )

            res = SegmentationResult(
                status=ModelStatus.READY,
                masks=[f_mask],
                combined_mask=mask,
                inference_time_ms=10.0,
                model_name="MockSAM2",
            )

            exported = save_segmentation_artifacts(
                image=img,
                result=res,
                source_name="unit_test_sample",
                output_dir=tmp_out,
            )

            self.assertTrue(Path(exported["overlay"]).is_file())
            self.assertTrue(Path(exported["metadata_json"]).is_file())
            self.assertEqual(len(exported["masks"]), 1)
            self.assertEqual(len(exported["rgba"]), 1)
            self.assertTrue(Path(exported["rgba"][0]).is_file())


if __name__ == "__main__":
    unittest.main()
