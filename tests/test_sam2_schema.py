"""
Unit tests for SAM 2 segmentation schemas and status codes.
Phase 3: SAM 2 Food Instance Segmentation
"""

import json
import unittest
import numpy as np

from src.detection.detector import ModelStatus
from src.segmentation.pipeline import FoodCropResult
from src.segmentation.segmenter import (
    FoodMask,
    SegmentationResult,
    SegmentationStatusCode,
)


class TestSAM2Schema(unittest.TestCase):
    def test_food_mask_creation_and_spatial_metrics(self):
        # 120x160 boolean canvas with 40x50 food rectangle
        mask = np.zeros((120, 160), dtype=bool)
        mask[20:60, 30:80] = True

        f_mask = FoodMask(
            mask=mask,
            class_name="pizza",
            class_id=4,
            confidence=0.92,
            prompt_bbox=[28.0, 18.0, 82.0, 62.0],
        )

        self.assertEqual(f_mask.area_pixels, 40 * 50)
        self.assertEqual(f_mask.class_name, "pizza")
        self.assertEqual(f_mask.class_id, 4)
        self.assertAlmostEqual(f_mask.confidence, 0.92, places=2)
        # Tight bounding box: [x1, y1, x2, y2]
        self.assertEqual(f_mask.bbox, [30.0, 20.0, 79.0, 59.0])
        self.assertEqual(f_mask.as_uint8().shape, (120, 160))
        self.assertEqual(f_mask.as_uint8()[30, 40], 255)
        self.assertEqual(f_mask.as_uint8()[0, 0], 0)

    def test_food_mask_validation_errors(self):
        # Non-2D array
        with self.assertRaises(ValueError):
            FoodMask(mask=np.zeros((10, 10, 3), dtype=bool))

        # Not a numpy array
        with self.assertRaises(ValueError):
            FoodMask(mask=[[True, False], [False, True]])

    def test_segmentation_status_codes(self):
        self.assertEqual(SegmentationStatusCode.SEGMENTATION_SUCCESS.value, "SEGMENTATION_SUCCESS")
        self.assertEqual(SegmentationStatusCode.SEGMENTATION_MODEL_NOT_AVAILABLE.value, "SEGMENTATION_MODEL_NOT_AVAILABLE")
        self.assertEqual(SegmentationStatusCode.NO_DETECTIONS.value, "NO_DETECTIONS")
        self.assertEqual(SegmentationStatusCode.INVALID_MASK.value, "INVALID_MASK")
        self.assertEqual(SegmentationStatusCode.MASK_TOO_SMALL.value, "MASK_TOO_SMALL")
        self.assertEqual(SegmentationStatusCode.CROP_FAILED.value, "CROP_FAILED")

    def test_segmentation_result_envelope_serialization(self):
        m1 = np.zeros((50, 50), dtype=bool)
        m1[10:30, 10:30] = True
        mask_obj = FoodMask(mask=m1, class_name="apple", class_id=1, confidence=0.88)

        res = SegmentationResult(
            status=ModelStatus.READY,
            status_code=SegmentationStatusCode.SEGMENTATION_SUCCESS.value,
            masks=[mask_obj],
            combined_mask=m1,
            inference_time_ms=125.4,
            model_name="SAM2FoodSegmenter",
            device="cpu",
            checkpoint="models/segmentation/sam2.1_t.pt",
        )

        self.assertTrue(res.is_success)
        self.assertEqual(res.count, 1)

        d = res.to_dict()
        self.assertEqual(d["status"], "READY")
        self.assertEqual(d["status_code"], "SEGMENTATION_SUCCESS")
        self.assertEqual(d["count"], 1)
        self.assertEqual(d["model_name"], "SAM2FoodSegmenter")
        self.assertEqual(d["device"], "cpu")
        self.assertTrue(d["has_combined_mask"])
        self.assertEqual(len(d["masks"]), 1)

        # Ensure JSON serializable
        json_str = json.dumps(d)
        self.assertIn("SEGMENTATION_SUCCESS", json_str)
        self.assertIn("apple", json_str)

    def test_food_crop_result_schema(self):
        crop = FoodCropResult(
            crop_id=1,
            class_id=2,
            class_name="rice",
            yolo_confidence=0.91,
            sam_quality_score=0.95,
            bbox=[10.0, 10.0, 100.0, 100.0],
            mask_tight_bbox=[12.0, 12.0, 98.0, 98.0],
            area_pixels=4500,
            relative_area=0.15,
            status=SegmentationStatusCode.SEGMENTATION_SUCCESS.value,
        )
        d = crop.to_dict()
        self.assertEqual(d["crop_id"], 1)
        self.assertEqual(d["class_name"], "rice")
        self.assertEqual(d["status"], "SEGMENTATION_SUCCESS")
        self.assertEqual(d["area_pixels"], 4500)


if __name__ == "__main__":
    unittest.main()
