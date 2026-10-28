"""
Unit tests for segmentation schema and mask representations (src.segmentation).
"""

import unittest
import numpy as np
from src.detection.detector import ModelStatus
from src.segmentation.segmenter import FoodMask, SegmentationResult


class TestSegmentationSchema(unittest.TestCase):
    def test_food_mask_computation(self):
        # 100x100 mask with 20x30 True rectangle
        m = np.zeros((100, 100), dtype=bool)
        m[10:40, 20:40] = True

        mask_obj = FoodMask(
            mask=m,
            class_name="chicken",
            class_id=1,
            confidence=0.95,
        )

        self.assertEqual(mask_obj.area_pixels, 30 * 20)
        self.assertEqual(mask_obj.bbox, [20.0, 10.0, 39.0, 39.0])
        self.assertEqual(mask_obj.as_uint8().shape, (100, 100))
        self.assertEqual(mask_obj.as_uint8()[15, 25], 255)
        self.assertEqual(mask_obj.as_uint8()[0, 0], 0)

    def test_segmentation_result_envelope(self):
        m = np.zeros((50, 50), dtype=bool)
        m[5:15, 5:15] = True
        mask_obj = FoodMask(mask=m, class_name="broccoli")

        result = SegmentationResult(
            status=ModelStatus.READY,
            masks=[mask_obj],
            combined_mask=m,
            model_name="TestSegmenter",
        )
        self.assertTrue(result.is_success)
        self.assertEqual(result.count, 1)
        r_dict = result.to_dict()
        self.assertEqual(r_dict["status"], "READY")
        self.assertEqual(len(r_dict["masks"]), 1)


if __name__ == "__main__":
    unittest.main()
