"""
Unit tests for food detection schemas and contracts (src.detection).
"""

import unittest
from src.detection.detector import (
    BoundingBox,
    DetectionResult,
    FoodDetection,
    ModelStatus,
)
from src.detection.yolo_detector import YOLO11FoodDetector


class TestDetectionSchema(unittest.TestCase):
    def test_bounding_box_logic(self):
        bbox = BoundingBox(10.0, 20.0, 50.0, 80.0)
        self.assertEqual(bbox.width, 40.0)
        self.assertEqual(bbox.height, 60.0)
        self.assertEqual(bbox.area, 2400.0)
        self.assertEqual(bbox.to_list(), [10.0, 20.0, 50.0, 80.0])

        # Clamp boundary
        clamped = BoundingBox(-5.0, 10.0, 150.0, 120.0).clamp(100, 100)
        self.assertEqual(clamped.x1, 0.0)
        self.assertEqual(clamped.x2, 100.0)
        self.assertEqual(clamped.y2, 100.0)

    def test_detection_schema_fields(self):
        det = FoodDetection(
            class_id=0,
            class_name="rice",
            confidence=0.88,
            bbox=[10.0, 15.0, 45.0, 60.0],
        )
        d_dict = det.to_dict()
        self.assertEqual(d_dict["class_id"], 0)
        self.assertEqual(d_dict["class_name"], "rice")
        self.assertEqual(d_dict["confidence"], 0.88)
        self.assertEqual(d_dict["bbox"], [10.0, 15.0, 45.0, 60.0])

    def test_yolo_unloaded_honest_reporting(self):
        detector = YOLO11FoodDetector(model_path="non_existent_yolo.pt")
        self.assertEqual(detector.status, ModelStatus.MODEL_NOT_LOADED)

        res = detector.predict("tests")
        self.assertFalse(res.is_success)
        self.assertEqual(len(res.detections), 0)


if __name__ == "__main__":
    unittest.main()
