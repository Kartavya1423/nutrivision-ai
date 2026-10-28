"""
Unit tests for model interfaces and honest reporting contracts.
Verifies that uninitialized/untrained models strictly report MODEL_NOT_LOADED or MODEL_NOT_TRAINED
and never return fabricated predictions.
"""

import unittest
import numpy as np
from PIL import Image

from src.classification import EfficientNetB4Classifier, MockFoodClassifier
from src.depth import DepthAnythingV2Estimator, MockDepthEstimator
from src.detection.detector import ModelStatus
from src.detection.mock_detector import MockFoodDetector
from src.detection.yolo_detector import YOLO11FoodDetector
from src.freshness import EfficientNetFreshnessDetector, MockFreshnessDetector
from src.segmentation.mock_segmenter import MockFoodSegmenter
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter


class TestModelInterfaces(unittest.TestCase):
    def setUp(self):
        # Create a blank dummy image (100x100 RGB) for testing interface contracts
        self.dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_yolo_detector_reports_not_loaded_when_weights_missing(self):
        detector = YOLO11FoodDetector(model_path="non_existent_weights.pt")
        self.assertFalse(detector.is_ready())
        self.assertEqual(detector.status, ModelStatus.MODEL_NOT_LOADED)

        result = detector.predict(self.dummy_img)
        self.assertEqual(result.status, ModelStatus.MODEL_NOT_LOADED)
        self.assertEqual(len(result.detections), 0)
        self.assertIn("not found", result.error_message.lower())

    def test_mock_detector_reports_not_trained(self):
        mock_det = MockFoodDetector()
        self.assertFalse(mock_det.is_ready())
        self.assertEqual(mock_det.status, ModelStatus.MODEL_NOT_TRAINED)

        result = mock_det.predict(self.dummy_img)
        self.assertEqual(result.status, ModelStatus.MODEL_NOT_TRAINED)
        self.assertEqual(len(result.detections), 0)
        self.assertIn("MODEL_NOT_TRAINED", result.error_message)

    def test_sam2_segmenter_reports_not_loaded_when_checkpoint_missing(self):
        segmenter = SAM2FoodSegmenter(checkpoint_path="non_existent_sam2.pt")
        self.assertFalse(segmenter.is_ready())
        self.assertEqual(segmenter.status, ModelStatus.MODEL_NOT_LOADED)

        result = segmenter.predict_masks(self.dummy_img)
        self.assertEqual(result.status, ModelStatus.MODEL_NOT_LOADED)
        self.assertEqual(len(result.masks), 0)

    def test_mock_segmenter_reports_not_trained(self):
        mock_seg = MockFoodSegmenter()
        self.assertFalse(mock_seg.is_ready())
        self.assertEqual(mock_seg.status, ModelStatus.MODEL_NOT_TRAINED)

        result = mock_seg.predict_masks(self.dummy_img)
        self.assertEqual(result.status, ModelStatus.MODEL_NOT_TRAINED)
        self.assertEqual(len(result.masks), 0)

    def test_classification_reports_not_loaded(self):
        classifier = EfficientNetB4Classifier(model_path="non_existent_effnet.pth")
        self.assertFalse(classifier.is_ready())
        self.assertEqual(classifier.status, ModelStatus.MODEL_NOT_TRAINED)

        result = classifier.classify(self.dummy_img)
        self.assertEqual(result.status, ModelStatus.MODEL_NOT_TRAINED)
        self.assertIsNone(result.result.class_id)

    def test_depth_reports_not_loaded(self):
        depth = DepthAnythingV2Estimator(model_path="non_existent_depth.pth")
        self.assertFalse(depth.is_ready())
        self.assertEqual(depth.status, ModelStatus.MODEL_NOT_LOADED)

        result = depth.estimate_depth(self.dummy_img)
        self.assertEqual(result.status, ModelStatus.MODEL_NOT_LOADED)
        self.assertIsNone(result.depth_map)

    def test_freshness_reports_not_loaded(self):
        freshness = EfficientNetFreshnessDetector(model_path="non_existent_fresh.pth")
        self.assertFalse(freshness.is_ready())
        self.assertEqual(freshness.status, ModelStatus.MODEL_NOT_TRAINED)

        result = freshness.assess_freshness(self.dummy_img)
        self.assertEqual(result.status, ModelStatus.MODEL_NOT_TRAINED)
        self.assertEqual(result.evaluation.state.value, "unknown")


if __name__ == "__main__":
    unittest.main()
