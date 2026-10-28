"""
Unit tests for food classification schema and contract (src.classification).
"""

import unittest
from src.classification.classifier import ClassificationOutput, ClassificationResult
from src.classification.efficientnet_classifier import EfficientNetB4Classifier
from src.classification.mock_classifier import MockFoodClassifier
from src.detection.detector import ModelStatus


class TestClassificationSchema(unittest.TestCase):
    def test_classification_schema_format(self):
        output = ClassificationOutput(
            class_id=42,
            class_name="grilled_salmon",
            confidence=0.96,
        )
        d = output.to_dict()
        self.assertEqual(d["class_id"], 42)
        self.assertEqual(d["class_name"], "grilled_salmon")
        self.assertEqual(d["confidence"], 0.96)

    def test_efficientnet_returns_not_trained(self):
        cls = EfficientNetB4Classifier(model_path="models/classification/unbuilt_model.pth")
        self.assertFalse(cls.is_ready())
        self.assertEqual(cls.status, ModelStatus.MODEL_NOT_TRAINED)

        dummy_img = [0]
        res = cls.classify(dummy_img)
        self.assertEqual(res.status, ModelStatus.MODEL_NOT_TRAINED)
        self.assertIsNone(res.result.class_id)
        self.assertIsNone(res.result.class_name)
        self.assertIsNone(res.result.confidence)

    def test_mock_classifier_returns_not_trained(self):
        mock_cls = MockFoodClassifier()
        self.assertEqual(mock_cls.status, ModelStatus.MODEL_NOT_TRAINED)
        res = mock_cls.classify("dummy")
        self.assertEqual(res.status, ModelStatus.MODEL_NOT_TRAINED)


if __name__ == "__main__":
    unittest.main()
