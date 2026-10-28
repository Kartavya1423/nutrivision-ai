"""
NutriVision AI — Food Freshness Detection Unit & Integration Tests (Phase 6)
Covers:
- Dataset loading and class mapping schema
- Data leakage detection
- EfficientNet model architecture and checkpoint serialization
- Single-image inference and probability normalization
- Low-confidence and edge-case handling
- End-to-end integration: YOLO11 -> SAM2 -> Food-only Crop -> Freshness
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from src.config.settings import get_settings
from src.detection.detector import ModelStatus
from src.freshness.freshness_detector import (
    BaseFreshnessDetector,
    EfficientNetFreshnessDetector,
    EfficientNetFreshnessModel,
    FreshnessEvaluation,
    FreshnessResult,
    FreshnessState,
)
from src.freshness.mock_freshness import MockFreshnessDetector
from src.pipeline.freshness_pipeline import FreshnessPipeline


class TestFreshnessDataset(unittest.TestCase):
    """Tests dataset directory structure, label schemas, and leakage detection."""

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.root = cls.settings.root_dir

    def test_freshness_classes_schema(self):
        schema_path = self.root / "models" / "freshness" / "freshness_classes.json"
        self.assertTrue(schema_path.is_file(), f"Missing schema at {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("classes", data)
        self.assertEqual(data["classes"]["0"], "fresh")
        self.assertEqual(data["classes"]["1"], "rotten")
        self.assertIn("subclasses", data)
        self.assertIn("freshapples", data["subclasses"])
        self.assertIn("rottenbanana", data["subclasses"])

    def test_dataset_audit_report_exists(self):
        audit_path = self.root / "outputs" / "freshness" / "dataset_audit_report.json"
        self.assertTrue(audit_path.is_file(), f"Missing audit report at {audit_path}")

        with open(audit_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["number_of_freshness_classes"], 2)
        self.assertEqual(report["number_of_food_classes"], 3)
        self.assertEqual(report["total_images"], 1200)
        self.assertEqual(report["split_sizes"]["train"], 840)
        self.assertEqual(report["split_sizes"]["validation"], 180)
        self.assertEqual(report["split_sizes"]["test"], 180)

    def test_data_leakage_report(self):
        leak_path = self.root / "outputs" / "freshness" / "data_leakage_report.json"
        self.assertTrue(leak_path.is_file(), f"Missing leakage report at {leak_path}")

        with open(leak_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertFalse(data["leakage_detected"], "Data leakage must NOT be detected across splits.")
        self.assertEqual(data["train_test_overlap_count"], 0)
        self.assertEqual(data["train_val_overlap_count"], 0)


class TestFreshnessModelArchitecture(unittest.TestCase):
    """Tests EfficientNet-B4 freshness model instantiation and inference shape."""

    def test_model_forward_shape(self):
        model = EfficientNetFreshnessModel(num_classes=2, pretrained=False)
        model.eval()

        dummy_tensor = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_tensor)

        self.assertEqual(output.shape, (2, 2))

    def test_gradcam_target_layer_exposed(self):
        model = EfficientNetFreshnessModel(num_classes=2, pretrained=False)
        target_layer = model.get_gradcam_target_layer()

        self.assertIsNotNone(target_layer, "Grad-CAM target layer must be exposed.")
        self.assertIsInstance(target_layer, torch.nn.Module)

    def test_checkpoint_saving_and_loading(self):
        model = EfficientNetFreshnessModel(num_classes=2, pretrained=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_freshness.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": 2,
                "image_size": 224,
                "class_mapping": {0: "fresh", 1: "rotten"},
            }, save_path)

            self.assertTrue(save_path.is_file())

            # Load into detector
            detector = EfficientNetFreshnessDetector(model_path=save_path)
            self.assertTrue(detector.is_ready())
            self.assertEqual(detector.status, ModelStatus.READY)


class TestFreshnessInference(unittest.TestCase):
    """Tests single-image inference, edge cases, and confidence thresholds."""

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.detector = EfficientNetFreshnessDetector(settings=cls.settings)
        # 100x100 RGB synthetic sample
        arr = np.random.randint(50, 220, (100, 100, 3), dtype=np.uint8)
        cls.test_img = Image.fromarray(arr)

    def test_inference_returns_valid_structure(self):
        if not self.detector.is_ready():
            self.skipTest("Trained checkpoint not yet finalized.")

        res = self.detector.assess_freshness(self.test_img, class_name="apple")
        self.assertTrue(res.is_success)
        self.assertEqual(res.status, ModelStatus.READY)
        self.assertIsNotNone(res.evaluation)
        self.assertIn("fresh", res.evaluation.probabilities)
        self.assertIn("rotten", res.evaluation.probabilities)
        # Probabilities sum to 1.0
        prob_sum = res.evaluation.probabilities["fresh"] + res.evaluation.probabilities["rotten"]
        self.assertAlmostEqual(prob_sum, 1.0, places=3)

    def test_missing_model_checkpoint_returns_not_trained(self):
        bad_detector = EfficientNetFreshnessDetector(model_path="non_existent_model_123.pth")
        self.assertFalse(bad_detector.is_ready())
        self.assertEqual(bad_detector.status, ModelStatus.MODEL_NOT_TRAINED)

        res = bad_detector.assess_freshness(self.test_img)
        self.assertEqual(res.status, ModelStatus.MODEL_NOT_TRAINED)
        self.assertEqual(res.evaluation.state, FreshnessState.UNKNOWN)

    def test_invalid_image_input(self):
        if not self.detector.is_ready():
            self.skipTest("Trained checkpoint not yet finalized.")

        res = self.detector.assess_freshness(np.array([]))
        self.assertEqual(res.status, ModelStatus.ERROR)
        self.assertIn("FOOD_CROP_UNAVAILABLE", str(res.error_message))

    def test_low_confidence_threshold_handling(self):
        # Force an impossibly high threshold (0.999) to verify LOW_CONFIDENCE behavior
        strict_detector = EfficientNetFreshnessDetector(
            settings=self.settings,
            confidence_threshold=0.999,
        )
        if not strict_detector.is_ready():
            self.skipTest("Trained checkpoint not yet finalized.")

        res = strict_detector.assess_freshness(self.test_img)
        self.assertTrue(res.is_success)
        self.assertEqual(res.evaluation.state, FreshnessState.UNKNOWN)
        self.assertIn("LOW_CONFIDENCE", res.evaluation.notes)


class TestFreshnessPipelineIntegration(unittest.TestCase):
    """Tests end-to-end YOLO11 -> SAM2 -> Food Crop -> Freshness pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()

    def test_mock_detector_interface(self):
        mock = MockFreshnessDetector()
        self.assertFalse(mock.is_ready())
        self.assertEqual(mock.status, ModelStatus.MODEL_NOT_TRAINED)

        dummy_img = Image.new("RGB", (64, 64), (100, 150, 200))
        res = mock.assess_freshness(dummy_img)
        self.assertEqual(res.status, ModelStatus.MODEL_NOT_TRAINED)
        self.assertEqual(res.evaluation.state, FreshnessState.UNKNOWN)

    def test_pipeline_instantiation(self):
        pipeline = FreshnessPipeline(settings=self.settings)
        self.assertIsNotNone(pipeline.detector)
        self.assertIsNotNone(pipeline.segmenter)
        self.assertIsNotNone(pipeline.freshness_detector)

    def test_food_crop_background_masking(self):
        pipeline = FreshnessPipeline(settings=self.settings)
        raw_img = np.full((100, 100, 3), 50, dtype=np.uint8)  # Dark background
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True  # Center food item
        raw_img[mask] = [200, 30, 30]  # Red food

        masked_crop, rgba_crop = pipeline.extract_food_only_crop(raw_img, [20, 20, 80, 80], mask)

        self.assertEqual(masked_crop.shape, (60, 60, 3))
        self.assertEqual(rgba_crop.shape, (60, 60, 4))
        # Non-food corner in mask should be white (255, 255, 255)
        # Check center is red
        self.assertEqual(masked_crop[30, 30, 0], 200)


if __name__ == "__main__":
    unittest.main()
