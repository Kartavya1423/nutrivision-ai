"""
NutriVision AI — Phase 4 EfficientNet-B4 Unit & Integration Tests
Validates classifier configuration, class mapping, dataset loading, preprocessing,
model creation, missing checkpoint handling, output schemas, Top-K extraction,
and low-confidence handling.
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch

from src.classification.classifier import (
    BaseFoodClassifier,
    ClassificationOutput,
    ClassificationResult,
    ClassificationStatus,
)
from src.classification.efficientnet_classifier import EfficientNetB4Classifier
from src.detection.detector import ModelStatus


# =====================================================================
# UNIT TESTS (No trained weights or heavy hardware required)
# =====================================================================

class TestEfficientNetUnit(unittest.TestCase):
    """Unit tests for Phase 4 classifier schema, preprocessing, and error handling."""

    def test_missing_checkpoint_reports_not_trained(self):
        """Missing checkpoint must report MODEL_NOT_TRAINED and is_ready == False."""
        classifier = EfficientNetB4Classifier(model_path="non_existent_weights.pth")
        self.assertFalse(classifier.is_ready())
        self.assertEqual(classifier.status, ModelStatus.MODEL_NOT_TRAINED)

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = classifier.classify(dummy_img)
        self.assertFalse(res.is_success)
        self.assertEqual(res.status, ModelStatus.MODEL_NOT_TRAINED)
        self.assertEqual(res.confidence_status, ClassificationStatus.CLASSIFICATION_MODEL_NOT_READY.value)
        self.assertIsNone(res.result.class_id)
        self.assertIsNone(res.result.class_name)

    def test_classification_output_and_result_schemas(self):
        """Validates ClassificationOutput and ClassificationResult dictionary structures."""
        out = ClassificationOutput(
            class_id=12,
            class_name="sushi",
            confidence=0.91234,
            top_k_candidates=[
                {"rank": 1, "class_id": 12, "class_name": "sushi", "confidence": 0.9123},
                {"rank": 2, "class_id": 8, "class_name": "rice", "confidence": 0.0512},
            ],
        )
        d = out.to_dict()
        self.assertEqual(d["class_id"], 12)
        self.assertEqual(d["class_name"], "sushi")
        self.assertEqual(d["confidence"], 0.9123)
        self.assertEqual(len(d["top_k_candidates"]), 2)

        res = ClassificationResult(
            status=ModelStatus.READY,
            result=out,
            inference_time_ms=18.5,
            model_name="EfficientNetB4Classifier",
            confidence_status=ClassificationStatus.CLASSIFICATION_SUCCESS.value,
            breakdown_ms={"preprocess_ms": 2.1, "forward_ms": 15.2, "postprocess_ms": 1.2, "total_ms": 18.5},
        )
        rd = res.to_dict()
        self.assertEqual(rd["status"], "READY")
        self.assertEqual(rd["confidence_status"], "CLASSIFICATION_SUCCESS")
        self.assertEqual(rd["inference_time_ms"], 18.5)
        self.assertIn("breakdown_ms", rd)
        self.assertEqual(rd["breakdown_ms"]["forward_ms"], 15.2)

    def test_rgba_preprocessing_compositing(self):
        """Verifies RGBA transparency is safely composited to 3-channel RGB tensor."""
        classifier = EfficientNetB4Classifier(model_path="non_existent.pth", image_size=224)
        # Create a 4-channel RGBA image with alpha channel
        rgba_img = np.zeros((80, 80, 4), dtype=np.uint8)
        rgba_img[:, :, 0] = 200  # Red
        rgba_img[:, :, 3] = 128  # 50% opacity
        pil_rgba = Image.fromarray(rgba_img, mode="RGBA")

        tensor = classifier._preprocess_crop(pil_rgba)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_numpy_rgb_preprocessing(self):
        """Verifies numpy RGB array is safely converted and normalized to (1, 3, H, W)."""
        classifier = EfficientNetB4Classifier(model_path="non_existent.pth", image_size=224)
        rgb_arr = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)

        tensor = classifier._preprocess_crop(rgb_arr)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_mock_checkpoint_loading_and_inference(self):
        """Builds a lightweight dummy checkpoint to verify weights loading, inference, and Top-K extraction."""
        import torchvision.models as models
        import torch.nn as nn

        num_classes = 5
        class_names = ["apple", "banana", "chicken", "rice", "salad"]
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        base = models.efficientnet_b4(weights=None)
        in_f = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_f, num_classes)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            ckpt = {
                "model_state_dict": base.state_dict(),
                "class_names": class_names,
                "class_to_idx": class_to_idx,
                "num_classes": num_classes,
                "image_size": 224,
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            }
            torch.save(ckpt, tmp_path)

            classifier = EfficientNetB4Classifier(model_path=tmp_path, confidence_threshold=0.10)
            self.assertTrue(classifier.is_ready())
            self.assertEqual(classifier.num_classes, 5)
            self.assertEqual(classifier.class_names, class_names)

            # Test inference on synthetic crop
            test_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            res = classifier.classify(test_crop, top_k=3)

            self.assertTrue(res.is_success)
            self.assertEqual(res.status, ModelStatus.READY)
            self.assertIn(res.result.class_name, class_names)
            self.assertGreaterEqual(res.result.confidence, 0.0)
            self.assertEqual(len(res.result.top_k_candidates), 3)
            self.assertIsNotNone(res.breakdown_ms)

            # Test Grad-CAM layer hooks
            target_layer = classifier.get_target_layer()
            self.assertIsNotNone(target_layer)
            hooks_dict = classifier.register_gradcam_hooks()
            self.assertIn("handles", hooks_dict)
            self.assertEqual(len(hooks_dict["handles"]), 2)
            for h in hooks_dict["handles"]:
                h.remove()

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_low_confidence_flagging(self):
        """Verifies that low prediction confidence sets CLASSIFICATION_LOW_CONFIDENCE."""
        import torchvision.models as models
        import torch.nn as nn

        num_classes = 3
        class_names = ["c1", "c2", "c3"]
        base = models.efficientnet_b4(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            torch.save({
                "model_state_dict": base.state_dict(),
                "class_names": class_names,
                "class_to_idx": {c: i for i, c in enumerate(class_names)},
                "num_classes": 3,
                "image_size": 224,
            }, tmp_path)

            # Set an impossibly high threshold (0.999) to force low-confidence
            classifier = EfficientNetB4Classifier(model_path=tmp_path, confidence_threshold=0.999)
            res = classifier.classify(np.zeros((100, 100, 3), dtype=np.uint8))
            self.assertEqual(res.confidence_status, ClassificationStatus.CLASSIFICATION_LOW_CONFIDENCE.value)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


# =====================================================================
# INTEGRATION TESTS
# =====================================================================

class TestClassificationIntegration(unittest.TestCase):
    """Integration tests verifying dataset manifests and pipeline contracts."""

    def test_classification_class_mapping_csv_exists(self):
        """Ensures the canonical classification class mapping CSV is populated and valid."""
        csv_path = Path("data/processed/classification/classification_class_mapping.csv")
        self.assertTrue(csv_path.is_file(), f"Missing class mapping: {csv_path}")

        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 198, f"Expected 198 classes, found {len(reader)}")
            # Verify columns
            required_cols = {"class_id", "class_name", "category", "train_count", "val_count", "test_count"}
            self.assertTrue(required_cols.issubset(set(reader[0].keys())))

    def test_classification_split_manifest_exists(self):
        """Ensures the classification split manifest index is populated and valid."""
        manifest_path = Path("data/processed/classification/classification_split_manifest.json")
        self.assertTrue(manifest_path.is_file(), f"Missing split manifest: {manifest_path}")

        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            self.assertIn("splits", manifest)
            self.assertIn("train", manifest["splits"])
            self.assertIn("val", manifest["splits"])
            self.assertIn("test", manifest["splits"])
            self.assertGreater(manifest["splits"]["train"], 1000)
            self.assertGreater(manifest["splits"]["val"], 500)
            self.assertGreater(manifest["splits"]["test"], 500)

    def test_zero_cross_split_leakage(self):
        """Verifies the leakage report confirms zero overlap between train, val, and test."""
        leak_report = Path("data/processed/classification/classification_leakage_report.json")
        self.assertTrue(leak_report.is_file(), f"Missing leakage report: {leak_report}")

        import json
        with open(leak_report, "r", encoding="utf-8") as f:
            rep = json.load(f)
            self.assertEqual(rep["status"], "PASS")
            self.assertEqual(rep["train_val_overlap"], 0)
            self.assertEqual(rep["train_test_overlap"], 0)
            self.assertEqual(rep["val_test_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
