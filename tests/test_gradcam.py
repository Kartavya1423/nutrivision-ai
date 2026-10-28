"""
NutriVision AI — Visual Explainability (Grad-CAM) Unit & Integration Tests (Phase 7)
Covers:
- Native PyTorch GradCAM calculation engine
- Target layer hook registration and gradient backpropagation
- Food classifier (EfficientNet-B4) Grad-CAM generation
- Freshness classifier (EfficientNetFreshnessModel) Grad-CAM generation
- Auto-predicted class vs explicit target class selection
- Heatmap normalization, bounds check [0, 1], and NaN/Inf prevention
- Colormap blending (JET, VIRIDIS) and alpha overlay generation
- 3-panel composite figure generation and artifact serialization
- Error handling: INVALID_IMAGE, INVALID_TARGET_CLASS, MODEL_NOT_READY
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from src.classification.efficientnet_classifier import EfficientNetB4Classifier
from src.config.settings import get_settings
from src.explainability.explainer import NutriVisionExplainer
from src.explainability.gradcam import GradCAM
from src.freshness.freshness_detector import EfficientNetFreshnessDetector


class TestGradCAMCore(unittest.TestCase):
    """Unit tests for native PyTorch GradCAM engine."""

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.food_cls = EfficientNetB4Classifier(settings=cls.settings)
        cls.fresh_cls = EfficientNetFreshnessDetector(settings=cls.settings)

        # Create synthetic 100x100 test crop image
        arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        cls.test_crop = Image.fromarray(arr)

    def test_gradcam_on_food_classifier(self):
        """Verify Grad-CAM produces normalized heatmap matching input dimensions on food classifier."""
        if not self.food_cls.is_ready():
            self.skipTest("EfficientNet-B4 classifier checkpoint not available.")

        target_layer = self.food_cls.get_target_layer()
        self.assertIsNotNone(target_layer)

        cam_engine = GradCAM(
            model=self.food_cls.model,
            target_layer=target_layer,
            device=self.food_cls.device,
        )

        res = cam_engine.generate(
            image=self.test_crop,
            class_names=self.food_cls.class_names,
            colormap="JET",
            alpha=0.5,
        )

        self.assertEqual(res["status"], "AVAILABLE")
        self.assertTrue(res["has_heatmap"])
        self.assertIsNone(res["error_code"])

        # Heatmap dimension check
        hm = res["heatmap_normalized"]
        self.assertEqual(hm.shape, (100, 100))
        self.assertEqual(hm.dtype, np.float32)

        # Bounds check [0.0, 1.0] and NaN check
        self.assertGreaterEqual(float(np.min(hm)), 0.0)
        self.assertLessEqual(float(np.max(hm)), 1.0)
        self.assertFalse(np.isnan(hm).any(), "Grad-CAM heatmap must not contain NaN values.")

        # Overlay check
        ov = res["overlay"]
        self.assertEqual(ov.shape, (100, 100, 3))
        self.assertEqual(ov.dtype, np.uint8)

        # 3-panel composite check
        comp = res["composite"]
        self.assertEqual(comp.shape[2], 3)
        self.assertGreater(comp.shape[1], 300)  # Width > 3x panel width

    def test_gradcam_on_freshness_classifier(self):
        """Verify Grad-CAM on Phase 6 fine-tuned freshness model."""
        if not self.fresh_cls.is_ready():
            self.skipTest("Freshness model checkpoint not available.")

        fr_layer = self.fresh_cls.model.get_gradcam_target_layer()
        self.assertIsNotNone(fr_layer)

        cam_engine = GradCAM(
            model=self.fresh_cls.model,
            target_layer=fr_layer,
            device=self.fresh_cls.device,
        )

        res = cam_engine.generate(
            image=self.test_crop,
            class_names=["fresh", "rotten"],
            colormap="VIRIDIS",
            alpha=0.6,
        )

        self.assertEqual(res["status"], "AVAILABLE")
        self.assertTrue(res["has_heatmap"])
        self.assertIn(res["predicted_class_name"], ["fresh", "rotten"])
        self.assertEqual(res["heatmap_normalized"].shape, (100, 100))

    def test_explicit_target_class(self):
        """Verify explicit target class can be chosen independently from predicted class."""
        if not self.fresh_cls.is_ready():
            self.skipTest("Freshness model checkpoint not available.")

        cam_engine = GradCAM(
            model=self.fresh_cls.model,
            target_layer=self.fresh_cls.model.get_gradcam_target_layer(),
            device=self.fresh_cls.device,
        )

        # Target explicitly Class 1 (rotten)
        res = cam_engine.generate(
            image=self.test_crop,
            target_class_id=1,
            class_names=["fresh", "rotten"],
        )

        self.assertEqual(res["status"], "AVAILABLE")
        self.assertEqual(res["target_class_id"], 1)
        self.assertEqual(res["target_class_name"], "rotten")

    def test_invalid_target_class_error(self):
        """Out of bounds class ID must return INVALID_TARGET_CLASS."""
        if not self.fresh_cls.is_ready():
            self.skipTest("Freshness model checkpoint not available.")

        cam_engine = GradCAM(
            model=self.fresh_cls.model,
            target_layer=self.fresh_cls.model.get_gradcam_target_layer(),
            device=self.fresh_cls.device,
        )

        res = cam_engine.generate(
            image=self.test_crop,
            target_class_id=999,  # Invalid for 2-class model
            class_names=["fresh", "rotten"],
        )

        self.assertEqual(res["status"], "GRADCAM_UNAVAILABLE")
        self.assertEqual(res["error_code"], "INVALID_TARGET_CLASS")

    def test_invalid_empty_image_error(self):
        """Empty image input must return INVALID_IMAGE."""
        if not self.food_cls.is_ready():
            self.skipTest("Classifier not available.")

        cam_engine = GradCAM(
            model=self.food_cls.model,
            target_layer=self.food_cls.get_target_layer(),
            device=self.food_cls.device,
        )

        res = cam_engine.generate(image=np.array([]))
        self.assertEqual(res["status"], "GRADCAM_UNAVAILABLE")
        self.assertEqual(res["error_code"], "INVALID_IMAGE")


class TestNutriVisionExplainer(unittest.TestCase):
    """Tests high-level explainer coordinator and artifact persistence."""

    @classmethod
    def setUpClass(cls):
        cls.settings = get_settings()
        cls.explainer = NutriVisionExplainer(settings=cls.settings)
        arr = np.random.randint(40, 220, (120, 120, 3), dtype=np.uint8)
        cls.test_img = Image.fromarray(arr)

    def test_explain_food_classification(self):
        res = self.explainer.explain_food_classification(self.test_img)
        self.assertEqual(res["status"], "AVAILABLE")
        self.assertIn("target_class_name", res)
        self.assertIn("confidence", res)

    def test_explain_freshness(self):
        res = self.explainer.explain_freshness(self.test_img)
        self.assertEqual(res["status"], "AVAILABLE")
        self.assertIn(res["predicted_class_name"], ["fresh", "rotten"])

    def test_save_artifacts(self):
        res = self.explainer.explain_freshness(self.test_img)
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = self.explainer.save_artifacts(res, output_dir=tmp_dir, file_prefix="test_item")
            self.assertIn("overlay_path", saved)
            self.assertIn("heatmap_path", saved)
            self.assertIn("composite_path", saved)
            self.assertIn("metadata_path", saved)

            for path_str in saved.values():
                self.assertTrue(Path(path_str).is_file())


if __name__ == "__main__":
    unittest.main()
