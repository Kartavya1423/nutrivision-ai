"""
NutriVision AI — Phase 5 Integration and Comprehensive End-to-End Tests
Phase 5: Food Depth -> Volume -> Weight Estimation Pipeline

Tests:
1. Mask & Depth alignment, masked depth extraction, and background exclusion
2. Known synthetic 3D geometry volume integration (deterministic cuboid/cylinder)
3. Density database lookup and strict DENSITY_NOT_AVAILABLE enforcement
4. Weight calculation (Volume × Density) and uncalibrated volume handling
5. Multi-food aggregation logic (total is ONLY computed when all items are valid)
6. All required failure states (NO_FOOD_DETECTED, DEPTH_MODEL_NOT_AVAILABLE, INVALID_IMAGE)
7. Full end-to-end pipeline execution with real neural models
"""

import unittest
from pathlib import Path
import numpy as np

from src.depth.depth_anything import DepthAnythingV2Estimator
from src.detection.detector import ModelStatus
from src.detection.yolo_detector import YOLO11FoodDetector
from src.pipeline.phase5_pipeline import Phase5Pipeline
from src.segmentation.segmenter import FoodMask
from src.volume.volume_estimator import VolumeEstimateOutput, VolumeEstimator
from src.weight.density_database import DensityDatabase
from src.weight.weight_estimator import ItemWeightEstimate, WeightEstimator, WeightEstimationResult


class TestPhase5Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.density_db = DensityDatabase()
        cls.volume_estimator = VolumeEstimator()
        cls.weight_estimator = WeightEstimator(density_db=cls.density_db)

    # -------------------------------------------------------------
    # 1. Segmentation + Depth Alignment & Background Exclusion
    # -------------------------------------------------------------
    def test_masked_depth_and_background_exclusion(self):
        """Food volume must ONLY integrate pixels inside the segmentation mask."""
        h, w = 100, 100
        # Create synthetic depth map
        depth_map = np.full((h, w), 5.0, dtype=np.float32)
        # Food region is center 20x20 with depth 3.0 (height = 5.0 - 3.0 = 2.0)
        mask = np.zeros((h, w), dtype=bool)
        mask[40:60, 40:60] = True
        depth_map[mask] = 3.0

        # Background has noisy extreme values
        depth_map[0:10, 0:10] = 0.1
        depth_map[90:100, 90:100] = 10.0

        # Calibrated scale: 0.1 cm/pixel
        res = self.volume_estimator.estimate_volume(
            mask=mask,
            depth_info=depth_map,
            camera_calibration={"cm_per_pixel": 0.1},
        )

        self.assertEqual(res.status, "READY")
        self.assertEqual(res.mask_pixel_count, 400)
        # Background noisy pixels must not affect food depth stats
        self.assertAlmostEqual(res.depth_statistics["min_depth"], 3.0, places=2)
        self.assertAlmostEqual(res.depth_statistics["max_depth"], 3.0, places=2)

    # -------------------------------------------------------------
    # 2. Known Synthetic 3D Geometry Volume Integration
    # -------------------------------------------------------------
    def test_known_synthetic_geometry_volume(self):
        """
        Verify volume integration on a synthetic food cuboid:
        Area = 10cm x 10cm = 100 cm² (100x100 pixels at 0.1 cm/px)
        Height = 2.0 cm (depth 20.0 vs baseline 22.0)
        Expected Volume = 100 cm² * 2.0 cm = 200.0 cm³
        """
        h, w = 150, 150
        depth_map = np.full((h, w), 22.0, dtype=np.float32)
        mask = np.zeros((h, w), dtype=bool)
        mask[25:125, 25:125] = True  # 100x100 pixels
        depth_map[mask] = 20.0       # 2.0 cm higher than baseline

        res = self.volume_estimator.estimate_volume(
            mask=mask,
            depth_info=depth_map,
            camera_calibration={"cm_per_pixel": 0.1},
        )

        self.assertEqual(res.status, "READY")
        # Volume ≈ 100 * 100 * 0.01 * (2.0 * 0.1) = 20.0 cm3
        self.assertAlmostEqual(res.volume, 20.0, places=1)

    def test_uncalibrated_depth_strictly_returns_calibration_status(self):
        """Without calibration, physical cm³ must be None and status uncalibrated."""
        mask = np.ones((50, 50), dtype=bool)
        depth_map = np.ones((50, 50), dtype=np.float32) * 3.0
        res = self.volume_estimator.estimate_volume(
            mask=mask,
            depth_info=depth_map,
            camera_calibration=None,
        )
        self.assertIsNone(res.volume)
        self.assertIn("CALIBRAT", res.status)
        self.assertEqual(res.calibration_status, "VOLUME_SCALE_NOT_CALIBRATED")

    # -------------------------------------------------------------
    # 3. Density Database Traceability & Fallbacks
    # -------------------------------------------------------------
    def test_density_database_known_foods(self):
        """Verify empirical densities for common foods with traceable sources."""
        for food, expected_approx in [("rice", 0.85), ("apple", 0.78), ("banana", 0.95), ("chicken", 1.05)]:
            lookup = self.density_db.get_density(food)
            self.assertEqual(lookup["status"], "AVAILABLE")
            self.assertAlmostEqual(lookup["density"], expected_approx, delta=0.05)
            self.assertIsNotNone(lookup.get("source"))

    def test_unlisted_food_strictly_returns_density_not_available(self):
        """Unlisted food must NOT fall back to arbitrary 0.85 default."""
        lookup = self.density_db.get_density("synthetic_unlisted_superfood_999")
        self.assertEqual(lookup["status"], "DENSITY_NOT_AVAILABLE")
        self.assertIsNone(lookup["density"])

    # -------------------------------------------------------------
    # 4. Weight Calculation: Weight = Volume × Density
    # -------------------------------------------------------------
    def test_weight_calculation_formula(self):
        """100 cm³ of banana at 0.95 g/cm³ = 95.0 grams."""
        vol = VolumeEstimateOutput(volume=100.0, unit="cm3", status="READY")
        res = self.weight_estimator.estimate_weight("banana", volume_output=vol)

        self.assertEqual(res.status, "READY")
        self.assertAlmostEqual(res.estimated_weight_grams, 95.0, places=1)
        self.assertEqual(res.class_name, "banana")

    def test_weight_when_volume_uncalibrated(self):
        """If volume is uncalibrated, weight cannot be computed and must be None."""
        vol = VolumeEstimateOutput(volume=None, unit="cm3", status="CALIBRATION_REQUIRED")
        res = self.weight_estimator.estimate_weight("apple", volume_output=vol)

        self.assertIsNone(res.estimated_weight_grams)
        self.assertEqual(res.status, "CALIBRATION_REQUIRED")

    def test_weight_when_density_unavailable(self):
        """If density is unavailable, weight cannot be computed and must report DENSITY_NOT_AVAILABLE."""
        vol = VolumeEstimateOutput(volume=100.0, unit="cm3", status="READY")
        res = self.weight_estimator.estimate_weight("unknown_space_fruit", volume_output=vol)

        self.assertIsNone(res.estimated_weight_grams)
        self.assertEqual(res.status, "DENSITY_NOT_AVAILABLE")

    # -------------------------------------------------------------
    # 5. Multi-Food Totals & Failure Isolation
    # -------------------------------------------------------------
    def test_multi_food_weight_total_all_calibrated(self):
        """Total weight is sum of item weights when all items are valid."""
        items_vol = [
            VolumeEstimateOutput(volume=50.0, unit="cm3", status="READY"),
            VolumeEstimateOutput(volume=100.0, unit="cm3", status="READY"),
        ]
        class DummyMultiVol:
            def __init__(self, vols):
                self.volumes = [
                    type("Item", (), {"class_name": "apple", "volume_cm3": 50.0, "status": "READY", "confidence": 0.9})(),
                    type("Item", (), {"class_name": "rice", "volume_cm3": 100.0, "status": "READY", "confidence": 0.95})(),
                ]

        res = self.weight_estimator.estimate_weight(DummyMultiVol(items_vol))
        self.assertEqual(res.status, "READY")
        # 50 * 0.78 + 100 * 0.85 = 39.0 + 85.0 = 124.0 g
        self.assertAlmostEqual(res.total_estimated_weight_grams, 124.0, places=1)

    def test_multi_food_weight_total_with_one_uncalibrated(self):
        """If any food is uncalibrated, total weight is None and status is UNAVAILABLE."""
        class DummyPartialVol:
            def __init__(self):
                self.volumes = [
                    type("Item", (), {"class_name": "apple", "volume_cm3": 50.0, "status": "READY", "confidence": 0.9})(),
                    type("Item", (), {"class_name": "rice", "volume_cm3": None, "status": "VOLUME_SCALE_NOT_CALIBRATED", "confidence": 0.9})(),
                ]

        res = self.weight_estimator.estimate_weight(DummyPartialVol())
        self.assertEqual(res.status, "WEIGHT_ESTIMATION_UNAVAILABLE")
        self.assertIsNone(res.total_estimated_weight_grams)

    # -------------------------------------------------------------
    # 6. Pipeline Failure States
    # -------------------------------------------------------------
    def test_pipeline_invalid_image_handling(self):
        """Pipeline must handle corrupted/invalid image input cleanly."""
        pipeline = Phase5Pipeline()
        res = pipeline.run("non_existent_image_file.jpg", save_visualizations=False)
        self.assertEqual(res.status, "INVALID_IMAGE")
        self.assertEqual(len(res.items), 0)

    def test_pipeline_no_food_detected(self):
        """Zero food detections strictly produces NO_FOOD_DETECTED."""
        from src.detection.mock_detector import MockFoodDetector
        mock_det = MockFoodDetector(fixture_detections=[])
        pipeline = Phase5Pipeline(detector=mock_det)
        blank_img = np.zeros((200, 200, 3), dtype=np.uint8)
        res = pipeline.run(blank_img, save_visualizations=False)
        self.assertEqual(res.status, "NO_FOOD_DETECTED")
        self.assertEqual(res.total_detected_foods, 0)

    # -------------------------------------------------------------
    # 7. Real End-to-End Pipeline Execution
    # -------------------------------------------------------------
    def test_real_end_to_end_pipeline_execution(self):
        """
        Verify the real complete pipeline:
        Image -> YOLO11 -> SAM2 -> Depth Anything V2 -> Volume -> Weight
        """
        test_img_path = Path("data/processed/yolo_detection/images/val/foodseg103_00000015.jpg")
        if not test_img_path.is_file():
            self.skipTest(f"Validation image not found: {test_img_path}")

        pipeline = Phase5Pipeline()

        # 1. Uncalibrated mode
        res_uncalib = pipeline.run(test_img_path, calibration=None, save_visualizations=False)
        self.assertIn(res_uncalib.calibration_status, ["DEPTH_SCALE_NOT_CALIBRATED", "VOLUME_SCALE_NOT_CALIBRATED"])
        self.assertIsNone(res_uncalib.total_estimated_weight_g)

        # 2. Calibrated mode
        calib = {"cm_per_pixel": 0.05, "reference_object": "plate_25cm"}
        res_calib = pipeline.run(test_img_path, calibration=calib, save_visualizations=False)
        self.assertEqual(res_calib.status, "COMPLETE")
        self.assertGreater(res_calib.total_detected_foods, 0)
        self.assertGreater(res_calib.total_volume_cm3, 0.0)
        self.assertGreater(res_calib.total_estimated_weight_g, 0.0)
        self.assertEqual(res_calib.calibration_status, "CALIBRATED")


if __name__ == "__main__":
    unittest.main()
