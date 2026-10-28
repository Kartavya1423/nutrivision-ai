"""
Unit tests for weight estimation and density database (src.weight).
Formula: weight = volume × density.
Verifies DENSITY_NOT_AVAILABLE and distinction between estimated and actual measured weight.
"""

import unittest
from src.volume.volume_estimator import VolumeEstimateOutput
from src.weight.density_database import DensityDatabase
from src.weight.weight_estimator import WeightEstimator


class TestWeightCalculation(unittest.TestCase):
    def setUp(self):
        self.density_db = DensityDatabase()
        self.estimator = WeightEstimator(density_db=self.density_db)

    def test_density_database_lookup(self):
        # Known food: cooked rice
        res = self.density_db.get_density("rice")
        self.assertEqual(res["status"], "AVAILABLE")
        self.assertGreater(res["density"], 0.0)

    def test_unlisted_food_returns_density_not_available(self):
        # Unlisted food item must return DENSITY_NOT_AVAILABLE
        res = self.density_db.get_density("unknown_alien_food_xyz")
        self.assertEqual(res["status"], "DENSITY_NOT_AVAILABLE")
        self.assertIsNone(res["density"])

    def test_weight_formula_with_valid_volume(self):
        # 100 cm³ of rice at 0.85 g/cm³ = 85.0 g
        vol_out = VolumeEstimateOutput(volume=100.0, unit="cm3", status="READY")
        item_wt = self.estimator.estimate_weight(
            class_name="rice",
            volume_output=vol_out,
            actual_measured_weight_grams=87.5,
        )

        d = item_wt.to_dict()
        self.assertEqual(d["status"], "READY")
        self.assertEqual(d["estimated_weight_grams"], 85.0)
        self.assertEqual(d["actual_measured_weight_grams"], 87.5)
        self.assertTrue(d["is_estimated"])

    def test_uncalibrated_volume_returns_volume_status(self):
        # If volume is CALIBRATION_REQUIRED, weight cannot be computed
        vol_out = VolumeEstimateOutput(volume=None, unit="cm3", status="CALIBRATION_REQUIRED")
        item_wt = self.estimator.estimate_weight(
            class_name="rice",
            volume_output=vol_out,
        )
        d = item_wt.to_dict()
        self.assertIsNone(d["estimated_weight_grams"])
        self.assertEqual(d["status"], "CALIBRATION_REQUIRED")


if __name__ == "__main__":
    unittest.main()
