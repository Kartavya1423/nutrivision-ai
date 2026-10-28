"""
Unit tests for 3D volume estimation schema (src.volume).
Enforces CALIBRATION_REQUIRED when depth is relative and camera/reference calibration is missing.
"""

import unittest
import numpy as np
from src.volume.volume_estimator import VolumeEstimateOutput, VolumeEstimator


class TestVolumeSchema(unittest.TestCase):
    def setUp(self):
        self.estimator = VolumeEstimator()
        self.mask = np.ones((50, 50), dtype=bool)
        self.depth_map = np.ones((50, 50), dtype=np.float32) * 2.0

    def test_uncalibrated_depth_returns_calibration_required(self):
        # Monocular depth without reference dimensions must report CALIBRATION_REQUIRED
        result = self.estimator.estimate_volume(
            mask=self.mask,
            depth_info=self.depth_map,
            camera_calibration=None,
            reference_info=None,
        )
        d = result.to_dict()
        self.assertIsNone(d["volume"])
        self.assertEqual(d["unit"], "cm3")
        self.assertEqual(d["status"], "CALIBRATION_REQUIRED")

    def test_missing_depth_reports_status(self):
        result = self.estimator.estimate_volume(
            mask=self.mask,
            depth_info=None,
        )
        d = result.to_dict()
        self.assertIsNone(d["volume"])
        self.assertEqual(d["status"], "DEPTH_NOT_AVAILABLE")

    def test_calibrated_volume_computation(self):
        # When calibration info is supplied (e.g. 0.05 cm per pixel)
        result = self.estimator.estimate_volume(
            mask=self.mask,
            depth_info=self.depth_map,
            camera_calibration={"cm_per_pixel": 0.05},
        )
        d = result.to_dict()
        self.assertEqual(d["unit"], "cm3")
        self.assertEqual(d["status"], "READY")


if __name__ == "__main__":
    unittest.main()
