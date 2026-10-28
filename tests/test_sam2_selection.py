"""
Unit tests for SAM 2 candidate mask selection strategy.
Phase 3: SAM 2 Food Instance Segmentation
"""

import unittest
import numpy as np

from src.segmentation.mask_selection import (
    compute_box_iou,
    compute_mask_tight_bbox,
    evaluate_candidate_mask,
    select_best_mask,
)


class TestSAM2MaskSelection(unittest.TestCase):
    def test_compute_box_iou(self):
        # Identical boxes
        box_a = [10.0, 10.0, 50.0, 50.0]
        self.assertAlmostEqual(compute_box_iou(box_a, box_a), 1.0)

        # Non-overlapping boxes
        box_b = [100.0, 100.0, 150.0, 150.0]
        self.assertAlmostEqual(compute_box_iou(box_a, box_b), 0.0)

        # 50% overlap box
        box_c = [10.0, 10.0, 30.0, 50.0]
        iou = compute_box_iou(box_a, box_c)
        self.assertAlmostEqual(iou, 0.5)

    def test_evaluate_candidate_mask_containment(self):
        # Image shape 100x100
        # Prompt box: [20, 20, 60, 60] (area = 1600)
        prompt_box = [20.0, 20.0, 60.0, 60.0]

        # Case 1: Mask perfectly inside box
        mask_good = np.zeros((100, 100), dtype=bool)
        mask_good[25:55, 25:55] = True
        res_good = evaluate_candidate_mask(
            mask=mask_good,
            prompt_bbox=prompt_box,
            image_shape=(100, 100),
            quality_score=0.95,
        )
        self.assertTrue(res_good["is_valid"])
        self.assertAlmostEqual(res_good["inside_ratio"], 1.0)
        self.assertGreater(res_good["composite_score"], 0.70)

        # Case 2: Mask leaking heavily into background outside box
        mask_leaking = np.zeros((100, 100), dtype=bool)
        mask_leaking[10:90, 10:90] = True  # Leaks far outside [20, 20, 60, 60]
        res_leaking = evaluate_candidate_mask(
            mask=mask_leaking,
            prompt_bbox=prompt_box,
            image_shape=(100, 100),
            quality_score=0.95,
        )
        self.assertTrue(res_leaking["is_valid"])
        self.assertLess(res_leaking["inside_ratio"], 0.40)
        # Good mask must score much higher than leaking mask
        self.assertGreater(res_good["composite_score"], res_leaking["composite_score"])

    def test_filter_mask_too_small(self):
        # 10 pixels < min_area 50
        mask_tiny = np.zeros((100, 100), dtype=bool)
        mask_tiny[10:12, 10:15] = True  # 10 pixels
        res = evaluate_candidate_mask(
            mask=mask_tiny,
            prompt_bbox=[10.0, 10.0, 30.0, 30.0],
            image_shape=(100, 100),
            min_area=50,
        )
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["failure_reason"], "MASK_TOO_SMALL")

    def test_filter_runaway_mask(self):
        # Mask covers almost entire 100x100 image (9800 pixels) when prompt is small
        mask_huge = np.ones((100, 100), dtype=bool)
        res = evaluate_candidate_mask(
            mask=mask_huge,
            prompt_bbox=[10.0, 10.0, 20.0, 20.0],
            image_shape=(100, 100),
        )
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["failure_reason"], "RUNAWAY_MASK")

    def test_deterministic_best_mask_selection(self):
        prompt_box = [20.0, 20.0, 60.0, 60.0]

        # Candidate 0: Too small (invalid)
        c0 = np.zeros((100, 100), dtype=bool)
        c0[20:22, 20:22] = True

        # Candidate 1: Moderate leak
        c1 = np.zeros((100, 100), dtype=bool)
        c1[15:75, 15:75] = True

        # Candidate 2: High fidelity, tight inside prompt
        c2 = np.zeros((100, 100), dtype=bool)
        c2[22:58, 22:58] = True

        candidates = [c0, c1, c2]
        scores = [0.80, 0.90, 0.95]

        best_m, best_score, meta = select_best_mask(
            candidate_masks=candidates,
            candidate_scores=scores,
            prompt_bbox=prompt_box,
            image_shape=(100, 100),
            min_mask_area=50,
        )

        self.assertIsNotNone(best_m)
        # Must pick candidate 2
        self.assertEqual(meta["candidate_index"], 2)
        self.assertGreater(best_score, 0.70)


if __name__ == "__main__":
    unittest.main()
