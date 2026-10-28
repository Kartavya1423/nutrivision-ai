"""
Unit tests for mask postprocessing utilities.
Phase 3: SAM 2 Food Instance Segmentation
"""

import unittest
import numpy as np

from src.segmentation.postprocessing import (
    clip_mask_to_image,
    constrain_mask_to_bbox,
    fill_mask_holes,
    postprocess_mask,
    remove_small_components,
)


class TestSAM2PostProcessing(unittest.TestCase):
    def test_clip_mask_to_image(self):
        mask = np.ones((80, 80), dtype=bool)
        clipped = clip_mask_to_image(mask, max_h=50, max_w=60)
        self.assertEqual(clipped.shape, (50, 60))
        self.assertTrue(clipped[0, 0])

    def test_remove_small_components(self):
        # Canvas 100x100
        mask = np.zeros((100, 100), dtype=bool)
        # Primary body: 40x40 = 1600 pixels
        mask[20:60, 20:60] = True
        # Tiny stray noise islands: 4 pixels, 8 pixels
        mask[2:4, 2:4] = True    # 4 pixels
        mask[90:92, 90:94] = True  # 8 pixels

        cleaned = remove_small_components(mask, min_size=32, keep_primary=True)

        # Main body must be preserved
        self.assertTrue(np.all(cleaned[20:60, 20:60]))
        # Stray noise must be eliminated
        self.assertFalse(np.any(cleaned[2:4, 2:4]))
        self.assertFalse(np.any(cleaned[90:92, 90:94]))

    def test_fill_mask_holes(self):
        # 50x50 solid rectangle with a 4x4 hole in the middle
        mask = np.zeros((60, 60), dtype=bool)
        mask[10:50, 10:50] = True
        mask[28:32, 28:32] = False  # 16-pixel hole

        self.assertFalse(mask[30, 30])
        filled = fill_mask_holes(mask, max_hole_area=30)
        self.assertTrue(filled[30, 30])
        # Outer boundary must not have expanded
        self.assertFalse(filled[5, 5])

    def test_constrain_mask_to_bbox(self):
        mask = np.zeros((100, 100), dtype=bool)
        # Food region inside box [20, 20, 50, 50]
        mask[20:50, 20:50] = True
        # Rogue leak far outside at [80, 80, 95, 95]
        mask[80:95, 80:95] = True

        constrained = constrain_mask_to_bbox(mask, bbox=[20.0, 20.0, 50.0, 50.0], margin_ratio=0.10)
        self.assertTrue(np.any(constrained[20:50, 20:50]))
        self.assertFalse(np.any(constrained[80:95, 80:95]))

    def test_full_postprocessing_pipeline(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:60, 20:60] = True
        mask[38:42, 38:42] = False  # internal hole
        mask[2:4, 2:4] = True      # isolated noise
        mask[85:95, 85:95] = True  # outside leak

        final = postprocess_mask(
            mask=mask,
            prompt_bbox=[20.0, 20.0, 60.0, 60.0],
            min_component_size=32,
            fill_holes=True,
            constrain_to_box=True,
            box_margin_ratio=0.10,
        )

        self.assertTrue(final[40, 40])  # Hole was filled
        self.assertFalse(final[2, 2])   # Noise eliminated
        self.assertFalse(final[90, 90]) # Leak constrained


if __name__ == "__main__":
    unittest.main()
