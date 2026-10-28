"""
Unit tests for reproducible split generation and official benchmark preservation (src.data.split_generator).
"""

import tempfile
import unittest
from pathlib import Path

from src.data.split_generator import SplitGenerator


class TestSplitGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.generator = SplitGenerator(splits_dir=self.root / "splits", seed=42)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_split_reproducibility(self):
        """Splits generated with identical seed must produce identical assignments."""
        dummy_paths = [self.root / f"img_{i:03d}.jpg" for i in range(100)]
        split1 = self.generator.generate_split("dataset_a", dummy_paths)

        gen2 = SplitGenerator(splits_dir=self.root / "splits", seed=42)
        split2 = gen2.generate_split("dataset_a", dummy_paths)

        self.assertEqual(split1.split_assignments, split2.split_assignments)
        self.assertEqual(split1.train_count, split2.train_count)
        self.assertEqual(split1.val_count, split2.val_count)
        self.assertEqual(split1.test_count, split2.test_count)

    def test_split_ratios_respected(self):
        """Split counts must closely mirror 70% / 15% / 15% defaults."""
        dummy_paths = [self.root / f"img_{i:03d}.jpg" for i in range(100)]
        split = self.generator.generate_split("dataset_b", dummy_paths)

        self.assertEqual(split.total_samples, 100)
        self.assertEqual(split.train_count, 70)
        self.assertEqual(split.val_count, 15)
        self.assertEqual(split.test_count, 15)

    def test_official_splits_preserved(self):
        """Official benchmark test sets must be preserved intact without leakage."""
        official_train = [self.root / "train" / f"tr_{i}.jpg" for i in range(80)]
        official_test = [self.root / "test" / f"te_{i}.jpg" for i in range(20)]
        all_paths = official_train + official_test

        split = self.generator.generate_split("official_bench", all_paths)

        self.assertTrue(split.official_splits_preserved)
        # All official test images must remain in test split
        for p in official_test:
            self.assertEqual(split.split_assignments[str(p)], "test")

        # Validation set must have been created strictly from official train candidates
        val_samples = [p for p, s in split.split_assignments.items() if s == "val"]
        self.assertGreater(len(val_samples), 0)
        for vp in val_samples:
            self.assertIn("train", vp.lower())


if __name__ == "__main__":
    unittest.main()
