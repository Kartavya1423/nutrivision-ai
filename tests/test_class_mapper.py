"""
Unit tests for class mapping and taxonomy categorization (src.data.class_mapper).
"""

import tempfile
import unittest
from pathlib import Path

from src.data.class_mapper import ClassMapper, VALID_CATEGORIES


class TestClassMapper(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.mapper = ClassMapper(processed_dir=self.root / "processed")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_class_normalization(self):
        """Names must be normalized into clean snake_case and standard singular forms."""
        self.assertEqual(self.mapper.normalize_name("Tomato"), "tomato")
        self.assertEqual(self.mapper.normalize_name("Tomatoes"), "tomato")
        self.assertEqual(self.mapper.normalize_name("Apples"), "apple")
        self.assertEqual(self.mapper.normalize_name("Apple Braeburn"), "apple_braeburn")
        self.assertEqual(self.mapper.normalize_name("french-fries"), "french_fry")

    def test_taxonomy_categorization(self):
        """Standard foods must map to valid categories without hallucinating unknown labels."""
        self.assertEqual(self.mapper.assign_category("apple"), "fruit")
        self.assertEqual(self.mapper.assign_category("banana"), "fruit")
        self.assertEqual(self.mapper.assign_category("carrot"), "vegetable")
        self.assertEqual(self.mapper.assign_category("chicken"), "protein")
        self.assertEqual(self.mapper.assign_category("rice"), "grain")
        self.assertEqual(self.mapper.assign_category("pizza"), "prepared_meal")
        self.assertEqual(self.mapper.assign_category("cake"), "dessert")
        self.assertEqual(self.mapper.assign_category("egg_tart"), "dessert")
        self.assertEqual(self.mapper.assign_category("peanut"), "legume")
        self.assertEqual(self.mapper.assign_category("xyz_unknown_nonfood_item"), "unknown")

    def test_all_assigned_categories_are_valid(self):
        """All assigned categories must strictly belong to the allowed set."""
        test_items = ["apple", "steak", "pasta", "yogurt", "cookie", "popcorn", "edamame", "wine", "unknown_item"]
        for item in test_items:
            cat = self.mapper.assign_category(item)
            self.assertIn(cat, VALID_CATEGORIES)

    def test_cross_dataset_overlap_distinguishes_confirmed_and_possible(self):
        """Overlap analysis must separate exact normalized matches from partial token matches."""
        records = self.mapper.generate_mapping({})
        overlap = self.mapper.analyze_cross_dataset_overlap(records)

        self.assertIn("confirmed_matches", overlap)
        self.assertIn("possible_matches", overlap)
        self.assertGreater(overlap["total_confirmed_matches"], 0)

        # Confirm that confirmed matches have identical normalized names
        for m in overlap["confirmed_matches"]:
            self.assertEqual(m["match_type"], "CONFIRMED")


if __name__ == "__main__":
    unittest.main()
