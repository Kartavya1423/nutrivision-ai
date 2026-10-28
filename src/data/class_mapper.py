"""
NutriVision AI — Unified Class Mapper & Cross-Dataset Taxonomy Analyzer
Phase 1: Dataset Inspection, Validation, Normalization, and Preparation

Implements non-destructive, reversible class mapping across heterogeneous food datasets:
- Generates data/processed/class_mapping.csv
- Assigns broad categories: fruit, vegetable, grain, legume, protein, dairy,
  prepared_meal, snack, dessert, other, unknown
- Analyzes cross-dataset class overlap: distinguishes CONFIRMED matches from POSSIBLE matches
- Preserves dataset-native class identities without destructive merging
"""

from collections import defaultdict
from dataclasses import dataclass
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from src.config.settings import get_settings

# Broad taxonomy categories allowed by specification
VALID_CATEGORIES = {
    "fruit",
    "vegetable",
    "grain",
    "legume",
    "protein",
    "dairy",
    "prepared_meal",
    "snack",
    "dessert",
    "other",
    "unknown"
}

# Canonical reference dictionaries for known food categories
CATEGORY_RULES: Dict[str, Set[str]] = {
    "fruit": {
        "apple", "banana", "orange", "grape", "strawberry", "blueberry", "raspberry", "blackberry",
        "mango", "pineapple", "watermelon", "melon", "cantaloupe", "honeydew", "peach", "plum",
        "pear", "cherry", "lemon", "lime", "kiwi", "pomegranate", "fig", "avocado", "coconut",
        "papaya", "guava", "passion_fruit", "apricot", "grapefruit", "date", "cranberry"
    },
    "vegetable": {
        "tomato", "potato", "onion", "garlic", "carrot", "broccoli", "cauliflower", "spinach",
        "lettuce", "cucumber", "bell_pepper", "pepper", "chili", "eggplant", "zucchini", "mushroom",
        "cabbage", "celery", "asparagus", "corn", "sweet_potato", "kale", "radish", "beetroot",
        "pumpkin", "squash", "peas", "green_beans", "scallion", "leek", "ginger", "olive"
    },
    "grain": {
        "rice", "bread", "toast", "noodle", "pasta", "spaghetti", "oatmeal", "cereal",
        "quinoa", "barley", "flour", "tortilla", "bagel", "croissant", "bun", "pancake", "waffle"
    },
    "legume": {
        "beans", "bean", "lentils", "chickpeas", "soybean", "edamame", "peanut", "black_beans", "kidney_beans", "red_bean"
    },
    "protein": {
        "chicken", "beef", "pork", "fish", "salmon", "tuna", "egg", "tofu", "shrimp",
        "turkey", "lamb", "duck", "bacon", "sausage", "ham", "steak", "meatball", "crab", "lobster"
    },
    "dairy": {
        "milk", "cheese", "yogurt", "butter", "cream", "sour_cream", "curd", "paneer", "parmesan", "cheese_butter"
    },
    "prepared_meal": {
        "pizza", "burger", "hamburger", "sandwich", "sandwiches", "salad", "soup", "curry", "ramen", "sushi",
        "burrito", "taco", "lasagna", "dumpling", "fried_rice", "paella", "stew", "hot_dog", "gratin"
    },
    "snack": {
        "chips", "popcorn", "pretzel", "crackers", "nuts", "almond", "walnut", "cashew", "pecan",
        "french_fries", "french_fry", "potato_chips", "biscuit"
    },
    "dessert": {
        "cake", "cookie", "ice_cream", "chocolate", "donut", "brownie", "pie", "muffin",
        "pudding", "candy", "pastry", "tart", "egg_tart", "custard_tart", "parfait", "cream_puff"
    },
    "other": {
        "wine", "milkshake", "coffee", "juice", "tea", "water", "other_food"
    }
}

# Canonical class list for FoodSeg103 (103 classes)
FOODSEG103_CANONICAL: List[str] = [
    "candy", "egg_tart", "french_fries", "chocolate", "biscuit", "popcorn", "pudding",
    "ice_cream", "cheese_butter", "cake", "wine", "milkshake", "coffee", "juice",
    "milk", "tea", "almond", "red_beans", "pecan", "cashew", "walnut", "peanut",
    "egg", "apple", "date", "apricot", "avocado", "banana", "strawberry", "cherry",
    "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pomegranate",
    "fig", "grape", "kiwi", "melon", "orange", "pear", "pepper", "pineapple",
    "plum", "watermelon", "radish", "carrot", "cucumber", "eggplant", "asparagus",
    "cabbage", "cauliflower", "broccoli", "celery", "sweet_potato", "corn",
    "zucchini", "garlic", "onion", "mushroom", "pumpkin", "tomato", "potato",
    "tart", "spring_rolls", "pork", "beef", "sausage", "chicken_duck", "fish",
    "shrimp", "seafood", "pasta_noodle", "bread", "pancake", "waffle", "pizza",
    "hamburger", "dumpling", "bread_steamed", "noodle", "rice", "curry", "soup",
    "congee", "tofu", "sushi", "salad", "lettuce", "spinach", "kale", "seaweed",
    "scallion", "ginger", "bean_sprouts", "green_beans", "peas", "mushroom_enoki",
    "chili", "other_food"
]

# Canonical category list for UECFoodPixComplete (102 classes)
UECFOODPIX_CANONICAL: List[str] = [
    "rice", "eel_rice", "pilaf", "chicken_rice", "fried_rice", "curry_rice",
    "sushi", "chirashi_sushi", "toast", "croissant", "roll_bread", "raisin_bread",
    "sandwiches", "hamburger", "pizza", "spaghetti", "ramen", "udon_noodle",
    "soba_noodle", "yakisoba", "pasta", "beef_curry", "pork_cutlet_curry",
    "tempura_bowl", "beef_bowl", "pork_cutlet_bowl", "omelet_rice", "egg_bowl",
    "steamed_egg_custard", "miso_soup", "potage", "minestrone", "clear_soup",
    "soup", "boiled_pork_slice", "sweet_and_sour_pork", "hamburger_steak",
    "steak", "sauteed_vegetables", "grilled_chicken", "fried_chicken", "roast_meat",
    "bbq_meat", "meat_dumplings", "spring_roll", "sashimi", "grilled_fish",
    "fried_fish", "teriyaki_fish", "simmered_fish", "tempura", "boiled_vegetables",
    "sauteed_spinach", "boiled_spinach", "salad", "potato_salad", "egg_salad",
    "macaroni_salad", "tofu", "agedashi_tofu", "cold_tofu", "seasoned_egg",
    "fried_egg", "tamagoyaki", "scrambled_egg", "steamed_dumplings", "meat_pie",
    "gratin", "croquette", "boiled_beans", "edamame", "boiled_corn", "french_fries",
    "potato_chips", "pizza_toast", "french_toast", "pancake", "waffle", "popcorn",
    "sponge_cake", "cheesecake", "apple_pie", "parfait", "ice_cream", "pudding",
    "cream_puff", "custard_tart", "donut", "cookie", "chocolate", "dumpling_sweet",
    "fruit", "apple", "orange", "banana", "strawberry", "grape", "melon",
    "watermelon", "pineapple", "kiwi", "peach"
]


@dataclass
class ClassMappingRecord:
    dataset: str
    original_class_id: str
    original_class_name: str
    normalized_class_name: str
    category: str
    mapping_confidence: str = "HIGH"
    notes: str = ""

    def to_row(self) -> List[str]:
        return [
            self.dataset,
            self.original_class_id,
            self.original_class_name,
            self.normalized_class_name,
            self.category,
            self.mapping_confidence,
            self.notes
        ]


class ClassMapper:
    """
    Manages unified food taxonomy mapping, normalization,
    and cross-dataset overlap reporting.
    """

    def __init__(self, processed_dir: Optional[Path] = None):
        settings = get_settings()
        self.processed_dir = Path(processed_dir).resolve() if processed_dir else settings.datasets.processed_data_root
        self.output_csv = self.processed_dir / "class_mapping.csv"

    # -------------------------------------------------------------------------
    # NORMALIZATION & TAXONOMY CLASSIFICATION
    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_name(raw_name: str) -> str:
        """
        Normalizes class name reversibly:
        - Lowercase, strip whitespace.
        - Replaces dashes/spaces with underscores.
        - Handles specific known plural/variety cases safely without over-merging.
        """
        name = raw_name.strip().lower()
        name = re.sub(r"[\s\-]+", "_", name)
        name = re.sub(r"[^\w_]", "", name)

        # Conservative plural stripping only for standard plurals
        if name.endswith("oes"):
            name = name[:-2]  # tomatoes -> tomato, potatoes -> potato
        elif name.endswith("ies"):
            name = name[:-3] + "y"  # strawberries -> strawberry
        elif name.endswith("s") and not name.endswith(("ss", "us", "is", "as")):
            name = name[:-1]  # apples -> apple, beans -> bean

        return name

    @staticmethod
    def assign_category(normalized_name: str) -> str:
        """
        Maps a normalized food item to broad category:
        fruit, vegetable, grain, legume, protein, dairy,
        prepared_meal, snack, dessert, other, unknown.
        """
        norm = normalized_name.lower()

        # 1. Exact matches first across all categories
        for cat, food_set in CATEGORY_RULES.items():
            if norm in food_set:
                return cat

        # 2. Specific compound dessert, snack, or meal heuristics
        if any(w in norm.split("_") for w in ["tart", "pie", "cake", "cookie", "pudding", "donut", "parfait", "pastry"]):
            return "dessert"
        if any(w in norm.split("_") for w in ["fries", "fry", "chips", "popcorn", "crackers", "pretzel"]):
            return "snack"
        if any(w in norm.split("_") for w in ["curry", "soup", "bowl", "rice", "noodle", "spaghetti", "pizza", "burger", "sandwich", "sandwiches", "salad"]):
            return "prepared_meal"

        # 3. Substring token matches
        for cat, food_set in CATEGORY_RULES.items():
            for item in food_set:
                if item in norm.split("_") or norm.startswith(f"{item}_") or norm.endswith(f"_{item}"):
                    return cat

        return "unknown"

    # -------------------------------------------------------------------------
    # DATASET SPECIFIC EXTRACTORS
    # -------------------------------------------------------------------------
    def generate_mapping(self, dataset_scans: Dict[str, Any]) -> List[ClassMappingRecord]:
        """Generates unified mapping records across all discovered and known target datasets."""
        records: List[ClassMappingRecord] = []

        # 1. FoodSeg103
        fs_scan = dataset_scans.get("FoodSeg103")
        fs_classes = fs_scan.class_names if (fs_scan and fs_scan.class_names) else FOODSEG103_CANONICAL
        for idx, cname in enumerate(fs_classes, start=1):
            norm = self.normalize_name(cname)
            cat = self.assign_category(norm)
            records.append(ClassMappingRecord(
                dataset="FoodSeg103",
                original_class_id=str(idx),
                original_class_name=cname,
                normalized_class_name=norm,
                category=cat,
                notes="Ingredient/item-level segmentation class."
            ))

        # 2. UECFoodPixComplete
        uec_scan = dataset_scans.get("UECFoodPixComplete")
        uec_classes = uec_scan.class_names if (uec_scan and uec_scan.class_names) else UECFOODPIX_CANONICAL
        for idx, cname in enumerate(uec_classes, start=1):
            norm = self.normalize_name(cname)
            cat = self.assign_category(norm)
            records.append(ClassMappingRecord(
                dataset="UECFoodPixComplete",
                original_class_id=str(idx),
                original_class_name=cname,
                normalized_class_name=norm,
                category=cat,
                notes="Real-world meal and dish-level segmentation class."
            ))

        # 3. Fruits360_100x100
        f360_scan = dataset_scans.get("Fruits360_100x100") or dataset_scans.get("Fruits360")
        if f360_scan and f360_scan.class_names:
            for idx, cname in enumerate(f360_scan.class_names, start=1):
                norm = self.normalize_name(cname)
                cat = self.assign_category(norm)
                records.append(ClassMappingRecord(
                    dataset="Fruits360_100x100",
                    original_class_id=str(idx),
                    original_class_name=cname,
                    normalized_class_name=norm,
                    category=cat,
                    notes="Produce variety classification class."
                ))

        # 4. Freshness Dataset (if scanned with classes)
        fr_scan = dataset_scans.get("freshness_dataset")
        if fr_scan and fr_scan.class_names:
            for idx, cname in enumerate(fr_scan.class_names, start=1):
                norm = self.normalize_name(cname)
                records.append(ClassMappingRecord(
                    dataset="freshness_dataset",
                    original_class_id=str(idx),
                    original_class_name=cname,
                    normalized_class_name=norm,
                    category="unknown",
                    notes="Freshness state class."
                ))

        # 5. Sample Benchmark (if scanned)
        sb_scan = dataset_scans.get("sample_benchmark")
        if sb_scan and sb_scan.class_names:
            for idx, cname in enumerate(sb_scan.class_names, start=1):
                norm = self.normalize_name(cname)
                cat = self.assign_category(norm)
                records.append(ClassMappingRecord(
                    dataset="sample_benchmark",
                    original_class_id=str(idx),
                    original_class_name=cname,
                    normalized_class_name=norm,
                    category=cat,
                    notes="Synthetic benchmark sample class."
                ))

        return records

    # -------------------------------------------------------------------------
    # CROSS-DATASET OVERLAP ANALYSIS
    # -------------------------------------------------------------------------
    def analyze_cross_dataset_overlap(
        self, records: List[ClassMappingRecord]
    ) -> Dict[str, Any]:
        """
        Analyzes classes shared across datasets.
        Separates confirmed matches (exact normalized match) from
        possible matches (partial token substring / similar stem).
        """
        # Group records by normalized name
        dataset_classes: Dict[str, Dict[str, str]] = {}
        for r in records:
            if r.dataset not in dataset_classes:
                dataset_classes[r.dataset] = {}
            dataset_classes[r.dataset][r.normalized_class_name] = r.original_class_name

        datasets = sorted(dataset_classes.keys())
        confirmed_matches = []
        possible_matches = []

        for i in range(len(datasets)):
            d1 = datasets[i]
            classes_d1 = dataset_classes[d1]
            for j in range(i + 1, len(datasets)):
                d2 = datasets[j]
                classes_d2 = dataset_classes[d2]

                # Exact normalized matches
                shared_norms = set(classes_d1.keys()).intersection(set(classes_d2.keys()))
                for norm in sorted(shared_norms):
                    confirmed_matches.append({
                        "dataset_a": d1,
                        "dataset_b": d2,
                        "shared_class": norm,
                        "class_a_original": classes_d1[norm],
                        "class_b_original": classes_d2[norm],
                        "match_type": "CONFIRMED",
                        "notes": "Exact semantic and normalized taxonomy match."
                    })

                # Possible matches (token overlaps, e.g. 'apple' in 'apple_pie' or 'juice')
                for norm1, orig1 in classes_d1.items():
                    if norm1 in shared_norms:
                        continue
                    tokens1 = set(norm1.split("_"))
                    for norm2, orig2 in classes_d2.items():
                        if norm2 in shared_norms:
                            continue
                        tokens2 = set(norm2.split("_"))
                        common_tokens = tokens1.intersection(tokens2)
                        # Require significant non-generic shared token
                        meaningful = common_tokens - {"food", "other", "sweet", "boiled", "fried", "slice", "bowl"}
                        if meaningful and len(norm1) > 3 and len(norm2) > 3:
                            possible_matches.append({
                                "dataset_a": d1,
                                "dataset_b": d2,
                                "class_a": norm1,
                                "class_b": norm2,
                                "shared_tokens": list(meaningful),
                                "match_type": "POSSIBLE",
                                "notes": f"Shared concept '{', '.join(meaningful)}'; not merged automatically."
                            })

        return {
            "total_confirmed_matches": len(confirmed_matches),
            "total_possible_matches": len(possible_matches),
            "confirmed_matches": confirmed_matches,
            "possible_matches": possible_matches[:50]  # Cap possible matches for readability
        }

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------
    def save_mapping_csv(self, records: List[ClassMappingRecord]) -> Path:
        """Saves unified class mapping table to data/processed/class_mapping.csv."""
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset",
                "original_class_id",
                "original_class_name",
                "normalized_class_name",
                "category",
                "mapping_confidence",
                "notes"
            ])
            for r in records:
                writer.writerow(r.to_row())

        return self.output_csv

    def save_dataset_classes_csv(self, dataset_scans: Dict[str, Any]) -> Path:
        """
        Saves dataset-native classes and their image counts to
        data/processed/dataset_classes.csv.
        """
        out_file = self.processed_dir / "dataset_classes.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for ds_name, scan in dataset_scans.items():
            if not getattr(scan, "is_available", False):
                continue
            dist = getattr(scan, "class_distribution", {})
            class_names = getattr(scan, "class_names", [])

            for idx, cname in enumerate(class_names, start=1):
                count = dist.get(cname, "")
                rows.append([ds_name, str(idx), cname, count])

        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "original_class_id", "original_class_name", "image_count"])
            for row in rows:
                writer.writerow(row)

        return out_file

    def save_class_overlap_csv(self, overlap_analysis: Dict[str, Any], records: List[ClassMappingRecord]) -> Path:
        """
        Saves cross-dataset class overlap to data/processed/class_overlap.csv.
        Lists confirmed matches, possible token matches, and unique classes.
        """
        out_file = self.processed_dir / "class_overlap.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset_a", "dataset_b", "class_a", "class_b", "match_type", "notes"])

            # 1. Confirmed matches
            for m in overlap_analysis.get("confirmed_matches", []):
                writer.writerow([
                    m["dataset_a"],
                    m["dataset_b"],
                    m["class_a_original"],
                    m["class_b_original"],
                    "CONFIRMED",
                    m["notes"]
                ])

            # 2. Possible matches
            for m in overlap_analysis.get("possible_matches", []):
                writer.writerow([
                    m["dataset_a"],
                    m["dataset_b"],
                    m["class_a"],
                    m["class_b"],
                    "POSSIBLE",
                    m["notes"]
                ])

            # 3. Dataset-unique classes
            # Classes that appeared in only one dataset
            class_datasets = defaultdict(set)
            for r in records:
                class_datasets[r.normalized_class_name].add(r.dataset)

            for norm_name, d_set in class_datasets.items():
                if len(d_set) == 1:
                    ds = list(d_set)[0]
                    writer.writerow([
                        ds,
                        "NONE",
                        norm_name,
                        "NONE",
                        "UNIQUE",
                        f"Class unique to {ds}."
                    ])

        return out_file
