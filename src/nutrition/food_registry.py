"""
NutriVision AI — Dataset-Driven Food Registry & Catalog Manager
Discovers and synchronizes supported food classes from:
1. Ground-truth YOLO11 / Classification taxonomy (198 classes in data.yaml)
2. Verified USDA FoodData Central offline cache (data/nutrition_cache/)
3. Empirical bulk density database (data/processed/food_density.json)

Features:
- Dynamically generates the Quick-Select food catalog from verified available dataset classes.
- Eliminates hardcoded fixed-size food lists.
- Synchronizes UI choices with model label taxonomy.
- Handles inconsistent casing, hyphens, and compound labels safely.
- Distinguishes verified nutrition records from unlisted foods without hallucination.
"""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config.settings import get_settings

logger = logging.getLogger("nutrivision.food_registry")

# Standard food icon mappings
FOOD_ICON_MAP: Dict[str, str] = {
    "pizza": "🍕",
    "burger": "🍔",
    "hamburger": "🍔",
    "salad": "🥗",
    "green_salad": "🥗",
    "sandwich": "🥪",
    "sandwiches": "🥪",
    "steak": "🥩",
    "beef_steak": "🥩",
    "cake": "🍰",
    "ice_cream": "🍦",
    "samosa": "🥟",
    "apple": "🍎",
    "banana": "🍌",
    "orange": "🍊",
    "guava": "🍈",
    "lime": "🍋",
    "pomegranate": "🫐",
    "carrot": "🥕",
    "broccoli": "🥦",
    "bread": "🍞",
    "chicken": "🍗",
    "egg": "🥚",
    "rice": "🍚",
    "fried_rice": "🍛",
    "corn": "🌽",
    "potato": "🥔",
    "tomato": "🍅",
    "cherry": "🍒",
    "plum": "🫐",
    "onion": "🧅",
    "mango": "🥭",
    "strawberry": "🍓",
    "watermelon": "🍉",
    "cucumber": "🥒",
    "french_fries": "🍟",
    "chocolate": "🍫",
    "sushi": "🍣",
    "cheese": "🧀",
    "fish": "🐟",
    "grapes": "🍇",
    "grape": "🍇",
    "soup": "🍲",
    "pasta": "🍝",
    "spaghetti": "🍝",
    "waffle": "🧇",
    "pancake": "🥞",
    "pineapple": "🍍",
    "spinach": "🥬",
    "lettuce": "🥗",
    "paneer": "🧀",
    "butter": "🧈",
}

# Empirical categories supported by data
FOOD_CATEGORIES: Dict[str, List[str]] = {
    "Popular Dishes": ["pizza", "burger", "hamburger", "salad", "sandwich", "steak", "samosa", "fried_rice", "sushi", "french_fries"],
    "Produce & Fruits": ["apple", "banana", "orange", "guava", "lime", "pomegranate", "cherry", "plum", "mango", "strawberry", "watermelon", "grape", "pineapple"],
    "Vegetables": ["carrot", "broccoli", "corn", "potato", "tomato", "onion", "cucumber", "spinach", "lettuce"],
    "Bakery & Desserts": ["bread", "cake", "ice_cream", "chocolate", "pancake", "waffle"],
    "Proteins & Dairy": ["chicken", "egg", "steak", "beef_steak", "cheese", "fish", "paneer", "butter"],
}


@dataclass
class SupportedFood:
    """Metadata representing a verified, dataset-driven food item."""
    key: str
    label: str
    icon: str
    category: str
    default_portion_g: float
    usda_fdc_id: Optional[int] = None
    usda_food_name: Optional[str] = None
    calories_per_100g: Optional[float] = None
    protein_per_100g: Optional[float] = None
    carbs_per_100g: Optional[float] = None
    fat_per_100g: Optional[float] = None
    fiber_per_100g: Optional[float] = None
    density_g_cm3: Optional[float] = None
    is_verified_nutrition: bool = False
    source: str = "USDA FoodData Central"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "icon": self.icon,
            "category": self.category,
            "default_portion_g": self.default_portion_g,
            "usda_fdc_id": self.usda_fdc_id,
            "usda_food_name": self.usda_food_name,
            "calories_per_100g": self.calories_per_100g,
            "protein_per_100g": self.protein_per_100g,
            "carbs_per_100g": self.carbs_per_100g,
            "fat_per_100g": self.fat_per_100g,
            "fiber_per_100g": self.fiber_per_100g,
            "density_g_cm3": self.density_g_cm3,
            "is_verified_nutrition": self.is_verified_nutrition,
            "source": self.source,
        }


class FoodRegistry:
    """Central authority for discovering, querying, and synchronizing food classes."""

    def __init__(self, root_dir: Optional[Path] = None):
        cfg = get_settings()
        self.root_dir = root_dir or cfg.root_dir
        self.cache_dir = self.root_dir / "data" / "nutrition_cache"
        self.density_file = self.root_dir / "data" / "processed" / "food_density.json"
        self.taxonomy_yaml = self.root_dir / "data" / "processed" / "yolo_detection" / "data.yaml"
        self.class_names_json = self.root_dir / "models" / "detection" / "yolo11" / "class_names.json"

        self._taxonomy_classes: List[str] = []
        self._cached_nutrition: Dict[str, Dict[str, Any]] = {}
        self._densities: Dict[str, float] = {}
        self._supported_foods: Dict[str, SupportedFood] = {}
        self._initialized: bool = False

        self.refresh()

    def refresh(self) -> None:
        """Scan disk repositories and refresh in-memory registries."""
        self._load_taxonomy()
        self._load_densities()
        self._load_cached_nutrition()
        self._build_supported_foods()
        self._initialized = True

    def _load_taxonomy(self) -> None:
        """Load 198 classes from data.yaml or class_names.json."""
        classes = []
        if self.taxonomy_yaml.is_file():
            try:
                import yaml
                with open(self.taxonomy_yaml, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    names = cfg.get("names", {})
                    if isinstance(names, dict):
                        classes = [str(names[k]).strip() for k in sorted(names.keys(), key=lambda x: int(x))]
                    elif isinstance(names, list):
                        classes = [str(n).strip() for n in names]
            except Exception as e:
                logger.warning(f"Failed to read taxonomy from {self.taxonomy_yaml}: {e}")

        if not classes and self.class_names_json.is_file():
            try:
                with open(self.class_names_json, "r", encoding="utf-8") as f:
                    classes = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read class_names.json: {e}")

        self._taxonomy_classes = classes

    def _load_densities(self) -> None:
        """Load bulk densities from food_density.json."""
        densities = {}
        if self.density_file.is_file():
            try:
                with open(self.density_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    raw = data.get("densities", {})
                    for k, v in raw.items():
                        norm_k = k.lower().strip().replace(" ", "_").replace("-", "_")
                        val = float(v.get("density_g_cm3", 0.0)) if isinstance(v, dict) else float(v)
                        if val > 0:
                            densities[norm_k] = val
            except Exception as e:
                logger.warning(f"Failed to load densities from {self.density_file}: {e}")
        self._densities = densities

    def _load_cached_nutrition(self) -> None:
        """Load authentic offline USDA records from nutrition_cache."""
        records = {}
        if self.cache_dir.is_dir():
            for p in self.cache_dir.glob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("status") == "AVAILABLE":
                            key = p.stem.lower().strip().replace("-", "_")
                            records[key] = data
                except Exception as e:
                    logger.debug(f"Skipping cache file {p}: {e}")
        self._cached_nutrition = records

    def _categorize_food(self, key: str) -> str:
        """Assign category based on taxonomy structure."""
        for cat, items in FOOD_CATEGORIES.items():
            if key in items or any(it in key for it in items):
                return cat
        # Fallback category inference
        if any(w in key for w in ["fruit", "berry", "melon", "apple", "banana", "plum", "cherry", "peach", "mango"]):
            return "Produce & Fruits"
        if any(w in key for w in ["vegetable", "greens", "carrot", "salad", "broccoli", "potato", "onion", "bean", "cabbage"]):
            return "Vegetables"
        if any(w in key for w in ["cake", "bread", "pastry", "tart", "cookie", "pie", "dessert", "sweet", "ice_cream", "pudding"]):
            return "Bakery & Desserts"
        if any(w in key for w in ["beef", "pork", "chicken", "meat", "steak", "egg", "fish", "salmon", "shrimp", "seafood"]):
            return "Proteins & Dairy"
        return "Prepared Dishes"

    def _build_supported_foods(self) -> None:
        """Build dictionary of verified supported foods combining cache, density, and taxonomy."""
        supported: Dict[str, SupportedFood] = {}

        # 1. First populate all foods with verified USDA nutrition records in cache
        for slug, rec in self._cached_nutrition.items():
            nutr = rec.get("nutrients_per_100g", {})
            norm_key = slug.replace(" ", "_")
            raw_name = rec.get("food_name") or slug
            label = raw_name.replace("_", " ").replace("-", " ").title().strip()
            # Clean special cases & canonical keys
            if norm_key in ["hamburger", "burger"]:
                norm_key = "burger"
                label = "Burger"
            elif norm_key in ["beef_steak", "steak"]:
                norm_key = "steak"
                label = "Steak"
            elif norm_key in ["sandwiches", "sandwich"]:
                norm_key = "sandwich"
                label = "Sandwich"
            elif norm_key in ["green_salad", "salad"]:
                norm_key = "salad"
                label = "Salad"
            elif norm_key == "ice_cream":
                label = "Ice Cream"
            elif norm_key == "fried_rice":
                label = "Fried Rice"

            icon = FOOD_ICON_MAP.get(norm_key, FOOD_ICON_MAP.get(slug, "🍽️"))
            cat = self._categorize_food(norm_key)
            density = self._densities.get(norm_key, self._densities.get(slug))

            # Default portion
            default_g = 150.0
            if cat == "Produce & Fruits":
                default_g = 120.0
            elif cat == "Vegetables":
                default_g = 100.0
            elif cat == "Bakery & Desserts":
                default_g = 110.0
            elif cat == "Proteins & Dairy":
                default_g = 180.0

            food_item = SupportedFood(
                key=norm_key,
                label=label,
                icon=icon,
                category=cat,
                default_portion_g=default_g,
                usda_fdc_id=rec.get("usda_fdc_id"),
                usda_food_name=rec.get("usda_food_name"),
                calories_per_100g=nutr.get("calories_kcal"),
                protein_per_100g=nutr.get("protein_g"),
                carbs_per_100g=nutr.get("carbohydrates_g"),
                fat_per_100g=nutr.get("fat_g"),
                fiber_per_100g=nutr.get("fiber_g"),
                density_g_cm3=density,
                is_verified_nutrition=True,
                source=rec.get("source", "USDA FoodData Central"),
            )
            supported[norm_key] = food_item

        # 2. Also register alias keys for lookups
        if "burger" in supported and "hamburger" not in supported:
            supported["hamburger"] = supported["burger"]
        if "steak" in supported and "beef_steak" not in supported:
            supported["beef_steak"] = supported["steak"]
        if "sandwich" in supported and "sandwiches" not in supported:
            supported["sandwiches"] = supported["sandwich"]
        if "salad" in supported and "green_salad" not in supported:
            supported["green_salad"] = supported["salad"]

        self._supported_foods = supported

    def get_supported_foods(self) -> List[SupportedFood]:
        """Return all verified supported foods, sorted alphabetically by label, deduplicated."""
        seen_labels = set()
        unique = []
        for f in sorted(self._supported_foods.values(), key=lambda x: x.label):
            if f.label not in seen_labels:
                seen_labels.add(f.label)
                unique.append(f)
        return unique

    def get_quick_catalog_foods(self, limit: int = 12) -> List[SupportedFood]:
        """
        Return the primary catalog food items for instant 1-click UI selection.
        Dynamically adapts to available verified classes without assuming only 8 exist.
        """
        all_foods = self.get_supported_foods()
        if not all_foods:
            return []

        # Prioritize key staple foods if present in supported list
        staples = ["pizza", "burger", "salad", "sandwich", "steak", "cake", "ice_cream", "samosa", "apple", "banana", "chicken", "carrot"]
        prioritized = []
        seen = set()

        for s in staples:
            if s in self._supported_foods and s not in seen:
                prioritized.append(self._supported_foods[s])
                seen.add(s)

        # Fill remaining slots up to limit from other supported foods
        for f in all_foods:
            if f.key not in seen and len(prioritized) < limit:
                prioritized.append(f)
                seen.add(f.key)

        return prioritized

    def get_all_taxonomy_classes(self) -> List[str]:
        """Return all 198 raw classes from the dataset taxonomy."""
        return self._taxonomy_classes.copy()

    def get_food(self, key: str) -> Optional[SupportedFood]:
        """Lookup metadata for a specific food key."""
        if not key:
            return None
        norm = key.lower().strip().replace(" ", "_").replace("-", "_")
        if norm in self._supported_foods:
            return self._supported_foods[norm]
        
        # Check if it is a valid taxonomy class without cached nutrition
        for c in self._taxonomy_classes:
            if c.lower().replace("-", "_") == norm:
                # Return unverified taxonomy food item
                label = c.replace("_", " ").title()
                return SupportedFood(
                    key=c,
                    label=label,
                    icon=FOOD_ICON_MAP.get(norm, "🍽️"),
                    category=self._categorize_food(norm),
                    default_portion_g=150.0,
                    is_verified_nutrition=False,
                    source="Dataset Taxonomy Only",
                )
        return None

    def search_foods(self, query: str) -> List[SupportedFood]:
        """Search across supported foods by prefix or keyword."""
        q = query.lower().strip()
        if not q:
            return self.get_supported_foods()
        results = []
        for f in self.get_supported_foods():
            if q in f.key.lower() or q in f.label.lower() or q in f.category.lower():
                results.append(f)
        return results

    def get_food_variants(self, key: str) -> List[Dict[str, Any]]:
        """Return all supported preparation/recipe variants for a food class."""
        from src.nutrition.fooddata_central import FoodDataCentralClient
        return FoodDataCentralClient.get_available_variants(key)


# Global singleton instance
_registry_instance: Optional[FoodRegistry] = None

def get_food_registry() -> FoodRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = FoodRegistry()
    return _registry_instance
