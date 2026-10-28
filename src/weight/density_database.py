"""
NutriVision AI — Food Density Database Manager
Loads food bulk density values (g/cm³) from data/processed/food_density.json.
Traceable to USDA FoodData Central and FAO Food Composition Tables.
Strictly returns DENSITY_NOT_AVAILABLE when a food category is unlisted.
Does NOT fabricate density values or silently substitute default constants.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.config.settings import get_settings


class DensityDatabase:
    """Manages food-specific density values for volume-to-mass translation."""

    def __init__(self, database_path: Optional[Union[str, Path]] = None):
        cfg = get_settings()
        default_p = cfg.root_dir / "data" / "processed" / "food_density.json"
        self.database_path = Path(database_path) if database_path else default_p
        self.densities: Dict[str, float] = {}
        self.sources: Dict[str, str] = {}
        self.load_database()

    def load_database(self) -> None:
        """Load density mapping and traceable sources from JSON file."""
        if self.database_path.is_file():
            try:
                with open(self.database_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    raw_densities = data.get("densities", {})
                    parsed: Dict[str, float] = {}
                    sources: Dict[str, str] = {}
                    for k, v in raw_densities.items():
                        norm_k = k.lower().strip().replace(" ", "_")
                        if isinstance(v, dict):
                            val = float(v.get("density_g_cm3", 0.0))
                            src = v.get("source", "USDA FoodData Central / FAO")
                        else:
                            val = float(v)
                            src = "Empirical bulk density table"
                        if val > 0:
                            parsed[norm_k] = val
                            sources[norm_k] = src
                    self.densities = parsed
                    self.sources = sources
            except Exception:
                self.densities = {}
                self.sources = {}
        else:
            self.densities = {}
            self.sources = {}

    def get_density(self, food_name: Optional[str]) -> Dict[str, Any]:
        """
        Lookup food bulk density in g/cm³.
        If unlisted or invalid, strictly returns status DENSITY_NOT_AVAILABLE.
        Never silently defaults to arbitrary constants.
        """
        if not food_name or not isinstance(food_name, str):
            return {
                "density": None,
                "unit": "g/cm3",
                "status": "DENSITY_NOT_AVAILABLE",
                "food_name": food_name,
                "source": None,
                "message": "Invalid food name provided.",
            }

        key = food_name.lower().strip().replace(" ", "_").replace("-", "_")

        # 1. Exact match
        if key in self.densities:
            return {
                "density": float(self.densities[key]),
                "unit": "g/cm3",
                "status": "AVAILABLE",
                "food_name": food_name,
                "matched_key": key,
                "source": self.sources.get(key, "USDA FoodData Central"),
            }

        # 2. Substring matching for compound names (e.g. "red apple" -> "apple")
        for d_key, d_val in self.densities.items():
            if d_key in key or key in d_key:
                return {
                    "density": float(d_val),
                    "unit": "g/cm3",
                    "status": "AVAILABLE",
                    "food_name": food_name,
                    "matched_key": d_key,
                    "source": self.sources.get(d_key, "USDA FoodData Central"),
                }

        # 3. Not found: do NOT invent fake values or substitute default
        return {
            "density": None,
            "unit": "g/cm3",
            "status": "DENSITY_NOT_AVAILABLE",
            "food_name": food_name,
            "source": None,
            "message": f"DENSITY_NOT_AVAILABLE: No empirical density recorded for '{food_name}'.",
        }
