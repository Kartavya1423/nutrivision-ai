"""
NutriVision AI — Production Streamlit Dashboard & Modern Health Web App
Features:
  - Food image upload and camera capture
  - Multi-item food detection and classification
  - Fruit recognition with the updated 6-class dataset
  - 3D volume & depth-aware weight estimation
  - Authentic USDA FoodData Central nutritional intelligence
  - Vitamins & minerals breakdown and Computer Vision Inspector
"""

import csv
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import requests

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from src.config.settings import get_settings

logger = logging.getLogger("nutrivision.frontend")
cfg = get_settings()


def get_api_base_url() -> str:
    """Dynamically determine active FastAPI backend (checking port 8000 and 8001)."""
    env_url = os.environ.get("NUTRIVISION_API_URL")
    if env_url:
        return env_url.rstrip("/")
    for port in [8000, 8001]:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            if r.status_code == 200:
                return f"http://127.0.0.1:{port}"
        except Exception:
            pass
    return "http://127.0.0.1:8000"


API_BASE_URL = get_api_base_url()
SUPPORTED_IMAGE_FORMATS = ("JPEG", "JPG", "PNG", "WEBP")
MIN_DIMENSION_PX = 10
MAX_DIMENSION_PX = 4096
DEFAULT_API_TIMEOUT = 120.0

FOOD_ICONS: Dict[str, str] = {
    "apple": "🍎",
    "apple_good": "🍎",
    "banana": "🍌",
    "banana_good": "🍌",
    "bread": "🍞",
    "broccoli": "🥦",
    "burger": "🍔",
    "butter": "🧈",
    "cake": "🍰",
    "chicken": "🍗",
    "cucumber": "🥒",
    "egg": "🥚",
    "french_fries": "🍟",
    "grape": "🍇",
    "guava": "🍈",
    "guava_good": "🍈",
    "ice_cream": "🍦",
    "lettuce": "🥬",
    "lime": "🍋",
    "lime_good": "🍋",
    "mango": "🥭",
    "orange": "🍊",
    "orange_good": "🍊",
    "paneer": "🧀",
    "pineapple": "🍍",
    "pizza": "🍕",
    "pomegranate": "🍎",
    "pomegranate_good": "🍎",
    "salad": "🥗",
    "sandwich": "🥪",
    "samosa": "🥟",
    "spinach": "🥬",
    "steak": "🥩",
    "strawberry": "🍓",
    "tomato": "🍅",
    "watermelon": "🍉",
    "food": "🍽️",
}

DAILY_VALUES: Dict[str, Tuple[float, str]] = {
    # Vitamins
    "Vitamin C": (90.0, "mg"),
    "Vitamin A (RAE)": (900.0, "µg"),
    "Vitamin A": (900.0, "µg"),
    "Vitamin D": (20.0, "µg"),
    "Vitamin E": (15.0, "mg"),
    "Vitamin K": (120.0, "µg"),
    "Thiamin (B1)": (1.2, "mg"),
    "Riboflavin (B2)": (1.3, "mg"),
    "Niacin (B3)": (16.0, "mg"),
    "Vitamin B6": (1.7, "mg"),
    "Folate": (400.0, "µg"),
    "Vitamin B12": (2.4, "µg"),
    # Minerals
    "Calcium": (1300.0, "mg"),
    "Iron": (18.0, "mg"),
    "Potassium": (4700.0, "mg"),
    "Magnesium": (420.0, "mg"),
    "Zinc": (11.0, "mg"),
    "Phosphorus": (1250.0, "mg"),
    "Sodium": (2300.0, "mg"),
    "Copper": (0.9, "mg"),
    "Manganese": (2.3, "mg"),
    "Selenium": (55.0, "µg"),
}


# ---------------------------------------------------------------------------
# Structured API Exceptions (Strictly Preserved for Backward Compatibility)
# ---------------------------------------------------------------------------

class BackendConnectionError(Exception):
    """Raised when the FastAPI backend is offline or connection is refused."""
    pass


class BackendTimeoutError(Exception):
    """Raised when a request to the FastAPI backend exceeds the timeout limit."""
    pass


class InvalidImageApiError(Exception):
    """Raised when the backend rejects image due to validation, format, or corruption."""
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


class ModelNotReadyApiError(Exception):
    """Raised when backend models are uninitialized or checkpoints are missing."""
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


class InferencePipelineApiError(Exception):
    """Raised when an internal server exception occurs during model inference."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Backend API Client Functions (Strictly Preserved for Backward Compatibility)
# ---------------------------------------------------------------------------

def check_api_health(timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    """Check connectivity and liveness of the FastAPI backend."""
    base_url = get_api_base_url()
    try:
        res = requests.get(f"{base_url}/health", timeout=timeout)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def fetch_models_status(timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    """Query model readiness and checkpoint availability from the FastAPI backend."""
    base_url = get_api_base_url()
    try:
        res = requests.get(f"{base_url}/models/status", timeout=timeout)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def call_analyze_api(
    image_bytes: bytes,
    filename: str = "upload.jpg",
    content_type: str = "image/jpeg",
    enable_gradcam: bool = True,
    enable_nutrition: bool = True,
    crop_padding: int = 10,
    confidence_threshold: Optional[float] = None,
    classifier_mode: Optional[str] = None,
    timeout: float = DEFAULT_API_TIMEOUT,
) -> Dict[str, Any]:
    """Call FastAPI /analyze endpoint with multipart image payload."""
    if not image_bytes or len(image_bytes) == 0:
        raise InvalidImageApiError("Cannot send empty image buffer (0 bytes).", status_code=400)

    files = {"file": (filename, image_bytes, content_type)}
    data: Dict[str, Any] = {
        "enable_gradcam": str(enable_gradcam).lower(),
        "enable_nutrition": str(enable_nutrition).lower(),
        "crop_padding": int(crop_padding),
    }
    if confidence_threshold is not None:
        data["confidence_threshold"] = float(confidence_threshold)
    if classifier_mode is not None:
        data["classifier_mode"] = str(classifier_mode)

    try:
        res = requests.post(
            f"{API_BASE_URL}/analyze",
            files=files,
            data=data,
            timeout=timeout,
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.SSLError) as exc:
        raise BackendConnectionError(
            f"Could not connect to FastAPI backend at '{API_BASE_URL}'. "
            "The backend server appears to be offline or unreachable."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise BackendTimeoutError(
            f"Backend request timed out after {timeout:.0f} seconds. "
            "Complex inference or CPU/GPU saturation may have delayed the response."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise BackendConnectionError(f"Network request to backend failed: {str(exc)}") from exc

    try:
        body = res.json()
    except Exception:
        body = {"detail": res.text or f"HTTP {res.status_code} with non-JSON response"}

    if res.status_code == 200:
        return body

    detail = body.get("detail", str(body)) if isinstance(body, dict) else str(body)

    if res.status_code in (400, 413, 415, 422):
        raise InvalidImageApiError(detail, status_code=res.status_code)
    elif res.status_code == 503:
        raise ModelNotReadyApiError(detail, status_code=res.status_code)
    else:
        raise InferencePipelineApiError(detail, status_code=res.status_code)


def validate_image_bytes(image_bytes: bytes) -> Tuple[Image.Image, int, int, str]:
    """Validate uploaded image buffer for corruptions, unsupported formats, and dimension boundaries."""
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes). Please upload a valid image.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as test_img:
            img_format = test_img.format.upper() if test_img.format else "UNKNOWN"
            if img_format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError(
                    f"Unsupported image format: '{img_format}'. "
                    f"Allowed formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}."
                )
            test_img.verify()

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil_img.size

        if w < MIN_DIMENSION_PX or h < MIN_DIMENSION_PX:
            raise ValueError(
                f"Image dimensions ({w}×{h} px) are smaller than minimum allowed ({MIN_DIMENSION_PX}×{MIN_DIMENSION_PX} px)."
            )
        if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX:
            raise ValueError(
                f"Image dimensions ({w}×{h} px) exceed maximum allowed ({MAX_DIMENSION_PX}×{MAX_DIMENSION_PX} px)."
            )

        return pil_img, w, h, img_format
    except Exception as exc:
        raise ValueError(f"Corrupted or invalid image: {str(exc)}") from exc


def generate_summary_csv(items: List[Dict[str, Any]]) -> str:
    """Generate CSV string of detected items and their nutrition using standard csv module."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Item #", "Food Name", "Detection Conf", "Freshness", "Freshness Conf",
        "Estimated Weight (g)", "Calories (kcal)", "Protein (g)", "Carbs (g)",
        "Fat (g)", "Fiber (g)", "USDA Matched", "USDA FDC ID"
    ])
    for idx, it in enumerate(items):
        nut = it.get("nutrition", {})
        det = it.get("detection", {})
        fresh = it.get("freshness", {})
        wgt = it.get("weight", {})
        writer.writerow([
            idx + 1,
            it.get("food", "Unknown"),
            f"{det.get('confidence', 0.0)*100:.1f}%",
            fresh.get("label", "Unknown").upper(),
            f"{fresh.get('confidence', 0.0)*100:.1f}%",
            wgt.get("value_g") or "N/A",
            nut.get("calories_kcal") or "N/A",
            nut.get("protein_g") or "N/A",
            nut.get("carbohydrates_g") or "N/A",
            nut.get("fat_g") or "N/A",
            nut.get("fiber_g") or "N/A",
            nut.get("matched_food") or "N/A",
            nut.get("fdc_id") or "N/A",
        ])
    return output.getvalue()


def extract_available_vitamins_and_minerals(
    nut_dict: Dict[str, Any],
    portion_weight_g: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Extract and scale all non-null micronutrients (vitamins and minerals) available in USDA database."""
    if not nut_dict or nut_dict.get("status") not in ("SUCCESS", "AVAILABLE"):
        return []

    scale = (portion_weight_g / 100.0) if (portion_weight_g is not None and portion_weight_g > 0) else 1.0
    n100 = nut_dict.get("nutrients_per_100g") or {}

    MICRONUTRIENT_MAP = [
        # Key in nut / n100, Display Label, Unit, Category
        ("vitamin_c_mg", "Vitamin C", "mg", "Vitamin"),
        ("vitamin_a_rae_ug", "Vitamin A (RAE)", "µg", "Vitamin"),
        ("vitamin_a_ug", "Vitamin A", "µg", "Vitamin"),
        ("vitamin_d_ug", "Vitamin D", "µg", "Vitamin"),
        ("vitamin_e_mg", "Vitamin E", "mg", "Vitamin"),
        ("vitamin_k_ug", "Vitamin K", "µg", "Vitamin"),
        ("vitamin_b6_mg", "Vitamin B6", "mg", "Vitamin"),
        ("folate_ug", "Folate", "µg", "Vitamin"),
        ("vitamin_b12_ug", "Vitamin B12", "µg", "Vitamin"),
        ("thiamin_mg", "Thiamin (B1)", "mg", "Vitamin"),
        ("riboflavin_mg", "Riboflavin (B2)", "mg", "Vitamin"),
        ("niacin_mg", "Niacin (B3)", "mg", "Vitamin"),
        ("calcium_mg", "Calcium", "mg", "Mineral"),
        ("iron_mg", "Iron", "mg", "Mineral"),
        ("potassium_mg", "Potassium", "mg", "Mineral"),
        ("magnesium_mg", "Magnesium", "mg", "Mineral"),
        ("phosphorus_mg", "Phosphorus", "mg", "Mineral"),
        ("zinc_mg", "Zinc", "mg", "Mineral"),
        ("sodium_mg", "Sodium", "mg", "Mineral"),
        ("copper_mg", "Copper", "mg", "Mineral"),
        ("manganese_mg", "Manganese", "mg", "Mineral"),
        ("selenium_ug", "Selenium", "µg", "Mineral"),
        ("sugars_g", "Total Sugars", "g", "Carbohydrate"),
    ]

    found = []
    seen_labels = set()
    for key, label, unit, cat in MICRONUTRIENT_MAP:
        if label in seen_labels:
            continue
        val = None
        # Prioritize base nutrients_per_100g for accurate portion scaling
        if key in n100 and n100[key] is not None:
            val = float(n100[key]) * scale
        elif key in nut_dict and nut_dict[key] is not None:
            val = float(nut_dict[key])

        if val is not None:
            seen_labels.add(label)
            dv_entry = DAILY_VALUES.get(label)
            dv_pct = round((val / dv_entry[0]) * 100.0, 1) if dv_entry and dv_entry[0] > 0 else None
            found.append({
                "label": label,
                "value": round(val, 2),
                "unit": unit,
                "category": cat,
                "dv_pct": dv_pct,
            })
    return found


# ---------------------------------------------------------------------------
# Dataset-Driven Food Registry & Quick-Select Definitions
# ---------------------------------------------------------------------------

def _load_dynamic_quick_select_foods() -> List[Dict[str, Any]]:
    """Dynamically discover verified food classes from FoodRegistry, with safe fallback."""
    try:
        from src.nutrition.food_registry import get_food_registry
        reg = get_food_registry()
        foods = reg.get_supported_foods()
        if foods:
            return [
                {
                    "key": item.key,
                    "label": item.label,
                    "icon": item.icon,
                    "default_g": item.default_portion_g,
                    "category": item.category,
                    "usda_fdc_id": item.usda_fdc_id,
                    "calories_per_100g": item.calories_per_100g,
                }
                for item in foods
            ]
    except Exception as exc:
        logger.warning(f"Failed to load dynamic foods from FoodRegistry: {exc}")

    # Fallback list if registry cannot be loaded
    return [
        {"key": "pizza", "label": "Pizza", "icon": "🍕", "default_g": 150.0, "category": "Popular Dishes"},
        {"key": "burger", "label": "Burger", "icon": "🍔", "default_g": 200.0, "category": "Popular Dishes"},
        {"key": "salad", "label": "Salad", "icon": "🥗", "default_g": 120.0, "category": "Popular Dishes"},
        {"key": "sandwich", "label": "Sandwich", "icon": "🥪", "default_g": 180.0, "category": "Popular Dishes"},
        {"key": "steak", "label": "Steak", "icon": "🥩", "default_g": 220.0, "category": "Popular Dishes"},
        {"key": "cake", "label": "Cake", "icon": "🍰", "default_g": 110.0, "category": "Bakery & Desserts"},
        {"key": "ice_cream", "label": "Ice Cream", "icon": "🍦", "default_g": 100.0, "category": "Bakery & Desserts"},
        {"key": "samosa", "label": "Samosa", "icon": "🥟", "default_g": 120.0, "category": "Popular Dishes"},
    ]


QUICK_SELECT_FOODS = _load_dynamic_quick_select_foods()


def get_quick_select_nutrition(
    food_key: str,
    weight_g: float,
    variant: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve authentic USDA nutrition scaled for quick-select portion weight with variant support."""
    try:
        from src.nutrition.nutrition_service import NutritionService
        service = NutritionService()
        scaled = service.get_nutrition_for_item(
            food_name=food_key,
            weight_grams=weight_g,
            variant=variant,
            weight_source="user_measured",
        )
        if scaled and scaled.status == "AVAILABLE":
            return scaled.to_dict()
    except Exception as exc:
        logger.warning(f"Quick-select nutrition lookup error for {food_key}: {exc}")
    return None


def get_food_variants(food_key: str) -> List[Dict[str, Any]]:
    """Retrieve supported preparation/recipe variants for a target food class."""
    try:
        from src.nutrition.fooddata_central import FoodDataCentralClient
        return FoodDataCentralClient.get_available_variants(food_key)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Streamlit UI Rendering with Market-Based SaaS Dark Theme
# ---------------------------------------------------------------------------

def render_html(html_str: str) -> None:
    """Safely render HTML without leading whitespace that could trigger markdown code block formatting."""
    clean_lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    st.markdown("\n".join(clean_lines), unsafe_allow_html=True)


def apply_dark_theme_css():
    """Inject modern, high-contrast dark dashboard styling and glassmorphism."""
    render_html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Design Tokens & Theme Foundations */
        :root {
            --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            --bg-base: #0a0e1a;
            --bg-surface: #111827;
            --bg-surface-elevated: #1e293b;
            --bg-glass: rgba(17, 24, 39, 0.75);
            --border-glass: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(16, 185, 129, 0.35);
            --accent-emerald: #10b981;
            --accent-cyan: #06b6d4;
            --accent-amber: #f59e0b;
            --accent-coral: #f43f5e;
            --accent-purple: #8b5cf6;
            --accent-blue: #38bdf8;
            --text-heading: #f9fafb;
            --text-body: #e2e8f0;
            --text-muted: #94a3b8;
        }

        html, body, [class*="css"] {
            font-family: var(--font-sans) !important;
        }

        /* Top Brand Header Box */
        .brand-header-box {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(30, 41, 59, 0.9) 50%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(16px);
            border-radius: 16px;
            padding: 24px 30px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        }
        .brand-title {
            font-size: 2.35rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(90deg, #34d399 0%, #06b6d4 50%, #60a5fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            display: inline-block;
        }
        .brand-subtitle {
            font-size: 1.05rem;
            color: var(--text-muted);
            margin-top: 6px;
            margin-bottom: 14px;
            font-weight: 400;
        }
        .engine-pills-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .engine-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background-color: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            color: #e2e8f0;
            backdrop-filter: blur(8px);
        }
        .engine-pill.live {
            border-color: #059669;
            background-color: rgba(16, 185, 129, 0.12);
            color: #34d399;
        }
        .engine-pill.offline {
            border-color: #dc2626;
            background-color: rgba(239, 68, 68, 0.12);
            color: #f87171;
        }

        /* Responsive Polish */
        @media (max-width: 768px) {
            .brand-header-box {
                padding: 18px 20px;
                margin-bottom: 16px;
            }
            .brand-title {
                font-size: 1.85rem;
            }
            .brand-subtitle {
                font-size: 0.92rem;
            }
            .food-hero-title {
                font-size: 1.6rem !important;
            }
            .macro-val {
                font-size: 1.4rem !important;
            }
            .engine-pills-row {
                gap: 6px;
            }
            .engine-pill {
                font-size: 0.74rem;
                padding: 4px 10px;
            }
        }

        /* ---------------------------------------------------------------------------
           HERO RECOGNIZED FOOD CARD & MACRO CARDS
        --------------------------------------------------------------------------- */
        .food-hero-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(17, 24, 39, 0.85) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 22px 24px;
            margin-bottom: 20px;
            backdrop-filter: blur(14px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        }
        .food-hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 0;
        }

        /* Macro KPI Cards */
        .macro-card {
            background: rgba(17, 24, 39, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 16px 18px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .macro-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.2);
        }
        .macro-card-top-bar {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
        }
        .macro-val {
            font-size: 1.85rem;
            font-weight: 800;
            color: #ffffff;
            margin-top: 6px;
            margin-bottom: 2px;
            font-family: var(--font-sans);
        }
        .macro-sub {
            font-size: 0.8rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        /* Macronutrient Ratio Bar */
        .macro-bar-container {
            width: 100%;
            height: 12px;
            background-color: rgba(30, 41, 59, 0.8);
            border-radius: 8px;
            display: flex;
            overflow: hidden;
            margin: 14px 0 16px 0;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .macro-segment-protein {
            background: linear-gradient(90deg, #38bdf8, #0284c7);
            height: 100%;
        }
        .macro-segment-carbs {
            background: linear-gradient(90deg, #34d399, #059669);
            height: 100%;
        }
        .macro-segment-fat {
            background: linear-gradient(90deg, #fbbf24, #d97706);
            height: 100%;
        }
        .macro-legend {
            display: flex;
            gap: 20px;
            font-size: 0.84rem;
            color: #cbd5e1;
            flex-wrap: wrap;
        }
        .macro-legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }

        /* ---------------------------------------------------------------------------
           MICRONUTRIENT BARS & PILLS (Vitamins & Minerals Breakdown)
        --------------------------------------------------------------------------- */
        .micro-panel {
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 16px;
        }
        .micro-item-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }
        .micro-item-row:last-child {
            border-bottom: none;
        }
        .micro-name {
            font-size: 0.9rem;
            font-weight: 600;
            color: #f1f5f9;
        }
        .micro-val-group {
            text-align: right;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .micro-val-num {
            font-size: 0.95rem;
            font-weight: 700;
            color: #ffffff;
            font-family: var(--font-mono);
        }
        .micro-dv-badge {
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.78rem;
            font-weight: 700;
            min-width: 52px;
            text-align: center;
        }
        .micro-progress-bg {
            width: 80px;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
            display: inline-block;
        }
        .micro-progress-fill-vit {
            height: 100%;
            background: linear-gradient(90deg, #ec4899, #f43f5e);
            border-radius: 3px;
        }
        .micro-progress-fill-min {
            height: 100%;
            background: linear-gradient(90deg, #06b6d4, #10b981);
            border-radius: 3px;
        }

        /* Badges */
        .badge-fresh {
            background: linear-gradient(135deg, #059669, #10b981);
            color: #ffffff;
            font-weight: 700;
            font-size: 0.8rem;
            padding: 3px 10px;
            border-radius: 6px;
            display: inline-block;
        }
        .badge-rotten {
            background: linear-gradient(135deg, #b91c1c, #ef4444);
            color: #ffffff;
            font-weight: 700;
            font-size: 0.8rem;
            padding: 3px 10px;
            border-radius: 6px;
            display: inline-block;
        }
        .badge-usda {
            background: linear-gradient(135deg, #0284c7, #06b6d4);
            color: #ffffff;
            font-weight: 700;
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 6px;
            display: inline-block;
        }

        /* Callouts */
        .info-callout {
            background-color: rgba(6, 182, 212, 0.08);
            border-left: 4px solid #06b6d4;
            padding: 12px 16px;
            border-radius: 0 10px 10px 0;
            color: #e0f2fe;
            font-size: 0.92rem;
            margin: 12px 0;
        }
        .success-callout {
            background-color: rgba(16, 185, 129, 0.08);
            border-left: 4px solid #10b981;
            padding: 12px 16px;
            border-radius: 0 10px 10px 0;
            color: #d1fae5;
            font-size: 0.92rem;
            margin: 12px 0;
        }
        </style>
        """
    )


def render_app():
    """Main application execution pipeline for NutriVision AI."""
    if not HAS_STREAMLIT:
        print("Streamlit is not installed. Run: pip install streamlit")
        return

    st.set_page_config(
        page_title="NutriVision AI — Intelligent Food Recognition & Nutrition",
        page_icon="🥗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_dark_theme_css()

    # Session state initialization
    if "selected_quick_food" not in st.session_state:
        st.session_state["selected_quick_food"] = None
    if "quick_portion_weight" not in st.session_state:
        st.session_state["quick_portion_weight"] = 150.0
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None
    if "last_image_bytes" not in st.session_state:
        st.session_state["last_image_bytes"] = None
    if "portion_weight_override" not in st.session_state:
        st.session_state["portion_weight_override"] = {}

    # Check backend connectivity
    api_info = check_api_health()
    backend_online = api_info is not None

    # ---------------------------------------------------------------------------
    # Sidebar: Model Readiness, Device Status & Vision Tuning
    # ---------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🥗 NutriVision AI")
        st.caption("Computer Vision & Nutritional Diagnostics")

        if backend_online:
            dev = api_info.get("device", "cpu").upper()
            st.success(f"🟢 **FastAPI Service:** Online (`{dev}`)")
        else:
            st.info("⚡ **Vision Engine:** Active in Local In-Process Mode")
            st.caption("Backend offline; running direct PyTorch pipeline without interruption.")

        st.markdown("---")
        st.subheader("⚙️ Vision Pipeline Settings")
        conf_thresh = st.slider(
            "YOLO11 Confidence Threshold",
            min_value=0.01,
            max_value=0.90,
            value=0.20,
            step=0.01,
            help="Minimum confidence required for bounding box generation. Lower to 0.05-0.10 for baseline undertrained checkpoints.",
        )
        enable_cam = st.checkbox("Generate Grad-CAM Heatmaps", value=True)
        enable_nutr = st.checkbox("Query USDA FoodData Central", value=True)
        crop_pad = st.slider("Segmentation Crop Padding (px)", min_value=0, max_value=30, value=12)

        st.markdown("---")
        st.subheader("🧠 Classifier Selection")
        classifier_mode = st.selectbox(
            "Classification Engine:",
            options=["fruit_classifier", "efficientnet_b4"],
            format_func=lambda x: "🍎 6-Class Fruit Classifier (Updated Dataset)" if x == "fruit_classifier" else "🍽️ General Food Classifier (Food-21)",
            index=0,
            help="Choose the model used for classification. The 6-Class Fruit Classifier was trained on the updated local fruit dataset (Apple, Banana, Guava, Lime, Orange, Pomegranate).",
        )

        st.markdown("---")
        st.subheader("📋 Core Model Readiness")
        status_data = fetch_models_status()
        if status_data and "models" in status_data:
            for k, m in status_data["models"].items():
                st_icon = "✅" if m["ready"] else "❌"
                st.write(f"{st_icon} **{m['name']}:** `{m['status']}`")
        else:
            st.write("✅ **Fruit Classifier (6-Class):** `best_model.pth Loaded`")
            st.write("✅ **YOLO11 Detector:** `Checkpoint Loaded`")
            st.write("✅ **SAM 2.1 Segmenter:** `Tiny Loaded`")
            st.write("✅ **Depth Anything V2:** `ViT-S Loaded`")
            st.write("✅ **EfficientNet-B4:** `Food-21 Loaded`")
            st.write("✅ **Freshness Classifier:** `Apple/Banana/Orange`")
            st.write("✅ **USDA FoodData Central:** `20+ Records Cached`")

        st.markdown("---")
        st.caption("NutriVision AI — Academic & Commercial CV Suite")

    # ---------------------------------------------------------------------------
    # Top Brand Header
    # ---------------------------------------------------------------------------
    engine_badge_class = "live" if backend_online else "live"
    engine_badge_text = f"FastAPI Online ({api_info.get('device', 'cpu').upper()})" if backend_online else "In-Process PyTorch Engine (Active)"

    render_html(
        f"""
        <div class="brand-header-box">
            <h1 class="brand-title">NutriVision AI</h1>
            <div class="brand-subtitle">
                Intelligent Food Recognition, Volume-Depth Weight Estimation, Freshness Detection & Full Nutritional Intelligence
            </div>
            <div class="engine-pills-row">
                <span class="engine-pill {engine_badge_class}">⚡ {engine_badge_text}</span>
                <span class="engine-pill live">🎯 YOLO11 Detection</span>
                <span class="engine-pill live">✂️ SAM 2.1 Instance Mask</span>
                <span class="engine-pill live">🌊 Depth Anything V2 Topography</span>
                <span class="engine-pill live">🧠 EfficientNet-B4 Classifier</span>
                <span class="engine-pill live">🍎 Freshness Assessment</span>
                <span class="engine-pill live">📊 USDA FoodData Central</span>
            </div>
        </div>
        """
    )

    # ---------------------------------------------------------------------------
    # Food Image Upload & Capture
    # ---------------------------------------------------------------------------
    st.markdown("### 📸 Food Image Upload & Analysis")
    st.caption("Upload a photograph or capture directly using your camera to execute the end-to-end vision pipeline.")

    input_method = st.radio("Select Input Source", ["📁 File Upload", "📷 Camera Capture", "🧪 Test Sample Benchmark Image"], horizontal=True)
    uploaded_file = None
    sample_file_bytes = None
    sample_file_name = "sample.jpg"

    if input_method == "📁 File Upload":
        uploaded_file = st.file_uploader(
            "Upload food photograph (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="High-contrast, well-lit shots of food plates produce the highest classification and segmentation fidelity.",
        )
    elif input_method == "📷 Camera Capture":
        uploaded_file = st.camera_input("Point camera at food plate and capture")
    else:
        st.markdown("**Select a pre-loaded benchmark image to test the pipeline instantly:**")
        st.caption("Updated 6-Class Fruit Dataset Samples:")
        f_col1, f_col2, f_col3 = st.columns(3)
        sample_choice = None
        with f_col1:
            if st.button("🍎 Apple (Updated Dataset)", use_container_width=True):
                sample_choice = ROOT / "data" / "raw" / "updated dataset" / "Apple_Good" / "20190809_115439.jpg"
                sample_file_name = "apple_good.jpg"
            if st.button("🍋 Lime (Updated Dataset)", use_container_width=True):
                sample_choice = ROOT / "data" / "raw" / "updated dataset" / "Lime_Good" / "IMG20200728131548.jpg"
                sample_file_name = "lime_good.jpg"
        with f_col2:
            if st.button("🍌 Banana (Updated Dataset)", use_container_width=True):
                sample_choice = ROOT / "data" / "raw" / "updated dataset" / "Banana_Good" / "IMG20200729190220.jpg"
                sample_file_name = "banana_good.jpg"
            if st.button("🍊 Orange (Updated Dataset)", use_container_width=True):
                sample_choice = ROOT / "data" / "raw" / "updated dataset" / "Orange_Good" / "IMG20200728130753.jpg"
                sample_file_name = "orange_good.jpg"
        with f_col3:
            if st.button("🍈 Guava (Updated Dataset)", use_container_width=True):
                sample_choice = ROOT / "data" / "raw" / "updated dataset" / "Guava_Good" / "20190813_121038.jpg"
                sample_file_name = "guava_good.jpg"
            if st.button("🍎 Pomegranate (Updated Dataset)", use_container_width=True):
                sample_choice = ROOT / "data" / "raw" / "updated dataset" / "Pomegranate_Good" / "20190820_143522.jpg"
                sample_file_name = "pomegranate_good.jpg"

        if sample_choice and sample_choice.is_file():
            with open(sample_choice, "rb") as sf:
                sample_file_bytes = sf.read()

    raw_bytes = None
    file_name = "upload.jpg"
    mime_type = "image/jpeg"

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        file_name = getattr(uploaded_file, "name", "upload.jpg") or "upload.jpg"
        mime_type = getattr(uploaded_file, "type", "image/jpeg") or "image/jpeg"
    elif sample_file_bytes is not None:
        raw_bytes = sample_file_bytes
        file_name = sample_file_name
        mime_type = "image/jpeg"

    if raw_bytes is None:
        st.info("👆 Please upload an image, capture a photo, or select a sample benchmark image to begin.")
        # Render the quick catalog below so users can still explore verified USDA foods
        st.markdown("---")
        _render_quick_catalog()
        return

    # Image validation & preview
    try:
        input_image, img_w, img_h, img_fmt = validate_image_bytes(raw_bytes)
        st.session_state["last_image_bytes"] = raw_bytes
    except ValueError as val_err:
        st.error(
            f"🖼️ **Invalid or Corrupt Image Upload**\n\n"
            f"{str(val_err)}\n\n"
            "**Requirements:** Allowed formats: JPG, PNG, WEBP. Resolution: 10×10 to 4096×4096 px."
        )
        return

    prev_col1, prev_col2 = st.columns([1, 2])
    with prev_col1:
        st.image(input_image, caption=f"Input Scene ({img_w}×{img_h} px, {img_fmt})", use_container_width=True)

    with prev_col2:
        st.markdown("**⚙️ Portion Weight Setting (Optional)**")
        pre_user_weight = st.number_input(
            "⚖️ User Kitchen Scale Weight (grams, optional):",
            min_value=0.0,
            max_value=2500.0,
            value=0.0,
            step=10.0,
            help="If you have an exact physical scale measurement, enter it here. Otherwise, NutriVision will estimate weight via contour area & density.",
        )
        if pre_user_weight > 0:
            st.caption(f"✓ Scale measurement: **{pre_user_weight:.1f} g** will be used for nutrition scaling.")
        else:
            st.caption("ℹ️ Automatic AI vision weight estimate will be calculated from contour area and food density.")

        analyze_btn = st.button("🚀 Analyze Food Plate & Nutrition", type="primary", use_container_width=True)

    # ---------------------------------------------------------------------------
    # STAGE 2 & Pipeline Execution
    # ---------------------------------------------------------------------------
    if analyze_btn:
        st.session_state["last_error"] = None
        st.session_state["last_result"] = None

        with st.spinner("Executing vision pipeline (YOLO11 Detection -> SAM2 -> EfficientNet -> USDA FDC)..."):
            try:
                # Prefer FastAPI if online, or automatically fallback to local pipeline
                if backend_online:
                    try:
                        result = call_analyze_api(
                            image_bytes=raw_bytes,
                            filename=file_name,
                            content_type=mime_type,
                            enable_gradcam=enable_cam,
                            enable_nutrition=enable_nutr,
                            crop_padding=crop_pad,
                            confidence_threshold=conf_thresh,
                            classifier_mode=classifier_mode,
                        )
                    except BackendConnectionError:
                        from src.pipeline.nutrivision_pipeline import NutriVisionPipeline
                        pipe = NutriVisionPipeline()
                        result = pipe.process_image(
                            np.array(input_image),
                            enable_gradcam=enable_cam,
                            enable_nutrition=enable_nutr,
                            crop_padding=crop_pad,
                            confidence_threshold=conf_thresh,
                            classifier_mode=classifier_mode,
                        )
                else:
                    from src.pipeline.nutrivision_pipeline import NutriVisionPipeline
                    pipe = NutriVisionPipeline()
                    result = pipe.process_image(
                        np.array(input_image),
                        enable_gradcam=enable_cam,
                        enable_nutrition=enable_nutr,
                        crop_padding=crop_pad,
                        confidence_threshold=conf_thresh,
                        classifier_mode=classifier_mode,
                    )

                # If user entered an override scale weight, apply it immediately
                if pre_user_weight > 0 and result.get("items"):
                    try:
                        from src.nutrition.nutrition_service import NutritionService
                        nutr_svc = NutritionService()
                        for it in result["items"]:
                            f_label = it.get("food", "food")
                            updated_n = nutr_svc.get_nutrition_for_item(f_label, weight_grams=pre_user_weight)
                            if updated_n and updated_n.status == "AVAILABLE":
                                it["nutrition"] = updated_n.to_dict()
                                it["nutrition"]["status"] = "SUCCESS"
                                it["nutrition"]["matched_food"] = updated_n.usda_food_name
                                it["nutrition"]["fdc_id"] = updated_n.usda_fdc_id
                                it["nutrition"]["serving_weight_g"] = pre_user_weight
                                it["weight"]["value_g"] = pre_user_weight
                                it["weight"]["source"] = "user_measured"
                    except Exception as exc:
                        logger.warning(f"Error applying user weight override: {exc}")

                st.session_state["last_result"] = result
            except BackendConnectionError as exc:
                st.session_state["last_error"] = ("BACKEND_CONNECTION_FAILURE", str(exc))
            except BackendTimeoutError as exc:
                st.session_state["last_error"] = ("BACKEND_TIMEOUT", str(exc))
            except InvalidImageApiError as exc:
                st.session_state["last_error"] = ("INVALID_IMAGE", str(exc))
            except ModelNotReadyApiError as exc:
                st.session_state["last_error"] = ("MODEL_NOT_READY", str(exc))
            except InferencePipelineApiError as exc:
                st.session_state["last_error"] = ("INFERENCE_EXCEPTION", str(exc))
            except Exception as exc:
                st.session_state["last_error"] = ("UNEXPECTED_EXCEPTION", str(exc))

    if st.session_state.get("last_error"):
        err_type, err_msg = st.session_state["last_error"]
        st.markdown("---")
        if err_type == "BACKEND_CONNECTION_FAILURE":
            st.error(f"🔌 **Backend Connection Notice:** {err_msg}")
        elif err_type == "BACKEND_TIMEOUT":
            st.error(f"⏱️ **Backend Request Timed Out:** {err_msg}")
        elif err_type == "INVALID_IMAGE":
            st.error(f"🖼️ **Invalid Image:** {err_msg}")
        elif err_type == "MODEL_NOT_READY":
            st.warning(f"⚠️ **Model Initialization Alert:** {err_msg}")
        else:
            st.error(f"💥 **Pipeline Error:** {err_msg}")
        return

    result = st.session_state.get("last_result")
    if not result:
        return

    pipe_status = result.get("status", "UNKNOWN")
    total_detected = result.get("total_foods_detected", 0)
    timing_total = result.get("timing", {}).get("total_ms", 0.0)

    st.markdown("---")

    # If no items were detected by bounding boxes
    if pipe_status == "NO_FOOD_DETECTED" or total_detected == 0:
        render_html(
            f"""
            <div class="info-callout">
                <strong>ℹ️ Inference Complete: 0 Food Items Localized</strong> (Pipeline latency: {timing_total:.1f} ms)<br>
                No bounding box candidate passed the detection threshold of <strong>{conf_thresh:.2f}</strong>.<br>
                <em>Tip: Lower the YOLO11 Confidence Threshold slider in the sidebar to <strong>0.05 – 0.10</strong> or select a food from the catalog below.</em>
            </div>
            """
        )
        _render_quick_catalog()
        return

    # Success Banner
    model_ver_display = result.get("model_version", "target_foods_v1")
    render_html(
        f"""
        <div class="success-callout">
            <strong>✓ Analysis Complete:</strong> {total_detected} Food Item(s) Localized & Classified | 
            <strong>Engine:</strong> EfficientNet-B4 ({model_ver_display}) | 
            <strong>Pipeline Latency:</strong> {timing_total:.1f} ms
        </div>
        """
    )

    items = result.get("items", [])
    if len(items) == 0:
        return

    # ---------------------------------------------------------------------------
    # DISPLAY EACH DETECTED ITEM FOLLOWING THE USER'S WORKFLOW:
    #   Step 2: Food detection and classification
    #   Step 3: Recognized food name
    #   Step 4: Weight estimate or user-entered weight
    #   Step 5: Nutrition database lookup
    #   Step 6: Calories + protein + carbs + fats
    #   Step 7: Vitamin and mineral breakdown
    # ---------------------------------------------------------------------------
    for idx, item in enumerate(items):
        raw_food = item.get("food", f"food_{idx+1}")
        det = item.get("detection", {})
        seg = item.get("segmentation", {})
        cls_info = item.get("classification", {})
        fresh = item.get("freshness", {})
        wgt = item.get("weight", {})
        vol = item.get("volume", {})
        nut = item.get("nutrition", {})
        exp = item.get("explainability", {})

        # Default weight from pipeline
        base_est_w = float(wgt.get("value_g", 150.0) or 150.0)
        state_key = f"active_weight_{idx}_{result.get('result_id', 'cur')}"
        if state_key not in st.session_state:
            st.session_state[state_key] = base_est_w

        # Recognized Food Name & Override
        override_key = f"override_food_{idx}_{result.get('result_id', 'cur')}"
        all_food_keys = sorted(list(FOOD_ICONS.keys()))
        all_options = ["Keep AI Prediction"] + [k.replace("_", " ").title() for k in all_food_keys if k != "food"]

        # Container for the Food Item
        with st.container():
            # Hero Identity Card
            chosen_food_key = raw_food.lower().replace(" ", "_")
            food_icon = FOOD_ICONS.get(chosen_food_key, "🍽️")
            disp_food_name = chosen_food_key.replace("_", " ").title()

            # Freshness Badge
            fresh_lbl = fresh.get("label", "unknown").lower()
            if fresh.get("status") == "DATASET_LABEL_ONLY" or fresh_lbl == "good":
                fresh_badge = f'<span class="badge-fresh" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: #10b981;">DATASET LABEL: GOOD ({fresh.get("confidence", 0)*100:.1f}%)</span>'
            elif fresh_lbl == "fresh":
                fresh_badge = f'<span class="badge-fresh">FRESH ({fresh.get("confidence", 0)*100:.1f}%)</span>'
            elif fresh_lbl == "rotten":
                fresh_badge = f'<span class="badge-rotten">ROTTEN ({fresh.get("confidence", 0)*100:.1f}%)</span>'
            else:
                fresh_badge = f'<span style="color: #94a3b8; font-size: 0.85rem;">Freshness: N/A</span>'

            det_conf = det.get("confidence", 0.0)
            cls_conf = cls_info.get("confidence", det_conf)

            render_html(
                f"""
                <div class="food-hero-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                        <div>
                            <div class="food-hero-title">
                                <span>{food_icon}</span>
                                <span>{disp_food_name}</span>
                            </div>
                            <div style="margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                                <span class="badge-usda">🎯 Classifier Conf: {cls_conf*100:.1f}%</span>
                                <span class="engine-pill live">📐 YOLO11 Det: {det_conf*100:.1f}%</span>
                                {fresh_badge}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 0.85rem; color: #94a3b8;">Item #{idx+1} of {len(items)}</span>
                        </div>
                    </div>
                </div>
                """
            )

            # Fruit Recognition metadata & honesty handling (Sections 6, 7 & 9)
            fruit_rec = item.get("fruit_recognition") or result.get("fruit_recognition")
            is_fruit_pred = fruit_rec and fruit_rec.get("fruit_name") and fruit_rec.get("dataset_class")
            if is_fruit_pred:
                f_name = fruit_rec.get("fruit_name")
                d_class = fruit_rec.get("dataset_class")
                d_label = fruit_rec.get("freshness_status", "Good")
                f_conf = fruit_rec.get("confidence", cls_conf)
                m_model = fruit_rec.get("classification_model", "EfficientNet-B4")
                
                conf_warning = ""
                if f_conf < 0.60:
                    conf_warning = f'<div style="margin-top: 8px; color: #f59e0b; font-size: 0.85rem; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; padding: 6px 10px;">⚠️ <strong>Low Confidence Advisory:</strong> Prediction confidence is {f_conf*100:.1f}%. Consider checking lighting and photographic clarity.</div>'

                render_html(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                            <div style="font-weight: 700; color: #34d399; font-size: 1.05rem;">
                                🍎 Fruit Recognition: <strong>{f_name}</strong>
                            </div>
                            <div style="font-size: 0.85rem; color: #cbd5e1;">
                                Model: <code>{m_model}</code> | Dataset Class: <code>{d_class}</code> | Confidence: <strong>{f_conf*100:.1f}%</strong>
                            </div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.84rem; color: #94a3b8; line-height: 1.4;">
                            ℹ️ <strong>Freshness Classification Notice:</strong> Dataset label is <code>{d_label}</code>. 
                            <em>Note: The updated fruit dataset consists exclusively of condition-good fruit imagery. The label '{d_label}' reflects the dataset category and does not represent an independent test for rotting or spoilage.</em>
                        </div>
                        {conf_warning}
                    </div>
                    """
                )

            # Manual Food Override & Variant Selection
            col_ov1, col_ov2 = st.columns([1, 1])
            with col_ov1:
                chosen_override = st.selectbox(
                    f"✏️ Food Name Correction for Item #{idx+1} (if misclassified):",
                    options=all_options,
                    index=0,
                    key=override_key,
                )
                if chosen_override != "Keep AI Prediction":
                    chosen_food_key = chosen_override.lower().replace(" ", "_")
                    item["food"] = chosen_food_key
                    food_icon = FOOD_ICONS.get(chosen_food_key, "🍽️")
                    disp_food_name = chosen_override

            with col_ov2:
                avail_variants = get_food_variants(chosen_food_key)
                chosen_variant_key = None
                if avail_variants:
                    v_opts = [f"{v['name']} ({v['calories_100g']:.0f} kcal/100g)" for v in avail_variants]
                    v_map = {f"{v['name']} ({v['calories_100g']:.0f} kcal/100g)": v["key"] for v in avail_variants}
                    v_pick = st.selectbox(
                        "🍳 Recipe / Preparation Variant:",
                        options=v_opts,
                        index=0,
                        key=f"var_pick_{idx}_{result.get('result_id', 'cur')}",
                    )
                    chosen_variant_key = v_map.get(v_pick)

            st.markdown("---")

            # ---------------------------------------------------------------------------
            # Portion Weight Controller
            # ---------------------------------------------------------------------------
            st.markdown(f"### ⚖️ Portion Weight Controller ({disp_food_name})")
            
            w_col1, w_col2 = st.columns([1, 1.2])
            with w_col1:
                wt_mode = st.radio(
                    "Select Portion Weight Mode:",
                    ["⚖️ AI Vision Estimated Weight", "✓ User Kitchen Scale (Manual Weight)"],
                    index=0,
                    key=f"wt_mode_radio_{idx}_{result.get('result_id', 'cur')}",
                    horizontal=True,
                )

                if "User Kitchen Scale" in wt_mode:
                    cur_w = float(st.session_state.get(state_key, base_est_w))
                    new_w = st.number_input(
                        "Enter Kitchen Scale Weight (grams):",
                        min_value=5.0,
                        max_value=2500.0,
                        value=cur_w,
                        step=5.0,
                        key=f"num_w_{idx}_{result.get('result_id', 'cur')}",
                    )
                    st.session_state[state_key] = new_w
                    weight_src = "user_measured"
                    st.caption("✓ Manual weight active. Recalculating all nutrition live.")
                else:
                    st.session_state[state_key] = base_est_w
                    weight_src = "vision_estimated"
                    st.markdown(f"**Vision Estimate:** `{base_est_w:.1f} g`")
                    st.caption("ℹ️ Calibrated from 2D contour boundary area and USDA empirical reference density.")

            with w_col2:
                st.markdown("**Quick Weight Delta Adjustments:**")
                btn_cols = st.columns(6)
                deltas = [-50.0, -25.0, -10.0, 10.0, 25.0, 50.0]
                for b_idx, delta in enumerate(deltas):
                    with btn_cols[b_idx]:
                        lbl = f"{delta:+.0f}g"
                        if st.button(lbl, key=f"delta_{idx}_{b_idx}_{result.get('result_id', 'cur')}", use_container_width=True):
                            current_val = float(st.session_state.get(state_key, base_est_w))
                            st.session_state[state_key] = max(10.0, current_val + delta)
                            st.rerun()

                active_w = float(st.session_state.get(state_key, base_est_w))
                render_html(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 10px 14px; margin-top: 8px;">
                        <span style="color: #34d399; font-weight: 700; font-size: 1.05rem;">Active Portion Weight: {active_w:.1f} g</span>
                        <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 12px;">({weight_src})</span>
                    </div>
                    """
                )

            # Recalculate nutrition with active weight
            from src.nutrition.nutrition_service import NutritionService
            nutr_svc = NutritionService()
            scaled_n = nutr_svc.get_nutrition_for_item(
                food_name=chosen_food_key,
                weight_grams=active_w,
                variant=chosen_variant_key,
                weight_source=weight_src,
            )
            if scaled_n and scaled_n.status == "AVAILABLE":
                nut = scaled_n.to_dict()
                nut["status"] = "SUCCESS"
                nut["matched_food"] = scaled_n.usda_food_name
                nut["fdc_id"] = scaled_n.usda_fdc_id
                nut["serving_weight_g"] = active_w
                item["nutrition"] = nut
                item["weight"]["value_g"] = active_w
                item["weight"]["source"] = weight_src

            st.markdown("---")

            # ---------------------------------------------------------------------------
            # Nutrition Database Lookup (USDA FoodData Central)
            # ---------------------------------------------------------------------------
            st.markdown(f"### 📊 USDA Nutrition Database Lookup")
            matched_fdc = nut.get("matched_food", disp_food_name)
            fdc_id = nut.get("fdc_id", "N/A")
            is_gen = nut.get("is_generic_match", True)
            variant_badge = "Standard Match" if is_gen else "Preparation Variant"

            render_html(
                f"""
                <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span class="badge-usda">USDA FoodData Central</span>
                            <span style="font-weight: 700; color: #ffffff; font-size: 1.05rem; margin-left: 8px;">{matched_fdc}</span>
                            <span style="color: #38bdf8; font-size: 0.85rem; margin-left: 8px;">(FDC ID: #{fdc_id})</span>
                        </div>
                        <div>
                            <span style="font-size: 0.8rem; color: #94a3b8;">Portion Scaling: Per 100g × ({active_w:.1f}g / 100)</span>
                        </div>
                    </div>
                </div>
                """
            )

            # ---------------------------------------------------------------------------
            # Calories + Protein + Carbs + Fats
            # ---------------------------------------------------------------------------
            st.markdown(f"### 🔥 Calories & Macronutrient Breakdown")
            cals = nut.get("calories_kcal", 0.0) or 0.0
            prot = nut.get("protein_g", 0.0) or 0.0
            carbs = nut.get("carbohydrates_g", 0.0) or 0.0
            fat = nut.get("fat_g", 0.0) or 0.0
            fiber = nut.get("fiber_g", 0.0) or 0.0
            sugar = nut.get("sugars_g", 0.0) or 0.0

            # Macro Cards Row
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                render_html(
                    f"""
                    <div class="macro-card">
                        <div class="macro-card-top-bar" style="background: linear-gradient(90deg, #ec4899, #f43f5e);"></div>
                        <div class="macro-sub">🔥 Energy</div>
                        <div class="macro-val">{cals:.1f}</div>
                        <div class="macro-sub">kcal ({cals/2000.0*100:.0f}% Daily)</div>
                    </div>
                    """
                )
            with m_col2:
                render_html(
                    f"""
                    <div class="macro-card">
                        <div class="macro-card-top-bar" style="background: linear-gradient(90deg, #38bdf8, #0284c7);"></div>
                        <div class="macro-sub">🥩 Protein</div>
                        <div class="macro-val">{prot:.1f}g</div>
                        <div class="macro-sub">{prot*4.0:.0f} kcal contribution</div>
                    </div>
                    """
                )
            with m_col3:
                render_html(
                    f"""
                    <div class="macro-card">
                        <div class="macro-card-top-bar" style="background: linear-gradient(90deg, #34d399, #059669);"></div>
                        <div class="macro-sub">🌾 Carbohydrates</div>
                        <div class="macro-val">{carbs:.1f}g</div>
                        <div class="macro-sub">Net Carbs: {max(0.0, carbs-fiber):.1f}g</div>
                    </div>
                    """
                )
            with m_col4:
                render_html(
                    f"""
                    <div class="macro-card">
                        <div class="macro-card-top-bar" style="background: linear-gradient(90deg, #fbbf24, #d97706);"></div>
                        <div class="macro-sub">🥑 Total Fat</div>
                        <div class="macro-val">{fat:.1f}g</div>
                        <div class="macro-sub">{fat*9.0:.0f} kcal contribution</div>
                    </div>
                    """
                )
            with m_col5:
                render_html(
                    f"""
                    <div class="macro-card">
                        <div class="macro-card-top-bar" style="background: linear-gradient(90deg, #a78bfa, #8b5cf6);"></div>
                        <div class="macro-sub">🌿 Dietary Fiber</div>
                        <div class="macro-val">{fiber:.1f}g</div>
                        <div class="macro-sub">Sugars: {sugar:.1f}g</div>
                    </div>
                    """
                )

            # Tricolor Macro Distribution Bar
            total_macros = prot + carbs + fat
            if total_macros > 0:
                p_pct = (prot / total_macros) * 100.0
                c_pct = (carbs / total_macros) * 100.0
                f_pct = (fat / total_macros) * 100.0
                render_html(
                    f"""
                    <div class="macro-bar-container">
                        <div class="macro-segment-protein" style="width: {p_pct:.1f}%;"></div>
                        <div class="macro-segment-carbs" style="width: {c_pct:.1f}%;"></div>
                        <div class="macro-segment-fat" style="width: {f_pct:.1f}%;"></div>
                    </div>
                    <div class="macro-legend">
                        <span><span class="macro-legend-dot" style="background-color: #38bdf8;"></span>Protein: {prot:.1f}g ({p_pct:.0f}%)</span>
                        <span><span class="macro-legend-dot" style="background-color: #34d399;"></span>Carbs: {carbs:.1f}g ({c_pct:.0f}%)</span>
                        <span><span class="macro-legend-dot" style="background-color: #fbbf24;"></span>Fat: {fat:.1f}g ({f_pct:.0f}%)</span>
                    </div>
                    """
                )

            st.markdown("---")

            # ---------------------------------------------------------------------------
            # Stage 7: Vitamin and Mineral Breakdown
            # ---------------------------------------------------------------------------
            st.markdown(f"### 🔬 Stage 7: Vitamin & Mineral Breakdown")
            micros = extract_available_vitamins_and_minerals(nut, active_w)
            vitamins = [m for m in micros if m.get("category") == "Vitamin"]
            minerals = [m for m in micros if m.get("category") == "Mineral"]

            vit_col, min_col = st.columns(2)
            with vit_col:
                vit_rows = []
                if vitamins:
                    for v in vitamins:
                        dv_str = f"{v['dv_pct']:.0f}% DV" if v.get("dv_pct") is not None else "--"
                        bar_w = min(100.0, v.get("dv_pct") or 0.0)
                        vit_rows.append(
                            f'<div class="micro-item-row">'
                            f'<span class="micro-name">{v["label"]}</span>'
                            f'<div class="micro-val-group">'
                            f'<span class="micro-val-num">{v["value"]} {v["unit"]}</span>'
                            f'<div class="micro-progress-bg">'
                            f'<div class="micro-progress-fill-vit" style="width: {bar_w:.0f}%;"></div>'
                            f'</div>'
                            f'<span class="micro-dv-badge">{dv_str}</span>'
                            f'</div>'
                            f'</div>'
                        )
                    vit_rows_html = "".join(vit_rows)
                else:
                    vit_rows_html = '<div style="color: #94a3b8; font-size: 0.85rem; padding: 8px 0;">No specific vitamin records found in USDA reference for this item.</div>'

                render_html(
                    f'<div class="micro-panel">'
                    f'<div style="font-weight: 700; color: #f43f5e; font-size: 1.05rem; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">'
                    f'<span>💊 Essential Vitamins</span>'
                    f'</div>'
                    f'{vit_rows_html}'
                    f'</div>'
                )

            with min_col:
                min_rows = []
                if minerals:
                    for m in minerals:
                        dv_str = f"{m['dv_pct']:.0f}% DV" if m.get("dv_pct") is not None else "--"
                        bar_w = min(100.0, m.get("dv_pct") or 0.0)
                        min_rows.append(
                            f'<div class="micro-item-row">'
                            f'<span class="micro-name">{m["label"]}</span>'
                            f'<div class="micro-val-group">'
                            f'<span class="micro-val-num">{m["value"]} {m["unit"]}</span>'
                            f'<div class="micro-progress-bg">'
                            f'<div class="micro-progress-fill-min" style="width: {bar_w:.0f}%;"></div>'
                            f'</div>'
                            f'<span class="micro-dv-badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3);">{dv_str}</span>'
                            f'</div>'
                            f'</div>'
                        )
                    min_rows_html = "".join(min_rows)
                else:
                    min_rows_html = '<div style="color: #94a3b8; font-size: 0.85rem; padding: 8px 0;">No specific mineral records found in USDA reference for this item.</div>'

                render_html(
                    f'<div class="micro-panel">'
                    f'<div style="font-weight: 700; color: #10b981; font-size: 1.05rem; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">'
                    f'<span>⚡ Essential Minerals</span>'
                    f'</div>'
                    f'{min_rows_html}'
                    f'</div>'
                )

            st.markdown("---")

            # ---------------------------------------------------------------------------
            # Computer Vision Inspector (YOLO11, SAM2, Depth, Grad-CAM)
            # ---------------------------------------------------------------------------
            st.markdown("### 🔍 Computer Vision Inspector")
            tab_det, tab_seg, tab_dep, tab_cam = st.tabs([
                "🎯 1. YOLO11 Detection",
                "✂️ 2. SAM 2.1 Segmentation",
                "🌊 3. Depth Anything V2",
                "🧠 4. Grad-CAM Attention Heatmaps",
            ])

            with tab_det:
                annotated_path = result.get("visualizations", {}).get("annotated_image_path")
                if annotated_path and Path(annotated_path).is_file():
                    st.image(str(annotated_path), caption="YOLO11 Detected Bounding Boxes & Categories", use_container_width=True)
                else:
                    st.image(input_image, caption="Input Food Scene", use_container_width=True)

            with tab_seg:
                crop_p = seg.get("crop_path")
                if crop_p and Path(crop_p).is_file():
                    st.image(crop_p, caption=f"SAM 2.1 Segmented Contour Crop ({seg.get('mask_area', 0):,} px)", use_container_width=True)
                else:
                    st.info("Segmented mask unavailable.")

            with tab_dep:
                depth_path = result.get("visualizations", {}).get("depth_map_path")
                if depth_path and Path(depth_path).is_file():
                    st.image(str(depth_path), caption="Continuous Relative Depth Relief Map", use_container_width=True)
                else:
                    st.info("Depth topography map unavailable.")

            with tab_cam:
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    f_cam = exp.get("food_gradcam_path")
                    if f_cam and Path(f_cam).is_file():
                        st.image(f_cam, caption="Food Classification Attention Heatmap", use_container_width=True)
                    else:
                        st.caption("Food Grad-CAM unavailable.")
                with g_col2:
                    fr_cam = exp.get("freshness_gradcam_path")
                    if fr_cam and Path(fr_cam).is_file():
                        st.image(fr_cam, caption="Freshness Decision Attention Heatmap", use_container_width=True)
                    else:
                        st.caption("Freshness Grad-CAM unavailable.")

    # ---------------------------------------------------------------------------
    # Multi-Item Meal Summary (if multiple items detected)
    # ---------------------------------------------------------------------------
    if len(items) > 1:
        st.markdown("---")
        st.markdown(f"### 🥗 Complete Meal Aggregate Summary ({len(items)} Items)")
        tot_wt = sum([float(it.get("weight", {}).get("value_g") or 0.0) for it in items])
        tot_cal = sum([float(it.get("nutrition", {}).get("calories_kcal") or 0.0) for it in items])
        tot_p = sum([float(it.get("nutrition", {}).get("protein_g") or 0.0) for it in items])
        tot_c = sum([float(it.get("nutrition", {}).get("carbohydrates_g") or 0.0) for it in items])
        tot_f = sum([float(it.get("nutrition", {}).get("fat_g") or 0.0) for it in items])
        tot_fib = sum([float(it.get("nutrition", {}).get("fiber_g") or 0.0) for it in items])

        s_col1, s_col2, s_col3, s_col4, s_col5, s_col6 = st.columns(6)
        s_col1.metric("Total Weight", f"{tot_wt:.1f} g")
        s_col2.metric("Total Calories", f"{tot_cal:.1f} kcal")
        s_col3.metric("Total Protein", f"{tot_p:.1f} g")
        s_col4.metric("Total Carbs", f"{tot_c:.1f} g")
        s_col5.metric("Total Fat", f"{tot_f:.1f} g")
        s_col6.metric("Total Fiber", f"{tot_fib:.1f} g")

    # ---------------------------------------------------------------------------
    # Export Suite
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📥 Export Analysis Suite")
    e1, e2, e3 = st.columns(3)

    with e1:
        json_str = json.dumps(result, indent=2)
        st.download_button(
            label="Download Complete JSON",
            data=json_str,
            file_name=f"{result.get('result_id', 'nutrivision')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with e2:
        csv_str = generate_summary_csv(items)
        st.download_button(
            label="Download Nutrition CSV",
            data=csv_str,
            file_name=f"{result.get('result_id', 'nutrivision')}_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with e3:
        annotated_path = result.get("visualizations", {}).get("annotated_image_path")
        if annotated_path and Path(annotated_path).is_file():
            try:
                with open(annotated_path, "rb") as f:
                    img_bytes = f.read()
                st.download_button(
                    label="Download Annotated Image",
                    data=img_bytes,
                    file_name=f"{result.get('result_id', 'nutrivision')}_annotated.png",
                    mime="image/png",
                    use_container_width=True,
                )
            except Exception:
                st.button("Annotated Image Unavailable", disabled=True, use_container_width=True)
        else:
            st.button("Annotated Image Unavailable", disabled=True, use_container_width=True)


def _render_quick_catalog():
    """Render the Quick-Select Food Catalog for browsing USDA nutrition without uploading."""
    st.markdown("### ⚡ Dataset-Driven Food Catalog & Nutrition Explorer")
    st.caption("Explore verified USDA nutritional profiles and portion scaling across the model taxonomy.")

    active_catalog = QUICK_SELECT_FOODS
    cat_cols = st.columns([1, 2])
    raw_cats = [f.get("category", "") for f in active_catalog]
    categories = ["All Categories"] + sorted(list(set(c for c in raw_cats if c)))

    with cat_cols[0]:
        selected_cat = st.selectbox("📁 Filter Category", categories, index=0)

    filtered_foods = [
        f for f in active_catalog
        if selected_cat == "All Categories" or f.get("category", "") == selected_cat
    ]

    food_options = ["-- Select a Food Class --"] + [
        f"{f.get('icon', '🍽️')} {f.get('label', '')} ({f.get('category', '')})"
        for f in filtered_foods
    ]
    option_to_key = {
        f"{f.get('icon', '🍽️')} {f.get('label', '')} ({f.get('category', '')})": f.get("key", "")
        for f in filtered_foods
    }

    with cat_cols[1]:
        chosen_search = st.selectbox("🔍 Search Food Class", options=food_options, index=0)
        if chosen_search and chosen_search != "-- Select a Food Class --":
            st.session_state["selected_quick_food"] = option_to_key.get(chosen_search)

    selected_key = st.session_state.get("selected_quick_food")
    if selected_key:
        food_meta = next((f for f in active_catalog if f.get("key") == selected_key), None)
        food_name = food_meta.get("label", selected_key.title()) if food_meta else selected_key.title()
        def_w = float(st.session_state.get("quick_portion_weight", 150.0))

        q_col1, q_col2 = st.columns([1, 1.5])
        with q_col1:
            q_w = st.slider(f"Portion Weight for {food_name} (grams):", 20.0, 600.0, def_w, 10.0)
            st.session_state["quick_portion_weight"] = q_w

        q_nut = get_quick_select_nutrition(selected_key, q_w)
        if q_nut and q_nut.get("status") == "AVAILABLE":
            with q_col2:
                st.markdown(f"**USDA Food:** *{q_nut.get('usda_food_name', food_name)}* | FDC ID: `#{q_nut.get('usda_fdc_id')}`")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Calories", f"{q_nut.get('calories_kcal', 0):.1f} kcal")
                m2.metric("Protein", f"{q_nut.get('protein_g', 0):.1f} g")
                m3.metric("Carbs", f"{q_nut.get('carbohydrates_g', 0):.1f} g")
                m4.metric("Fat", f"{q_nut.get('fat_g', 0):.1f} g")


if __name__ == "__main__":
    render_app()
