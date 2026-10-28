"""NutriVision AI - Fruit Single-Image Prediction Script
Runs inference on a single image using the trained 6-class fruit classifier.
Returns structured JSON output including fruit name, dataset class, freshness note,
confidence, and nutrition information fetched from USDA FoodData Central.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Project root path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.classification.fruit_classifier import FruitClassifier
from src.config.settings import settings
from src.nutrition.nutrition_service import get_nutrition_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("predict_fruit")


def predict_fruit_image(
    image_path: Path | str,
    checkpoint_path: Path | str | None = None,
    with_nutrition: bool = True,
    serving_weight_g: float = 150.0,
) -> dict:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at '{image_path}'")

    ckpt = Path(checkpoint_path) if checkpoint_path else Path(settings.model_paths.fruit_classifier_path)
    if not ckpt.is_absolute():
        ckpt = ROOT_DIR / ckpt

    if not ckpt.exists():
        raise FileNotFoundError(
            f"Fruit classifier checkpoint not found at '{ckpt}'. "
            "Please train the model first with 'python scripts/train_fruit_classifier.py'."
        )

    classifier = FruitClassifier.from_checkpoint(ckpt)
    prediction = classifier.predict_image_path(image_path)

    response = {
        "fruit_name": prediction["fruit_name"],
        "dataset_class": prediction["dataset_class"],
        "freshness_status": prediction["freshness_status"],
        "freshness_note": prediction["freshness_note"],
        "confidence": prediction["confidence"],
        "classification_model": prediction["classification_model"],
        "is_low_confidence": prediction["is_low_confidence"],
        "nutrition": {
            "calories": None,
            "protein": None,
            "carbohydrates": None,
            "fat": None,
        },
    }

    if with_nutrition:
        try:
            nutrition_svc = get_nutrition_service()
            fruit_name = prediction["fruit_name"].lower()
            nutr = nutrition_svc.get_nutrition_for_item(fruit_name, serving_weight_g)
            if nutr and nutr.status == "AVAILABLE":
                response["nutrition"] = {
                    "calories": nutr.calories_kcal,
                    "protein": nutr.protein_g,
                    "carbohydrates": nutr.carbohydrates_g,
                    "fat": nutr.fat_g,
                    "fiber": nutr.fiber_g,
                    "serving_weight_g": serving_weight_g,
                    "fdc_id": nutr.usda_fdc_id,
                    "data_source": nutr.source,
                }
            else:
                response["nutrition_status"] = getattr(nutr, "status", "UNAVAILABLE")
        except Exception as e:
            logger.warning(f"Could not retrieve nutrition for '{prediction['fruit_name']}': {e}")
            response["nutrition_error"] = str(e)

    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fruit class for a single image")
    parser.add_argument("image_path", type=str, help="Path to input fruit image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(settings.model_paths.fruit_classifier_path),
        help="Path to fruit classifier checkpoint",
    )
    parser.add_argument(
        "--no-nutrition",
        action="store_true",
        help="Disable USDA nutrition analysis lookup",
    )
    parser.add_argument(
        "--weight-g",
        type=float,
        default=150.0,
        help="Estimated serving weight in grams for nutrition calculation",
    )

    args = parser.parse_args()

    result = predict_fruit_image(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint,
        with_nutrition=not args.no_nutrition,
        serving_weight_g=args.weight_g,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
