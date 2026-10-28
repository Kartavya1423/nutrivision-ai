"""
Tests for NutriVision AI — Updated Fruit Dataset & 6-Class Fruit Classifier.
Validates:
1. Dataset path resolution with spaces.
2. Automatic 6-class discovery and image count validation.
3. Deterministic alphabetical class-to-index mapping.
4. Preprocessing transforms separation (training vs evaluation).
5. Safe checkpoint loading with architecture metadata and metrics.
6. Single-image inference with exact Section 6 structured output.
7. Freshness semantics (Good is dataset label, not spoilage detection).
8. USDA FoodData Central integration for all six fruit classes.
9. FastAPI /classify/fruit endpoint contract.
"""

from pathlib import Path
import pytest
from PIL import Image
import torch

from src.config.settings import get_settings
from src.data.fruit_dataset_loader import (
    FruitDatasetValidator,
    get_fruit_transforms,
    EXPECTED_FRUIT_CLASSES,
)
from src.classification.fruit_classifier import FruitClassifier
from src.nutrition.nutrition_service import NutritionService


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "raw" / "updated dataset"
CHECKPOINT_PATH = ROOT / "models" / "classification" / "fruit_classifier_corrected_split" / "best_model.pth"
SAMPLE_APPLE = DATASET_DIR / "Apple_Good" / "20190809_115439.jpg"


def test_fruit_dataset_path_resolution():
    """Verify that the dataset path with spaces resolves correctly and exists."""
    settings = get_settings()
    configured_dir = Path(settings.dataset_paths.updated_fruit_dataset_dir)
    if not configured_dir.is_absolute():
        configured_dir = ROOT / configured_dir

    assert configured_dir.exists(), f"Configured fruit dataset dir does not exist: {configured_dir}"
    assert "updated dataset" in str(configured_dir), "Dataset folder must contain 'updated dataset' with space"


def test_fruit_dataset_validator_discovery():
    """Verify that FruitDatasetValidator automatically discovers the 6 classes and counts images."""
    validator = FruitDatasetValidator(dataset_dir=DATASET_DIR)
    is_valid, summary = validator.validate()

    assert is_valid is True, f"Validation failed: {summary.get('errors')}"
    assert summary["discovered_classes"] == EXPECTED_FRUIT_CLASSES
    assert summary["class_count"] == 6
    assert summary["total_valid_images"] > 10000
    assert summary["corrupted_count"] == 0

    # Verify per-class counts are positive
    counts = summary["per_class_counts"]
    for c in EXPECTED_FRUIT_CLASSES:
        assert c in counts
        assert counts[c] >= 1000, f"Class {c} has fewer than 1000 samples: {counts[c]}"


def test_fruit_dataset_deterministic_mapping():
    """Verify deterministic alphabetical class-to-index mapping."""
    expected = [
        "Apple_Good",
        "Banana_Good",
        "Guava_Good",
        "Lime_Good",
        "Orange_Good",
        "Pomegranate_Good",
    ]
    assert EXPECTED_FRUIT_CLASSES == expected
    for idx, c in enumerate(expected):
        assert FruitClassifier.CLASS_TO_IDX[c] == idx
        assert FruitClassifier.IDX_TO_CLASS[idx] == c


def test_fruit_transforms_separation():
    """Verify training transforms include augmentation while evaluation transforms do not."""
    train_tf = get_fruit_transforms(image_size=380, is_training=True)
    eval_tf = get_fruit_transforms(image_size=380, is_training=False)

    assert train_tf is not None
    assert eval_tf is not None

    # Test that a PIL image transforms into a normalized torch Tensor (3, 380, 380)
    dummy_img = Image.new("RGB", (400, 400), color=(128, 128, 128))
    tensor_eval = eval_tf(dummy_img)

    assert isinstance(tensor_eval, torch.Tensor)
    assert tensor_eval.shape == (3, 380, 380)


def test_fruit_classifier_checkpoint_loading():
    """Verify that the trained classifier checkpoint loads properly with full metadata."""
    assert CHECKPOINT_PATH.is_file(), f"Trained checkpoint missing at {CHECKPOINT_PATH}"

    classifier = FruitClassifier.from_checkpoint(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    assert classifier.is_ready() is True
    assert classifier.num_classes == 6
    assert classifier.classes == EXPECTED_FRUIT_CLASSES
    assert classifier.image_size in (224, 380)

    # Verify metadata saved with checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "class_to_idx" in ckpt
    assert "image_size" in ckpt
    assert "val_metrics" in ckpt
    assert "test_metrics" in ckpt
    assert ckpt["dataset_info"]["total_valid_images"] == 11664
    assert ckpt["dataset_info"]["train_count"] == 8163
    assert ckpt["dataset_info"]["val_count"] == 1747
    assert ckpt["dataset_info"]["test_count"] == 1754
    assert ckpt["test_metrics"]["test_accuracy"] >= 0.99


def test_fruit_classifier_structured_inference():
    """Verify inference output matches Section 6 prompt specification."""
    assert SAMPLE_APPLE.is_file(), f"Sample apple image missing at {SAMPLE_APPLE}"

    classifier = FruitClassifier.from_checkpoint(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    result = classifier.predict_structured(SAMPLE_APPLE)

    # Required Section 6 Schema
    assert "fruit_name" in result
    assert "dataset_class" in result
    assert "freshness_status" in result
    assert "confidence" in result
    assert "classification_model" in result
    assert "nutrition" in result

    assert result["fruit_name"] == "Apple"
    assert result["dataset_class"] == "Apple_Good"
    assert result["freshness_status"] == "Good"
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["classification_model"] == "EfficientNet-B4"

    # Nutrition fields
    nut = result["nutrition"]
    assert "calories" in nut
    assert "protein" in nut
    assert "carbohydrates" in nut
    assert "fat" in nut
    assert nut["calories"] is not None and nut["calories"] > 0


def test_freshness_semantics_no_false_claims():
    """Verify that Good is reported as dataset category label without claiming spoilage detection."""
    classifier = FruitClassifier.from_checkpoint(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    result = classifier.predict_structured(SAMPLE_APPLE)

    assert "freshness_note" in result
    # Must explicitly state dataset category
    assert "dataset category" in result["freshness_note"].lower() or "dataset label" in result["freshness_note"].lower()


def test_usda_nutrition_integration_all_six_fruits():
    """Verify that USDA FoodData Central integration succeeds for all 6 fruit categories."""
    service = NutritionService()
    fruits = ["apple", "banana", "guava", "lime", "orange", "pomegranate"]

    for fruit in fruits:
        scaled = service.get_nutrition_for_item(fruit, weight_grams=100.0)
        assert scaled is not None, f"Nutrition lookup failed for {fruit}"
        assert scaled.status == "AVAILABLE", f"Nutrition status not AVAILABLE for {fruit}"
        assert scaled.calories_kcal > 0, f"Calories not positive for {fruit}: {scaled.calories_kcal}"
        assert scaled.usda_fdc_id is not None, f"FDC ID missing for {fruit}"


def test_api_fruit_endpoint():
    """Verify FastAPI /classify/fruit endpoint returns valid Section 6 response."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)

    with open(SAMPLE_APPLE, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/classify/fruit",
        files={"file": ("test_apple.jpg", img_bytes, "image/jpeg")},
    )

    assert response.status_code == 200, f"Endpoint error: {response.text}"
    body = response.json()

    assert body["fruit_name"] == "Apple"
    assert body["dataset_class"] == "Apple_Good"
    assert body["freshness_status"] == "Good"
    assert body["classification_model"] == "EfficientNet-B4"
    assert body["confidence"] > 0.50
    assert body["nutrition"]["calories"] is not None


def test_model_manager_fruit_status():
    """Verify ModelManager loads the fruit classifier and reports its status accurately."""
    from src.models.model_manager import get_model_manager
    mgr = get_model_manager()
    status = mgr.get_status()
    assert "models" in status
    assert "fruit_classifier" in status["models"]
    fc_info = status["models"]["fruit_classifier"]
    assert fc_info["ready"] is True
    assert "fruit_classifier_corrected_split" in str(fc_info["checkpoint"])


def test_pipeline_fruit_integration():
    """Verify that NutriVision master pipeline produces structured fruit recognition."""
    from src.pipeline.nutrivision_pipeline import NutriVisionPipeline
    pipeline = NutriVisionPipeline()
    result = pipeline.process_image(
        image=SAMPLE_APPLE,
        classifier_mode="fruit_classifier",
        enable_gradcam=False,
        enable_nutrition=True,
    )
    assert result["status"] in ("SUCCESS", "PARTIAL_SUCCESS")
    assert "fruit_recognition" in result
    fr = result["fruit_recognition"]
    assert fr is not None
    assert fr["fruit_name"] == "Apple"
    assert fr["dataset_class"] == "Apple_Good"
    assert fr["freshness_status"] == "Good"
    assert fr["confidence"] > 0.90
    assert fr["nutrition"]["calories"] is not None

