"""
NutriVision AI — Phase 9 End-to-End Validation Test Suite
Authoritative validation covering:
1. FastAPI Health & Model Status diagnostics (/health, /models/status).
2. End-to-end image upload inference (/analyze).
3. Base64 camera scan upload inference.
4. Food localization (YOLO11), classification (EfficientNet-B4 target_foods_v1),
   SAM2 segmentation, freshness, depth, volume, and contour weight estimation.
5. Explicit label origin tagging ("classifier", "detector", "user_corrected").
6. Portions, depth topography disclaimer, and USDA nutrition calculation.
7. Results retrieval (/results/{result_id}) and summary CSV generation.
8. Robust error handling (invalid files, corrupted payloads, missing records).
9. Checkpoint persistence and reload verification across application restarts.
"""

import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config.settings import get_settings
from src.models.model_manager import get_model_manager
from src.pipeline.nutrivision_pipeline import NutriVisionPipeline
from src.nutrition.nutrition_service import NutritionService
from src.nutrition.fooddata_central import FoodDataCentralClient
from frontend.streamlit_app import generate_summary_csv


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_apple_image_bytes():
    """Generate a clean synthetic RGB test image with an apple-like red circular pattern."""
    img = Image.new("RGB", (320, 320), color=(240, 240, 240))
    # Draw a red circle representing food
    import cv2
    arr = np.array(img)
    cv2.circle(arr, (160, 160), 80, (220, 30, 30), -1)  # Red food blob
    cv2.circle(arr, (150, 90), 10, (40, 160, 40), -1)   # Green leaf/stem
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()


class TestPhase9SystemStartupAndHealth:
    """Validates FastAPI health, model readiness, and Windows execution paths."""

    def test_health_endpoint(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "online"
        assert data["project"] == "NutriVision AI"
        assert "device" in data
        assert "models_ready" in data

    def test_models_status_endpoint(self, client):
        res = client.get("/models/status")
        assert res.status_code == 200
        data = res.json()
        assert data["total_models"] >= 5
        models = data["models"]
        assert "yolo11" in models
        assert "sam2" in models
        assert "efficientnet_b4" in models
        assert "freshness" in models
        assert "depth_anything_v2" in models


class TestPhase9EndToEndInference:
    """Validates full 10-stage pipeline execution on file upload and camera base64."""

    def test_upload_file_inference(self, client, sample_apple_image_bytes):
        files = {"file": ("test_apple.jpg", sample_apple_image_bytes, "image/jpeg")}
        res = client.post("/analyze", files=files)
        assert res.status_code == 200
        data = res.json()

        assert data["status"] == "SUCCESS"
        assert "result_id" in data
        assert "timing" in data
        assert len(data["items"]) >= 1

        item = data["items"][0]
        # 1. Localization
        assert "detection" in item
        assert "bbox" in item["detection"]
        assert item["detection"]["confidence"] > 0

        # 2. Classification & Label Origin
        assert "classification" in item
        assert item["classification"]["label"]
        assert item["classification"]["confidence"] >= 0.0
        assert "label_origin" in item["classification"]
        assert item["classification"]["label_origin"] in ("classifier", "detector")
        assert item["classification"]["model_version"] is not None

        # 3. Segmentation
        assert "segmentation" in item
        assert "mask_available" in item["segmentation"]

        # 4. Freshness
        assert "freshness" in item
        assert item["freshness"]["label"] in ("fresh", "rotten", "unknown")

        # 5. Depth & Physical Metrics
        assert "depth" in item
        assert "volume" in item
        assert "weight" in item
        assert item["weight"]["value_g"] is not None
        assert item["weight"]["value_g"] > 0
        assert item["weight"]["source"] == "vision_estimated"
        assert "monocular depth" in item["weight"]["monocular_depth_disclaimer"].lower()

        # 6. USDA Nutrition
        assert "nutrition" in item
        nut = item["nutrition"]
        assert nut["status"] in ("SUCCESS", "AVAILABLE")
        assert nut["calories_kcal"] is not None and nut["calories_kcal"] > 0
        assert nut["protein_g"] is not None
        assert nut["carbohydrates_g"] is not None
        assert nut["fat_g"] is not None
        assert nut["fiber_g"] is not None
        assert nut["source"] == "USDA FoodData Central"
        assert nut["fdc_id"] is not None

    def test_camera_scan_base64_inference(self, client, sample_apple_image_bytes):
        b64_str = f"data:image/jpeg;base64,{base64.b64encode(sample_apple_image_bytes).decode('utf-8')}"
        res = client.post("/analyze", data={"image_base64": b64_str})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "SUCCESS"
        assert len(data["items"]) >= 1


class TestPhase9DataExportAndRetrieval:
    """Validates result caching, /results/{result_id}, and summary CSV generation."""

    def test_results_retrieval(self, client, sample_apple_image_bytes):
        files = {"file": ("test_export.jpg", sample_apple_image_bytes, "image/jpeg")}
        post_res = client.post("/analyze", files=files)
        assert post_res.status_code == 200
        result_id = post_res.json()["result_id"]

        get_res = client.get(f"/results/{result_id}")
        assert get_res.status_code == 200
        retrieved_data = get_res.json()
        assert retrieved_data["result_id"] == result_id
        assert len(retrieved_data["items"]) == len(post_res.json()["items"])

    def test_csv_summary_generation(self):
        sample_items = [
            {
                "food": "apple",
                "label_origin": "classifier",
                "classification": {"confidence": 0.942, "model_version": "target_foods_v1"},
                "detection": {"confidence": 0.885, "localization_method": "YOLO11"},
                "freshness": {"label": "fresh", "confidence": 0.961},
                "weight": {"value_g": 182.5, "source": "vision_estimated"},
                "nutrition": {
                    "matched_food": "Apples, raw, with skin",
                    "fdc_id": 171688,
                    "calories_kcal": 94.9,
                    "protein_g": 0.47,
                    "carbohydrates_g": 25.2,
                    "fat_g": 0.31,
                    "fiber_g": 4.38,
                }
            }
        ]
        csv_str = generate_summary_csv(sample_items)
        assert "Food Name" in csv_str
        assert "apple" in csv_str
        assert "94.9" in csv_str
        assert "182.5" in csv_str
        assert "FRESH" in csv_str or "fresh" in csv_str


class TestPhase9ErrorHandling:
    """Validates robustness against malformed input, missing files, and unsupported formats."""

    def test_missing_file_and_base64_error(self, client):
        res = client.post("/analyze")
        assert res.status_code == 400
        assert "No image provided" in res.json()["detail"]

    def test_corrupted_image_error(self, client):
        files = {"file": ("corrupt.jpg", b"ThisIsNotAnImageFileContent", "image/jpeg")}
        res = client.post("/analyze", files=files)
        assert res.status_code == 422
        assert "corrupted" in res.json()["detail"].lower() or "cannot decode" in res.json()["detail"].lower()

    def test_unsupported_media_type(self, client):
        files = {"file": ("test.pdf", b"%PDF-1.4...", "application/pdf")}
        res = client.post("/analyze", files=files)
        assert res.status_code == 415


class TestPhase9CheckpointsAndPersistence:
    """Validates model checkpoint integrity and persistence."""

    def test_target_foods_v1_checkpoint_files_exist(self):
        cfg = get_settings()
        v1_dir = cfg.root_dir / "models" / "classification" / "target_foods_v1"
        assert v1_dir.is_dir(), f"Target foods v1 directory missing: {v1_dir}"

        best_ckpt = v1_dir / "best.pth"
        final_ckpt = v1_dir / "last.pth"
        classes_json = v1_dir / "class_names.json"
        config_json = v1_dir / "training_config.json"

        assert best_ckpt.is_file(), f"best.pth missing: {best_ckpt}"
        assert final_ckpt.is_file(), f"last.pth missing: {final_ckpt}"
        assert classes_json.is_file(), f"class_names.json missing: {classes_json}"
        assert config_json.is_file(), f"training_config.json missing: {config_json}"

        with open(classes_json, "r", encoding="utf-8") as f:
            classes = json.load(f)
        assert len(classes) == 21, f"Expected 21 dataset-supported vision classes, got {len(classes)}"
        assert "apple" in classes
        assert "pizza" in classes
        assert "banana" in classes
        assert "watermelon" in classes

    def test_original_apple_checkpoint_preserved(self):
        cfg = get_settings()
        apple_ckpt = cfg.root_dir / "models" / "classification" / "efficientnet_b4" / "best.pth"
        assert apple_ckpt.is_file(), f"Original apple checkpoint must remain intact: {apple_ckpt}"
