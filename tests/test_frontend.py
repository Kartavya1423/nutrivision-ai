"""
NutriVision AI — Streamlit Frontend Component & Integration Tests (Phase 1 & Phase 8)
Verifies:
- Streamlit application module import
- Summary CSV generation from structured items (including empty items)
- API health and model status parsing
- Image upload request construction and parameter serialization
- Valid image processing & complete response parsing
- Invalid image handling (empty buffer, corrupt bytes, unsupported types)
- Backend unavailable & timeout handling (structured exceptions)
- Empty detection response parsing
- Model not ready & inference exception handling
- Frontend image bytes validation (dimensions, formats, corruptions)
"""

import io
import json
import unittest
from unittest.mock import MagicMock, patch
import requests

from frontend.streamlit_app import (
    BackendConnectionError,
    BackendTimeoutError,
    InferencePipelineApiError,
    InvalidImageApiError,
    ModelNotReadyApiError,
    call_analyze_api,
    check_api_health,
    extract_available_vitamins_and_minerals,
    fetch_models_status,
    generate_summary_csv,
    render_html,
    validate_image_bytes,
)
from PIL import Image


class TestFrontendComponents(unittest.TestCase):
    def setUp(self):
        # Create a small valid JPEG for testing
        img = Image.new("RGB", (100, 100), color=(180, 60, 60))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        self.valid_jpeg_bytes = buf.getvalue()

    def test_summary_csv_generation(self):
        """Verify tabular CSV export format from pipeline item dictionaries."""
        sample_items = [
            {
                "food": "apple",
                "detection": {"confidence": 0.95, "bbox": [10, 20, 100, 200]},
                "freshness": {"label": "fresh", "confidence": 0.92},
                "weight": {"value_g": 150.0},
                "nutrition": {
                    "calories_kcal": 78.0,
                    "protein_g": 0.39,
                    "carbohydrates_g": 20.7,
                    "fat_g": 0.25,
                    "fiber_g": 3.6,
                    "matched_food": "Apples, fuji, with skin, raw",
                    "fdc_id": 1750340,
                },
            },
            {
                "food": "banana",
                "detection": {"confidence": 0.88, "bbox": [150, 50, 280, 220]},
                "freshness": {"label": "fresh", "confidence": 0.89},
                "weight": {"value_g": 120.0},
                "nutrition": {
                    "calories_kcal": 106.8,
                    "protein_g": 1.31,
                    "carbohydrates_g": 27.41,
                    "fat_g": 0.40,
                    "fiber_g": 3.12,
                    "matched_food": "Bananas, raw",
                    "fdc_id": 173944,
                },
            },
        ]

        csv_str = generate_summary_csv(sample_items)
        self.assertIn("Item #,Food Name,Detection Conf,Freshness", csv_str)
        self.assertIn("apple", csv_str)
        self.assertIn("banana", csv_str)
        self.assertIn("150.0", csv_str)
        self.assertIn("106.8", csv_str)
        self.assertIn("1750340", csv_str)

    def test_summary_csv_handles_empty_items(self):
        """CSV generator handles empty item list gracefully."""
        csv_str = generate_summary_csv([])
        self.assertIn("Item #,Food Name", csv_str)

    @patch("frontend.streamlit_app.requests.get")
    def test_api_health_check_parsing(self, mock_get):
        """Verify API health response parsing."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "status": "online",
            "project": "NutriVision AI",
            "device": "cpu",
        }
        res = check_api_health()
        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "online")
        self.assertEqual(res["device"], "cpu")

    @patch("frontend.streamlit_app.requests.get")
    def test_fetch_models_status_parsing(self, mock_get):
        """Verify model readiness diagnostic parsing."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "status": "READY",
            "models": {
                "yolo11": {"name": "YOLO11", "ready": True},
                "sam2": {"name": "SAM2", "ready": True},
            },
        }
        res = fetch_models_status()
        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "READY")
        self.assertTrue(res["models"]["yolo11"]["ready"])

    # --- Phase 1 Integration Tests ---

    @patch("frontend.streamlit_app.requests.post")
    def test_image_upload_request_construction(self, mock_post):
        """Verify multipart request structure, filenames, MIME types, and form data fields."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "status": "SUCCESS",
            "result_id": "nv_test_123",
            "total_foods_detected": 1,
            "items": [],
            "timing": {"total_ms": 150.0},
        }

        res = call_analyze_api(
            image_bytes=self.valid_jpeg_bytes,
            filename="plate.jpg",
            content_type="image/jpeg",
            enable_gradcam=True,
            enable_nutrition=False,
            crop_padding=15,
            confidence_threshold=0.35,
        )

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn("/analyze", args[0])
        self.assertIn("files", kwargs)
        self.assertIn("file", kwargs["files"])
        self.assertEqual(kwargs["files"]["file"][0], "plate.jpg")
        self.assertEqual(kwargs["files"]["file"][1], self.valid_jpeg_bytes)
        self.assertEqual(kwargs["files"]["file"][2], "image/jpeg")

        data = kwargs["data"]
        self.assertEqual(data["enable_gradcam"], "true")
        self.assertEqual(data["enable_nutrition"], "false")
        self.assertEqual(data["crop_padding"], 15)
        self.assertEqual(data["confidence_threshold"], 0.35)
        self.assertEqual(res["status"], "SUCCESS")

    @patch("frontend.streamlit_app.requests.post")
    def test_valid_image_processing(self, mock_post):
        """Verify successful response parsing with detected items."""
        sample_response = {
            "status": "SUCCESS",
            "result_id": "nv_20261016_001",
            "total_foods_detected": 1,
            "items": [
                {
                    "food": "apple",
                    "detection": {"confidence": 0.94, "bbox": [10, 10, 80, 80]},
                    "freshness": {"label": "fresh", "confidence": 0.91},
                    "weight": {"value_g": 145.0},
                    "nutrition": {"status": "SUCCESS", "calories_kcal": 75.4},
                }
            ],
            "meal_summary": {"total_calories_kcal": 75.4, "total_weight_g": 145.0},
            "timing": {"total_ms": 320.5},
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = sample_response

        result = call_analyze_api(self.valid_jpeg_bytes)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["total_foods_detected"], 1)
        self.assertEqual(result["items"][0]["food"], "apple")
        self.assertEqual(result["meal_summary"]["total_calories_kcal"], 75.4)

    def test_invalid_image_empty_buffer(self):
        """Verify immediate rejection of empty (0-byte) image buffer."""
        with self.assertRaises(InvalidImageApiError) as ctx:
            call_analyze_api(b"")
        self.assertIn("empty", str(ctx.exception).lower())

    @patch("frontend.streamlit_app.requests.post")
    def test_invalid_image_handling_backend_422(self, mock_post):
        """Verify backend HTTP 422 corrupted image response raises InvalidImageApiError."""
        mock_post.return_value.status_code = 422
        mock_post.return_value.json.return_value = {
            "detail": "Corrupted or invalid image data: cannot identify image file"
        }

        with self.assertRaises(InvalidImageApiError) as ctx:
            call_analyze_api(b"corrupt_bytes")
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("Corrupted or invalid image", str(ctx.exception))

    @patch("frontend.streamlit_app.requests.post")
    def test_unsupported_media_type_handling_415(self, mock_post):
        """Verify backend HTTP 415 unsupported file raises InvalidImageApiError."""
        mock_post.return_value.status_code = 415
        mock_post.return_value.json.return_value = {
            "detail": "Unsupported file type: 'text/plain'. Allowed types: JPG, PNG, WEBP."
        }

        with self.assertRaises(InvalidImageApiError) as ctx:
            call_analyze_api(b"hello world", content_type="text/plain")
        self.assertEqual(ctx.exception.status_code, 415)
        self.assertIn("Unsupported file type", str(ctx.exception))

    @patch("frontend.streamlit_app.requests.post")
    def test_backend_unavailable(self, mock_post):
        """Verify connection failure raises BackendConnectionError with diagnostic info."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused to 127.0.0.1:8000")

        with self.assertRaises(BackendConnectionError) as ctx:
            call_analyze_api(self.valid_jpeg_bytes)
        self.assertIn("offline or unreachable", str(ctx.exception))

    @patch("frontend.streamlit_app.requests.post")
    def test_backend_timeout(self, mock_post):
        """Verify request timeout raises BackendTimeoutError."""
        mock_post.side_effect = requests.exceptions.Timeout("Read timed out after 120s")

        with self.assertRaises(BackendTimeoutError) as ctx:
            call_analyze_api(self.valid_jpeg_bytes)
        self.assertIn("timed out", str(ctx.exception).lower())

    @patch("frontend.streamlit_app.requests.post")
    def test_empty_detection_response(self, mock_post):
        """Verify successful response parsing when 0 food items are localized."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "status": "NO_FOOD_DETECTED",
            "result_id": "nv_empty_001",
            "total_foods_detected": 0,
            "items": [],
            "meal_summary": {"total_items": 0, "total_calories_kcal": 0.0},
            "timing": {"total_ms": 110.0, "detection_ms": 85.0},
        }

        result = call_analyze_api(self.valid_jpeg_bytes)
        self.assertEqual(result["status"], "NO_FOOD_DETECTED")
        self.assertEqual(result["total_foods_detected"], 0)
        self.assertEqual(len(result["items"]), 0)
        self.assertEqual(result["timing"]["detection_ms"], 85.0)

    @patch("frontend.streamlit_app.requests.post")
    def test_inference_exception_handling_500(self, mock_post):
        """Verify backend HTTP 500 raises InferencePipelineApiError."""
        mock_post.return_value.status_code = 500
        mock_post.return_value.json.return_value = {
            "detail": "Inference pipeline execution error: CUDA out of memory"
        }

        with self.assertRaises(InferencePipelineApiError) as ctx:
            call_analyze_api(self.valid_jpeg_bytes)
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("CUDA out of memory", str(ctx.exception))

    @patch("frontend.streamlit_app.requests.post")
    def test_model_not_ready_handling_503(self, mock_post):
        """Verify backend HTTP 503 raises ModelNotReadyApiError."""
        mock_post.return_value.status_code = 503
        mock_post.return_value.json.return_value = {
            "detail": "YOLO11 food detector model is not initialized"
        }

        with self.assertRaises(ModelNotReadyApiError) as ctx:
            call_analyze_api(self.valid_jpeg_bytes)
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("not initialized", str(ctx.exception))

    # --- Frontend Image Validation Function Tests ---

    def test_validate_image_bytes_valid(self):
        """Valid JPEG bytes are successfully decoded and return PIL Image."""
        pil_img, w, h, fmt = validate_image_bytes(self.valid_jpeg_bytes)
        self.assertEqual(w, 100)
        self.assertEqual(h, 100)
        self.assertEqual(fmt, "JPEG")
        self.assertEqual(pil_img.mode, "RGB")

    def test_validate_image_bytes_corrupted(self):
        """Corrupted byte sequence raises ValueError with informative message."""
        with self.assertRaises(ValueError) as ctx:
            validate_image_bytes(b"NOT_A_VALID_IMAGE_BYTES_12345")
        self.assertIn("Corrupted or invalid image", str(ctx.exception))

    def test_validate_image_bytes_too_small(self):
        """Image smaller than 10x10 px is rejected."""
        small_img = Image.new("RGB", (5, 5), color=(100, 100, 100))
        buf = io.BytesIO()
        small_img.save(buf, format="PNG")
        with self.assertRaises(ValueError) as ctx:
            validate_image_bytes(buf.getvalue())
        self.assertIn("smaller than minimum", str(ctx.exception))

    def test_quick_select_foods_definition(self):
        """Verify all 8 mandatory quick-select catalog options exist."""
        from frontend.streamlit_app import QUICK_SELECT_FOODS, get_quick_select_nutrition
        keys = [f["key"] for f in QUICK_SELECT_FOODS]
        expected = ["pizza", "burger", "salad", "sandwich", "steak", "cake", "ice_cream", "samosa"]
        for exp in expected:
            self.assertIn(exp, keys, f"Missing required quick-select option: {exp}")

    def test_quick_select_nutrition_calculation(self):
        """Verify authentic USDA nutrition scaling for all 8 quick-select foods."""
        from frontend.streamlit_app import QUICK_SELECT_FOODS, get_quick_select_nutrition
        for food in QUICK_SELECT_FOODS:
            k = food["key"]
            nutr = get_quick_select_nutrition(k, weight_g=150.0)
            self.assertIsNotNone(nutr, f"Nutrition returned None for {k}")
            self.assertEqual(nutr.get("status"), "AVAILABLE", f"Status not AVAILABLE for {k}")
            self.assertGreater(nutr.get("calories_kcal", 0), 0, f"Calories should be > 0 for {k}")
            self.assertIn("protein_g", nutr)
            self.assertIn("carbohydrates_g", nutr)
            self.assertIn("fat_g", nutr)


    def test_extract_available_vitamins_and_minerals(self):
        """Verify dynamic scaling and categorizing of vitamins and minerals."""
        sample_nut = {
            "status": "SUCCESS",
            "nutrients_per_100g": {
                "vitamin_c_mg": 4.6,
                "vitamin_a_rae_ug": 3.0,
                "calcium_mg": 6.0,
                "potassium_mg": 107.0,
                "iron_mg": 0.12,
            },
        }
        # Scaled for 200g portion (2x)
        micros = extract_available_vitamins_and_minerals(sample_nut, portion_weight_g=200.0)
        labels = [m["label"] for m in micros]
        self.assertIn("Vitamin C", labels)
        self.assertIn("Vitamin A (RAE)", labels)
        self.assertIn("Calcium", labels)
        self.assertIn("Potassium", labels)

        vit_c = next(m for m in micros if m["label"] == "Vitamin C")
        self.assertEqual(vit_c["category"], "Vitamin")
        self.assertAlmostEqual(vit_c["value"], 9.2, places=2)
        self.assertEqual(vit_c["unit"], "mg")
        self.assertIsNotNone(vit_c["dv_pct"])

        calcium = next(m for m in micros if m["label"] == "Calcium")
        self.assertEqual(calcium["category"], "Mineral")
        self.assertAlmostEqual(calcium["value"], 12.0, places=2)

    @patch("frontend.streamlit_app.st.markdown")
    def test_render_html_no_code_blocks(self, mock_markdown):
        """Verify render_html strips leading whitespace so lines cannot trigger code blocks."""
        indented_html = """
            <div class="micro-panel">
                <div class="micro-item-row">
                    <span class="micro-name">Vitamin C</span>
                </div>
            </div>
        """
        render_html(indented_html)
        mock_markdown.assert_called_once()
        rendered_arg = mock_markdown.call_args[0][0]
        for line in rendered_arg.splitlines():
            # No line should start with 4 or more spaces (CommonMark indented code block rule)
            self.assertFalse(line.startswith("    "), f"Line has 4+ spaces of leading indent: {line}")


if __name__ == "__main__":
    unittest.main()

