"""
NutriVision AI — Comprehensive FastAPI Test Suite (Phase 8)
Tests:
- GET /
- GET /health and /api/v1/health
- GET /models/status and /api/v1/models/status
- POST /analyze with valid image
- POST /analyze with invalid/corrupted bytes
- POST /analyze with unsupported file type
- POST /analyze/options query parameters
- GET /results/{id} existing and non-existent IDs
"""

import io
import json
from pathlib import Path
import unittest

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from api.main import app, pipeline


class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # Create a small valid test image buffer
        img = Image.new("RGB", (100, 100), color=(200, 50, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        cls.valid_image_bytes = buf.getvalue()

    def test_root_endpoint(self):
        """GET / returns application metadata and discovery links."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("project", data)
        self.assertEqual(data["status"], "OPERATIONAL")
        self.assertIn("endpoints", data)

    def test_health_endpoints(self):
        """GET /health and GET /api/v1/health return online status."""
        for endpoint in ["/health", "/api/v1/health"]:
            res = self.client.get(endpoint)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "online")
            self.assertIn("device", data)
            self.assertIn("timestamp", data)

    def test_models_status_endpoints(self):
        """GET /models/status reports ready status of deep learning models."""
        for endpoint in ["/models/status", "/api/v1/models/status"]:
            res = self.client.get(endpoint)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn(data["status"], ["READY", "PARTIALLY_READY"])
            self.assertIn("models", data)
            self.assertIn("yolo11", data["models"])
            self.assertIn("sam2", data["models"])
            self.assertIn("efficientnet_b4", data["models"])
            self.assertIn("freshness", data["models"])
            self.assertIn("depth_anything_v2", data["models"])

    def test_analyze_valid_image(self):
        """POST /analyze executes the full pipeline and returns complete schema."""
        files = {"file": ("test_apple.jpg", self.valid_image_bytes, "image/jpeg")}
        data = {
            "enable_gradcam": "false",  # Speed up unit test
            "enable_nutrition": "true",
            "crop_padding": "10",
        }
        res = self.client.post("/analyze", files=files, data=data)
        self.assertEqual(res.status_code, 200)
        res_data = res.json()
        self.assertIn("status", res_data)
        self.assertIn("result_id", res_data)
        self.assertIn("items", res_data)
        self.assertIn("meal_summary", res_data)
        self.assertIn("timing", res_data)
        self.assertIn("total_ms", res_data["timing"])

    def test_analyze_unsupported_file_type(self):
        """POST /analyze rejects non-image files with HTTP 415."""
        files = {"file": ("script.sh", b"echo hello", "text/x-shellscript")}
        res = self.client.post("/analyze", files=files)
        self.assertEqual(res.status_code, 415)
        self.assertIn("Unsupported file type", res.json()["detail"])

    def test_analyze_corrupted_image(self):
        """POST /analyze rejects corrupted bytes with HTTP 422."""
        files = {"file": ("corrupt.jpg", b"NOT_AN_IMAGE_RANDOM_GARBAGE", "image/jpeg")}
        res = self.client.post("/analyze", files=files)
        self.assertEqual(res.status_code, 422)
        self.assertIn("Corrupted or invalid image", res.json()["detail"])

    def test_analyze_empty_file(self):
        """POST /analyze rejects empty 0-byte file with HTTP 400."""
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        res = self.client.post("/analyze", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json()["detail"].lower())

    def test_analyze_with_options_endpoint(self):
        """POST /analyze/options accepts query parameters."""
        files = {"file": ("test.png", self.valid_image_bytes, "image/png")}
        params = {
            "enable_gradcam": False,
            "enable_nutrition": True,
            "crop_padding": 5,
        }
        res = self.client.post("/analyze/options", files=files, params=params)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("result_id", data)

    def test_get_results_by_id(self):
        """GET /results/{id} retrieves saved result or 404."""
        # 1. Non-existent ID returns 404
        res_404 = self.client.get("/results/non_existent_result_id_12345")
        self.assertEqual(res_404.status_code, 404)

        # 2. Analyze an image to create an ID, then retrieve it
        files = {"file": ("test.jpg", self.valid_image_bytes, "image/jpeg")}
        analyze_res = self.client.post("/analyze", files=files, data={"enable_gradcam": "false"})
        res_id = analyze_res.json()["result_id"]

        get_res = self.client.get(f"/results/{res_id}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["result_id"], res_id)

    def test_analyze_base64_valid(self):
        """POST /analyze accepts a valid base64 image in form data."""
        import base64
        b64_str = base64.b64encode(self.valid_image_bytes).decode("utf-8")
        data = {
            "image_base64": f"data:image/jpeg;base64,{b64_str}",
            "enable_gradcam": "false",
        }
        res = self.client.post("/analyze", data=data)
        self.assertEqual(res.status_code, 200)
        res_data = res.json()
        self.assertIn("status", res_data)
        self.assertIn("result_id", res_data)

    def test_analyze_base64_invalid(self):
        """POST /analyze rejects invalid base64 image with HTTP 422."""
        data = {"image_base64": "NOT_A_VALID_BASE64_STRING!!!"}
        res = self.client.post("/analyze", data=data)
        self.assertEqual(res.status_code, 422)

    def test_analyze_zero_detections_schema(self):
        """POST /analyze on non-food solid image returns NO_FOOD_DETECTED with valid structure."""
        files = {"file": ("blank.jpg", self.valid_image_bytes, "image/jpeg")}
        res = self.client.post("/analyze", files=files, data={"enable_gradcam": "false"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "NO_FOOD_DETECTED")
        self.assertEqual(data["total_foods_detected"], 0)
        self.assertEqual(len(data["items"]), 0)
        self.assertIn("timing", data)
        self.assertIn("meal_summary", data)

    def test_analyze_pipeline_exception_handling(self):
        """POST /analyze returns HTTP 500 with informative detail if pipeline raises exception."""
        from unittest.mock import patch
        files = {"file": ("test.jpg", self.valid_image_bytes, "image/jpeg")}
        with patch.object(pipeline, "process_image", side_effect=RuntimeError("GPU driver failure")):
            res = self.client.post("/analyze", files=files)
            self.assertEqual(res.status_code, 500)
            self.assertIn("Inference pipeline execution error", res.json()["detail"])
            self.assertIn("GPU driver failure", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()
