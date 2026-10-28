"""
Unit tests for API endpoints and status reporting (api.main).
"""

import unittest
from api.main import app, cfg
from fastapi.testclient import TestClient


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint_response(self):
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["project"], "NutriVision AI")

    def test_models_status_endpoint_reports_status(self):
        res = self.client.get("/api/v1/models/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn(data["status"], ["READY", "PARTIALLY_READY"])
        self.assertIn("models", data)
        self.assertIn("yolo11", data["models"])
        self.assertIn("sam2", data["models"])


if __name__ == "__main__":
    unittest.main()
