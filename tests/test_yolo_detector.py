"""
NutriVision AI — YOLO11 Food Detector Interface Tests
Phase 2: YOLO11 Food Detection Pipeline
"""

import unittest
from pathlib import Path
import numpy as np

from src.detection.detector import ModelStatus, DetectionResult, FoodDetection
from src.detection.yolo_detector import YOLO11FoodDetector


class TestYOLO11FoodDetector(unittest.TestCase):
    """Tests YOLO11 detector envelope behavior, status checking, and camera array inputs."""

    def test_missing_weights_returns_model_not_loaded(self):
        """Unloaded/missing checkpoint strictly returns MODEL_NOT_LOADED status."""
        detector = YOLO11FoodDetector(model_path=Path("non_existent_weights.pt"))
        self.assertFalse(detector.is_ready())
        self.assertEqual(detector._status, ModelStatus.MODEL_NOT_LOADED)

        # Predicting when model is not loaded returns MODEL_NOT_LOADED with 0 detections
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.predict(dummy_frame)
        self.assertEqual(result.status, ModelStatus.MODEL_NOT_LOADED)
        self.assertEqual(result.count, 0)
        self.assertEqual(len(result.detections), 0)
        self.assertFalse(result.is_success)

    def test_detect_alias(self):
        """detect() alias produces identical result envelope to predict()."""
        detector = YOLO11FoodDetector(model_path=Path("non_existent_weights.pt"))
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res1 = detector.predict(dummy_frame)
        res2 = detector.detect(dummy_frame)
        self.assertEqual(res1.status, res2.status)
        self.assertEqual(res1.count, res2.count)

    def test_detection_result_serialization(self):
        """DetectionResult and FoodDetection serialize properly."""
        det = FoodDetection(
            class_id=12,
            class_name="pizza",
            confidence=0.9412,
            bbox=[120.5, 80.2, 540.1, 430.7]
        )
        res = DetectionResult(
            status=ModelStatus.READY,
            detections=[det],
            image_shape=(480, 640, 3),
            inference_time_ms=15.4,
            model_name="YOLO11FoodDetector"
        )
        d = res.to_dict()
        self.assertEqual(d["status"], "READY")
        self.assertEqual(d["count"], 1)
        self.assertEqual(d["detections"][0]["class_name"], "pizza")
        self.assertAlmostEqual(d["detections"][0]["confidence"], 0.9412, places=3)


    def test_checkpoint_loading_and_cpu_inference(self):
        """Verifies checkpoint loading and CPU inference on numpy array and PIL Image."""
        model_path = Path("models/detection/yolo11/best.pt")
        if not model_path.is_file():
            model_path = Path("models/detection/yolo11/phase2_fixed_run/weights/best.pt")

        if model_path.is_file():
            detector = YOLO11FoodDetector(
                model_path=model_path,
                conf_threshold=0.001,
                device="cpu"
            )
            self.assertTrue(detector.is_ready())
            self.assertEqual(detector._status, ModelStatus.READY)

            # Test numpy array
            frame = np.full((320, 320, 3), 128, dtype=np.uint8)
            res_np = detector.predict(frame)
            self.assertTrue(res_np.is_success)
            self.assertEqual(res_np.status, ModelStatus.READY)
            self.assertGreater(res_np.inference_time_ms, 0.0)

            # Test PIL image
            from PIL import Image
            pil_img = Image.fromarray(frame)
            res_pil = detector.detect(pil_img)
            self.assertTrue(res_pil.is_success)
            self.assertEqual(res_pil.status, ModelStatus.READY)

    def test_confidence_filtering(self):
        """High confidence threshold filters out weak detections."""
        model_path = Path("models/detection/yolo11/best.pt")
        if not model_path.is_file():
            model_path = Path("models/detection/yolo11/phase2_fixed_run/weights/best.pt")

        if model_path.is_file():
            detector = YOLO11FoodDetector(
                model_path=model_path,
                conf_threshold=0.999,  # Impossibly high threshold
                device="cpu"
            )
            frame = np.full((320, 320, 3), 100, dtype=np.uint8)
            res = detector.predict(frame)
            self.assertTrue(res.is_success)
            self.assertEqual(res.count, 0)
            self.assertEqual(len(res.detections), 0)

    def test_dataset_yaml_and_class_consistency(self):
        """Verifies dataset.yaml contains exactly 198 contiguous unique classes matching CSV."""
        import csv
        import yaml

        yaml_path = Path("data/processed/yolo_detection/dataset.yaml")
        self.assertTrue(yaml_path.is_file())

        with open(yaml_path, "r", encoding="utf-8") as f:
            ds = yaml.safe_load(f)

        names = ds.get("names", {})
        self.assertEqual(len(names), 198)
        self.assertEqual(sorted(list(names.keys())), list(range(198)))
        self.assertEqual(len(set(names.values())), 198)

        mapping_csv = Path("data/processed/detection_class_mapping.csv")
        if mapping_csv.is_file():
            with open(mapping_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            csv_classes = {int(r["detector_class_id"]): r["detector_class_name"] for r in rows}
            for cid, cname in names.items():
                self.assertIn(cid, csv_classes)
                self.assertEqual(names[cid], csv_classes[cid])

    def test_detect_and_crop_clamping(self):
        """detect_and_crop produces crops that never exceed image boundaries."""
        from src.detection.crop_utils import crop_bounding_box
        dummy_img = np.zeros((200, 300, 3), dtype=np.uint8)

        # Box extending outside boundaries: [-20, -10, 350, 250]
        out_bbox = [-20.0, -10.0, 350.0, 250.0]
        crop_np, coords = crop_bounding_box(dummy_img, out_bbox)
        x1, y1, x2, y2 = coords
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 300)
        self.assertLessEqual(y2, 200)
        self.assertEqual(crop_np.shape[0], y2 - y1)
        self.assertEqual(crop_np.shape[1], x2 - x1)

    def test_repaired_model_loading_and_interface_contract(self):
        """Verifies the repaired checkpoint loads and returns standard FoodDetection objects."""
        repaired_path = Path("models/detection/yolo11/repair_run/best.pt")
        if not repaired_path.is_file():
            repaired_path = Path("models/detection/yolo11/production/best.pt")

        self.assertTrue(repaired_path.is_file(), "A valid detection checkpoint must exist")
        detector = YOLO11FoodDetector(
            model_path=repaired_path,
            conf_threshold=0.001,
            device="cpu"
        )
        self.assertTrue(detector.is_ready())
        
        # Test on a real image
        test_img = Path("data/raw/sample_benchmark/train/fresh_apple/fresh_apple_train_0.jpg")
        self.assertTrue(test_img.is_file())
        res = detector.predict(str(test_img))
        self.assertTrue(res.is_success)
        self.assertIsInstance(res.detections, list)
        self.assertIn("image_shape", res.to_dict())

    def test_invalid_image_inputs_return_error_status(self):
        """Passing non-existent file or corrupted input returns ERROR status without crashing."""
        detector = YOLO11FoodDetector(
            model_path=Path("models/detection/yolo11/repair_run/best.pt"),
            device="cpu"
        )
        # Non-existent file
        res = detector.predict("non_existent_file_12345.jpg")
        self.assertEqual(res.status, ModelStatus.ERROR)
        self.assertIn("Invalid image input", res.error_message)

    def test_bounding_box_clamping_to_image_dimensions(self):
        """Ensures any detected box is strictly clamped within [0, 0, width, height]."""
        from src.detection.detector import BoundingBox
        w, h = 640, 480
        # Wild coordinates from uncalibrated head: [-50, -20, 700, 550]
        wild_box = BoundingBox(-50.0, -20.0, 700.0, 550.0).clamp(w, h)
        self.assertEqual(wild_box.x1, 0.0)
        self.assertEqual(wild_box.y1, 0.0)
        self.assertEqual(wild_box.x2, 640.0)
        self.assertEqual(wild_box.y2, 480.0)


if __name__ == "__main__":
    unittest.main()
