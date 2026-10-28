"""
NutriVision AI — Mock Food Detector
Interface placeholder and testing harness.
Explicitly reports MODEL_NOT_TRAINED by default. Does NOT fabricate fake predictions.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from src.detection.detector import (
    BaseFoodDetector,
    DetectionResult,
    FoodDetection,
    ModelStatus,
)
from src.utils.image_ops import load_image


class MockFoodDetector(BaseFoodDetector):
    """
    Mock detector designed for architecture verification and testing.
    Honest reporting: Returns MODEL_NOT_TRAINED by default.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        fixture_detections: Optional[List[FoodDetection]] = None,
        force_status: Optional[ModelStatus] = None,
        status: Optional[ModelStatus] = None,
    ):
        super().__init__(model_path=model_path)
        self.fixture_detections = fixture_detections or []
        effective_status = force_status or status
        self._status = effective_status or (
            ModelStatus.READY if fixture_detections is not None else ModelStatus.MODEL_NOT_TRAINED
        )
        if self._status == ModelStatus.MODEL_NOT_TRAINED:
            self._error_msg = (
                "MODEL_NOT_TRAINED: Food detector model has not been trained. "
                "No fake food predictions are generated."
            )

    def load_model(self) -> bool:
        if self._status == ModelStatus.MODEL_NOT_TRAINED:
            return False
        return True

    def predict(self, image: Union[str, Path, np.ndarray, Image.Image]) -> DetectionResult:
        t0 = time.perf_counter()
        np_img, _ = load_image(image)
        h, w = np_img.shape[:2]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if self._status == ModelStatus.MODEL_NOT_TRAINED:
            return DetectionResult(
                status=ModelStatus.MODEL_NOT_TRAINED,
                detections=[],
                image_shape=(h, w, 3),
                inference_time_ms=elapsed_ms,
                model_name="MockFoodDetector",
                error_message=self._error_msg,
            )

        return DetectionResult(
            status=self._status,
            detections=self.fixture_detections,
            image_shape=(h, w, 3),
            inference_time_ms=elapsed_ms,
            model_name="MockFoodDetector",
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "mock_detector",
            "status": self._status.value,
            "is_ready": self.is_ready(),
            "has_fixtures": len(self.fixture_detections) > 0,
        }
