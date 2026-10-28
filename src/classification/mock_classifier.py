"""
NutriVision AI — Mock Food Classifier
Testing fixture. Returns MODEL_NOT_TRAINED by default. Does NOT fabricate predictions.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from src.classification.classifier import (
    BaseFoodClassifier,
    ClassificationOutput,
    ClassificationResult,
)
from src.detection.detector import ModelStatus


class MockFoodClassifier(BaseFoodClassifier):
    """
    Mock classifier explicitly returning MODEL_NOT_TRAINED by default.
    """

    def __init__(
        self,
        fixture_output: Optional[ClassificationOutput] = None,
        force_status: Optional[ModelStatus] = None,
    ):
        super().__init__()
        self.fixture_output = fixture_output
        self._status = force_status or (
            ModelStatus.READY if fixture_output is not None else ModelStatus.MODEL_NOT_TRAINED
        )
        if self._status == ModelStatus.MODEL_NOT_TRAINED:
            self._error_msg = (
                "MODEL_NOT_TRAINED: Food classifier has not been trained yet. "
                "No fake predictions are generated."
            )

    def load_model(self) -> bool:
        return self._status == ModelStatus.READY

    def classify(
        self,
        food_crop: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 5,
    ) -> ClassificationResult:
        t0 = time.perf_counter()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if self._status == ModelStatus.MODEL_NOT_TRAINED:
            return ClassificationResult(
                status=ModelStatus.MODEL_NOT_TRAINED,
                result=ClassificationOutput(class_id=None, class_name=None, confidence=None),
                inference_time_ms=elapsed_ms,
                model_name="MockFoodClassifier",
                error_message=self._error_msg,
            )

        return ClassificationResult(
            status=self._status,
            result=self.fixture_output,
            inference_time_ms=elapsed_ms,
            model_name="MockFoodClassifier",
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "mock_classifier",
            "status": self._status.value,
            "is_ready": self.is_ready(),
        }
