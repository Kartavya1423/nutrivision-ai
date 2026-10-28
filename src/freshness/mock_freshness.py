"""
NutriVision AI — Mock Freshness Detector
Testing fixture returning MODEL_NOT_TRAINED by default.
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image

from src.detection.detector import ModelStatus
from src.freshness.freshness_detector import (
    BaseFreshnessDetector,
    FreshnessEvaluation,
    FreshnessResult,
    FreshnessState,
)


class MockFreshnessDetector(BaseFreshnessDetector):
    """
    Mock freshness detector reporting MODEL_NOT_TRAINED by default.
    """

    def __init__(
        self,
        fixture_evaluation: Optional[FreshnessEvaluation] = None,
        force_status: Optional[ModelStatus] = None,
    ):
        super().__init__()
        self.fixture_evaluation = fixture_evaluation
        self._status = force_status or (
            ModelStatus.READY if fixture_evaluation is not None else ModelStatus.MODEL_NOT_TRAINED
        )
        if self._status == ModelStatus.MODEL_NOT_TRAINED:
            self._error_msg = (
                "MODEL_NOT_TRAINED: Freshness classifier has not been trained yet. "
                "No fake freshness or spoilage predictions are generated."
            )

    def load_model(self) -> bool:
        return self._status == ModelStatus.READY

    def assess_freshness(
        self,
        food_crop: Union[str, Path, np.ndarray, Image.Image],
        class_name: str = "food",
        item_index: int = 0,
    ) -> FreshnessResult:
        t0 = time.perf_counter()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if self._status == ModelStatus.MODEL_NOT_TRAINED:
            return FreshnessResult(
                status=ModelStatus.MODEL_NOT_TRAINED,
                evaluation=FreshnessEvaluation(state=FreshnessState.UNKNOWN, confidence=None),
                inference_time_ms=elapsed_ms,
                model_name="MockFreshnessDetector",
                error_message=self._error_msg,
            )

        return FreshnessResult(
            status=self._status,
            evaluation=self.fixture_evaluation,
            inference_time_ms=elapsed_ms,
            model_name="MockFreshnessDetector",
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "mock_freshness",
            "status": self._status.value,
            "is_ready": self.is_ready(),
        }
