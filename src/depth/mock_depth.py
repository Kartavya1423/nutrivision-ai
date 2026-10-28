"""
NutriVision AI — Mock Depth Estimator
Testing fixture. Reports MODEL_NOT_TRAINED / MODEL_NOT_LOADED. Does NOT fabricate physical depth.
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image

from src.depth.depth_estimator import BaseDepthEstimator, DepthResult, DepthType
from src.detection.detector import ModelStatus


class MockDepthEstimator(BaseDepthEstimator):
    """
    Mock depth estimator for interface testing.
    """

    def __init__(
        self,
        fixture_depth_map: Optional[np.ndarray] = None,
        is_metric: bool = False,
        force_status: Optional[ModelStatus] = None,
    ):
        super().__init__()
        self.fixture_depth_map = fixture_depth_map
        self.is_metric = is_metric
        self._status = force_status or (
            ModelStatus.READY if fixture_depth_map is not None else ModelStatus.MODEL_NOT_TRAINED
        )
        if self._status == ModelStatus.MODEL_NOT_TRAINED:
            self._error_msg = (
                "MODEL_NOT_TRAINED: Depth model is not configured. "
                "Do not claim relative depth is automatically metric depth in cm."
            )

    def load_model(self) -> bool:
        return self._status == ModelStatus.READY

    def estimate_depth(self, image: Union[str, Path, np.ndarray, Image.Image]) -> DepthResult:
        t0 = time.perf_counter()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if self._status == ModelStatus.MODEL_NOT_TRAINED:
            return DepthResult(
                status=ModelStatus.MODEL_NOT_TRAINED,
                depth_map=None,
                depth_type=DepthType.RELATIVE,
                is_metric_calibrated=False,
                inference_time_ms=elapsed_ms,
                model_name="MockDepthEstimator",
                error_message=self._error_msg,
            )

        d = self.fixture_depth_map
        return DepthResult(
            status=self._status,
            depth_map=d,
            depth_type=DepthType.METRIC if self.is_metric else DepthType.RELATIVE,
            is_metric_calibrated=self.is_metric,
            min_depth=float(np.min(d)) if d is not None else 0.0,
            max_depth=float(np.max(d)) if d is not None else 0.0,
            mean_depth=float(np.mean(d)) if d is not None else 0.0,
            inference_time_ms=elapsed_ms,
            model_name="MockDepthEstimator",
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "mock_depth",
            "status": self._status.value,
            "is_ready": self.is_ready(),
        }
