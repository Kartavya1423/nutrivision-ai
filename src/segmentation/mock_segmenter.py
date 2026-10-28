"""
NutriVision AI — Mock Food Segmenter
Testing fixture and architectural placeholder.
Explicitly reports MODEL_NOT_TRAINED by default. Does NOT fabricate masks.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from src.detection.detector import ModelStatus
from src.segmentation.segmenter import (
    BaseFoodSegmenter,
    FoodMask,
    SegmentationResult,
)
from src.utils.image_ops import load_image


class MockFoodSegmenter(BaseFoodSegmenter):
    """
    Mock segmenter for interface verification and pipeline testing.
    Honest reporting: returns MODEL_NOT_TRAINED by default.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        fixture_masks: Optional[List[FoodMask]] = None,
        force_status: Optional[ModelStatus] = None,
        device: Optional[str] = None,
    ):
        super().__init__(checkpoint_path=checkpoint_path)
        self.device = device or "cpu"
        self.fixture_masks = fixture_masks or []
        self._status = force_status or (
            ModelStatus.READY if fixture_masks is not None else ModelStatus.MODEL_NOT_TRAINED
        )
        if self._status == ModelStatus.MODEL_NOT_TRAINED:
            self._error_msg = (
                "MODEL_NOT_TRAINED: Segmentation model (SAM 2) has not been trained or downloaded. "
                "No fake masks are generated."
            )

    def load_model(self) -> bool:
        return self._status == ModelStatus.READY

    def predict_masks(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        bboxes: Optional[List[List[float]]] = None,
        class_names: Optional[List[str]] = None,
    ) -> SegmentationResult:
        t0 = time.perf_counter()
        np_img, _ = load_image(image)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if self._status == ModelStatus.MODEL_NOT_TRAINED:
            return SegmentationResult(
                status=ModelStatus.MODEL_NOT_TRAINED,
                masks=[],
                inference_time_ms=elapsed_ms,
                model_name="MockFoodSegmenter",
                error_message=self._error_msg,
            )

        return SegmentationResult(
            status=self._status,
            masks=self.fixture_masks,
            inference_time_ms=elapsed_ms,
            model_name="MockFoodSegmenter",
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "mock_segmenter",
            "status": self._status.value,
            "is_ready": self.is_ready(),
            "has_fixtures": len(self.fixture_masks) > 0,
        }
