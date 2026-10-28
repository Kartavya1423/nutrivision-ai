"""
NutriVision AI — Food Classification Base Interface and Schemas
Receives food-only cropped image and outputs fine-grained food category recognition.
Strictly reports MODEL_NOT_TRAINED until real weights are supplied.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from src.detection.detector import ModelStatus
from src.utils.image_ops import load_image


class ClassificationStatus(str, Enum):
    """Fine-grained classification status flags."""
    CLASSIFICATION_SUCCESS = "CLASSIFICATION_SUCCESS"
    CLASSIFICATION_LOW_CONFIDENCE = "CLASSIFICATION_LOW_CONFIDENCE"
    CLASSIFICATION_MODEL_NOT_READY = "CLASSIFICATION_MODEL_NOT_READY"
    CLASSIFICATION_FAILED = "CLASSIFICATION_FAILED"


@dataclass
class ClassificationOutput:
    """Individual food classification schema."""
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    top_k_candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4) if self.confidence is not None else None,
            "top_k_candidates": self.top_k_candidates,
        }


@dataclass
class ClassificationResult:
    """Standardized envelope for food classification."""
    status: ModelStatus
    result: Optional[ClassificationOutput] = None
    inference_time_ms: float = 0.0
    model_name: str = "BaseFoodClassifier"
    error_message: Optional[str] = None
    confidence_status: Optional[str] = None
    breakdown_ms: Optional[Dict[str, float]] = None

    @property
    def is_success(self) -> bool:
        return self.status == ModelStatus.READY and self.result is not None

    def to_dict(self) -> Dict[str, Any]:
        res_dict = self.result.to_dict() if self.result else {
            "class_id": None,
            "class_name": None,
            "confidence": None,
            "top_k_candidates": [],
        }
        return {
            "status": self.status.value if isinstance(self.status, ModelStatus) else str(self.status),
            "model_name": self.model_name,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "confidence_status": self.confidence_status or (
                ClassificationStatus.CLASSIFICATION_SUCCESS.value if self.is_success else ClassificationStatus.CLASSIFICATION_MODEL_NOT_READY.value
            ),
            "result": res_dict,
            "breakdown_ms": {k: round(v, 2) for k, v in self.breakdown_ms.items()} if self.breakdown_ms else None,
            "error_message": self.error_message,
        }


class BaseFoodClassifier(ABC):
    """
    Abstract interface for food classification models.
    Input: food-only image patch.
    Output: class_id, class_name, confidence.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self._status: ModelStatus = ModelStatus.MODEL_NOT_TRAINED
        self._error_msg: Optional[str] = None

    @abstractmethod
    def load_model(self) -> bool:
        """Load trained classifier weights."""
        pass

    @abstractmethod
    def classify(
        self,
        food_crop: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 5,
    ) -> ClassificationResult:
        """
        Classify a food-only cropped image.
        Args:
            food_crop: Food-only image (background removed or cropped patch).
            top_k: Number of top candidate classes to return.
        """
        pass

    def is_ready(self) -> bool:
        return self._status == ModelStatus.READY

    @property
    def status(self) -> ModelStatus:
        return self._status

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass
