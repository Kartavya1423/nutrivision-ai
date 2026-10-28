"""
NutriVision AI — Food Detection Base Interface and Schemas
Defines abstract contracts for multi-item food localization models.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.utils.image_ops import load_image


class ModelStatus(str, Enum):
    """Execution and readiness status for modular CV models."""
    READY = "READY"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    MODEL_NOT_TRAINED = "MODEL_NOT_TRAINED"
    ERROR = "ERROR"


@dataclass
class BoundingBox:
    """Represents an axis-aligned bounding box [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        # Ensure ordering
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> List[float]:
        return [round(self.x1, 2), round(self.y1, 2), round(self.x2, 2), round(self.y2, 2)]

    def to_int_list(self) -> List[int]:
        return [int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))]

    def clamp(self, max_width: int, max_height: int) -> "BoundingBox":
        return BoundingBox(
            x1=max(0.0, min(float(max_width), self.x1)),
            y1=max(0.0, min(float(max_height), self.y1)),
            x2=max(0.0, min(float(max_width), self.x2)),
            y2=max(0.0, min(float(max_height), self.y2)),
        )


@dataclass
class FoodDetection:
    """Individual food item detection output."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    track_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(c), 2) for c in self.bbox],
            "track_id": self.track_id,
            "metadata": self.metadata,
        }


@dataclass
class DetectionResult:
    """Standardized envelope returned by any food detector."""
    status: ModelStatus
    detections: List[FoodDetection] = field(default_factory=list)
    image_shape: Optional[Tuple[int, int, int]] = None
    inference_time_ms: float = 0.0
    model_name: str = "BaseDetector"
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status == ModelStatus.READY

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def status_code(self) -> str:
        if self.status != ModelStatus.READY:
            return self.status.value
        if self.count == 0:
            return "NO_FOOD_DETECTED"
        if all(d.confidence < 0.05 for d in self.detections):
            return "LOW_CONFIDENCE"
        return "SUCCESS"

    def to_api_contract(self) -> Dict[str, Any]:
        """Returns standard production API contract conforming to Part 17 & 18."""
        return {
            "status": self.status.value,
            "status_code": self.status_code,
            "detections": [
                {
                    "class_name": d.class_name,
                    "class_id": d.class_id,
                    "confidence": round(float(d.confidence), 4),
                    "bbox": [round(float(c), 2) for c in d.bbox]
                }
                for d in self.detections
            ]
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "status_code": self.status_code,
            "count": self.count,
            "model_name": self.model_name,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "image_shape": list(self.image_shape) if self.image_shape else None,
            "detections": [d.to_dict() for d in self.detections],
            "error_message": self.error_message,
        }


class BaseFoodDetector(ABC):
    """Abstract interface for all food detection backends."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self._status: ModelStatus = ModelStatus.MODEL_NOT_LOADED
        self._error_msg: Optional[str] = None

    @abstractmethod
    def load_model(self) -> bool:
        """
        Load model weights into memory/device.
        
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        pass

    @abstractmethod
    def predict(self, image: Union[str, Path, np.ndarray, Image.Image]) -> DetectionResult:
        """
        Execute detection on input image.
        
        Args:
            image: Image file path, numpy array, or PIL Image.
            
        Returns:
            DetectionResult: Envelope with detections and status.
        """
        pass

    def is_ready(self) -> bool:
        """Check if model is loaded and ready to serve inference."""
        return self._status == ModelStatus.READY

    @property
    def status(self) -> ModelStatus:
        """Current lifecycle status of the detector."""
        return self._status

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return diagnostic metadata about the model."""
        pass
