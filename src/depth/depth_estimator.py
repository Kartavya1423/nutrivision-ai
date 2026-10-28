"""
NutriVision AI — Monocular Depth Estimation Interface
Predicts depth maps for 3D food volume reconstruction using Depth Anything V2.
Crucial distinction:
- 'relative depth': Normalized inverse/relative depth output by monocular vision models (unitless).
- 'metric depth': Physical distance in centimeters (cm), requiring camera intrinsics or reference calibration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image

from src.detection.detector import ModelStatus
from src.utils.image_ops import load_image


class DepthType(str, Enum):
    RELATIVE = "relative"  # Monocular depth (unitless inverse/normalized)
    METRIC = "metric"      # Calibrated physical depth in cm


@dataclass
class DepthResult:
    """Envelope containing predicted depth map and metric calibration details."""
    status: ModelStatus
    depth_map: Optional[np.ndarray] = None  # 2D float array
    depth_type: DepthType = DepthType.RELATIVE
    is_metric_calibrated: bool = False
    reference_object_used: Optional[str] = None
    calibration_status: str = "DEPTH_SCALE_NOT_CALIBRATED"
    min_depth: float = 0.0
    max_depth: float = 0.0
    mean_depth: float = 0.0
    median_depth: float = 0.0
    percentiles: Optional[Dict[str, float]] = None
    inference_time_ms: float = 0.0
    model_name: str = "BaseDepthEstimator"
    device: str = "cpu"
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status == ModelStatus.READY and self.depth_map is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "depth_type": self.depth_type.value,
            "is_metric_calibrated": self.is_metric_calibrated,
            "calibration_status": self.calibration_status,
            "reference_object_used": self.reference_object_used,
            "has_depth_map": self.depth_map is not None,
            "shape": list(self.depth_map.shape) if self.depth_map is not None else None,
            "min_depth": round(float(self.min_depth), 4),
            "max_depth": round(float(self.max_depth), 4),
            "mean_depth": round(float(self.mean_depth), 4),
            "median_depth": round(float(self.median_depth), 4),
            "percentiles": {k: round(float(v), 4) for k, v in self.percentiles.items()} if self.percentiles else None,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "device": self.device,
            "error_message": self.error_message,
        }

    def render_colormap(self) -> Optional[np.ndarray]:
        """Convert depth map to RGB visualization (H, W, 3)."""
        if self.depth_map is None:
            return None
        d = self.depth_map.copy()
        d_min, d_max = float(np.min(d)), float(np.max(d))
        if d_max > d_min:
            norm = (d - d_min) / (d_max - d_min)
        else:
            norm = np.zeros_like(d)
        norm_uint8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        try:
            import cv2
            color_bgr = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_INFERNO)
            return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            r = np.clip(255 * (1.5 - np.abs(norm * 4 - 3)), 0, 255).astype(np.uint8)
            g = np.clip(255 * (1.5 - np.abs(norm * 4 - 2)), 0, 255).astype(np.uint8)
            b = np.clip(255 * (1.5 - np.abs(norm * 4 - 1)), 0, 255).astype(np.uint8)
            return np.stack([r, g, b], axis=-1)


class BaseDepthEstimator(ABC):
    """Abstract base class for monocular depth models."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self._status: ModelStatus = ModelStatus.MODEL_NOT_LOADED
        self._error_msg: Optional[str] = None

    @abstractmethod
    def load_model(self) -> bool:
        pass

    @abstractmethod
    def estimate_depth(self, image: Union[str, Path, np.ndarray, Image.Image]) -> DepthResult:
        pass

    def is_ready(self) -> bool:
        return self._status == ModelStatus.READY

    @property
    def status(self) -> ModelStatus:
        return self._status

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass
