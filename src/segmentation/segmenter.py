"""
NutriVision AI — Food Segmentation Base Interface and Schemas
Phase 3: SAM 2 Food Instance Segmentation

Defines contracts, status codes, and data structures for promptable
food instance segmentation using SAM 2.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.detection.detector import ModelStatus
from src.utils.image_ops import load_image


class SegmentationStatusCode(str, Enum):
    """Machine-readable status codes for segmentation lifecycle and inference."""
    SEGMENTATION_SUCCESS = "SEGMENTATION_SUCCESS"
    SEGMENTATION_MODEL_NOT_AVAILABLE = "SEGMENTATION_MODEL_NOT_AVAILABLE"
    SEGMENTATION_MODEL_NOT_LOADED = "SEGMENTATION_MODEL_NOT_LOADED"
    NO_DETECTIONS = "NO_DETECTIONS"
    SEGMENTATION_FAILED = "SEGMENTATION_FAILED"
    INVALID_MASK = "INVALID_MASK"
    MASK_TOO_SMALL = "MASK_TOO_SMALL"
    CROP_FAILED = "CROP_FAILED"
    INVALID_IMAGE = "INVALID_IMAGE"


@dataclass
class FoodMask:
    """Represents a segmented food item mask and associated spatial metrics."""
    mask: np.ndarray  # 2D boolean array (H, W) where True indicates food pixel
    class_name: str = "food"
    class_id: int = 0
    confidence: float = 0.0  # Quality score / predicted IoU from SAM 2
    bbox: Optional[List[float]] = None  # Tight bbox enclosing the mask [x1, y1, x2, y2]
    prompt_bbox: Optional[List[float]] = None  # Input YOLO prompt bbox [x1, y1, x2, y2]
    area_pixels: int = 0
    relative_area: float = 0.0
    status: str = SegmentationStatusCode.SEGMENTATION_SUCCESS.value
    crop_rgb: Optional[np.ndarray] = None
    crop_rgba: Optional[np.ndarray] = None
    crop_bbox: Optional[List[int]] = None
    selection_strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.mask, np.ndarray) or self.mask.ndim != 2:
            raise ValueError("Mask must be a 2D NumPy array.")
        # Ensure boolean mask
        if self.mask.dtype != bool:
            self.mask = self.mask > 0
        if self.area_pixels == 0:
            self.area_pixels = int(np.sum(self.mask))
        if self.bbox is None and self.area_pixels > 0:
            self.bbox = self.get_bounding_box()

    def get_bounding_box(self) -> List[float]:
        """Compute tight bounding box [x1, y1, x2, y2] enclosing the mask."""
        if self.area_pixels == 0:
            return [0.0, 0.0, 0.0, 0.0]
        rows = np.any(self.mask, axis=1)
        cols = np.any(self.mask, axis=0)
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        if len(y_indices) == 0 or len(x_indices) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        y1, y2 = float(y_indices[0]), float(y_indices[-1])
        x1, x2 = float(x_indices[0]), float(x_indices[-1])
        return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]

    def as_uint8(self) -> np.ndarray:
        """Return mask as uint8 array (255 for food, 0 for background)."""
        return (self.mask.astype(np.uint8)) * 255

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox,
            "prompt_bbox": self.prompt_bbox,
            "area_pixels": self.area_pixels,
            "relative_area": round(float(self.relative_area), 4),
            "shape": list(self.mask.shape),
            "status": self.status,
            "selection_strategy": self.selection_strategy,
            "crop_bbox": self.crop_bbox,
            "has_crop_rgb": self.crop_rgb is not None,
            "has_crop_rgba": self.crop_rgba is not None,
            "metadata": self.metadata,
        }


@dataclass
class SegmentationResult:
    """Envelope containing all instance masks and multi-item composites."""
    status: Union[ModelStatus, SegmentationStatusCode, str]
    status_code: str = SegmentationStatusCode.SEGMENTATION_SUCCESS.value
    masks: List[FoodMask] = field(default_factory=list)
    combined_mask: Optional[np.ndarray] = None
    inference_time_ms: float = 0.0
    model_name: str = "BaseSegmenter"
    device: str = "cpu"
    checkpoint: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        # Keep status and status_code coherent
        if isinstance(self.status, (ModelStatus, SegmentationStatusCode)):
            raw_val = self.status.value
        else:
            raw_val = str(self.status)

        # If status_code was default but status indicates error/unloaded
        if self.status_code == SegmentationStatusCode.SEGMENTATION_SUCCESS.value and raw_val != ModelStatus.READY.value:
            if raw_val in (ModelStatus.MODEL_NOT_LOADED.value, SegmentationStatusCode.SEGMENTATION_MODEL_NOT_LOADED.value):
                self.status_code = SegmentationStatusCode.SEGMENTATION_MODEL_NOT_LOADED.value
            else:
                self.status_code = raw_val

        # Ensure ModelStatus compatibility
        if self.status_code in (SegmentationStatusCode.SEGMENTATION_SUCCESS.value, SegmentationStatusCode.NO_DETECTIONS.value):
            self.status = ModelStatus.READY
        elif self.status_code in (SegmentationStatusCode.SEGMENTATION_MODEL_NOT_LOADED.value, ModelStatus.MODEL_NOT_LOADED.value):
            self.status = ModelStatus.MODEL_NOT_LOADED
        elif self.status_code in (SegmentationStatusCode.SEGMENTATION_FAILED.value, SegmentationStatusCode.INVALID_IMAGE.value, ModelStatus.ERROR.value):
            self.status = ModelStatus.ERROR

    @property
    def is_success(self) -> bool:
        return self.status == ModelStatus.READY or self.status_code == SegmentationStatusCode.SEGMENTATION_SUCCESS.value

    @property
    def count(self) -> int:
        return len(self.masks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, (ModelStatus, SegmentationStatusCode)) else str(self.status),
            "status_code": self.status_code,
            "count": self.count,
            "model_name": self.model_name,
            "device": self.device,
            "checkpoint": self.checkpoint,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "masks": [m.to_dict() for m in self.masks],
            "has_combined_mask": self.combined_mask is not None,
            "error_message": self.error_message,
        }

    def render_overlay(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Overlay distinct colored transparent masks onto the original image.
        Returns uint8 RGB numpy array.
        """
        np_img, _ = load_image(image)
        h, w = np_img.shape[:2]
        canvas = np_img.astype(np.float32).copy()

        # Distinct color palette for multiple food items (RGB)
        palette = [
            (230, 25, 75),   # Red
            (60, 180, 75),   # Green
            (255, 225, 25),  # Yellow
            (0, 130, 200),   # Blue
            (245, 130, 48),  # Orange
            (145, 30, 180),  # Purple
            (70, 240, 240),  # Cyan
            (240, 50, 230),  # Magenta
            (210, 245, 60),  # Lime
            (250, 190, 212), # Pink
        ]

        for idx, food_mask in enumerate(self.masks):
            m = food_mask.mask
            if m.shape[:2] != (h, w):
                continue
            color = np.array(palette[idx % len(palette)], dtype=np.float32)
            canvas[m] = canvas[m] * (1.0 - alpha) + color * alpha

        return np.clip(canvas, 0, 255).astype(np.uint8)


class BaseFoodSegmenter(ABC):
    """Abstract interface for SAM2 / prompt-based food instance segmentation."""

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = None
        self._status: ModelStatus = ModelStatus.MODEL_NOT_LOADED
        self._error_msg: Optional[str] = None

    @abstractmethod
    def load_model(self) -> bool:
        """Load segmentation model checkpoints into memory."""
        pass

    @abstractmethod
    def predict_masks(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        bboxes: Optional[List[List[float]]] = None,
        class_names: Optional[List[str]] = None,
    ) -> SegmentationResult:
        """
        Segment one or more food regions.
        
        Args:
            image: Original RGB image.
            bboxes: Optional bounding boxes [x1, y1, x2, y2] from YOLO11 detector.
            class_names: Optional class names matching bboxes.
            
        Returns:
            SegmentationResult containing individual FoodMask items.
        """
        pass

    def is_ready(self) -> bool:
        return self._status == ModelStatus.READY

    @property
    def status(self) -> ModelStatus:
        return self._status

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return diagnostic metadata about segmentation backend."""
        pass
