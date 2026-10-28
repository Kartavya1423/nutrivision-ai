"""NutriVision AI Food Detection Module"""

from src.detection.detector import (
    BaseFoodDetector,
    BoundingBox,
    DetectionResult,
    FoodDetection,
    ModelStatus,
)
from src.detection.mock_detector import MockFoodDetector
from src.detection.yolo_detector import YOLO11FoodDetector
from src.detection.mask_to_yolo import MaskToYOLOConverter, YOLOBox, MaskConversionResult
from src.detection.crop_utils import crop_bounding_box, save_detection_crops
from src.detection.detection_config import MaskConversionConfig, YOLO11TrainingConfig, DetectionInferenceConfig

__all__ = [
    "BaseFoodDetector",
    "BoundingBox",
    "FoodDetection",
    "DetectionResult",
    "ModelStatus",
    "YOLO11FoodDetector",
    "MockFoodDetector",
    "MaskToYOLOConverter",
    "YOLOBox",
    "MaskConversionResult",
    "crop_bounding_box",
    "save_detection_crops",
    "MaskConversionConfig",
    "YOLO11TrainingConfig",
    "DetectionInferenceConfig",
]
