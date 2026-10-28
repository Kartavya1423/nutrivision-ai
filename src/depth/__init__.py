"""NutriVision AI Depth Estimation Module"""

from src.depth.depth_anything import DepthAnythingV2Estimator
from src.depth.depth_estimator import (
    BaseDepthEstimator,
    DepthResult,
    DepthType,
)
from src.depth.mock_depth import MockDepthEstimator

__all__ = [
    "BaseDepthEstimator",
    "DepthResult",
    "DepthType",
    "DepthAnythingV2Estimator",
    "MockDepthEstimator",
]
