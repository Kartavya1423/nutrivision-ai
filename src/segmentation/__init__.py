"""
NutriVision AI — Food Segmentation Module
Phase 3: SAM 2 Food Instance Segmentation
"""

from src.segmentation.crop_export import save_segmentation_artifacts
from src.segmentation.mask_selection import evaluate_candidate_mask, select_best_mask
from src.segmentation.mock_segmenter import MockFoodSegmenter
from src.segmentation.pipeline import FoodCropResult, SegmentedFoodObject, YOLOToSAMPipeline
from src.segmentation.postprocessing import postprocess_mask
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.segmentation.segmenter import (
    BaseFoodSegmenter,
    FoodMask,
    SegmentationResult,
    SegmentationStatusCode,
)
from src.segmentation.visualizer import create_segmentation_comparison_panel

__all__ = [
    "BaseFoodSegmenter",
    "FoodMask",
    "SegmentationResult",
    "SegmentationStatusCode",
    "SAM2FoodSegmenter",
    "MockFoodSegmenter",
    "YOLOToSAMPipeline",
    "SegmentedFoodObject",
    "FoodCropResult",
    "select_best_mask",
    "evaluate_candidate_mask",
    "postprocess_mask",
    "save_segmentation_artifacts",
    "create_segmentation_comparison_panel",
]
