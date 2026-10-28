"""
NutriVision AI — Unified End-to-End Pipeline Orchestrator
Coordinates multi-stage computer vision workflow:
Detection -> Segmentation -> Food Crop -> Classification -> Depth -> Volume -> Weight -> Freshness -> Nutrition.
Strictly reports honest statuses across all pipeline stages without fabricating predictions.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from src.classification.classifier import BaseFoodClassifier
from src.classification.efficientnet_classifier import EfficientNetB4Classifier as EfficientNetFoodClassifier
from src.config.settings import Settings, get_settings
from src.depth.depth_anything import DepthAnythingV2Estimator
from src.depth.depth_estimator import BaseDepthEstimator
from src.detection.detector import BaseFoodDetector, ModelStatus
from src.detection.yolo_detector import YOLO11FoodDetector
from src.explainability.explainer import GradCAMExplainer
from src.freshness.freshness_detector import BaseFreshnessDetector, EfficientNetFreshnessDetector
from src.nutrition.nutrition_service import MealNutritionReport, NutritionService
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.segmentation.segmenter import BaseFoodSegmenter
from src.utils.crop import extract_food_region
from src.utils.image_ops import load_image
from src.volume.volume_estimator import VolumeEstimator, VolumeResult
from src.weight.weight_estimator import WeightEstimator, WeightResult


@dataclass
class PipelineItemAnalysis:
    """Aggregated analysis for a single detected and segmented food item."""
    item_index: int
    class_name: str
    confidence: float
    bbox: List[float]
    has_mask: bool
    mask_pixels: int = 0
    crop_image: Optional[Image.Image] = None
    crop_rgba_arr: Optional[np.ndarray] = None
    volume_cm3: float = 0.0
    weight_grams: float = 0.0
    freshness_score: float = 0.0
    freshness_state: str = "unknown"
    nutrition: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_index": self.item_index,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox,
            "has_mask": self.has_mask,
            "mask_pixels": self.mask_pixels,
            "volume_cm3": round(float(self.volume_cm3), 2),
            "weight_grams": round(float(self.weight_grams), 1),
            "freshness_score": round(float(self.freshness_score), 4),
            "freshness_state": self.freshness_state,
            "nutrition": self.nutrition,
        }


@dataclass
class PipelineResult:
    """Consolidated report across the full NutriVision AI pipeline."""
    status: ModelStatus
    stage_statuses: Dict[str, str]
    total_items: int = 0
    items: List[PipelineItemAnalysis] = field(default_factory=list)
    total_meal_weight_grams: float = 0.0
    meal_nutrition: Optional[Dict[str, Any]] = None
    inference_time_ms: float = 0.0
    error_message: Optional[str] = None
    artifacts_saved: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "stage_statuses": self.stage_statuses,
            "total_items": self.total_items,
            "items": [item.to_dict() for item in self.items],
            "total_meal_weight_grams": round(float(self.total_meal_weight_grams), 1),
            "meal_nutrition": self.meal_nutrition,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "error_message": self.error_message,
            "artifacts_saved": self.artifacts_saved,
        }


class NutriVisionPipeline:
    """
    End-to-end multi-task vision pipeline for food recognition and dietary tracking.
    Modular design allows each model to be trained and plugged in independently.
    """

    def __init__(
        self,
        detector: Optional[BaseFoodDetector] = None,
        segmenter: Optional[BaseFoodSegmenter] = None,
        classifier: Optional[BaseFoodClassifier] = None,
        depth_estimator: Optional[BaseDepthEstimator] = None,
        freshness_detector: Optional[BaseFreshnessDetector] = None,
        volume_estimator: Optional[VolumeEstimator] = None,
        weight_estimator: Optional[WeightEstimator] = None,
        nutrition_service: Optional[NutritionService] = None,
        explainer: Optional[GradCAMExplainer] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()

        # Modular CV components
        self.detector = detector or YOLO11FoodDetector(settings=self.settings)
        self.segmenter = segmenter or SAM2FoodSegmenter(settings=self.settings)
        self.classifier = classifier or EfficientNetFoodClassifier(settings=self.settings)
        self.depth_estimator = depth_estimator or DepthAnythingV2Estimator(settings=self.settings)
        self.freshness_detector = freshness_detector or EfficientNetFreshnessDetector(settings=self.settings)
        
        # Volumetric, mass, and nutrition services
        self.volume_estimator = volume_estimator or VolumeEstimator(
            reference_object=self.settings.weight.reference_object,
            reference_dimension_cm=self.settings.weight.reference_dimension_cm,
        )
        self.weight_estimator = weight_estimator or WeightEstimator(settings=self.settings)
        self.nutrition_service = nutrition_service or NutritionService(settings=self.settings)
        self.explainer = explainer or GradCAMExplainer(settings=self.settings)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Diagnostic summary of all model backends and dependencies."""
        return {
            "detection": self.detector.get_model_info(),
            "segmentation": self.segmenter.get_model_info(),
            "classification": self.classifier.get_model_info(),
            "depth": self.depth_estimator.get_model_info(),
            "freshness": self.freshness_detector.get_model_info(),
            "all_ready": (
                self.detector.is_ready()
                and self.segmenter.is_ready()
                and self.classifier.is_ready()
                and self.depth_estimator.is_ready()
                and self.freshness_detector.is_ready()
            ),
        }

    def analyze_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        save_artifacts: bool = False,
        crop_padding: int = 10,
    ) -> PipelineResult:
        """
        Execute full pipeline on an uploaded or camera-captured image.
        """
        t0 = time.perf_counter()
        np_img, pil_img = load_image(image)
        h, w = np_img.shape[:2]

        stage_statuses: Dict[str, str] = {
            "detection": self.detector.status.value,
            "segmentation": self.segmenter.status.value,
            "classification": self.classifier.status.value,
            "depth": self.depth_estimator.status.value,
            "freshness": self.freshness_detector.status.value,
        }

        # 1. Food Detection
        det_result = self.detector.predict(np_img)
        stage_statuses["detection"] = det_result.status.value

        if not det_result.is_success or len(det_result.detections) == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return PipelineResult(
                status=det_result.status,
                stage_statuses=stage_statuses,
                total_items=0,
                items=[],
                inference_time_ms=elapsed_ms,
                error_message=det_result.error_message or "No food items detected.",
            )

        bboxes = [d.bbox for d in det_result.detections]
        class_names = [d.class_name for d in det_result.detections]

        # 2. Food Segmentation (SAM 2)
        seg_result = self.segmenter.predict_masks(np_img, bboxes=bboxes, class_names=class_names)
        stage_statuses["segmentation"] = seg_result.status.value

        # 3. Depth Estimation
        depth_result = self.depth_estimator.estimate_depth(np_img)
        stage_statuses["depth"] = depth_result.status.value

        # 4. Volume Estimation
        vol_result = self.volume_estimator.estimate_volume(
            masks=seg_result.masks if seg_result.is_success else [],
            depth_map=depth_result.depth_map if depth_result.is_success else None,
            image_shape=(h, w),
        )

        # 5. Weight Estimation
        weight_result = self.weight_estimator.estimate_weight(vol_result)

        # 6. Food-Only Crops, Freshness, and Nutrition per item
        items_analysis: List[PipelineItemAnalysis] = []
        nutrition_input_list: List[Dict[str, Any]] = []
        artifacts_saved: Dict[str, str] = {}

        for idx, det in enumerate(det_result.detections):
            # Associated mask
            matching_mask = seg_result.masks[idx] if seg_result.is_success and idx < len(seg_result.masks) else None
            
            crop_arr, crop_pil = None, None
            if matching_mask is not None:
                save_p = (
                    self.settings.outputs.crops / f"item_{idx}_{int(time.time())}.png"
                    if save_artifacts else None
                )
                crop_arr, crop_pil = extract_food_region(
                    image=np_img,
                    mask=matching_mask,
                    padding=crop_padding,
                    save_path=save_p,
                    save_visualization=save_artifacts,
                )
                if save_p:
                    artifacts_saved[f"crop_{idx}"] = str(save_p)

            # Volume & Weight
            matching_weight = (
                weight_result.weights[idx]
                if weight_result.status == ModelStatus.READY and idx < len(weight_result.weights)
                else None
            )
            item_grams = matching_weight.weight_grams if matching_weight else 100.0
            item_vol = matching_weight.volume_cm3 if matching_weight else 0.0

            # Freshness
            fresh_score = 0.0
            fresh_state = "unknown"
            if crop_arr is not None:
                fr_res = self.freshness_detector.assess_freshness(crop_arr, class_name=det.class_name, item_index=idx)
                if fr_res.is_success and len(fr_res.assessments) > 0:
                    fresh_score = fr_res.assessments[0].freshness_score
                    fresh_state = fr_res.assessments[0].state.value

            # Nutrition
            nutrients = self.nutrition_service.get_nutrition_for_item(det.class_name, item_grams)
            nutrition_input_list.append({"name": det.class_name, "weight_grams": item_grams})

            items_analysis.append(
                PipelineItemAnalysis(
                    item_index=idx,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    has_mask=matching_mask is not None,
                    mask_pixels=matching_mask.area_pixels if matching_mask else 0,
                    crop_image=crop_pil,
                    crop_rgba_arr=crop_arr,
                    volume_cm3=item_vol,
                    weight_grams=item_grams,
                    freshness_score=fresh_score,
                    freshness_state=fresh_state,
                    nutrition=nutrients.to_dict(),
                )
            )

        # 7. Meal Nutrition Aggregation
        meal_report = self.nutrition_service.analyze_meal(nutrition_input_list)
        total_meal_grams = sum(i.weight_grams for i in items_analysis)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            status=ModelStatus.READY,
            stage_statuses=stage_statuses,
            total_items=len(items_analysis),
            items=items_analysis,
            total_meal_weight_grams=total_meal_grams,
            meal_nutrition=meal_report.to_dict(),
            inference_time_ms=elapsed_ms,
            artifacts_saved=artifacts_saved,
        )
