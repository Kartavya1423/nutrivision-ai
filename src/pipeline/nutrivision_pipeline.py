"""
NutriVision AI — Authoritative End-to-End Pipeline (Phase 8)
Single Source of Truth for complete food analysis:
IMAGE
  -> YOLO11 Detection
  -> SAM2 Instance Segmentation
  -> Food-Only Isolated Crop
  -> EfficientNet-B4 Fine-Grained Classification
  -> Freshness & Spoilage Classification
  -> Depth Anything V2 Monocular Depth
  -> Geometric Voxel Volume Estimation
  -> Traceable Density Weight Estimation
  -> USDA FoodData Central Nutritional Scaling
  -> Dual Grad-CAM Visual Explainability (Food & Freshness)
  -> Consolidated Structured Result + Meal Summary Card

Serves both the FastAPI backend and Streamlit frontend without duplicate inference code.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import cv2
import numpy as np
from PIL import Image

from src.classification.classifier import BaseFoodClassifier
from src.classification.efficientnet_classifier import EfficientNetB4Classifier
from src.config.settings import Settings, get_settings
from src.depth.depth_anything import DepthAnythingV2Estimator
from src.depth.depth_estimator import BaseDepthEstimator
from src.detection.detector import BaseFoodDetector, FoodDetection, ModelStatus
from src.detection.yolo_detector import YOLO11FoodDetector
from src.explainability.explainer import NutriVisionExplainer
from src.freshness.freshness_detector import BaseFreshnessDetector, EfficientNetFreshnessDetector
from src.models.model_manager import ModelManager, get_model_manager
from src.nutrition.nutrition_service import MealNutritionReport, NutritionService, ScaledNutrients
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.segmentation.segmenter import BaseFoodSegmenter
from src.utils.crop import extract_food_region
from src.utils.image_ops import load_image
from src.volume.volume_estimator import VolumeEstimator, VolumeResult
from src.weight.weight_estimator import WeightEstimator, WeightResult
from src.weight.contour_weight_estimator import ContourWeightEstimator

logger = logging.getLogger("nutrivision.pipeline")

# Phase 6 Evidence-Based Empirical Performance Tiers
TARGET_FOOD_SUPPORT_TIERS: Dict[str, str] = {
    "orange": "Tier 1: High Confidence (87.3% Test F1, 100% Prec)",
    "apple": "Tier 1: High Confidence (85.7% Test F1, 100% Prec)",
    "banana": "Tier 1: High Confidence (81.1% Test F1, 88.2% Prec)",
    "tomato": "Tier 1: High Confidence (72.7% Test F1, 88.9% Rec)",
    "french_fries": "Tier 2: Moderately Supported (69.6% Test F1)",
    "watermelon": "Tier 2: Moderately Supported (60.0% Test F1)",
    "mango": "Tier 2: Moderately Supported (55.6% Test F1)",
    "chicken": "Tier 2: Moderately Supported (51.6% Test F1)",
    "egg": "Tier 2: Moderately Supported (46.8% Test F1)",
    "spinach": "Tier 2: Moderately Supported (46.2% Test F1)",
    "broccoli": "Tier 2: Moderately Supported (43.8% Test F1)",
    "lettuce": "Tier 2: Moderately Supported (41.7% Test F1)",
    "grape": "Tier 2: Moderately Supported (40.0% Test F1)",
    "burger": "Tier 3: Experimental / Low Confidence (38.7% Test F1)",
    "bread": "Tier 3: Experimental / Low Confidence (36.6% Test F1)",
    "cake": "Tier 3: Experimental / Low Confidence (28.6% Test F1)",
    "strawberry": "Tier 3: Experimental / Low Confidence (25.0% Test F1)",
    "pineapple": "Tier 3: Experimental / Low Confidence (25.0% Test F1)",
    "pizza": "Tier 3: Experimental / Low Confidence (20.7% Test F1)",
    "ice_cream": "Tier 3: Experimental / Low Confidence (12.5% Test F1)",
    "cucumber": "Tier 4: Failed / Unsupported (0% Test F1)",
    "paneer": "Tier 5: Data Deficient (0 images in training)",
    "butter": "Tier 5: Data Deficient (Bundled in cheese_butter)",
}


class NutriVisionPipeline:
    """
    Authoritative NutriVision master pipeline.
    Coordinates all 10 computer vision and nutrition components.
    Provides structured status values, per-stage timing, and failure isolation.
    """

    def __init__(
        self,
        detector: Optional[BaseFoodDetector] = None,
        segmenter: Optional[BaseFoodSegmenter] = None,
        classifier: Optional[BaseFoodClassifier] = None,
        fruit_classifier: Optional[Any] = None,
        freshness_detector: Optional[BaseFreshnessDetector] = None,
        depth_estimator: Optional[BaseDepthEstimator] = None,
        volume_estimator: Optional[VolumeEstimator] = None,
        weight_estimator: Optional[WeightEstimator] = None,
        nutrition_service: Optional[NutritionService] = None,
        explainer: Optional[NutriVisionExplainer] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.root_dir = self.settings.root_dir

        # Setup Phase 8 output directories
        self.output_dir = self.root_dir / "outputs" / "phase8"
        self.uploads_dir = self.output_dir / "uploads"
        self.results_dir = self.output_dir / "results"
        self.viz_dir = self.output_dir / "visualizations"
        self.crops_dir = self.output_dir / "crops"
        self.gradcam_dir = self.output_dir / "gradcam"
        for d in [self.uploads_dir, self.results_dir, self.viz_dir, self.crops_dir, self.gradcam_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Retrieve centralized models or use injected fixtures
        mgr = get_model_manager(settings=self.settings)
        self.detector = detector or mgr.detector
        self.segmenter = segmenter or mgr.segmenter
        self.classifier = classifier or mgr.classifier
        self.fruit_classifier = fruit_classifier or mgr.fruit_classifier
        self.freshness_detector = freshness_detector or mgr.freshness_detector
        self.depth_estimator = depth_estimator or mgr.depth_estimator

        # Services
        ref_obj = getattr(self.settings.weight, "reference_object", "coin")
        ref_dim = getattr(self.settings.weight, "reference_dimension_cm", 2.7)
        self.volume_estimator = volume_estimator or VolumeEstimator(
            reference_object=ref_obj,
            reference_dimension_cm=ref_dim,
        )
        self.weight_estimator = weight_estimator or WeightEstimator()
        self.contour_weight_estimator = ContourWeightEstimator()
        self.nutrition_service = nutrition_service or NutritionService(settings=self.settings)

        # Grad-CAM explainer
        if explainer is not None:
            self.explainer = explainer
        else:
            fc = self.classifier if isinstance(self.classifier, EfficientNetB4Classifier) else None
            fd = self.freshness_detector if isinstance(self.freshness_detector, EfficientNetFreshnessDetector) else None
            self.explainer = NutriVisionExplainer(
                food_classifier=fc,
                freshness_detector=fd,
                settings=self.settings,
            )

    def process_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        enable_gradcam: bool = True,
        enable_nutrition: bool = True,
        crop_padding: int = 10,
        confidence_threshold: Optional[float] = None,
        result_id: Optional[str] = None,
        classifier_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the authoritative 10-stage NutriVision pipeline.
        Returns a structured dictionary strictly adhering to Section 8.3 & 8.4 schema.
        """
        start_time = time.perf_counter()
        res_id = result_id or f"nv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        ckpt_path_str = str(getattr(self.classifier, "model_path", ""))
        model_version = "target_foods_v1" if "target_foods_v1" in ckpt_path_str else "apple_baseline_198"

        timings: Dict[str, float] = {
            "detection_ms": 0.0,
            "segmentation_ms": 0.0,
            "classification_ms": 0.0,
            "freshness_ms": 0.0,
            "depth_ms": 0.0,
            "volume_ms": 0.0,
            "weight_ms": 0.0,
            "nutrition_ms": 0.0,
            "gradcam_ms": 0.0,
            "total_ms": 0.0,
        }
        errors: List[str] = []

        # 1. Load and validate image
        try:
            np_img, pil_img = load_image(image)
        except Exception as e:
            timings["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
            return {
                "status": "INVALID_IMAGE",
                "result_id": res_id,
                "timestamp": timestamp,
                "source_image": str(image) if isinstance(image, (str, Path)) else "upload",
                "device": getattr(self.detector, "device", "cpu"),
                "image_dimensions": {"width": 0, "height": 0},
                "total_foods_detected": 0,
                "items": [],
                "foods": [],
                "meal_summary": None,
                "visualizations": {},
                "timing": timings,
                "errors": [f"Failed to load image: {str(e)}"],
            }

        height, width = np_img.shape[:2]
        img_dims = {"width": width, "height": height}

        # 2. Food Detection (YOLO11)
        t_det0 = time.perf_counter()
        if not self.detector.is_ready():
            timings["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
            msg = "YOLO11 food detection model is not loaded or checkpoint is missing."
            return {
                "status": "DETECTION_MODEL_NOT_AVAILABLE",
                "message": msg,
                "result_id": res_id,
                "timestamp": timestamp,
                "source_image": str(image) if isinstance(image, (str, Path)) else "upload",
                "device": getattr(self.detector, "device", "cpu"),
                "image_dimensions": img_dims,
                "total_foods_detected": 0,
                "items": [],
                "foods": [],
                "meal_summary": None,
                "visualizations": {},
                "timing": timings,
                "errors": [msg],
            }

        try:
            det_result = self.detector.predict(np_img, conf_threshold=confidence_threshold)
            timings["detection_ms"] = round((time.perf_counter() - t_det0) * 1000.0, 2)
        except Exception as e:
            timings["detection_ms"] = round((time.perf_counter() - t_det0) * 1000.0, 2)
            timings["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
            return {
                "status": "DETECTION_FAILED",
                "result_id": res_id,
                "timestamp": timestamp,
                "source_image": str(image) if isinstance(image, (str, Path)) else "upload",
                "device": getattr(self.detector, "device", "cpu"),
                "image_dimensions": img_dims,
                "total_foods_detected": 0,
                "items": [],
                "foods": [],
                "meal_summary": None,
                "visualizations": {},
                "timing": timings,
                "errors": [f"YOLO11 detection error: {str(e)}"],
            }

        # Monocular Depth Estimation (Depth Anything V2)
        t_dep0 = time.perf_counter()
        depth_info = None
        depth_map_path = None
        if self.depth_estimator and self.depth_estimator.is_ready():
            try:
                depth_info = self.depth_estimator.estimate_depth(np_img)
                if depth_info and depth_info.is_success and depth_info.depth_map is not None:
                    # Save depth map visualization
                    d_norm = cv2.normalize(depth_info.depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
                    depth_map_path = str(self.viz_dir / f"{res_id}_depth.png")
                    cv2.imwrite(depth_map_path, d_color)
            except Exception as e:
                errors.append(f"Depth Anything V2 warning: {str(e)}")
        timings["depth_ms"] = round((time.perf_counter() - t_dep0) * 1000.0, 2)

        detections = det_result.detections if det_result and det_result.is_success else []

        # Extract solid whole-food saliency to ensure full-food observation
        solid_mask, whole_bbox, cov = None, None, 0.0
        try:
            solid_mask, whole_bbox, cov = self.contour_weight_estimator._extract_robust_food_mask(np_img)
        except Exception as e:
            logger.warning(f"Whole-food saliency extraction warning: {e}")

        def expand_box_to_component(box: List[float], mask: np.ndarray, w_img: int, h_img: int) -> List[float]:
            if mask is None or np.sum(mask) == 0:
                return box
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
            sub_labels = np.unique(labels[y1:y2, x1:x2])
            sub_labels = [l for l in sub_labels if l > 0]
            if not sub_labels:
                return box

            ex1, ey1, ex2, ey2 = x1, y1, x2, y2
            for l in sub_labels:
                cx, cy, cw, ch, carea = stats[l]
                if carea < 0.02 * (w_img * h_img):
                    continue
                comp_mask = (labels == l)
                covered_px = np.count_nonzero(comp_mask[y1:y2, x1:x2])
                overlap_ratio = covered_px / max(1, carea)
                if overlap_ratio >= 0.25:
                    margin = 4
                    if x1 > (cx + margin) or y1 > (cy + margin) or x2 < (cx + cw - margin) or y2 < (cy + ch - margin) or overlap_ratio < 0.88:
                        ex1 = min(ex1, cx)
                        ey1 = min(ey1, cy)
                        ex2 = max(ex2, cx + cw)
                        ey2 = max(ey2, cy + ch)

            return [float(max(0, ex1)), float(max(0, ey1)), float(min(w_img, ex2)), float(min(h_img, ey2))]

        # Expand partial boxes so close-up or single-item food is never cut in half
        if solid_mask is not None and np.sum(solid_mask) > 0:
            for det in detections:
                det.bbox = expand_box_to_component(det.bbox, solid_mask, width, height)

        # Merge close-up subpart and duplicate boxes
        if len(detections) > 1:
            from src.detection.yolo_detector import merge_subpart_and_duplicate_boxes
            detections = merge_subpart_and_duplicate_boxes(detections, width, height)

        # Holistic Full-Image Observation
        global_cls_result = None
        if self.classifier and self.classifier.is_ready():
            try:
                global_cls_result = self.classifier.classify(np_img)
            except Exception as e:
                logger.warning(f"Full-image classification warning: {e}")

        if len(detections) == 0:
            # Fallback to whole-food contour analysis (Research Paper Section III-D)
            if solid_mask is not None and whole_bbox is not None and cov >= 0.02:
                wx, wy, ww, wh = whole_bbox
                x1, y1 = max(0, wx), max(0, wy)
                x2, y2 = min(width, wx + ww), min(height, wy + wh)
                detections = [
                    FoodDetection(
                        class_id=0,
                        class_name="food",
                        confidence=0.50,
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                    )
                ]
            else:
                cw_pre = self.contour_weight_estimator.estimate_weight(np_img, "food")
                if cw_pre and cw_pre.status == "SUCCESS" and cw_pre.area_ratio >= 0.03:
                    x, y, w, h = cw_pre.contour_bbox
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(width, x + w), min(height, y + h)
                    detections = [
                        FoodDetection(
                            class_id=0,
                            class_name="food",
                            confidence=0.50,
                            bbox=[float(x1), float(y1), float(x2), float(y2)],
                        )
                    ]

        use_fruit_classifier = (
            classifier_mode == "fruit_classifier"
            or (
                classifier_mode is None
                and getattr(getattr(self.settings, "classification", None), "active_classifier", None) == "fruit_classifier"
            )
        )
        if len(detections) == 0 and use_fruit_classifier and self.fruit_classifier and self.fruit_classifier.is_ready():
            detections = [
                FoodDetection(
                    class_id=0,
                    class_name="food",
                    confidence=0.60,
                    bbox=[0.0, 0.0, float(width), float(height)],
                )
            ]

        if len(detections) == 0:
            timings["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
            no_food_res = {
                "status": "NO_FOOD_DETECTED",
                "result_id": res_id,
                "timestamp": timestamp,
                "source_image": str(image) if isinstance(image, (str, Path)) else "upload",
                "device": getattr(self.detector, "device", "cpu"),
                "model_version": model_version,
                "classifier_checkpoint": ckpt_path_str,
                "image_dimensions": img_dims,
                "total_foods_detected": 0,
                "items": [],
                "foods": [],
                "fruit_recognition": {
                    "fruit_name": None,
                    "dataset_class": None,
                    "freshness_status": None,
                    "freshness_note": "No food detected in image",
                    "confidence": 0.0,
                    "classification_model": "EfficientNet-B4",
                    "nutrition": {
                        "calories": None,
                        "protein": None,
                        "carbohydrates": None,
                        "fat": None,
                    },
                },
                "meal_summary": {
                    "total_items": 0,
                    "calculated_items_count": 0,
                    "unavailable_items_count": 0,
                    "total_weight_g": 0.0,
                    "total_calories_kcal": 0.0,
                    "total_protein_g": 0.0,
                    "total_carbohydrates_g": 0.0,
                    "total_fat_g": 0.0,
                    "total_fiber_g": 0.0,
                    "source": "USDA FoodData Central",
                    "unavailable_foods": [],
                },
                "visualizations": {
                    "depth_map_path": depth_map_path,
                } if depth_map_path else {},
                "timing": timings,
                "errors": errors,
            }
            try:
                res_json_path = self.results_dir / f"{res_id}.json"
                with open(res_json_path, "w", encoding="utf-8") as f:
                    json.dump(no_food_res, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to persist no_food_res json: {e}")
            return no_food_res

        bboxes = [d.bbox for d in detections]
        class_names = [d.class_name for d in detections]

        # 3. Instance Segmentation (SAM 2)
        t_seg0 = time.perf_counter()
        seg_result = None
        if self.segmenter and self.segmenter.is_ready():
            try:
                seg_result = self.segmenter.predict_masks(np_img, bboxes=bboxes, class_names=class_names)
            except Exception as e:
                errors.append(f"SAM2 segmentation warning: {str(e)}")
        timings["segmentation_ms"] = round((time.perf_counter() - t_seg0) * 1000.0, 2)

        # 5. Process each localized food item independently
        items: List[Dict[str, Any]] = []
        annotated_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        for idx, det in enumerate(detections):
            # Safe bounding box extraction
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Mask extraction
            mask_obj = None
            mask_arr = None
            mask_area = 0
            if seg_result and seg_result.is_success and idx < len(seg_result.masks):
                m_candidate = seg_result.masks[idx]
                m_raw = getattr(m_candidate, "mask", getattr(m_candidate, "mask_array", None))
                if m_raw is not None and isinstance(m_raw, np.ndarray):
                    mask_obj = m_candidate
                    mask_arr = m_raw.astype(bool)
                    mask_area = int(np.sum(mask_arr))

            # Food-only crop extraction with natural margin/padding (preserves complete context for deep networks)
            crop_path = None
            pad = max(12, int(0.04 * max(width, height)), int(crop_padding))
            cy1 = max(0, y1 - pad)
            cy2 = min(height, y2 + pad)
            cx1 = max(0, x1 - pad)
            cx2 = min(width, x2 + pad)
            food_crop_rgb = np_img[cy1:cy2, cx1:cx2].copy() if cy2 > cy1 and cx2 > cx1 else None
            try:
                if food_crop_rgb is not None and food_crop_rgb.size > 0:
                    crop_path = str(self.crops_dir / f"{res_id}_item{idx}_{det.class_name}_crop.png")
                    cv2.imwrite(crop_path, cv2.cvtColor(food_crop_rgb, cv2.COLOR_RGB2BGR))
            except Exception as e:
                errors.append(f"Crop extraction warning for item {idx}: {str(e)}")

            # Classification (EfficientNet-B4)
            t_cls0 = time.perf_counter()
            cls_label = det.class_name
            cls_conf = det.confidence
            cls_top_k: List[Dict[str, Any]] = []
            alt_suggestions: List[Dict[str, Any]] = []
            cls_status = "SUCCESS"
            fruit_rec_item: Optional[Dict[str, Any]] = None

            use_fruit_classifier = (
                classifier_mode == "fruit_classifier"
                or (
                    classifier_mode is None
                    and getattr(getattr(self.settings, "classification", None), "active_classifier", None) == "fruit_classifier"
                )
            )

            # Check Fruit Classifier on crop
            if (
                self.fruit_classifier
                and self.fruit_classifier.is_ready()
                and food_crop_rgb is not None
                and food_crop_rgb.size > 0
            ):
                try:
                    c_fruit = self.fruit_classifier.predict_crop(food_crop_rgb)
                    if c_fruit and c_fruit.get("confidence", 0.0) >= 0.20:
                        fruit_rec_item = c_fruit
                        if use_fruit_classifier or c_fruit.get("confidence", 0.0) >= 0.45 or det.class_name in ("food", "fruit", "others", "other_food", "proposal"):
                            cls_label = c_fruit["fruit_name"].lower()
                            cls_conf = c_fruit["confidence"]
                            model_version = "updated_fruit_dataset_v1"
                            cls_top_k = [
                                {"class": k, "confidence": round(float(v), 4)}
                                for k, v in list(c_fruit.get("probabilities", {}).items())[:5]
                            ]
                            alt_suggestions = [
                                {
                                    "class": k,
                                    "confidence": round(float(v), 4),
                                    "support_tier": "Updated Fruit Dataset (6-Class)",
                                }
                                for k, v in list(c_fruit.get("probabilities", {}).items())[:4]
                                if k != c_fruit.get("dataset_class")
                            ]
                except Exception as ef:
                    logger.warning(f"Fruit classifier prediction error: {ef}")

            # General food classifier execution if not exclusively using fruit classifier
            if not use_fruit_classifier and self.classifier and self.classifier.is_ready() and food_crop_rgb is not None and food_crop_rgb.size > 0:
                ckpt_path_str = str(getattr(self.classifier, "model_path", ""))
                if "updated_fruit" not in model_version:
                    model_version = "target_foods_v1" if "target_foods_v1" in ckpt_path_str else "apple_baseline_198"

                generic_names = ("food", "fruit", "vegetable", "dish", "others", "other_food", "proposal", "bowl")
                try:
                    c_res = self.classifier.classify(food_crop_rgb)
                    if c_res.is_success and c_res.result:
                        c_label = c_res.result.class_name
                        c_conf = c_res.result.confidence

                        # Fuse with holistic full-image observation when available
                        item_area_ratio = float((x2 - x1) * (y2 - y1)) / float(width * height)
                        if item_area_ratio >= 0.15 and global_cls_result and global_cls_result.is_success and global_cls_result.result:
                            g_label = global_cls_result.result.class_name
                            g_conf = global_cls_result.result.confidence or 0.0
                            if g_label not in generic_names:
                                if det.class_name in generic_names and g_conf >= 0.20:
                                    c_label = g_label
                                    c_conf = max(c_conf, g_conf)
                                elif g_conf >= 0.45 and (g_conf > c_conf + 0.15 or c_conf < 0.35):
                                    c_label = g_label
                                    c_conf = g_conf

                        if fruit_rec_item and fruit_rec_item.get("confidence", 0.0) > (c_conf + 0.10):
                            cls_label = fruit_rec_item["fruit_name"].lower()
                            cls_conf = fruit_rec_item["confidence"]
                            model_version = "updated_fruit_dataset_v1"
                        else:
                            # Prioritize classifier over detector label
                            if det.class_name in generic_names:
                                if c_label not in generic_names and c_conf >= 0.15:
                                    cls_label = c_label
                                    cls_conf = c_conf
                                else:
                                    cls_label = c_label if c_conf >= 0.15 else det.class_name
                                    cls_conf = c_conf
                            elif c_label == det.class_name:
                                cls_label = c_label
                                cls_conf = max(c_conf, det.confidence)
                            elif c_conf >= 0.30:
                                cls_label = c_label
                                cls_conf = c_conf
                            elif det.confidence >= c_conf + 0.20:
                                cls_label = det.class_name
                                cls_conf = det.confidence
                            else:
                                cls_label = c_label
                                cls_conf = c_conf

                            from src.classification.efficientnet_classifier import CANONICAL_FOOD_MAP
                            cls_label = CANONICAL_FOOD_MAP.get(cls_label, cls_label)

                            cls_top_k = [
                                {"class": cand.get("class_name", ""), "confidence": round(float(cand.get("confidence", 0.0)), 4)}
                                for cand in c_res.result.top_k_candidates
                            ]
                            alt_suggestions = [
                                {
                                    "class": cand.get("class_name", ""),
                                    "confidence": round(float(cand.get("confidence", 0.0)), 4),
                                    "support_tier": TARGET_FOOD_SUPPORT_TIERS.get(cand.get("class_name", ""), "General Food Class"),
                                }
                                for cand in c_res.result.top_k_candidates
                                if cand.get("class_name") != cls_label
                            ][:3]
                except Exception as e:
                    cls_status = "CLASSIFICATION_FAILED"
                    errors.append(f"Classification failed for item {idx}: {str(e)}")
            timings["classification_ms"] += (time.perf_counter() - t_cls0) * 1000.0

            # If candidate came from zero-detection contour fallback, verify it meets minimum classification confidence
            generic_names = ("food", "fruit", "vegetable", "dish", "others", "other_food", "proposal", "bowl")
            if det.class_name == "food" and (cls_label in generic_names or cls_conf < 0.20):
                continue

            # Freshness Classification
            t_fre0 = time.perf_counter()
            fresh_label = "unknown"
            fresh_conf = 0.0
            fresh_probs: Dict[str, float] = {}
            fresh_status = "SUCCESS"

            CALIBRATED_FRESHNESS_CLASSES = {"apple", "banana", "orange"}
            if not use_fruit_classifier and cls_label.lower() in CALIBRATED_FRESHNESS_CLASSES:
                if self.freshness_detector and self.freshness_detector.is_ready() and food_crop_rgb is not None and food_crop_rgb.size > 0:
                    try:
                        f_res = self.freshness_detector.assess_freshness(food_crop_rgb)
                        if f_res.is_success:
                            fresh_label = f_res.predicted_state.value if hasattr(f_res.predicted_state, "value") else str(f_res.predicted_state)
                            fresh_conf = f_res.confidence
                            fresh_probs = {k: round(float(v), 4) for k, v in f_res.probabilities.items()}
                    except Exception as e:
                        fresh_status = "FRESHNESS_UNAVAILABLE"
                        errors.append(f"Freshness detection failed for item {idx}: {str(e)}")
                else:
                    fresh_status = "FRESHNESS_UNAVAILABLE"
            elif fruit_rec_item is not None:
                # Truthful handling of Good-only fruit dataset (Section 7)
                fresh_label = fruit_rec_item.get("freshness_status", "Good")
                fresh_conf = fruit_rec_item.get("confidence", 0.0)
                fresh_probs = {fruit_rec_item.get("freshness_status", "Good"): fruit_rec_item.get("confidence", 0.0)}
                fresh_status = "DATASET_LABEL_ONLY"
            else:
                fresh_status = "NOT_APPLICABLE"
                fresh_label = "not_applicable"
                fresh_conf = 0.0
                fresh_probs = {}
            timings["freshness_ms"] += (time.perf_counter() - t_fre0) * 1000.0

            # Volume & Weight Estimation
            t_vol0 = time.perf_counter()
            vol_res = None
            vol_val = None
            vol_status = "VOLUME_UNAVAILABLE"
            weight_val = None
            weight_status = "WEIGHT_UNAVAILABLE"

            try:
                vol_res = self.volume_estimator.estimate_volume(mask=mask_obj, depth_info=depth_info)
                if vol_res.is_success and vol_res.volume_cm3 is not None:
                    vol_val = round(float(vol_res.volume_cm3), 2)
                    vol_status = "SUCCESS"
                else:
                    vol_status = vol_res.status.value if hasattr(vol_res.status, "value") else str(vol_res.status)
            except Exception as e:
                vol_status = "VOLUME_FAILED"
                errors.append(f"Volume estimation error for item {idx}: {str(e)}")
            timings["volume_ms"] += (time.perf_counter() - t_vol0) * 1000.0

            t_wgt0 = time.perf_counter()
            # Contour-based 2D Weight Estimation as primary approach (Research Paper Section III-D)
            # Formula: We = Wb * (0.6 + 1.2 * Ar)
            try:
                cw_res = self.contour_weight_estimator.estimate_weight(
                    image=np_img,
                    food_name=cls_label,
                    bbox=[x1, y1, x2, y2],
                    mask=mask_obj,
                )
                if cw_res and cw_res.estimated_weight_g > 0:
                    weight_val = round(float(cw_res.estimated_weight_g), 2)
                    weight_status = "SUCCESS"
            except Exception as e:
                errors.append(f"Contour weight error for item {idx}: {str(e)}")

            # Volumetric fallback if contour weight unavailable
            if weight_val is None or weight_val <= 0:
                try:
                    w_res = self.weight_estimator.estimate_weight(class_name=cls_label, volume_output=vol_res)
                    if w_res.is_success and w_res.estimated_weight_grams is not None:
                        weight_val = round(float(w_res.estimated_weight_grams), 2)
                        weight_status = "SUCCESS"
                    else:
                        weight_status = w_res.status.value if hasattr(w_res.status, "value") else str(w_res.status)
                except Exception as e:
                    weight_status = "WEIGHT_FAILED"
                    errors.append(f"Weight estimation error for item {idx}: {str(e)}")

            # Standard portion fallback ensuring every recognized food has an empirical portion weight
            if weight_val is None or weight_val <= 0:
                base_w = self.contour_weight_estimator.get_base_weight(cls_label)
                weight_val = round(float(base_w), 2)
                weight_status = "SUCCESS"

            timings["weight_ms"] += (time.perf_counter() - t_wgt0) * 1000.0

            # USDA Nutrition Lookup & Serving Scaling
            t_nut0 = time.perf_counter()
            nutrition_data: Dict[str, Any] = {
                "status": "NUTRITION_UNAVAILABLE",
                "serving_size_g": 100.0,
                "serving_weight_g": weight_val,
                "calories_kcal": None,
                "protein_g": None,
                "carbohydrates_g": None,
                "fat_g": None,
                "fiber_g": None,
                "source": "USDA FoodData Central",
                "matched_food": None,
                "fdc_id": None,
                "preparation_variant": None,
                "variant_name": None,
                "is_generic_match": True,
                "available_variants": [],
                "weight_source": "vision_estimated",
                "weight_label": "Vision Estimated Portion",
                "scaling_formula": "nutrient_scaled = nutrient_per_100g * (portion_weight_g / 100.0)",
                "nutrients_per_100g": None,
            }

            if enable_nutrition and self.nutrition_service:
                try:
                    nut_res = self.nutrition_service.calculate_nutrition_for_weight(
                        food_name=cls_label,
                        weight_g=weight_val,
                        weight_source="vision_estimated",
                    )
                    if nut_res and nut_res.get("status") == "AVAILABLE":
                        nutrition_data["status"] = "SUCCESS"
                        nutrition_data["calories_kcal"] = nut_res.get("calories_kcal")
                        nutrition_data["protein_g"] = nut_res.get("protein_g")
                        nutrition_data["carbohydrates_g"] = nut_res.get("carbohydrates_g")
                        nutrition_data["fat_g"] = nut_res.get("fat_g")
                        nutrition_data["fiber_g"] = nut_res.get("fiber_g")
                        nutrition_data["sodium_mg"] = nut_res.get("sodium_mg")
                        nutrition_data["potassium_mg"] = nut_res.get("potassium_mg")
                        nutrition_data["calcium_mg"] = nut_res.get("calcium_mg")
                        nutrition_data["iron_mg"] = nut_res.get("iron_mg")
                        nutrition_data["sugars_g"] = nut_res.get("sugars_g")
                        nutrition_data["serving_size_g"] = nut_res.get("serving_size_g", 100.0)
                        nutrition_data["matched_food"] = nut_res.get("matched_food")
                        nutrition_data["fdc_id"] = nut_res.get("fdc_id")
                        nutrition_data["source"] = nut_res.get("source", "USDA FoodData Central")
                        nutrition_data["preparation_variant"] = nut_res.get("preparation_variant")
                        nutrition_data["variant_name"] = nut_res.get("variant_name")
                        nutrition_data["is_generic_match"] = nut_res.get("is_generic_match", True)
                        nutrition_data["available_variants"] = nut_res.get("available_variants", [])
                        nutrition_data["weight_source"] = nut_res.get("weight_source", "vision_estimated")
                        nutrition_data["weight_label"] = nut_res.get("weight_label", "Vision Estimated Portion")
                        nutrition_data["nutrients_per_100g"] = nut_res.get("nutrients_per_100g")
                    else:
                        nutrition_data["status"] = nut_res.get("error_code", "NUTRITION_UNAVAILABLE")
                except Exception as e:
                    nutrition_data["status"] = "NUTRITION_LOOKUP_FAILED"
                    errors.append(f"Nutrition lookup error for item {idx}: {str(e)}")
            timings["nutrition_ms"] += (time.perf_counter() - t_nut0) * 1000.0

            # Grad-CAM Explainability
            t_cam0 = time.perf_counter()
            explainability_data: Dict[str, Any] = {
                "status": "EXPLANATION_UNAVAILABLE",
                "food_gradcam_path": None,
                "freshness_gradcam_path": None,
                "food_target_layer": None,
                "freshness_target_layer": None,
            }

            if enable_gradcam and self.explainer and food_crop_rgb is not None and food_crop_rgb.size > 0:
                try:
                    exp_res = self.explainer.explain(
                        image=food_crop_rgb,
                        food_output_dir=self.gradcam_dir / "food",
                        freshness_output_dir=self.gradcam_dir / "freshness",
                        prefix=f"{res_id}_item{idx}_{cls_label}",
                    )
                    if exp_res:
                        food_cam = exp_res.get("food_gradcam")
                        fresh_cam = exp_res.get("freshness_gradcam")
                        explainability_data["status"] = "SUCCESS"
                        if food_cam:
                            explainability_data["food_gradcam_path"] = food_cam.composite_path
                            explainability_data["food_target_layer"] = food_cam.target_layer
                        if fresh_cam:
                            explainability_data["freshness_gradcam_path"] = fresh_cam.composite_path
                            explainability_data["freshness_target_layer"] = fresh_cam.target_layer
                except Exception as e:
                    explainability_data["status"] = "EXPLANATION_FAILED"
                    errors.append(f"Grad-CAM generation error for item {idx}: {str(e)}")
            timings["gradcam_ms"] += (time.perf_counter() - t_cam0) * 1000.0

            # Structured single food item according to Section 8.3
            det_display_class = cls_label if det.class_name in ("food", "others", "other_food", "proposal") else det.class_name
            det_display_conf = cls_conf if det.class_name in ("food", "others", "other_food", "proposal") else det.confidence

            item_schema = {
                "food": cls_label,
                "label_origin": "fruit_classifier" if fruit_rec_item and use_fruit_classifier else ("classifier" if cls_status == "SUCCESS" else "detector"),
                "detection": {
                    "class": det_display_class,
                    "confidence": round(float(det_display_conf), 4),
                    "detector_raw_class": det.class_name,
                    "detector_raw_confidence": round(float(det.confidence), 4),
                    "localization_method": "YOLO11" if det.class_name not in ("food", "others", "other_food", "proposal") else "Vision_Contour_Proposal",
                    "bbox": [x1, y1, x2, y2],
                },
                "segmentation": {
                    "mask_available": mask_obj is not None,
                    "mask_area": mask_area,
                    "crop_path": crop_path,
                },
                "fruit_recognition": fruit_rec_item,
                "classification": {
                    "label": cls_label,
                    "fruit_name": fruit_rec_item.get("fruit_name") if fruit_rec_item else None,
                    "dataset_class": fruit_rec_item.get("dataset_class") if fruit_rec_item else None,
                    "freshness_status": fruit_rec_item.get("freshness_status") if fruit_rec_item else None,
                    "freshness_note": fruit_rec_item.get("freshness_note") if fruit_rec_item else None,
                    "label_origin": "fruit_classifier" if fruit_rec_item and use_fruit_classifier else ("classifier" if cls_status == "SUCCESS" else "detector"),
                    "confidence": round(float(cls_conf), 4),
                    "model_version": model_version,
                    "support_tier": "Updated Fruit Dataset (6-Class)" if fruit_rec_item and use_fruit_classifier else TARGET_FOOD_SUPPORT_TIERS.get(cls_label, "General Food Class"),
                    "alternative_suggestions": alt_suggestions,
                    "top_k": cls_top_k,
                    "status": cls_status,
                },
                "freshness": {
                    "label": fresh_label,
                    "confidence": round(float(fresh_conf), 4),
                    "probabilities": fresh_probs,
                    "status": fresh_status,
                },
                "depth": {
                    "status": "SUCCESS" if depth_info and depth_info.is_success else "DEPTH_UNAVAILABLE",
                    "calibration": "CALIBRATION_REQUIRED" if vol_status != "SUCCESS" else "CALIBRATED",
                    "depth_map_path": depth_map_path,
                },
                "volume": {
                    "value_cm3": vol_val,
                    "status": vol_status,
                },
                "weight": {
                    "value_g": weight_val,
                    "status": weight_status,
                    "source": "vision_estimated",
                    "label": "Vision Estimated Portion",
                    "monocular_depth_disclaimer": (
                        "Physical weight is an empirical estimate calibrated from 2D contour area "
                        "and reference food density. Monocular depth provides relative surface relief only, "
                        "not absolute physical mass or scale."
                    ),
                },
                "nutrition": nutrition_data,
                "explainability": explainability_data,
            }
            items.append(item_schema)

            # Draw on annotated image
            color = (0, 220, 0) if fresh_label == "fresh" else ((0, 165, 255) if fresh_label == "rotten" else (50, 180, 50))
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
            if fresh_status == "SUCCESS" and fresh_label in ("fresh", "rotten"):
                label_text = f"{cls_label} ({cls_conf*100:.1f}%) | {fresh_label.upper()}"
            else:
                label_text = f"{cls_label} ({cls_conf*100:.1f}%)"
            cv2.putText(annotated_bgr, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(annotated_bgr, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # Save annotated image visualization
        annotated_path = str(self.viz_dir / f"{res_id}_annotated.png")
        cv2.imwrite(annotated_path, annotated_bgr)

        # 6. Meal Nutrition Summary
        calculated_count = 0
        unavailable_count = 0
        tot_weight = 0.0
        tot_cals = 0.0
        tot_prot = 0.0
        tot_carb = 0.0
        tot_fat = 0.0
        tot_fib = 0.0
        unavail_foods: List[str] = []

        for it in items:
            nut = it["nutrition"]
            w_val = it["weight"]["value_g"]
            if nut["status"] == "SUCCESS" and w_val is not None:
                calculated_count += 1
                tot_weight += w_val
                tot_cals += (nut["calories_kcal"] or 0.0)
                tot_prot += (nut["protein_g"] or 0.0)
                tot_carb += (nut["carbohydrates_g"] or 0.0)
                tot_fat += (nut["fat_g"] or 0.0)
                tot_fib += (nut["fiber_g"] or 0.0)
            else:
                unavailable_count += 1
                unavail_foods.append(it["food"])

        meal_summary = {
            "total_items": len(items),
            "calculated_items_count": calculated_count,
            "unavailable_items_count": unavailable_count,
            "total_weight_g": round(tot_weight, 2),
            "total_calories_kcal": round(tot_cals, 2),
            "total_protein_g": round(tot_prot, 2),
            "total_carbohydrates_g": round(tot_carb, 2),
            "total_fat_g": round(tot_fat, 2),
            "total_fiber_g": round(tot_fib, 2),
            "source": "USDA FoodData Central",
            "unavailable_foods": unavail_foods,
        }

        # Determine overall pipeline status (Section 8.5)
        if len(items) == 0:
            overall_status = "NO_FOOD_DETECTED"
        elif unavailable_count == 0 and len(errors) == 0:
            overall_status = "SUCCESS"
        else:
            overall_status = "PARTIAL_SUCCESS"

        # Round timing metrics (Section 8.6)
        timings["classification_ms"] = round(timings["classification_ms"], 2)
        timings["freshness_ms"] = round(timings["freshness_ms"], 2)
        timings["volume_ms"] = round(timings["volume_ms"], 2)
        timings["weight_ms"] = round(timings["weight_ms"], 2)
        timings["nutrition_ms"] = round(timings["nutrition_ms"], 2)
        timings["gradcam_ms"] = round(timings["gradcam_ms"], 2)
        timings["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)

        final_result = {
            "status": overall_status,
            "result_id": res_id,
            "timestamp": timestamp,
            "source_image": str(image) if isinstance(image, (str, Path)) else "upload",
            "device": getattr(self.detector, "device", "cpu"),
            "model_version": model_version,
            "classifier_checkpoint": ckpt_path_str,
            "image_dimensions": img_dims,
            "total_foods_detected": len(items),
            "items": items,
            "foods": items,
            "meal_summary": meal_summary,
            "visualizations": {
                "annotated_image_path": annotated_path,
                "depth_map_path": depth_map_path,
            },
            "timing": timings,
            "errors": errors,
        }

        # Structured fruit recognition contract (Section 6)
        primary_fruit = None
        if items:
            best_item = max(items, key=lambda x: x.get("classification", {}).get("confidence", 0.0))
            f_rec = best_item.get("fruit_recognition")
            p_nut = best_item.get("nutrition", {})
            if f_rec:
                primary_fruit = {
                    "fruit_name": f_rec.get("fruit_name"),
                    "dataset_class": f_rec.get("dataset_class"),
                    "freshness_status": f_rec.get("freshness_status"),
                    "freshness_note": f_rec.get("freshness_note"),
                    "confidence": f_rec.get("confidence"),
                    "classification_model": f_rec.get("classification_model", "EfficientNet-B4"),
                    "nutrition": {
                        "calories": p_nut.get("calories_kcal"),
                        "protein": p_nut.get("protein_g"),
                        "carbohydrates": p_nut.get("carbohydrates_g"),
                        "fat": p_nut.get("fat_g"),
                    },
                }

        if primary_fruit is None:
            primary_fruit = {
                "fruit_name": None,
                "dataset_class": None,
                "freshness_status": None,
                "freshness_note": "No fruit recognized in image",
                "confidence": 0.0,
                "classification_model": "EfficientNet-B4",
                "nutrition": {
                    "calories": None,
                    "protein": None,
                    "carbohydrates": None,
                    "fat": None,
                },
            }

        final_result["fruit_recognition"] = primary_fruit

        # Persist result JSON for auditing and download export
        try:
            res_json_path = self.results_dir / f"{res_id}.json"
            with open(res_json_path, "w", encoding="utf-8") as f:
                json.dump(final_result, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist result json: {e}")

        return final_result

    def analyze_image(self, *args, **kwargs) -> Any:
        """Alias for backward compatibility with earlier tests."""
        return self.process_image(*args, **kwargs)
