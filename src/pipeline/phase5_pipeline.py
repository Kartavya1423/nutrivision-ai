"""
NutriVision AI — Phase 5 End-to-End Pipeline
Coordinates:
    Image -> YOLO11 -> SAM2 -> Depth Anything V2 -> Food Masked Depth -> Volume -> Density -> Weight

Strict Project Rules:
- Monocular depth is relative; metric scale requires calibration.
- If uncalibrated, reports DEPTH_SCALE_NOT_CALIBRATED and VOLUME_SCALE_NOT_CALIBRATED.
- If food is unlisted in density database, reports DENSITY_NOT_AVAILABLE.
- If any component is uncalibrated/unavailable, total weight is None and status is WEIGHT_ESTIMATION_UNAVAILABLE.
- Produces 7-panel visualization under outputs/phase5/visualizations/.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.config.settings import Settings, get_settings
from src.depth.depth_anything import DepthAnythingV2Estimator
from src.depth.depth_estimator import BaseDepthEstimator, DepthResult, DepthType
from src.detection.detector import BaseFoodDetector, ModelStatus
from src.detection.yolo_detector import YOLO11FoodDetector
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.segmentation.segmenter import BaseFoodSegmenter, FoodMask
from src.utils.image_ops import load_image
from src.volume.volume_estimator import VolumeEstimateOutput, VolumeEstimator
from src.weight.density_database import DensityDatabase
from src.weight.weight_estimator import ItemWeightEstimate, WeightEstimator


@dataclass
class Phase5ItemResult:
    """Detailed Phase 5 pipeline output for a single food item."""
    item_index: int
    class_name: str
    confidence: float
    bbox: List[float]
    has_mask: bool
    mask_pixel_count: int
    depth_stats: Optional[Dict[str, Any]]
    volume_cm3: Optional[float]
    relative_volume: Optional[float]
    volume_status: str
    density_g_cm3: Optional[float]
    density_source: Optional[str]
    density_status: str
    estimated_weight_g: Optional[float]
    weight_status: str
    calibration_status: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_index": self.item_index,
            "food": self.class_name,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(c), 2) for c in self.bbox],
            "has_mask": self.has_mask,
            "mask_pixel_count": self.mask_pixel_count,
            "depth_stats": self.depth_stats,
            "volume_cm3": round(float(self.volume_cm3), 2) if self.volume_cm3 is not None else None,
            "relative_volume": round(float(self.relative_volume), 2) if self.relative_volume is not None else None,
            "volume_status": self.volume_status,
            "density_g_cm3": round(float(self.density_g_cm3), 4) if self.density_g_cm3 is not None else None,
            "density_source": self.density_source,
            "density_status": self.density_status,
            "estimated_weight_g": round(float(self.estimated_weight_g), 2) if self.estimated_weight_g is not None else None,
            "weight_status": self.weight_status,
            "calibration_status": self.calibration_status,
            "error_message": self.error_message,
        }


@dataclass
class Phase5PipelineResult:
    """Top-level Phase 5 structured pipeline envelope."""
    status: str
    source_image: str
    image_width: int
    image_height: int
    inference_time_ms: float
    depth_model: str
    depth_checkpoint: Optional[str]
    device: str
    calibration_status: str
    total_detected_foods: int
    items: List[Phase5ItemResult]
    total_estimated_weight_g: Optional[float] = None
    total_volume_cm3: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    visualization_files: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "source_image": self.source_image,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_time_ms": round(float(self.inference_time_ms), 2),
            "depth_model": self.depth_model,
            "depth_checkpoint": self.depth_checkpoint,
            "device": self.device,
            "calibration_status": self.calibration_status,
            "total_detected_foods": self.total_detected_foods,
            "total_estimated_weight_g": round(float(self.total_estimated_weight_g), 2) if self.total_estimated_weight_g is not None else None,
            "total_volume_cm3": round(float(self.total_volume_cm3), 2) if self.total_volume_cm3 is not None else None,
            "items": [it.to_dict() for it in self.items],
            "warnings": self.warnings,
            "visualization_files": self.visualization_files,
            "error_message": self.error_message,
        }


class Phase5Pipeline:
    """
    Dedicated Phase 5 Food Depth -> Volume -> Weight Estimation Pipeline.
    Supports single-food and multi-food images with real inference and strict calibration enforcement.
    """

    def __init__(
        self,
        detector: Optional[BaseFoodDetector] = None,
        segmenter: Optional[BaseFoodSegmenter] = None,
        depth_estimator: Optional[BaseDepthEstimator] = None,
        volume_estimator: Optional[VolumeEstimator] = None,
        weight_estimator: Optional[WeightEstimator] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()

        # Modular components
        if detector is not None:
            self.detector = detector
        else:
            cand_p2 = [
                self.settings.root_dir / "models" / "detection" / "yolo11" / "phase2_final_run" / "weights" / "best.pt",
                self.settings.root_dir / "models" / "detection" / "yolo11" / "phase2_fixed_run" / "weights" / "best.pt",
                self.settings.root_dir / "models" / "detection" / "yolo11" / "best.pt",
            ]
            det_p = next((p for p in cand_p2 if p.is_file()), self.settings.model_paths.detection_model_path)
            self.detector = YOLO11FoodDetector(model_path=det_p, conf_threshold=0.001, settings=self.settings)
        self.segmenter = segmenter or SAM2FoodSegmenter(settings=self.settings)
        self.depth_estimator = depth_estimator or DepthAnythingV2Estimator(settings=self.settings)
        self.volume_estimator = volume_estimator or VolumeEstimator()
        self.weight_estimator = weight_estimator or WeightEstimator(
            density_db=DensityDatabase(),
            settings=self.settings,
        )

    def run(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        calibration: Optional[Dict[str, Any]] = None,
        save_visualizations: bool = True,
        output_dir: str = "outputs/phase5",
    ) -> Phase5PipelineResult:
        """
        Execute full Phase 5 food weight estimation workflow:
            1. Validate input image
            2. Run YOLO11 detection
            3. Run SAM2 segmentation
            4. Run Depth Anything V2 monocular depth
            5. Mask depth per food item
            6. Estimate relative or calibrated 3D volume
            7. Query food density database
            8. Estimate physical weight (g)
            9. Generate 7-panel visualization
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        # 1. Validate and load image
        try:
            np_rgb, pil_img = load_image(image)
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return Phase5PipelineResult(
                status="INVALID_IMAGE",
                source_image=str(image) if isinstance(image, (str, Path)) else "in_memory_image",
                image_width=0,
                image_height=0,
                inference_time_ms=elapsed,
                depth_model="DepthAnythingV2",
                depth_checkpoint=None,
                device="cpu",
                calibration_status="DEPTH_SCALE_NOT_CALIBRATED",
                total_detected_foods=0,
                items=[],
                error_message=f"Failed to load image: {str(e)}",
            )

        orig_h, orig_w = np_rgb.shape[:2]
        img_name = Path(image).stem if isinstance(image, (str, Path)) else f"image_{int(time.time())}"

        # 2. YOLO11 Food Detection
        det_result = self.detector.predict(np_rgb)
        detections = getattr(det_result, "detections", [])

        if len(detections) == 0:
            elapsed = (time.perf_counter() - t0) * 1000.0
            # Run depth map for diagnostics even if no food detected
            depth_res = self.depth_estimator.estimate_depth(np_rgb) if self.depth_estimator.is_ready() else None
            return Phase5PipelineResult(
                status="NO_FOOD_DETECTED",
                source_image=str(image) if isinstance(image, (str, Path)) else img_name,
                image_width=orig_w,
                image_height=orig_h,
                inference_time_ms=elapsed,
                depth_model="Depth Anything V2",
                depth_checkpoint=str(getattr(self.depth_estimator, "model_path", None)),
                device=getattr(self.depth_estimator, "device", "cpu"),
                calibration_status="DEPTH_SCALE_NOT_CALIBRATED",
                total_detected_foods=0,
                items=[],
                warnings=["No food objects detected by YOLO11 in the input image."],
            )

        # 3. SAM2 Instance Segmentation
        bboxes = [d.bbox for d in detections]
        class_names = [d.class_name for d in detections]
        seg_result = self.segmenter.predict_masks(np_rgb, bboxes=bboxes, class_names=class_names)
        masks: List[FoodMask] = getattr(seg_result, "masks", [])

        if len(masks) == 0:
            warnings.append("SAM2 failed to segment food masks from detected bounding boxes.")

        # 4. Depth Anything V2 Monocular Depth Estimation
        if self.depth_estimator is None or not self.depth_estimator.is_ready():
            elapsed = (time.perf_counter() - t0) * 1000.0
            return Phase5PipelineResult(
                status="DEPTH_MODEL_NOT_AVAILABLE",
                source_image=str(image) if isinstance(image, (str, Path)) else img_name,
                image_width=orig_w,
                image_height=orig_h,
                inference_time_ms=elapsed,
                depth_model="Depth Anything V2",
                depth_checkpoint=str(getattr(self.depth_estimator, "model_path", None)),
                device="cpu",
                calibration_status="DEPTH_SCALE_NOT_CALIBRATED",
                total_detected_foods=len(detections),
                items=[],
                error_message="Depth Anything V2 model checkpoint is not loaded.",
            )

        depth_result: DepthResult = self.depth_estimator.estimate_depth(np_rgb, calibration=calibration)
        if not depth_result.is_success or depth_result.depth_map is None:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return Phase5PipelineResult(
                status="DEPTH_INFERENCE_FAILED",
                source_image=str(image) if isinstance(image, (str, Path)) else img_name,
                image_width=orig_w,
                image_height=orig_h,
                inference_time_ms=elapsed,
                depth_model="Depth Anything V2",
                depth_checkpoint=str(getattr(self.depth_estimator, "model_path", None)),
                device=getattr(self.depth_estimator, "device", "cpu"),
                calibration_status="DEPTH_SCALE_NOT_CALIBRATED",
                total_detected_foods=len(detections),
                items=[],
                error_message=f"Depth inference failed: {depth_result.error_message}",
            )

        raw_depth_map = depth_result.depth_map
        depth_is_metric = depth_result.is_metric_calibrated
        global_calib_status = "CALIBRATED" if depth_is_metric else "DEPTH_SCALE_NOT_CALIBRATED"

        if not depth_is_metric and calibration is None:
            warnings.append("DEPTH_SCALE_NOT_CALIBRATED: Monocular depth is relative; physical volume and weight require metric calibration.")

        # 5. Food Mask Alignment, Volume Estimation, Density Lookup & Weight Estimation
        item_results: List[Phase5ItemResult] = []
        combined_food_mask = np.zeros((orig_h, orig_w), dtype=bool)

        for idx, det in enumerate(detections):
            # Associate detection with corresponding mask
            matching_mask = None
            if idx < len(masks):
                matching_mask = masks[idx]
            elif masks:
                matching_mask = masks[0]

            if matching_mask is not None and np.sum(matching_mask.mask) > 0:
                mask_arr = matching_mask.mask
                has_m = True
                combined_food_mask |= mask_arr
            else:
                # If SAM2 mask is missing, create bbox mask
                has_m = False
                mask_arr = np.zeros((orig_h, orig_w), dtype=bool)
                x1, y1, x2, y2 = [int(round(c)) for c in det.bbox]
                mask_arr[max(0, y1):min(orig_h, y2), max(0, x1):min(orig_w, x2)] = True
                warnings.append(f"SEGMENTATION_NOT_AVAILABLE: Using bbox fallback for item {idx} ({det.class_name}).")

            # 6. Volume Estimation
            vol_out: VolumeEstimateOutput = self.volume_estimator.estimate_volume(
                mask=mask_arr,
                depth_info=raw_depth_map,
                camera_calibration=calibration,
            )

            # Determine explicit volume status
            vol_status = "READY" if vol_out.status == "READY" else "VOLUME_SCALE_NOT_CALIBRATED"

            # 7. Density Lookup & Weight Estimation
            weight_out: ItemWeightEstimate = self.weight_estimator.estimate_weight(
                class_name=det.class_name,
                volume_output=vol_out,
            )

            item_calib_status = "CALIBRATED" if vol_status == "READY" else "VOLUME_SCALE_NOT_CALIBRATED"

            item_results.append(
                Phase5ItemResult(
                    item_index=idx,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    has_mask=has_m,
                    mask_pixel_count=vol_out.mask_pixel_count,
                    depth_stats=vol_out.depth_statistics,
                    volume_cm3=vol_out.volume,
                    relative_volume=vol_out.relative_volume,
                    volume_status=vol_status,
                    density_g_cm3=weight_out.density_g_cm3,
                    density_source=weight_out.density_source,
                    density_status="AVAILABLE" if weight_out.density_g_cm3 else "DENSITY_NOT_AVAILABLE",
                    estimated_weight_g=weight_out.estimated_weight_grams,
                    weight_status=weight_out.status,
                    calibration_status=item_calib_status,
                    error_message=weight_out.error_message,
                )
            )

        # 8. Determine Overall Pipeline Status & Multi-Food Totals
        all_calibrated = len(item_results) > 0 and all(it.weight_status == "READY" for it in item_results)
        if all_calibrated:
            total_wt = sum(it.estimated_weight_g for it in item_results if it.estimated_weight_g is not None)
            total_vol = sum(it.volume_cm3 for it in item_results if it.volume_cm3 is not None)
            overall_status = "COMPLETE"
            overall_calib = "CALIBRATED"
        else:
            total_wt = None
            total_vol = None
            overall_calib = "VOLUME_SCALE_NOT_CALIBRATED"
            missing_types = set()
            for it in item_results:
                if it.volume_status != "READY":
                    missing_types.add(it.volume_status)
                if it.density_status != "AVAILABLE":
                    missing_types.add(it.density_status)
                if it.weight_status != "READY":
                    missing_types.add(it.weight_status)
            overall_status = "VOLUME_SCALE_NOT_CALIBRATED" if "VOLUME_SCALE_NOT_CALIBRATED" in missing_types else "WEIGHT_ESTIMATION_UNAVAILABLE"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 9. Generate 7-Panel Visualizations
        vis_paths: Dict[str, str] = {}
        if save_visualizations:
            vis_paths = self._generate_visualizations(
                np_rgb=np_rgb,
                detections=detections,
                masks=masks,
                depth_result=depth_result,
                item_results=item_results,
                img_name=img_name,
                output_dir=output_dir,
            )

        return Phase5PipelineResult(
            status=overall_status,
            source_image=str(image) if isinstance(image, (str, Path)) else img_name,
            image_width=orig_w,
            image_height=orig_h,
            inference_time_ms=elapsed_ms,
            depth_model="Depth Anything V2",
            depth_checkpoint=str(getattr(self.depth_estimator, "model_path", None)),
            device=getattr(self.depth_estimator, "device", "cpu"),
            calibration_status=overall_calib,
            total_detected_foods=len(detections),
            items=item_results,
            total_estimated_weight_g=total_wt,
            total_volume_cm3=total_vol,
            warnings=warnings,
            visualization_files=vis_paths,
        )

    def _generate_visualizations(
        self,
        np_rgb: np.ndarray,
        detections: List[Any],
        masks: List[FoodMask],
        depth_result: DepthResult,
        item_results: List[Phase5ItemResult],
        img_name: str,
        output_dir: str,
    ) -> Dict[str, str]:
        """
        Generate and save 7 distinct visualization panels:
            1. Original RGB image
            2. YOLO bounding boxes
            3. SAM2 food mask overlay
            4. Depth map (colormap)
            5. Masked depth map (food only)
            6. Food region used for volume calculation
            7. Final composite analysis visualization
        """
        out_base = Path(output_dir) / "visualizations"
        out_base.mkdir(parents=True, exist_ok=True)
        h, w = np_rgb.shape[:2]

        # Panel 1: Original Image
        p1_orig = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

        # Panel 2: YOLO Bounding Boxes
        p2_det = p1_orig.copy()
        colors = [(0, 255, 128), (255, 128, 0), (0, 128, 255), (255, 0, 128), (128, 255, 0)]
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(round(c)) for c in det.bbox]
            col = colors[i % len(colors)]
            cv2.rectangle(p2_det, (x1, y1), (x2, y2), col, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(p2_det, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        # Panel 3: SAM2 Segmentation Masks
        p3_seg = p1_orig.copy()
        combined_mask = np.zeros((h, w), dtype=bool)
        for i, m in enumerate(masks):
            if hasattr(m, "mask"):
                m_arr = m.mask
                combined_mask |= m_arr
                col = colors[i % len(colors)]
                overlay = p3_seg.copy()
                overlay[m_arr] = col
                cv2.addWeighted(overlay, 0.45, p3_seg, 0.55, 0, p3_seg)
                # Contours
                contours, _ = cv2.findContours(m_arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(p3_seg, contours, -1, col, 2)

        # Panel 4: Depth Map (Colormap)
        depth_rgb = depth_result.render_colormap()
        p4_depth = cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR) if depth_rgb is not None else np.zeros_like(p1_orig)

        # Panel 5: Masked Depth Map (Depth isolated to food region only)
        p5_masked_depth = p4_depth.copy()
        if np.any(combined_mask):
            p5_masked_depth[~combined_mask] = 0
        else:
            # Fallback to bbox if masks empty
            bbox_mask = np.zeros((h, w), dtype=bool)
            for det in detections:
                x1, y1, x2, y2 = [int(round(c)) for c in det.bbox]
                bbox_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
            p5_masked_depth[~bbox_mask] = 0

        # Panel 6: Food region used for volume calculation (Height gradient overlay)
        p6_volume_region = p1_orig.copy()
        if np.any(combined_mask) and depth_result.depth_map is not None:
            depth_arr = depth_result.depth_map
            food_depths = depth_arr[combined_mask]
            if len(food_depths) > 0:
                base_depth = float(np.percentile(food_depths, 95))
                height_norm = np.clip((base_depth - depth_arr) / max(0.1, (base_depth - float(np.min(food_depths)))), 0, 1)
                height_color = cv2.applyColorMap((height_norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
                overlay = p6_volume_region.copy()
                overlay[combined_mask] = height_color[combined_mask]
                cv2.addWeighted(overlay, 0.65, p6_volume_region, 0.35, 0, p6_volume_region)

        # Panel 7: Final Result Visualization (Composite with Metrics Table)
        # Create 2x4 grid or 7 panels side-by-side
        thumb_w, thumb_h = 320, 240
        def resize_thumb(img, title):
            res = cv2.resize(img, (thumb_w, thumb_h))
            cv2.rectangle(res, (0, 0), (thumb_w, 24), (20, 20, 20), -1)
            cv2.putText(res, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            return res

        t1 = resize_thumb(p1_orig, "1. Input RGB Image")
        t2 = resize_thumb(p2_det, f"2. YOLO11 ({len(detections)} items)")
        t3 = resize_thumb(p3_seg, f"3. SAM2 Masks ({len(masks)} masks)")
        t4 = resize_thumb(p4_depth, "4. Depth Anything V2")
        t5 = resize_thumb(p5_masked_depth, "5. Masked Food Depth")
        t6 = resize_thumb(p6_volume_region, "6. Voxel Volume Region")

        # Metric summary panel (card)
        t7_card = np.full((thumb_h, thumb_w, 3), 30, dtype=np.uint8)
        cv2.rectangle(t7_card, (0, 0), (thumb_w, 24), (45, 45, 45), -1)
        cv2.putText(t7_card, "7. Pipeline Summary", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
        y_pos = 45
        for it in item_results[:4]:
            txt1 = f"{it.class_name}: Vol={it.volume_cm3 or 'UNCL'}cm3"
            txt2 = f"  Dens={it.density_g_cm3 or 'N/A'}g/cm3 | Wt={it.estimated_weight_g or 'UNCL'}g"
            cv2.putText(t7_card, txt1, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            cv2.putText(t7_card, txt2, (10, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)
            y_pos += 35
        calib_txt = f"Calib: {item_results[0].calibration_status if item_results else 'N/A'}"
        cv2.putText(t7_card, calib_txt, (10, thumb_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)

        # Build 2-row grid (Row 1: 4 panels, Row 2: 3 panels + info)
        blank = np.full((thumb_h, thumb_w, 3), 20, dtype=np.uint8)
        cv2.putText(blank, "NutriVision AI Phase 5", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        row1 = np.hstack([t1, t2, t3, t4])
        row2 = np.hstack([t5, t6, t7_card, blank])
        composite = np.vstack([row1, row2])

        # Save files
        composite_path = out_base / f"{img_name}_phase5_7panel.png"
        cv2.imwrite(str(composite_path), composite)

        p1_path = out_base / f"{img_name}_1_original.png"
        p2_path = out_base / f"{img_name}_2_yolo_boxes.png"
        p3_path = out_base / f"{img_name}_3_sam2_masks.png"
        p4_path = out_base / f"{img_name}_4_depth_map.png"
        p5_path = out_base / f"{img_name}_5_masked_depth.png"
        p6_path = out_base / f"{img_name}_6_volume_region.png"
        p7_path = out_base / f"{img_name}_7_summary_card.png"

        cv2.imwrite(str(p1_path), p1_orig)
        cv2.imwrite(str(p2_path), p2_det)
        cv2.imwrite(str(p3_path), p3_seg)
        cv2.imwrite(str(p4_path), p4_depth)
        cv2.imwrite(str(p5_path), p5_masked_depth)
        cv2.imwrite(str(p6_path), p6_volume_region)
        cv2.imwrite(str(p7_path), t7_card)

        return {
            "composite_7panel": str(composite_path.resolve()),
            "panel_1_original": str(p1_path.resolve()),
            "panel_2_yolo_boxes": str(p2_path.resolve()),
            "panel_3_sam2_masks": str(p3_path.resolve()),
            "panel_4_depth_map": str(p4_path.resolve()),
            "panel_5_masked_depth": str(p5_path.resolve()),
            "panel_6_volume_region": str(p6_path.resolve()),
            "panel_7_summary_card": str(p7_path.resolve()),
        }
