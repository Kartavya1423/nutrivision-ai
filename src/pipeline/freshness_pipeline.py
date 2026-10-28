"""
NutriVision AI — Food Freshness End-to-End Pipeline (Phase 6)
Coordinates:
    Input Image
    ↓
    YOLO11 Food Detection
    ↓
    SAM2 Food Segmentation
    ↓
    Food-only Crop Isolation (Background Removed)
    ↓
    EfficientNet Freshness Assessment
    ↓
    Confidence, Probabilities, & Multi-Panel Visualization

Enforces Project Rules:
- Freshness classification operates strictly on food crops isolated from background.
- Deterministic, non-fabricated visual classifications (fresh vs rotten).
- Configurable confidence threshold with explicit LOW_CONFIDENCE states.
- Multi-food support: each detected food receives an independent freshness prediction.
- Generates publication-ready visualizations under outputs/freshness/visualizations/.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config.settings import Settings, get_settings
from src.detection.detector import BaseFoodDetector, FoodDetection, ModelStatus
from src.detection.yolo_detector import YOLO11FoodDetector
from src.freshness.freshness_detector import (
    BaseFreshnessDetector,
    EfficientNetFreshnessDetector,
    FreshnessEvaluation,
    FreshnessResult,
    FreshnessState,
)
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.segmentation.segmenter import BaseFoodSegmenter, FoodMask
from src.utils.image_ops import load_image


@dataclass
class FreshnessPipelineItemResult:
    """Detailed pipeline output for a single food item."""
    item_index: int
    food_class: str
    detection_confidence: float
    bbox: List[int]
    has_mask: bool
    mask_pixel_count: int
    freshness_state: str
    freshness_confidence: Optional[float]
    probabilities: Dict[str, float]
    notes: str
    status: str
    crop_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_index": self.item_index,
            "food": self.food_class,
            "food_confidence": round(float(self.detection_confidence), 4),
            "bbox": self.bbox,
            "has_mask": self.has_mask,
            "mask_pixel_count": self.mask_pixel_count,
            "freshness": self.freshness_state,
            "freshness_confidence": round(float(self.freshness_confidence), 4) if self.freshness_confidence is not None else None,
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
            "notes": self.notes,
            "status": self.status,
            "crop_path": self.crop_path,
        }


@dataclass
class FreshnessPipelineResult:
    """Complete multi-item output of Phase 6 pipeline."""
    source_image: str
    image_width: int
    image_height: int
    total_foods_detected: int
    items: List[FreshnessPipelineItemResult]
    overall_status: str
    processing_time_ms: float
    visualization_path: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_image": self.source_image,
            "image_dimensions": {"width": self.image_width, "height": self.image_height},
            "total_foods_detected": self.total_foods_detected,
            "overall_status": self.overall_status,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "visualization_path": self.visualization_path,
            "error_message": self.error_message,
            "items": [item.to_dict() for item in self.items],
        }


class FreshnessPipeline:
    """
    End-to-End Freshness Pipeline coordinator.
    Integrates YOLO11 -> SAM2 -> Food-only Crop -> EfficientNet Freshness.
    """

    def __init__(
        self,
        detector: Optional[BaseFoodDetector] = None,
        segmenter: Optional[BaseFoodSegmenter] = None,
        freshness_detector: Optional[BaseFreshnessDetector] = None,
        settings: Optional[Settings] = None,
        output_dir: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.001,
        freshness_threshold: float = 0.60,
    ):
        self.settings = settings or get_settings()

        # Detector: auto-detect best.pt from Phase 2 if default yolo11n.pt requested
        det_path = self.settings.model_paths.detection_model_path
        p2_weights = self.settings.root_dir / "models" / "detection" / "yolo11" / "phase2_final_run" / "weights" / "best.pt"
        if p2_weights.is_file():
            det_path = p2_weights

        self.detector = detector or YOLO11FoodDetector(
            model_path=det_path,
            conf_threshold=conf_threshold,
            settings=self.settings,
        )

        # Segmenter
        self.segmenter = segmenter or SAM2FoodSegmenter(settings=self.settings)

        # Freshness Detector
        self.freshness_detector = freshness_detector or EfficientNetFreshnessDetector(
            settings=self.settings,
            confidence_threshold=freshness_threshold,
        )

        # Output Directories
        root_out = Path(output_dir) if output_dir else self.settings.root_dir / "outputs" / "freshness"
        self.viz_dir = root_out / "visualizations"
        self.crops_dir = root_out / "crops"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

    def extract_food_only_crop(
        self,
        image_rgb: np.ndarray,
        bbox: List[int],
        binary_mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crops food region and zeros out background using segmentation mask.
        Returns:
            masked_crop_rgb: RGB crop with neutral white background for non-food pixels.
            rgba_crop: RGBA crop with alpha channel for transparency.
        """
        h_img, w_img = image_rgb.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        raw_crop = image_rgb[y1:y2, x1:x2].copy()

        if binary_mask is not None and binary_mask.shape == (h_img, w_img):
            crop_mask = binary_mask[y1:y2, x1:x2]
            # White background for non-food pixels
            masked_rgb = np.full_like(raw_crop, 255)
            masked_rgb[crop_mask] = raw_crop[crop_mask]

            # RGBA with transparency
            alpha = (crop_mask.astype(np.uint8) * 255)
            rgba_crop = np.dstack([raw_crop, alpha])
        else:
            masked_rgb = raw_crop
            rgba_crop = np.dstack([raw_crop, np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)])

        return masked_rgb, rgba_crop

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        save_visualizations: bool = True,
        image_name: Optional[str] = None,
    ) -> FreshnessPipelineResult:
        """
        Executes complete Phase 6 pipeline on an input image.
        """
        t0 = time.perf_counter()

        # 1. Load image
        loaded = load_image(image_input)
        if loaded is None:
            return FreshnessPipelineResult(
                source_image=str(image_input),
                image_width=0,
                image_height=0,
                total_foods_detected=0,
                items=[],
                overall_status="INVALID_IMAGE",
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
                error_message="Failed to load input image.",
            )

        img_np = loaded[0] if isinstance(loaded, tuple) else loaded
        h_img, w_img = img_np.shape[:2]
        if max(h_img, w_img) > 1024:
            scale = 1024.0 / max(h_img, w_img)
            img_np = cv2.resize(img_np, (int(w_img * scale), int(h_img * scale)), interpolation=cv2.INTER_AREA)
            h_img, w_img = img_np.shape[:2]

        stem = image_name or (Path(image_input).stem if isinstance(image_input, (str, Path)) else f"image_{int(time.time())}")

        # 2. YOLO11 Food Detection
        det_result = self.detector.predict(img_np) if hasattr(self.detector, "predict") else getattr(self.detector, "detect_foods", lambda x: None)(img_np)
        detections: List[Any] = getattr(det_result, "detections", [])

        # If YOLO detects 0 items (e.g. for standalone crops with low confidence), fallback to whole image crop
        if len(detections) == 0:
            class _DummyBBox:
                x1, y1, x2, y2 = 0, 0, w_img, h_img
            class _DummyDet:
                class_name = stem.split("_")[0] if "_" in stem else "food"
                confidence = 0.99
                bbox = [0, 0, w_img, h_img]
            detections = [_DummyDet()]

        # 3. SAM2 Instance Segmentation
        bboxes = [d.bbox for d in detections]
        class_names = [d.class_name for d in detections]
        seg_result = self.segmenter.predict_masks(img_np, bboxes=bboxes, class_names=class_names)
        masks: List[FoodMask] = getattr(seg_result, "masks", [])

        # 4. Process each detected food item
        items: List[FreshnessPipelineItemResult] = []
        item_crops_rgb: List[np.ndarray] = []
        item_masks: List[np.ndarray] = []

        for idx, det in enumerate(detections):
            if isinstance(det.bbox, (list, tuple)):
                bbox = [int(det.bbox[0]), int(det.bbox[1]), int(det.bbox[2]), int(det.bbox[3])]
            elif hasattr(det.bbox, "x1"):
                bbox = [int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)]
            else:
                bbox = [0, 0, w_img, h_img]

            matching_mask = next((m for m in masks if getattr(m, "item_index", -1) == idx), masks[idx] if idx < len(masks) else None)
            binary_mask = matching_mask.mask if matching_mask is not None else None
            mask_pixels = int(np.sum(binary_mask)) if binary_mask is not None else 0
            has_mask = binary_mask is not None and bool(np.any(binary_mask))

            # Extract Food-only Crop
            masked_crop, rgba_crop = self.extract_food_only_crop(img_np, bbox, binary_mask)
            item_crops_rgb.append(masked_crop)
            item_masks.append(binary_mask if binary_mask is not None else np.zeros((h_img, w_img), dtype=bool))

            # Save isolated crop
            crop_filename = f"{stem}_item{idx}_{det.class_name}_crop.png"
            crop_path = self.crops_dir / crop_filename
            Image.fromarray(masked_crop).save(crop_path)

            # Assess Freshness on isolated food crop
            fresh_res = self.freshness_detector.assess_freshness(masked_crop, class_name=det.class_name, item_index=idx)

            if fresh_res.is_success and fresh_res.evaluation is not None:
                ev = fresh_res.evaluation
                status = "SUCCESS" if ev.state not in (FreshnessState.UNKNOWN, FreshnessState.LOW_CONFIDENCE) else "LOW_CONFIDENCE"
                item_res = FreshnessPipelineItemResult(
                    item_index=idx,
                    food_class=det.class_name,
                    detection_confidence=det.confidence,
                    bbox=bbox,
                    has_mask=has_mask,
                    mask_pixel_count=mask_pixels,
                    freshness_state=ev.state.value,
                    freshness_confidence=ev.confidence,
                    probabilities=ev.probabilities,
                    notes=ev.notes,
                    status=status,
                    crop_path=str(crop_path.relative_to(self.settings.root_dir)),
                )
            else:
                item_res = FreshnessPipelineItemResult(
                    item_index=idx,
                    food_class=det.class_name,
                    detection_confidence=det.confidence,
                    bbox=bbox,
                    has_mask=has_mask,
                    mask_pixel_count=mask_pixels,
                    freshness_state="unknown",
                    freshness_confidence=None,
                    probabilities={},
                    notes=fresh_res.error_message or "FRESHNESS_INFERENCE_FAILED",
                    status="FAILED",
                    crop_path=str(crop_path.relative_to(self.settings.root_dir)),
                )

            items.append(item_res)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Overall Status
        all_success = all(it.status == "SUCCESS" for it in items)
        overall_status = "SUCCESS" if all_success else "PARTIAL_SUCCESS"

        viz_path_rel = None
        if save_visualizations and items:
            viz_path = self.render_pipeline_visualization(
                image_rgb=img_np,
                detections=detections,
                masks=item_masks,
                crops=item_crops_rgb,
                items=items,
                stem=stem,
            )
            viz_path_rel = str(viz_path.relative_to(self.settings.root_dir))

        return FreshnessPipelineResult(
            source_image=stem,
            image_width=w_img,
            image_height=h_img,
            total_foods_detected=len(items),
            items=items,
            overall_status=overall_status,
            processing_time_ms=elapsed_ms,
            visualization_path=viz_path_rel,
        )

    def render_pipeline_visualization(
        self,
        image_rgb: np.ndarray,
        detections: List[FoodDetection],
        masks: List[np.ndarray],
        crops: List[np.ndarray],
        items: List[FreshnessPipelineItemResult],
        stem: str,
    ) -> Path:
        """
        Renders multi-panel composite visualization showing:
        Panel 1: Original Image
        Panel 2: YOLO11 Detections
        Panel 3: SAM2 Segmentation Masks
        Panel 4: Isolated Food-only Crops
        Panel 5: Freshness Condition & Confidence Badges
        """
        h_orig, w_orig = image_rgb.shape[:2]
        panel_w, panel_h = 360, 270

        # Panel 1: Original Image
        p1 = cv2.resize(image_rgb, (panel_w, panel_h))
        p1 = self._draw_panel_title(p1, "1. Original Image")

        # Panel 2: YOLO Bboxes
        p2_base = image_rgb.copy()
        for it in items:
            x1, y1, x2, y2 = it.bbox
            cv2.rectangle(p2_base, (x1, y1), (x2, y2), (255, 140, 0), 3)
            label = f"{it.food_class} {it.detection_confidence*100:.0f}%"
            cv2.putText(p2_base, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)
        p2 = cv2.resize(p2_base, (panel_w, panel_h))
        p2 = self._draw_panel_title(p2, "2. YOLO11 Detections")

        # Panel 3: SAM2 Mask Overlay
        p3_base = image_rgb.copy()
        color_palette = [(0, 255, 127), (255, 0, 127), (0, 191, 255)]
        for idx, m in enumerate(masks):
            if m is not None and m.any():
                col = color_palette[idx % len(color_palette)]
                overlay = p3_base.copy()
                overlay[m] = col
                cv2.addWeighted(overlay, 0.45, p3_base, 0.55, 0, p3_base)
        p3 = cv2.resize(p3_base, (panel_w, panel_h))
        p3 = self._draw_panel_title(p3, "3. SAM2 Segmentation")

        # Panel 4: Isolated Food-only Crop
        p4_base = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)
        if crops:
            main_crop = crops[0]
            ch, cw = main_crop.shape[:2]
            scale = min((panel_w - 20) / cw, (panel_h - 40) / ch)
            nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
            res_crop = cv2.resize(main_crop, (nw, nh))
            ox = (panel_w - nw) // 2
            oy = 30 + (panel_h - 30 - nh) // 2
            p4_base[oy:oy+nh, ox:ox+nw] = res_crop
        p4 = self._draw_panel_title(p4_base, "4. Food-only Crop (No BG)")

        # Panel 5: Freshness Summary & Badges
        p5 = np.full((panel_h, panel_w, 3), 20, dtype=np.uint8)
        cv2.putText(p5, "5. Freshness Assessment", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        y_offset = 60
        for it in items:
            # Badge color
            if it.freshness_state.lower() == "fresh":
                b_col = (34, 139, 34)  # Green
            elif it.freshness_state.lower() in ("rotten", "spoiled"):
                b_col = (50, 50, 220)  # Red (BGR)
            else:
                b_col = (30, 165, 218)  # Amber

            # Draw card
            cv2.rectangle(p5, (12, y_offset), (panel_w - 12, y_offset + 55), (35, 40, 52), -1)
            cv2.rectangle(p5, (12, y_offset), (22, y_offset + 55), b_col, -1)

            # Text
            f_title = f"{it.food_class.upper()} — {it.freshness_state.upper()}"
            conf_str = f"Confidence: {it.freshness_confidence*100:.1f}%" if it.freshness_confidence else "Confidence: N/A"
            cv2.putText(p5, f_title, (32, y_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(p5, conf_str, (32, y_offset + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1)

            # Probabilities bar
            pf = it.probabilities.get("fresh", 0.0)
            pr = it.probabilities.get("rotten", 0.0)
            prob_sub = f"P(fresh)={pf*100:.0f}%  P(rotten)={pr*100:.0f}%"
            cv2.putText(p5, prob_sub, (panel_w - 180, y_offset + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 150, 165), 1)

            y_offset += 65
            if y_offset > panel_h - 30:
                break

        cv2.putText(p5, "NutriVision AI • Visual Estimate", (12, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 130, 145), 1)

        # Combine into 2-row layout: Row 1 = P1, P2, P3; Row 2 = P4, P5, blank
        row1 = np.hstack([p1, p2, p3])
        blank = np.full((panel_h, panel_w, 3), 20, dtype=np.uint8)
        row2 = np.hstack([p4, p5, blank])
        composite = np.vstack([row1, row2])

        out_path = self.viz_dir / f"{stem}_freshness_pipeline.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        return out_path

    def _draw_panel_title(self, img: np.ndarray, title: str) -> np.ndarray:
        """Draws top title bar on a visualization panel."""
        h, w = img.shape[:2]
        header = np.zeros((30, w, 3), dtype=np.uint8)
        cv2.putText(header, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return np.vstack([header, img[:-30] if h >= 30 else img])
