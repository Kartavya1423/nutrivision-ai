"""
NutriVision AI — YOLO11 → SAM 2 Integrated Pipeline
STEP 4: Food Segmentation Integration and Validation

Coordinates detection prompting, instance segmentation, and food-only crop extraction:
DetectionResult
   ↓
Prompt Conversion & BBox Validation
   ↓
SAM2 Food Segmenter
   ↓
SegmentationResult
   ↓
SegmentedFoodObject Items (with file-based mask references)
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.config.settings import Settings, get_settings
from src.detection.detector import BaseFoodDetector, DetectionResult, ModelStatus
from src.detection.yolo_detector import YOLO11FoodDetector
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.segmentation.segmenter import (
    BaseFoodSegmenter,
    FoodMask,
    SegmentationResult,
    SegmentationStatusCode,
)
from src.utils.image_ops import load_image


@dataclass
class SegmentedFoodObject:
    """
    NutriVision AI — Step 4 Standardized Food Object Schema.
    Strictly preserves detector predictions separate from downstream classification,
    and references saved mask image files instead of embedding raw pixel arrays.
    """
    food_id: str
    class_name: str
    class_id: int
    detection_confidence: float
    bbox_xyxy: List[float]
    mask_reference: Optional[str] = None
    mask_area_pixels: int = 0
    mask_quality_score: float = 0.0
    segmentation_status: str = SegmentationStatusCode.SEGMENTATION_SUCCESS.value
    mask_tight_bbox: Optional[List[float]] = None
    relative_area: float = 0.0
    crop_reference: Optional[str] = None
    crop_rgb: Optional[np.ndarray] = None
    crop_rgba: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Backward compatibility properties for FoodCropResult callers
    @property
    def crop_id(self) -> int:
        try:
            return int(self.food_id.replace("food_", ""))
        except Exception:
            return 1

    @property
    def yolo_confidence(self) -> float:
        return self.detection_confidence

    @property
    def bbox(self) -> List[float]:
        return self.bbox_xyxy

    @property
    def sam_quality_score(self) -> float:
        return self.mask_quality_score

    @property
    def status(self) -> str:
        return self.segmentation_status

    def to_dict(self) -> Dict[str, Any]:
        """Returns clean JSON-serializable dictionary without large pixel arrays."""
        return {
            "food_id": self.food_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "detection_confidence": round(float(self.detection_confidence), 4),
            "bbox_xyxy": [round(float(c), 2) for c in self.bbox_xyxy],
            "mask_reference": str(self.mask_reference) if self.mask_reference else None,
            "mask_area_pixels": int(self.mask_area_pixels),
            "mask_quality_score": round(float(self.mask_quality_score), 4),
            "segmentation_status": self.segmentation_status,
        }

    def to_extended_dict(self) -> Dict[str, Any]:
        """Returns dictionary with additional downstream geometric & crop metadata."""
        d = self.to_dict()
        d.update({
            "crop_id": self.crop_id,
            "mask_tight_bbox": self.mask_tight_bbox,
            "relative_area": round(float(self.relative_area), 4),
            "crop_reference": str(self.crop_reference) if self.crop_reference else None,
            "has_crop_rgb": self.crop_rgb is not None,
            "has_crop_rgba": self.crop_rgba is not None,
            "metadata": self.metadata,
        })
        return d


# Backward compatibility class for Phase 3 FoodCropResult callers
class FoodCropResult(SegmentedFoodObject):
    """Backward compatibility wrapper for Phase 3 FoodCropResult schema."""
    def __init__(
        self,
        crop_id: int = 1,
        food_id: Optional[str] = None,
        class_id: int = 0,
        class_name: str = "food",
        detection_confidence: Optional[float] = None,
        yolo_confidence: Optional[float] = None,
        bbox_xyxy: Optional[List[float]] = None,
        bbox: Optional[List[float]] = None,
        mask_reference: Optional[str] = None,
        mask_area_pixels: Optional[int] = None,
        area_pixels: Optional[int] = None,
        mask_quality_score: Optional[float] = None,
        sam_quality_score: Optional[float] = None,
        segmentation_status: Optional[str] = None,
        status: Optional[str] = None,
        mask_tight_bbox: Optional[List[float]] = None,
        relative_area: float = 0.0,
        crop_reference: Optional[str] = None,
        crop_rgb: Optional[np.ndarray] = None,
        crop_rgba: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        f_id = food_id if food_id is not None else f"food_{crop_id}"
        conf = detection_confidence if detection_confidence is not None else (yolo_confidence if yolo_confidence is not None else 1.0)
        box = bbox_xyxy if bbox_xyxy is not None else (bbox if bbox is not None else [0.0, 0.0, 0.0, 0.0])
        area = mask_area_pixels if mask_area_pixels is not None else (area_pixels if area_pixels is not None else 0)
        q_score = mask_quality_score if mask_quality_score is not None else (sam_quality_score if sam_quality_score is not None else 0.0)
        seg_status = segmentation_status if segmentation_status is not None else (status if status is not None else SegmentationStatusCode.SEGMENTATION_SUCCESS.value)
        super().__init__(
            food_id=f_id,
            class_name=class_name,
            class_id=class_id,
            detection_confidence=conf,
            bbox_xyxy=box,
            mask_reference=mask_reference,
            mask_area_pixels=area,
            mask_quality_score=q_score,
            segmentation_status=seg_status,
            mask_tight_bbox=mask_tight_bbox,
            relative_area=relative_area,
            crop_reference=crop_reference,
            crop_rgb=crop_rgb,
            crop_rgba=crop_rgba,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["crop_id"] = self.crop_id
        d["area_pixels"] = self.mask_area_pixels
        d["status"] = self.segmentation_status
        return d


class YOLOToSAMPipeline:
    """
    Decoupled pipeline orchestrator bridging YOLO11 object localization
    with SAM 2 pixel-level instance segmentation.
    """

    def __init__(
        self,
        detector: Optional[BaseFoodDetector] = None,
        segmenter: Optional[BaseFoodSegmenter] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.detector = detector or YOLO11FoodDetector(settings=self.settings)
        self.segmenter = segmenter or SAM2FoodSegmenter(settings=self.settings)

    def process_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        conf_threshold: Optional[float] = None,
        crop_padding: int = 10,
        generate_crops: bool = True,
        save_artifacts: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        source_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes YOLO11 detection followed by prompt-guided SAM 2 segmentation.

        Args:
            image: Original RGB image input.
            conf_threshold: Optional detector confidence threshold override.
            crop_padding: Pixel padding around food boundaries.
            generate_crops: Whether to compute RGBA food-only crops.
            save_artifacts: Whether to write mask PNGs and crops to disk.
            output_dir: Base directory for segmentation artifacts (e.g. outputs/segmentation).
            source_name: Stem filename for saved artifacts (e.g. 'plate_01').

        Returns:
            Structured dictionary containing detection, segmentation, and food objects.
        """
        t0 = time.perf_counter()

        # 1. Load image
        try:
            np_img, _ = load_image(image)
        except Exception as e:
            return {
                "status": SegmentationStatusCode.INVALID_IMAGE.value,
                "message": f"Failed to load image: {str(e)}",
                "inference_time_ms": 0.0,
                "detection_time_ms": 0.0,
                "segmentation_time_ms": 0.0,
                "detection_count": 0,
                "segmentation_count": 0,
                "food_objects": [],
                "crops": [],
                "detections": [],
                "segmentations": [],
                "exported_files": {},
            }

        h, w = np_img.shape[:2]

        # Determine source stem for artifact export
        if source_name is None:
            if isinstance(image, (str, Path)):
                source_name = Path(image).stem
            else:
                source_name = f"food_sample_{int(time.time())}"
        stem = Path(source_name).stem

        # 2. YOLO11 Food Detection
        if not self.detector.is_ready():
            return {
                "status": "DETECTION_MODEL_NOT_AVAILABLE",
                "message": "YOLO11 food detector checkpoint is not loaded.",
                "inference_time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "detection_time_ms": 0.0,
                "segmentation_time_ms": 0.0,
                "detection_count": 0,
                "segmentation_count": 0,
                "food_objects": [],
                "crops": [],
                "detections": [],
                "segmentations": [],
                "exported_files": {},
            }

        # Override confidence threshold if requested
        orig_thresh = getattr(self.detector, "conf_threshold", 0.25)
        if conf_threshold is not None:
            self.detector.conf_threshold = conf_threshold

        t_det_start = time.perf_counter()
        try:
            det_result: DetectionResult = self.detector.detect(np_img)
        finally:
            self.detector.conf_threshold = orig_thresh
        det_time_ms = round((time.perf_counter() - t_det_start) * 1000.0, 2)

        if not det_result.is_success or len(det_result.detections) == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": SegmentationStatusCode.NO_DETECTIONS.value,
                "message": det_result.error_message or "No food items detected by YOLO11.",
                "inference_time_ms": round(elapsed_ms, 2),
                "detection_time_ms": det_time_ms,
                "segmentation_time_ms": 0.0,
                "detection_count": 0,
                "segmentation_count": 0,
                "food_objects": [],
                "crops": [],
                "detections": [],
                "segmentations": [],
                "detection_result": det_result,
                "exported_files": {},
            }

        # 3. Extract and validate bounding box prompts
        valid_indices: List[int] = []
        bboxes: List[List[float]] = []
        class_names: List[str] = []
        confidences: List[float] = []

        for idx, det in enumerate(det_result.detections):
            box = det.bbox
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(c) for c in box]
            # Validate non-degenerate box
            if x2 <= x1 or y2 <= y1:
                continue
            # Clamp to image boundaries
            x1_c = max(0.0, min(float(w - 1), x1))
            y1_c = max(0.0, min(float(h - 1), y1))
            x2_c = max(x1_c + 1.0, min(float(w), x2))
            y2_c = max(y1_c + 1.0, min(float(h), y2))

            bboxes.append([x1_c, y1_c, x2_c, y2_c])
            class_names.append(det.class_name)
            confidences.append(det.confidence)
            valid_indices.append(idx)

        if not bboxes:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": SegmentationStatusCode.INVALID_MASK.value,
                "message": "All detector bounding boxes were invalid or degenerate.",
                "inference_time_ms": round(elapsed_ms, 2),
                "detection_time_ms": det_time_ms,
                "segmentation_time_ms": 0.0,
                "detection_count": len(det_result.detections),
                "segmentation_count": 0,
                "food_objects": [],
                "crops": [],
                "detections": [d.to_dict() for d in det_result.detections],
                "segmentations": [],
                "detection_result": det_result,
                "exported_files": {},
            }

        # 4. SAM 2 Prompt-Guided Segmentation
        if not self.segmenter.is_ready():
            return {
                "status": SegmentationStatusCode.SEGMENTATION_MODEL_NOT_AVAILABLE.value,
                "message": "SAM 2 segmentation model is not loaded.",
                "inference_time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "detection_time_ms": det_time_ms,
                "segmentation_time_ms": 0.0,
                "detection_count": len(det_result.detections),
                "segmentation_count": 0,
                "food_objects": [],
                "crops": [],
                "detections": [d.to_dict() for d in det_result.detections],
                "segmentations": [],
                "detection_result": det_result,
                "exported_files": {},
            }

        t_seg_start = time.perf_counter()
        seg_result: SegmentationResult = self.segmenter.predict_masks(
            image=np_img,
            bboxes=bboxes,
            class_names=class_names,
            confidences=confidences,
            crop_padding=crop_padding,
            generate_crops=generate_crops,
        )
        seg_time_ms = round((time.perf_counter() - t_seg_start) * 1000.0, 2)

        # 5. Handle artifact saving if requested or output_dir provided
        exported_files: Dict[str, Any] = {"masks": [], "crops": [], "rgba": []}
        save_paths: Dict[int, Dict[str, str]] = {}

        if save_artifacts or output_dir is not None:
            out_base = Path(output_dir) if output_dir else self.settings.outputs.segmentation
            masks_dir = out_base / "masks"
            crops_dir = out_base / "crops"
            rgba_dir = out_base / "rgba"
            for d in (masks_dir, crops_dir, rgba_dir):
                d.mkdir(parents=True, exist_ok=True)

            for idx, f_mask in enumerate(seg_result.masks):
                f_id_str = f"food_{idx+1}"
                c_clean = f_mask.class_name.replace(" ", "_").lower()
                prefix = f"{stem}_{f_id_str}_{c_clean}"

                # Save 8-bit PNG mask
                mask_p = masks_dir / f"{prefix}_mask.png"
                mask_u8 = f_mask.as_uint8()
                Image.fromarray(mask_u8).save(mask_p)
                exported_files["masks"].append(str(mask_p))

                paths_dict = {"mask": str(mask_p)}

                # Save RGB crop
                if f_mask.crop_rgb is not None:
                    crop_p = crops_dir / f"{prefix}.jpg"
                    Image.fromarray(f_mask.crop_rgb).save(crop_p, quality=92)
                    exported_files["crops"].append(str(crop_p))
                    paths_dict["crop_rgb"] = str(crop_p)

                # Save RGBA crop
                if f_mask.crop_rgba is not None:
                    rgba_p = rgba_dir / f"{prefix}_rgba.png"
                    Image.fromarray(f_mask.crop_rgba, mode="RGBA").save(rgba_p)
                    exported_files["rgba"].append(str(rgba_p))
                    paths_dict["crop_rgba"] = str(rgba_p)

                save_paths[idx] = paths_dict

        # 6. Assemble SegmentedFoodObject items
        food_objects: List[SegmentedFoodObject] = []
        for idx, f_mask in enumerate(seg_result.masks):
            det_idx = valid_indices[idx] if idx < len(valid_indices) else idx
            det = det_result.detections[det_idx] if det_idx < len(det_result.detections) else None

            paths = save_paths.get(idx, {})
            mask_ref = paths.get("mask")
            crop_ref = paths.get("crop_rgba") or paths.get("crop_rgb")

            food_obj = SegmentedFoodObject(
                food_id=f"food_{idx+1}",
                class_name=f_mask.class_name,
                class_id=f_mask.class_id,
                detection_confidence=det.confidence if det else 0.0,
                bbox_xyxy=f_mask.prompt_bbox or (det.bbox if det else [0.0, 0.0, 0.0, 0.0]),
                mask_reference=mask_ref,
                mask_area_pixels=f_mask.area_pixels,
                mask_quality_score=f_mask.confidence,
                segmentation_status=f_mask.status,
                mask_tight_bbox=f_mask.bbox or [0.0, 0.0, 0.0, 0.0],
                relative_area=f_mask.relative_area,
                crop_reference=crop_ref,
                crop_rgb=f_mask.crop_rgb,
                crop_rgba=f_mask.crop_rgba,
                metadata=f_mask.metadata,
            )
            food_objects.append(food_obj)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "status": SegmentationStatusCode.SEGMENTATION_SUCCESS.value,
            "inference_time_ms": round(elapsed_ms, 2),
            "detection_time_ms": det_time_ms,
            "segmentation_time_ms": seg_time_ms,
            "detection_count": len(det_result.detections),
            "segmentation_count": len(seg_result.masks),
            "food_objects": food_objects,
            "crops": food_objects,  # Backward compatibility
            "detections": [d.to_dict() for d in det_result.detections],
            "segmentations": seg_result.masks,
            "detection_result": det_result,
            "segmentation_result": seg_result,
            "image_shape": [h, w, 3],
            "exported_files": exported_files,
        }
