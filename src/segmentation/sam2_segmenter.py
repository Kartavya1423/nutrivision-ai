"""
NutriVision AI — SAM 2 (Segment Anything Model 2) Food Segmenter
Phase 3: SAM 2 Food Instance Segmentation

Performs prompt-guided instance segmentation using bounding boxes from YOLO11.
Never fabricates masks; strictly returns standard status codes when checkpoints
or detections are unavailable.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from src.config.settings import Settings, get_settings
from src.detection.detector import ModelStatus
from src.segmentation.mask_selection import select_best_mask
from src.segmentation.postprocessing import postprocess_mask
from src.segmentation.segmenter import (
    BaseFoodSegmenter,
    FoodMask,
    SegmentationResult,
    SegmentationStatusCode,
)
from src.utils.crop import extract_food_region
from src.utils.image_ops import load_image


class SAM2FoodSegmenter(BaseFoodSegmenter):
    """
    Production-grade food instance segmenter powered by Meta's SAM 2 architecture.
    Accepts bounding box prompts from YOLO11 to generate precise pixel-level masks,
    isolating food regions and generating alpha-transparent RGBA crops.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_config: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        min_mask_area: int = 50,
        max_candidate_masks: int = 3,
        selection_strategy: str = "iou_containment_confidence",
        box_margin: float = 0.05,
        apply_postprocessing: bool = True,
        settings: Optional[Settings] = None,
    ):
        cfg = settings or get_settings()
        ckpt = checkpoint_path or cfg.model_paths.segmentation_model_path
        super().__init__(checkpoint_path=ckpt)

        self.model_config = model_config or "sam2.1_t.pt"
        self.mask_threshold = mask_threshold or cfg.segmentation.mask_threshold
        self.min_mask_area = min_mask_area or cfg.segmentation.min_mask_area
        self.max_candidate_masks = max_candidate_masks or cfg.segmentation.max_candidate_masks
        self.selection_strategy = selection_strategy or cfg.segmentation.selection_strategy
        self.box_margin = box_margin or cfg.segmentation.box_margin
        self.apply_postprocessing = apply_postprocessing if apply_postprocessing is not None else cfg.segmentation.apply_postprocessing

        # Device detection: prefer CUDA when available, gracefully fall back to CPU
        desired_device = (device or cfg.resolve_device()).lower()
        if desired_device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = desired_device

        # Model instance
        self.backend = "ultralytics_sam"
        self.model = None

        # Attempt to load model
        self.load_model()

    def load_model(self) -> bool:
        """
        Validate checkpoint existence and load SAM 2 predictor.
        Follows strict Phase 3 rules: never fabricates masks or mock weights.
        """
        if self.checkpoint_path is None or not Path(self.checkpoint_path).is_file():
            self._status = ModelStatus.MODEL_NOT_LOADED
            self._error_msg = (
                f"SAM 2 checkpoint not found at: '{self.checkpoint_path}'. "
                "Phase 3 requires a valid SAM 2 checkpoint (e.g. sam2.1_t.pt). "
                "Do not fabricate masks."
            )
            self.model = None
            return False

        try:
            # 1. Attempt loading via Ultralytics SAM (supports sam2.1_t.pt, sam2_b.pt, etc.)
            load_path = self.checkpoint_path
            if "sam2_hiera_base_plus" in str(load_path):
                alt = Path(load_path).parent / "sam2.1_t.pt"
                if alt.is_file():
                    load_path = alt

            from ultralytics import SAM
            self.model = SAM(str(load_path))
            # Put model on specified device
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            self.backend = "ultralytics_sam"
            self._status = ModelStatus.READY
            self._error_msg = None
            return True
        except ImportError:
            pass
        except Exception as e:
            err_str = str(e)
            # Try secondary Meta SAM 2 import if ultralytics fails
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                sam_model = build_sam2(self.model_config, str(self.checkpoint_path), device=self.device)
                self.model = SAM2ImagePredictor(sam_model)
                self.backend = "meta_sam2"
                self._status = ModelStatus.READY
                self._error_msg = None
                return True
            except Exception as e2:
                self._status = ModelStatus.ERROR
                self._error_msg = f"Failed to load SAM 2 model: {err_str}; Meta SAM2 fallback failed: {str(e2)}"
                self.model = None
                return False

        self._status = ModelStatus.ERROR
        self._error_msg = "Could not initialize SAM 2 backend."
        self.model = None
        return False

    def predict_masks(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        bboxes: Optional[List[List[float]]] = None,
        class_names: Optional[List[str]] = None,
        confidences: Optional[List[float]] = None,
        crop_padding: int = 10,
        generate_crops: bool = True,
    ) -> SegmentationResult:
        """
        Segments food items using YOLO11 bounding box prompts.
        
        Args:
            image: Original RGB input image.
            bboxes: List of bounding boxes [x1, y1, x2, y2].
            class_names: Optional class names matching bboxes.
            confidences: Optional detector confidence scores.
            crop_padding: Pixel padding for food-only RGBA crops.
            generate_crops: If True, computes isolated food crops and RGBA arrays.

        Returns:
            SegmentationResult containing individual FoodMask items.
        """
        t0 = time.perf_counter()

        # 1. Image validation
        try:
            np_img, _ = load_image(image)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SegmentationResult(
                status=ModelStatus.ERROR,
                status_code=SegmentationStatusCode.INVALID_IMAGE.value,
                masks=[],
                inference_time_ms=elapsed_ms,
                model_name=f"SAM2FoodSegmenter ({self.backend})",
                device=self.device,
                checkpoint=str(self.checkpoint_path),
                error_message=f"Invalid image input: {str(e)}",
            )

        h, w = np_img.shape[:2]
        total_pixels = float(h * w)

        # 2. Model readiness check
        if not self.is_ready():
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SegmentationResult(
                status=self._status,
                status_code=SegmentationStatusCode.SEGMENTATION_MODEL_NOT_LOADED.value,
                masks=[],
                inference_time_ms=elapsed_ms,
                model_name=f"SAM2FoodSegmenter ({self.backend})",
                device=self.device,
                checkpoint=str(self.checkpoint_path) if self.checkpoint_path else None,
                error_message=self._error_msg or "SAM 2 model is not loaded.",
            )

        # 3. Handle empty detections
        if bboxes is None or len(bboxes) == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SegmentationResult(
                status=ModelStatus.READY,
                status_code=SegmentationStatusCode.NO_DETECTIONS.value,
                masks=[],
                combined_mask=np.zeros((h, w), dtype=bool),
                inference_time_ms=elapsed_ms,
                model_name=f"SAM2FoodSegmenter ({self.backend})",
                device=self.device,
                checkpoint=str(self.checkpoint_path),
                error_message="No bounding boxes provided for segmentation.",
            )

        food_masks: List[FoodMask] = []
        combined = np.zeros((h, w), dtype=bool)

        try:
            for idx, box in enumerate(bboxes):
                c_name = class_names[idx] if class_names and idx < len(class_names) else f"food_{idx}"
                det_conf = float(confidences[idx]) if confidences and idx < len(confidences) else 1.0

                # Ensure box format [x1, y1, x2, y2]
                bx1, by1, bx2, by2 = [float(c) for c in box]
                # Clamp prompt box to image dimensions
                bx1 = max(0.0, min(float(w - 1), bx1))
                by1 = max(0.0, min(float(h - 1), by1))
                bx2 = max(bx1 + 1.0, min(float(w), bx2))
                by2 = max(by1 + 1.0, min(float(h), by2))
                clamped_box = [bx1, by1, bx2, by2]

                # Run SAM 2 prediction
                candidate_masks: List[np.ndarray] = []
                candidate_scores: List[float] = []

                if self.backend == "ultralytics_sam":
                    pred_res = self.model.predict(np_img, bboxes=[clamped_box], verbose=False)[0]
                    if pred_res.masks is not None and len(pred_res.masks.data) > 0:
                        m_tensor = pred_res.masks.data.cpu().numpy()
                        for m_idx in range(len(m_tensor)):
                            candidate_masks.append(m_tensor[m_idx] > 0)
                        if pred_res.boxes is not None and pred_res.boxes.conf is not None:
                            candidate_scores = pred_res.boxes.conf.cpu().numpy().tolist()
                        else:
                            candidate_scores = [1.0] * len(candidate_masks)
                elif self.backend == "meta_sam2":
                    self.model.set_image(np_img)
                    box_np = np.array(clamped_box, dtype=np.float32)
                    masks, scores, _ = self.model.predict(box=box_np, multimask_output=True)
                    for m_idx in range(len(masks)):
                        candidate_masks.append(masks[m_idx] > 0)
                    candidate_scores = [float(s) for s in scores]

                if not candidate_masks:
                    # Failed to produce any mask for this box
                    food_masks.append(
                        FoodMask(
                            mask=np.zeros((h, w), dtype=bool),
                            class_name=c_name,
                            class_id=idx,
                            confidence=0.0,
                            bbox=[0.0, 0.0, 0.0, 0.0],
                            prompt_bbox=clamped_box,
                            area_pixels=0,
                            relative_area=0.0,
                            status=SegmentationStatusCode.SEGMENTATION_FAILED.value,
                            metadata={"error": "Model produced 0 masks for prompt box."},
                        )
                    )
                    continue

                # 4. Deterministic candidate selection
                best_mask, best_score, sel_meta = select_best_mask(
                    candidate_masks=candidate_masks,
                    candidate_scores=candidate_scores,
                    prompt_bbox=clamped_box,
                    image_shape=(h, w),
                    min_mask_area=self.min_mask_area,
                    strategy=self.selection_strategy,
                )

                if best_mask is None:
                    # Mask failed validity (e.g. MASK_TOO_SMALL or RUNAWAY_MASK)
                    status_reason = sel_meta.get("status", SegmentationStatusCode.MASK_TOO_SMALL.value)
                    food_masks.append(
                        FoodMask(
                            mask=np.zeros((h, w), dtype=bool),
                            class_name=c_name,
                            class_id=idx,
                            confidence=0.0,
                            bbox=[0.0, 0.0, 0.0, 0.0],
                            prompt_bbox=clamped_box,
                            area_pixels=0,
                            relative_area=0.0,
                            status=status_reason,
                            metadata=sel_meta,
                        )
                    )
                    continue

                # 5. Mask post-processing
                if self.apply_postprocessing:
                    final_mask = postprocess_mask(
                        best_mask,
                        prompt_bbox=clamped_box,
                        min_component_size=32,
                        fill_holes=True,
                        max_hole_size=150,
                        constrain_to_box=True,
                        box_margin_ratio=self.box_margin,
                    )
                else:
                    final_mask = best_mask

                area = int(np.sum(final_mask))
                if area < self.min_mask_area:
                    food_masks.append(
                        FoodMask(
                            mask=np.zeros((h, w), dtype=bool),
                            class_name=c_name,
                            class_id=idx,
                            confidence=0.0,
                            bbox=[0.0, 0.0, 0.0, 0.0],
                            prompt_bbox=clamped_box,
                            area_pixels=0,
                            relative_area=0.0,
                            status=SegmentationStatusCode.MASK_TOO_SMALL.value,
                            metadata={"postprocessed_area": area, "min_required": self.min_mask_area},
                        )
                    )
                    continue

                rel_area = area / total_pixels if total_pixels > 0 else 0.0
                combined = np.logical_or(combined, final_mask)

                # 6. Extract food-only RGBA and RGB crops
                crop_rgb, crop_rgba = None, None
                crop_coords = None

                if generate_crops:
                    try:
                        crop_rgba, _ = extract_food_region(
                            image=np_img,
                            mask=final_mask,
                            padding=crop_padding,
                            return_rgba=True,
                        )
                        crop_rgb, _ = extract_food_region(
                            image=np_img,
                            mask=final_mask,
                            padding=crop_padding,
                            return_rgba=False,
                        )
                        # Determine tight crop box in parent coordinates
                        tight_box = compute_mask_tight_bbox(final_mask)
                        crop_coords = [int(round(c)) for c in tight_box]
                    except Exception as e:
                        sel_meta["crop_error"] = str(e)

                food_mask_obj = FoodMask(
                    mask=final_mask,
                    class_name=c_name,
                    class_id=idx,
                    confidence=best_score,
                    bbox=None,  # Computed automatically in FoodMask.__post_init__
                    prompt_bbox=clamped_box,
                    area_pixels=area,
                    relative_area=rel_area,
                    status=SegmentationStatusCode.SEGMENTATION_SUCCESS.value,
                    crop_rgb=crop_rgb,
                    crop_rgba=crop_rgba,
                    crop_bbox=crop_coords,
                    selection_strategy=self.selection_strategy,
                    metadata={
                        "yolo_confidence": round(det_conf, 4),
                        "selection_metadata": sel_meta,
                    },
                )
                food_masks.append(food_mask_obj)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SegmentationResult(
                status=ModelStatus.READY,
                status_code=SegmentationStatusCode.SEGMENTATION_SUCCESS.value,
                masks=food_masks,
                combined_mask=combined,
                inference_time_ms=elapsed_ms,
                model_name=f"SAM2FoodSegmenter ({self.backend})",
                device=self.device,
                checkpoint=str(self.checkpoint_path),
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SegmentationResult(
                status=ModelStatus.ERROR,
                status_code=SegmentationStatusCode.SEGMENTATION_FAILED.value,
                masks=[],
                inference_time_ms=elapsed_ms,
                model_name=f"SAM2FoodSegmenter ({self.backend})",
                device=self.device,
                checkpoint=str(self.checkpoint_path),
                error_message=f"SAM 2 inference error: {str(e)}",
            )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "sam2",
            "backend": self.backend,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "status": self._status.value,
            "device": self.device,
            "is_ready": self.is_ready(),
            "mask_threshold": self.mask_threshold,
            "min_mask_area": self.min_mask_area,
            "selection_strategy": self.selection_strategy,
            "apply_postprocessing": self.apply_postprocessing,
            "error_message": self._error_msg,
        }


def compute_mask_tight_bbox(mask: np.ndarray) -> List[float]:
    """Helper to derive tight bounding box [x1, y1, x2, y2] from 2D boolean mask."""
    if not np.any(mask):
        return [0.0, 0.0, 0.0, 0.0]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    if len(y_indices) == 0 or len(x_indices) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(x_indices[0]), float(y_indices[0]), float(x_indices[-1]), float(y_indices[-1])]
