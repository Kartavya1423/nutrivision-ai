"""
NutriVision AI — Ultralytics YOLO11 Food Detector
Modular detector for multi-item food localization.
Never fabricates detections; reports MODEL_NOT_LOADED when weights are missing.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.config.settings import Settings, get_settings
from src.detection.detector import (
    BaseFoodDetector,
    BoundingBox,
    DetectionResult,
    FoodDetection,
    ModelStatus,
)
import src.utils.torch_patch
from src.utils.image_ops import load_image


COCO_FOOD_CLASSES = {
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "bowl",
}


def merge_subpart_and_duplicate_boxes(
    detections: List[FoodDetection],
    img_w: int,
    img_h: int,
) -> List[FoodDetection]:
    """
    Merge contiguous sub-parts of the same food item (e.g. broccoli florets + stalk,
    carrot root + greens, burger bun + patty, close-up split food halves) and suppress duplicate detections.
    Prevents single food items from being erroneously fragmented into multiple items, especially in close-up shots.
    """
    if len(detections) <= 1:
        return detections

    generic_classes = {"food", "others", "other_food", "vegetable", "fruit", "dish", "proposal", "bowl"}
    area_img = float(max(1, img_w * img_h))

    def should_merge(det_a: FoodDetection, det_b: FoodDetection) -> Tuple[bool, Optional[str], Optional[int]]:
        b_a = det_a.bbox
        b_b = det_b.bbox
        x1_a, y1_a, x2_a, y2_a = b_a
        x1_b, y1_b, x2_b, y2_b = b_b

        w_a, h_a = max(1.0, x2_a - x1_a), max(1.0, y2_a - y1_a)
        w_b, h_b = max(1.0, x2_b - x1_b), max(1.0, y2_b - y1_b)
        area_a = w_a * h_a
        area_b = w_b * h_b

        xi1, yi1 = max(x1_a, x1_b), max(y1_a, y1_b)
        xi2, yi2 = min(x2_a, x2_b), min(y2_a, y2_b)
        inter_w = max(0.0, xi2 - xi1)
        inter_h = max(0.0, yi2 - yi1)
        inter_area = inter_w * inter_h

        min_area = min(area_a, area_b)
        iom = inter_area / min_area if min_area > 0 else 0.0
        union_area = area_a + area_b - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0

        c_a = det_a.class_name
        c_b = det_b.class_name

        # Resolve priority class name
        if c_a not in generic_classes and c_b in generic_classes:
            best_cls = c_a
            best_id = det_a.class_id
        elif c_b not in generic_classes and c_a in generic_classes:
            best_cls = c_b
            best_id = det_b.class_id
        elif det_a.confidence >= det_b.confidence:
            best_cls = c_a
            best_id = det_a.class_id
        else:
            best_cls = c_b
            best_id = det_b.class_id

        # 1. Any meaningful overlap or containment (one box overlaps the other)
        # In food photography, separate items rarely overlap by > 18%; overlapping boxes are dual proposals or subparts
        if iom >= 0.18 or iou >= 0.15:
            return True, best_cls, best_id

        # Calculate bounding box proximity gaps
        horiz_gap = max(0.0, max(x1_a, x1_b) - min(x2_a, x2_b)) if xi1 > xi2 else 0.0
        vert_gap = max(0.0, max(y1_a, y1_b) - min(y2_a, y2_b)) if yi1 > yi2 else 0.0
        min_w, max_w = min(w_a, w_b), max(w_a, w_b)
        min_h, max_h = min(h_a, h_b), max(h_a, h_b)

        # 2. Vertical split / stacking (e.g. top and bottom halves of close-up fruit, burger, or stalk + head)
        horiz_ratio = inter_w / min_w if min_w > 0 else 0.0
        if horiz_ratio >= 0.25 and (inter_h > 0 or vert_gap <= 0.20 * max_h):
            return True, best_cls, best_id

        # 3. Horizontal split / stacking (e.g. left and right halves of close-up bread, pizza, banana, or meat)
        vert_ratio = inter_h / min_h if min_h > 0 else 0.0
        if vert_ratio >= 0.25 and (inter_w > 0 or horiz_gap <= 0.20 * max_w):
            return True, best_cls, best_id

        # 4. Close-Up Dominance: In close-up shots, the food covers a large area of the image.
        # If boxes are adjacent or close together and together form a significant food region, merge them!
        union_box_w = max(x2_a, x2_b) - min(x1_a, x1_b)
        union_box_h = max(y2_a, y2_b) - min(y1_a, y1_b)
        union_box_area = union_box_w * union_box_h

        if union_box_area >= 0.18 * area_img:
            # Check if boxes are within close proximity
            max_img_dim = max(float(img_w), float(img_h))
            if horiz_gap <= 0.15 * max_img_dim and vert_gap <= 0.15 * max_img_dim:
                return True, best_cls, best_id

        # 5. Dual-detection image rule: when exactly 2 boxes are detected in the entire image,
        # close-up food is frequently split into 2 items.
        if len(detections) == 2:
            if (area_a + area_b) >= 0.15 * area_img or union_box_area >= 0.20 * area_img:
                if horiz_gap <= 0.20 * float(img_w) and vert_gap <= 0.20 * float(img_h):
                    return True, best_cls, best_id

        # 6. Same food class or generic food category with loose proximity
        same_or_generic = (c_a == c_b) or (c_a in generic_classes) or (c_b in generic_classes)
        if same_or_generic:
            if horiz_gap <= 0.12 * max_w and vert_gap <= 0.12 * max_h:
                return True, best_cls, best_id

        # 7. Known related vegetable/produce parts
        produce_group = {"broccoli", "lettuce", "spinach", "cabbage", "celery", "kale"}
        if c_a in produce_group and c_b in produce_group:
            if horiz_ratio >= 0.20 and (inter_h > 0 or vert_gap <= 0.20 * max_h):
                if "broccoli" in (c_a, c_b):
                    return True, "broccoli", (det_a.class_id if c_a == "broccoli" else det_b.class_id)
                return True, best_cls, best_id

        return False, None, None

    # Iterative pairwise merging
    merged_list = list(detections)
    changed = True
    while changed:
        changed = False
        n = len(merged_list)
        for i in range(n):
            for j in range(i + 1, n):
                det_i = merged_list[i]
                det_j = merged_list[j]
                merge_ok, best_cls, best_id = should_merge(det_i, det_j)
                if merge_ok:
                    u_x1 = max(0.0, min(det_i.bbox[0], det_j.bbox[0]))
                    u_y1 = max(0.0, min(det_i.bbox[1], det_j.bbox[1]))
                    u_x2 = min(float(img_w), max(det_i.bbox[2], det_j.bbox[2]))
                    u_y2 = min(float(img_h), max(det_i.bbox[3], det_j.bbox[3]))
                    new_conf = max(det_i.confidence, det_j.confidence)

                    merged_det = FoodDetection(
                        class_id=best_id if best_id is not None else det_i.class_id,
                        class_name=best_cls or det_i.class_name,
                        confidence=new_conf,
                        bbox=[u_x1, u_y1, u_x2, u_y2],
                    )
                    merged_list[i] = merged_det
                    merged_list.pop(j)
                    changed = True
                    break
            if changed:
                break

    return merged_list


class YOLO11FoodDetector(BaseFoodDetector):
    """
    Food detector based on Ultralytics YOLO11.
    Strictly checks checkpoint presence and dependency readiness.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        image_size: Optional[int] = None,
        device: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        cfg = settings or get_settings()
        path = model_path or cfg.model_paths.detection_model_path
        super().__init__(model_path=path)

        self.conf_threshold = conf_threshold if conf_threshold is not None else cfg.detection.confidence_threshold
        self.iou_threshold = iou_threshold if iou_threshold is not None else cfg.detection.iou_threshold
        self.image_size = image_size if image_size is not None else cfg.detection.image_size
        self.device = device or cfg.resolve_device()
        self.max_detections = cfg.detection.max_detections

        self.settings = cfg
        self.coco_model = None

        # Attempt safe initial load check
        self.load_model()

    def load_model(self) -> bool:
        """
        Verify weight checkpoint exists and load YOLO model.
        Returns False and sets status appropriately if weights are missing.
        """
        if self.model_path is None or not self.model_path.is_file():
            self._status = ModelStatus.MODEL_NOT_LOADED
            self._error_msg = (
                f"YOLO11 weights file not found at: '{self.model_path}'. "
                "Phase 0 architecture is initialized, but model has not yet been trained or loaded. "
                "Do not fabricate detections."
            )
            self.model = None
            return False

        try:
            from ultralytics import YOLO  # Dynamic import
            self.model = YOLO(str(self.model_path))
            self._status = ModelStatus.READY
            self._error_msg = None

            # Load auxiliary COCO model for high-precision produce & containers
            self.coco_model = None
            coco_candidates = [
                self.settings.root_dir / "models" / "detection" / "yolo11" / "coco_backup.pt",
                self.settings.root_dir / "yolo11n.pt",
                self.settings.root_dir / "models" / "detection" / "yolo11n.pt",
            ]
            for cand in coco_candidates:
                if cand.is_file():
                    try:
                        self.coco_model = YOLO(str(cand))
                        break
                    except Exception:
                        pass

            return True
        except ImportError:
            self._status = ModelStatus.MODEL_NOT_LOADED
            self._error_msg = (
                "The 'ultralytics' package is not installed. "
                "Install via: pip install ultralytics"
            )
            self.model = None
            return False
        except Exception as e:
            self._status = ModelStatus.ERROR
            self._error_msg = f"Failed to load YOLO model: {str(e)}"
            self.model = None
            return False

    @property
    def class_names(self) -> List[str]:
        """Returns the list of class names configured in the YOLO detector."""
        if self.model is not None and hasattr(self.model, "names") and self.model.names:
            return list(self.model.names.values())
        return []

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        conf_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """
        Run detection on input image.
        If model is not loaded, returns empty detections with MODEL_NOT_LOADED status.
        """
        t0 = time.perf_counter()

        # Validate image input
        try:
            np_img, pil_img = load_image(image)
        except Exception as e:
            return DetectionResult(
                status=ModelStatus.ERROR,
                detections=[],
                inference_time_ms=0.0,
                model_name="YOLO11FoodDetector",
                error_message=f"Invalid image input: {str(e)}",
            )

        h, w = np_img.shape[:2]
        shape = (h, w, 3)

        if not self.is_ready():
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DetectionResult(
                status=self._status,
                detections=[],
                image_shape=shape,
                inference_time_ms=elapsed_ms,
                model_name="YOLO11FoodDetector",
                error_message=self._error_msg or "Model weights not available.",
            )

        try:
            effective_conf = conf_threshold if conf_threshold is not None else self.conf_threshold

            # Ultralytics interprets numpy ndarray as BGR format.
            # load_image returns RGB, so convert to BGR to preserve correct color spaces.
            if np_img.ndim == 3 and np_img.shape[2] == 3:
                bgr_img = np_img[:, :, ::-1]
            else:
                bgr_img = np_img

            detections: List[FoodDetection] = []

            # 1. Primary Model Predictions (198 fine-grained food classes)
            results = self.model.predict(
                source=bgr_img,
                conf=effective_conf,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.device,
                max_det=self.max_detections,
                verbose=False,
            )

            if results and len(results) > 0:
                boxes = results[0].boxes
                names = results[0].names or {}
                is_coco_model = len(names) == 80 and names.get(0) == "person"

                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0].item())
                        cls_id = int(box.cls[0].item())
                        cls_name = names.get(cls_id, f"food_{cls_id}")

                        if is_coco_model and cls_name not in COCO_FOOD_CLASSES:
                            continue
                        if cls_name == "bowl":
                            cls_name = "food"

                        bbox_obj = BoundingBox(xyxy[0], xyxy[1], xyxy[2], xyxy[3]).clamp(w, h)
                        detections.append(
                            FoodDetection(
                                class_id=cls_id,
                                class_name=cls_name,
                                confidence=conf,
                                bbox=bbox_obj.to_list(),
                            )
                        )

            # 2. Auxiliary Pretrained Model Predictions (high-recall produce, meals & food containers)
            if self.coco_model is not None and self.coco_model is not self.model:
                try:
                    c_results = self.coco_model.predict(
                        source=bgr_img,
                        conf=max(0.12, effective_conf),
                        iou=self.iou_threshold,
                        imgsz=self.image_size,
                        device=self.device,
                        max_det=self.max_detections,
                        verbose=False,
                    )
                    if c_results and len(c_results) > 0:
                        c_boxes = c_results[0].boxes
                        c_names = c_results[0].names or {}
                        if c_boxes is not None:
                            for box in c_boxes:
                                xyxy = box.xyxy[0].tolist()
                                conf = float(box.conf[0].item())
                                cls_id = int(box.cls[0].item())
                                cls_name = c_names.get(cls_id, f"food_{cls_id}")

                                if cls_name not in COCO_FOOD_CLASSES:
                                    continue
                                if cls_name == "bowl":
                                    cls_name = "food"

                                bbox_obj = BoundingBox(xyxy[0], xyxy[1], xyxy[2], xyxy[3]).clamp(w, h)
                                new_box = bbox_obj.to_list()

                                # Overlap deduplication via IoU
                                overlap = False
                                for existing in detections:
                                    ex_b = existing.bbox
                                    xi1 = max(new_box[0], ex_b[0])
                                    yi1 = max(new_box[1], ex_b[1])
                                    xi2 = min(new_box[2], ex_b[2])
                                    yi2 = min(new_box[3], ex_b[3])
                                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                                    box_area1 = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
                                    box_area2 = (ex_b[2] - ex_b[0]) * (ex_b[3] - ex_b[1])
                                    union_area = max(1.0, box_area1 + box_area2 - inter_area)
                                    iou = inter_area / union_area
                                    min_area = min(box_area1, box_area2)
                                    iom = inter_area / min_area if min_area > 0 else 0.0

                                    if iou > 0.15 or iom > 0.20:
                                        overlap = True
                                        if existing.class_name in ("food", "others", "other_food") and cls_name not in ("food", "others", "other_food"):
                                            existing.class_name = cls_name
                                            existing.confidence = max(existing.confidence, conf)
                                            existing.bbox = new_box
                                        elif conf > existing.confidence:
                                            existing.confidence = conf
                                        break

                                if not overlap:
                                    detections.append(
                                        FoodDetection(
                                            class_id=cls_id,
                                            class_name=cls_name,
                                            confidence=conf,
                                            bbox=new_box,
                                        )
                                    )
                except Exception as e:
                    pass

            # Merge sub-parts and connected components of single food items
            detections = merge_subpart_and_duplicate_boxes(detections, w, h)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DetectionResult(
                status=ModelStatus.READY,
                detections=detections,
                image_shape=shape,
                inference_time_ms=elapsed_ms,
                model_name="YOLO11FoodDetector",
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DetectionResult(
                status=ModelStatus.ERROR,
                detections=[],
                image_shape=shape,
                inference_time_ms=elapsed_ms,
                model_name="YOLO11FoodDetector",
                error_message=f"Detection inference failed: {str(e)}",
            )

    def detect(self, image: Union[str, Path, np.ndarray, Image.Image]) -> DetectionResult:
        """Alias for predict conforming to the detect(image) specification."""
        return self.predict(image)

    def detect_and_crop(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        output_dir: Union[str, Path] = "outputs/detection/crops",
        source_image_name: Optional[str] = None,
        padding_ratio: float = 0.0,
    ) -> Tuple[DetectionResult, List[Dict[str, Any]]]:
        """Runs detection and generates food-only crop images with metadata."""
        from src.detection.crop_utils import save_detection_crops
        res = self.predict(image)
        crops = []
        if res.is_success and res.detections:
            crops = save_detection_crops(
                image,
                res.detections,
                output_dir=output_dir,
                source_image_name=source_image_name,
                padding_ratio=padding_ratio,
            )
        return res, crops

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "yolo11",
            "model_path": str(self.model_path) if self.model_path else None,
            "status": self._status.value,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "image_size": self.image_size,
            "max_detections": self.max_detections,
            "is_ready": self.is_ready(),
            "error_message": self._error_msg,
        }
