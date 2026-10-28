"""
NutriVision AI — Segmentation Mask to YOLO Bounding Box Converter
Phase 2: YOLO11 Food Detection Pipeline

Converts pixel-level segmentation masks (FoodSeg103, UECFoodPixComplete)
into spatial bounding boxes normalized to the YOLO detection format:
    <class_id> <x_center> <y_center> <width> <height>

Applies connected-component analysis to distinguish spatially separated food items
without fabricating instance annotations. All output labels are marked as
DERIVED_FROM_SEGMENTATION.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.detection.detection_config import MaskConversionConfig


@dataclass
class YOLOBox:
    """Represents a single normalized YOLO detection bounding box."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: Optional[float] = None
    native_class_id: Optional[int] = None
    pixel_area: int = 0
    is_clipped: bool = False

    def to_yolo_line(self) -> str:
        """Formats box as standard YOLO detection text label."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "native_class_id": self.native_class_id,
            "x_center": round(self.x_center, 6),
            "y_center": round(self.y_center, 6),
            "width": round(self.width, 6),
            "height": round(self.height, 6),
            "pixel_area": self.pixel_area,
            "is_clipped": self.is_clipped,
        }

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validates that coordinates are within [0, 1] and positive."""
        if not (0.0 <= self.x_center <= 1.0):
            return False, f"x_center out of bounds: {self.x_center}"
        if not (0.0 <= self.y_center <= 1.0):
            return False, f"y_center out of bounds: {self.y_center}"
        if not (0.0 < self.width <= 1.0):
            return False, f"width out of bounds: {self.width}"
        if not (0.0 < self.height <= 1.0):
            return False, f"height out of bounds: {self.height}"
        return True, None


@dataclass
class MaskConversionResult:
    """Outcome and diagnostics of a mask-to-YOLO conversion."""
    image_path: Path
    mask_path: Path
    boxes: List[YOLOBox] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    boxes_generated: int = 0
    invalid_boxes: int = 0
    clipped_boxes: int = 0
    is_empty_mask: bool = False
    classes_found: List[int] = field(default_factory=list)
    annotation_type: str = "derived"
    bbox_source: str = "derived_from_mask"
    instance_separation: str = "connected_components"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": str(self.image_path),
            "mask": str(self.mask_path),
            "image_size": [self.image_width, self.image_height],
            "boxes_generated": self.boxes_generated,
            "invalid_boxes": self.invalid_boxes,
            "clipped_boxes": self.clipped_boxes,
            "is_empty_mask": self.is_empty_mask,
            "classes_found": self.classes_found,
            "annotation_type": self.annotation_type,
            "bbox_source": self.bbox_source,
            "instance_separation": self.instance_separation,
            "boxes": [b.to_dict() for b in self.boxes],
            "error": self.error_message,
        }


class MaskToYOLOConverter:
    """
    Converts semantic segmentation masks to YOLO bounding-box annotations.
    """

    def __init__(self, config: Optional[MaskConversionConfig] = None):
        self.config = config or MaskConversionConfig()

    @staticmethod
    def load_mask_array(mask_path: Union[str, Path]) -> np.ndarray:
        """
        Loads a mask file into a 2D integer numpy array of class IDs.
        Handles single-channel grayscale (FoodSeg103) and RGBA (UECFoodPixComplete).
        """
        path = Path(mask_path)
        if not path.is_file():
            raise FileNotFoundError(f"Mask file not found: {path}")

        with Image.open(path) as img:
            arr = np.array(img)

        # Multi-channel handling: UECFoodPix stores class ID in channel 0
        if arr.ndim == 3:
            arr = arr[:, :, 0]

        return arr.astype(np.int32)

    @staticmethod
    def _extract_component_boxes(binary_mask: np.ndarray, connectivity: int = 8) -> List[Tuple[int, int, int, int, int]]:
        """
        Extracts bounding boxes (x_min, y_min, x_max, y_max, area) for each connected component.
        Uses cv2.connectedComponentsWithStats (C++ accelerated) or fallback.
        """
        try:
            import cv2
            u8 = binary_mask.astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                u8, connectivity=connectivity
            )
            boxes = []
            for i in range(1, num_labels):
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                area = int(stats[i, cv2.CC_STAT_AREA])
                boxes.append((x, y, x + w, y + h, area))
            return boxes
        except ImportError:
            pass

        try:
            from scipy.ndimage import label, generate_binary_structure
            struct = generate_binary_structure(2, 2 if connectivity == 8 else 1)
            labeled, num_features = label(binary_mask, structure=struct)
            boxes = []
            for comp_id in range(1, num_features + 1):
                coords = np.argwhere(labeled == comp_id)
                if len(coords) == 0:
                    continue
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0) + 1
                area = len(coords)
                boxes.append((int(x_min), int(y_min), int(x_max), int(y_max), area))
            return boxes
        except ImportError:
            coords = np.argwhere(binary_mask)
            if len(coords) == 0:
                return []
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1
            area = len(coords)
            return [(int(x_min), int(y_min), int(x_max), int(y_max), area)]

    def convert(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
        class_mapping: Optional[Dict[int, int]] = None,
    ) -> MaskConversionResult:
        """
        Converts a single image-mask pair to YOLO detection bounding boxes.

        Args:
            image_path: Path to RGB input image.
            mask_path: Path to segmentation mask PNG.
            class_mapping: Optional dict {native_class_id: detector_class_id}.
                           If None, native class IDs are preserved.

        Returns:
            MaskConversionResult with normalized YOLO boxes and diagnostics.
        """
        img_p = Path(image_path)
        msk_p = Path(mask_path)
        result = MaskConversionResult(image_path=img_p, mask_path=msk_p)

        try:
            with Image.open(img_p) as img:
                img_w, img_h = img.size
                result.image_width = img_w
                result.image_height = img_h
        except Exception as e:
            result.error_message = f"Failed reading image dimensions: {e}"
            return result

        try:
            mask_arr = self.load_mask_array(msk_p)
        except Exception as e:
            result.error_message = f"Failed loading mask array: {e}"
            return result

        mask_h, mask_w = mask_arr.shape[:2]
        if mask_w != img_w or mask_h != img_h:
            result.error_message = f"Dimension mismatch: image is ({img_w}, {img_h}), mask is ({mask_w}, {mask_h})."
            return result

        unique_classes = np.unique(mask_arr)
        # 0 is strictly background
        food_classes = [int(c) for c in unique_classes if int(c) != 0]
        result.classes_found = food_classes

        if not food_classes:
            result.is_empty_mask = True
            return result

        yolo_boxes: List[YOLOBox] = []

        for native_cls in food_classes:
            binary = (mask_arr == native_cls)
            detector_cls = class_mapping.get(native_cls, native_cls) if class_mapping else native_cls

            if self.config.use_connected_components:
                raw_boxes = self._extract_component_boxes(binary, connectivity=self.config.connectivity)
                for x_min, y_min, x_max, y_max, area in raw_boxes:
                    # Filter noise
                    if area < self.config.min_area_pixels:
                        continue
                    if (x_max - x_min) < self.config.min_dimension_pixels or (y_max - y_min) < self.config.min_dimension_pixels:
                        continue

                    # Check clipping / clamp
                    clipped = False
                    if x_min < 0 or y_min < 0 or x_max > img_w or y_max > img_h:
                        clipped = True
                        result.clipped_boxes += 1
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(img_w, x_max)
                        y_max = min(img_h, y_max)

                    # Normalized YOLO coordinates
                    bw = x_max - x_min
                    bh = y_max - y_min
                    xc = (x_min + x_max) / 2.0 / img_w
                    yc = (y_min + y_max) / 2.0 / img_h
                    nw = bw / img_w
                    nh = bh / img_h

                    box = YOLOBox(
                        class_id=detector_cls,
                        x_center=xc,
                        y_center=yc,
                        width=nw,
                        height=nh,
                        native_class_id=native_cls,
                        pixel_area=area,
                        is_clipped=clipped,
                    )

                    valid, err = box.validate()
                    if valid:
                        yolo_boxes.append(box)
                    else:
                        result.invalid_boxes += 1
            else:
                # Single box covering all pixels of class
                coords = np.argwhere(binary)
                if len(coords) == 0:
                    continue
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0) + 1
                area = len(coords)

                if area < self.config.min_area_pixels:
                    continue

                bw = x_max - x_min
                bh = y_max - y_min
                xc = (x_min + x_max) / 2.0 / img_w
                yc = (y_min + y_max) / 2.0 / img_h
                nw = bw / img_w
                nh = bh / img_h

                box = YOLOBox(
                    class_id=detector_cls,
                    x_center=xc,
                    y_center=yc,
                    width=nw,
                    height=nh,
                    native_class_id=native_cls,
                    pixel_area=area,
                )
                valid, _ = box.validate()
                if valid:
                    yolo_boxes.append(box)
                else:
                    result.invalid_boxes += 1

        result.boxes = yolo_boxes
        result.boxes_generated = len(yolo_boxes)
        return result

    @staticmethod
    def write_yolo_label_file(output_path: Union[str, Path], boxes: List[YOLOBox]) -> Path:
        """
        Writes YOLO format text label file.
        Format per line: <class_id> <x_center> <y_center> <width> <height>
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        lines = [b.to_yolo_line() for b in boxes]
        with open(out_p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        return out_p
