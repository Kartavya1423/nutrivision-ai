"""
NutriVision AI — Food Region Cropping Utility
Phase 2: YOLO11 Food Detection Pipeline

Extracts food-only regions from bounding boxes for downstream
segmentation (SAM 2), classification, freshness, and volume estimation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.detection.detector import BoundingBox, FoodDetection
from src.utils.image_ops import load_image


def crop_bounding_box(
    image: Union[str, Path, np.ndarray, Image.Image],
    bbox: Union[BoundingBox, List[float], Tuple[float, float, float, float]],
    padding_ratio: float = 0.0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crops a rectangular bounding box region from an image.
    Clamps safely to image dimensions.

    Args:
        image: File path, numpy ndarray, or PIL Image.
        bbox: BoundingBox object or [x1, y1, x2, y2].
        padding_ratio: Fractional padding around the box (default 0.0).

    Returns:
        Tuple of (cropped numpy array in RGB, clamped (x1, y1, x2, y2)).
    """
    np_img, _ = load_image(image)
    h, w = np_img.shape[:2]

    if isinstance(bbox, BoundingBox):
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    else:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    # Apply padding if requested
    bw = x2 - x1
    bh = y2 - y1
    if padding_ratio > 0:
        pad_x = bw * padding_ratio
        pad_y = bh * padding_ratio
        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y

    # Clamp integers
    ix1 = max(0, min(w, int(round(x1))))
    iy1 = max(0, min(h, int(round(y1))))
    ix2 = max(0, min(w, int(round(x2))))
    iy2 = max(0, min(h, int(round(y2))))

    if ix2 <= ix1 or iy2 <= iy1:
        raise ValueError(
            f"Invalid crop dimensions after clamping: [{ix1}, {iy1}, {ix2}, {iy2}] "
            f"for image size ({w}, {h})."
        )

    cropped = np_img[iy1:iy2, ix1:ix2]
    return cropped, (ix1, iy1, ix2, iy2)


def save_detection_crops(
    image: Union[str, Path, np.ndarray, Image.Image],
    detections: List[FoodDetection],
    output_dir: Union[str, Path],
    source_image_name: Optional[str] = None,
    padding_ratio: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Crops all detected food objects from an image and persists them
    with metadata for Phase 3 (SAM2 / Classification / Nutrition).

    Returns:
        List of metadata dictionaries for each crop created.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source_image_name is None:
        if isinstance(image, (str, Path)):
            source_image_name = Path(image).stem
        else:
            source_image_name = "frame"

    results: List[Dict[str, Any]] = []

    for idx, det in enumerate(detections, start=1):
        try:
            cropped_np, clamped_coords = crop_bounding_box(
                image, det.bbox, padding_ratio=padding_ratio
            )
            crop_filename = f"{source_image_name}_food_{idx:02d}_{det.class_name}.jpg"
            crop_path = out_dir / crop_filename

            # Save JPEG crop
            crop_pil = Image.fromarray(cropped_np)
            crop_pil.save(crop_path, "JPEG", quality=95)

            meta = {
                "source_image": str(source_image_name),
                "crop_id": idx,
                "crop_file": str(crop_path.name),
                "crop_path": str(crop_path),
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": round(float(det.confidence), 4),
                "original_bbox": [round(float(c), 2) for c in det.bbox],
                "clamped_bbox": list(clamped_coords),
                "crop_width": cropped_np.shape[1],
                "crop_height": cropped_np.shape[0],
            }

            # Write sidecar JSON
            meta_path = out_dir / f"{source_image_name}_food_{idx:02d}_{det.class_name}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            results.append(meta)
        except Exception as e:
            # Skip invalid crop and log
            continue

    return results
