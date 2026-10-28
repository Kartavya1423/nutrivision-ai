"""
NutriVision AI — Segmentation Artifacts Exporter
Phase 3: SAM 2 Food Instance Segmentation

Exports segmentation masks, colored overlays, isolated RGB crops,
transparent RGBA crops, and JSON sidecar metadata to outputs/segmentation/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
from PIL import Image

from src.config.settings import Settings, get_settings
from src.segmentation.segmenter import FoodMask, SegmentationResult
from src.utils.image_ops import load_image


def save_segmentation_artifacts(
    image: Union[str, Path, np.ndarray, Image.Image],
    result: SegmentationResult,
    source_name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """
    Saves all visual and structured artifacts produced by SAM 2.

    Args:
        image: Original input image.
        result: SegmentationResult containing FoodMask items.
        source_name: Base stem name for output files (e.g. 'foodseg103_00000001').
        output_dir: Base segmentation outputs directory.
        settings: Settings instance.

    Returns:
        Dictionary containing paths to all exported files.
    """
    cfg = settings or get_settings()
    base_out = Path(output_dir) if output_dir else cfg.outputs.segmentation
    
    # Subdirectories
    masks_dir = base_out / "masks"
    overlays_dir = base_out / "overlays"
    crops_dir = base_out / "crops"
    rgba_dir = base_out / "rgba"
    predictions_dir = base_out / "predictions"
    reports_dir = base_out / "reports"

    for d in (masks_dir, overlays_dir, crops_dir, rgba_dir, predictions_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    np_img, _ = load_image(image)
    stem = source_name or "food_sample"
    if "." in stem:
        stem = Path(stem).stem

    exported_files = {
        "source_name": stem,
        "masks": [],
        "crops": [],
        "rgba": [],
        "overlay": None,
        "metadata_json": None,
    }

    # 1. Save colored overlay if there are masks
    if result.masks:
        overlay_arr = result.render_overlay(np_img, alpha=0.45)
        overlay_path = overlays_dir / f"{stem}_overlay.jpg"
        Image.fromarray(overlay_arr).save(overlay_path, quality=92)
        exported_files["overlay"] = str(overlay_path)

    # 2. Save individual item artifacts
    item_metadata_list: List[Dict[str, Any]] = []

    for idx, f_mask in enumerate(result.masks):
        c_name = f_mask.class_name.replace(" ", "_").lower()
        prefix = f"{stem}_food_{idx+1:02d}_{c_name}"

        mask_path = masks_dir / f"{prefix}_mask.png"
        crop_rgb_path = crops_dir / f"{prefix}.jpg"
        crop_rgba_path = rgba_dir / f"{prefix}_rgba.png"

        # Save mask as 8-bit PNG (0 = bg, 255 = food)
        mask_u8 = f_mask.as_uint8()
        Image.fromarray(mask_u8).save(mask_path)
        exported_files["masks"].append(str(mask_path))

        # Save crops if generated
        if f_mask.crop_rgb is not None:
            Image.fromarray(f_mask.crop_rgb).save(crop_rgb_path, quality=92)
            exported_files["crops"].append(str(crop_rgb_path))

        if f_mask.crop_rgba is not None:
            Image.fromarray(f_mask.crop_rgba, mode="RGBA").save(crop_rgba_path)
            exported_files["rgba"].append(str(crop_rgba_path))

        # Build sidecar metadata dictionary
        item_meta = {
            "source_image": stem,
            "detection_id": idx + 1,
            "class_id": f_mask.class_id,
            "class_name": f_mask.class_name,
            "yolo_confidence": f_mask.metadata.get("yolo_confidence", 0.0),
            "sam_quality_score": round(float(f_mask.confidence), 4),
            "status": f_mask.status,
            "prompt_bbox": f_mask.prompt_bbox,
            "mask_tight_bbox": f_mask.bbox,
            "mask_area_pixels": f_mask.area_pixels,
            "relative_area": f_mask.relative_area,
            "selection_strategy": f_mask.selection_strategy,
            "output_paths": {
                "mask": str(mask_path),
                "crop_rgb": str(crop_rgb_path) if f_mask.crop_rgb is not None else None,
                "crop_rgba": str(crop_rgba_path) if f_mask.crop_rgba is not None else None,
            },
            "model_info": {
                "model_name": result.model_name,
                "device": result.device,
                "checkpoint": result.checkpoint,
            },
        }

        # Individual item JSON sidecar
        item_sidecar_path = reports_dir / f"{prefix}_meta.json"
        with open(item_sidecar_path, "w", encoding="utf-8") as f:
            json.dump(item_meta, f, indent=2)

        item_metadata_list.append(item_meta)

    # Master image metadata report
    master_report = {
        "source_image": stem,
        "total_detections": len(result.masks),
        "status": result.status_code,
        "inference_time_ms": round(result.inference_time_ms, 2),
        "model_name": result.model_name,
        "device": result.device,
        "checkpoint": result.checkpoint,
        "exported_files": exported_files,
        "items": item_metadata_list,
    }
    master_meta_path = reports_dir / f"{stem}_metadata.json"
    with open(master_meta_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    exported_files["metadata_json"] = str(master_meta_path)
    return exported_files
