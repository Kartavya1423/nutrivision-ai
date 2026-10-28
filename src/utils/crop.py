"""
NutriVision AI — Food-Only Crop Module
Extracts isolated food regions using binary segmentation masks.
Removes irrelevant background, supports configurable padding, preserves image immutability,
and generates optional side-by-side or isolated food visualizations.
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.utils.image_ops import load_image


def extract_food_region(
    image: Union[str, Path, np.ndarray, Image.Image],
    mask: Union[np.ndarray, Any],
    padding: int = 0,
    background_fill: Tuple[int, ...] = (0, 0, 0, 0),
    return_rgba: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    save_visualization: bool = False,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Extract isolated food region by applying binary mask and cropping to bounding box.
    
    Args:
        image: Original image (path, numpy array RGB, or PIL Image).
        mask: 2D binary mask (numpy array bool/uint8 or FoodMask object).
        padding: Pixel padding around food bounding box (clamped to boundaries).
        background_fill: Color fill for background pixels (RGBA 4-tuple or RGB 3-tuple).
        return_rgba: If True, returns RGBA with transparent background.
                     If False, returns RGB with background filled with background_fill.
        save_path: Optional file path to save extracted crop.
        save_visualization: If True and save_path is provided, also saves a 3-panel
                            comparison (Original, Mask, Isolated Food).

    Returns:
        Tuple[np.ndarray, Image.Image]: (Extracted food array, PIL Image)
    """
    # 1. Standardize and copy image to ensure original is preserved
    orig_np, _ = load_image(image)
    h, w = orig_np.shape[:2]

    # 2. Extract and validate binary mask
    if hasattr(mask, "mask"):
        binary_mask = mask.mask
    else:
        binary_mask = mask

    if not isinstance(binary_mask, np.ndarray) or binary_mask.ndim != 2:
        raise ValueError(f"Mask must be a 2D NumPy array. Got: {type(binary_mask)}")

    if binary_mask.shape[:2] != (h, w):
        raise ValueError(
            f"Dimension mismatch: image shape is ({h}, {w}), but mask shape is {binary_mask.shape[:2]}"
        )

    binary_bool = binary_mask > 0

    # 3. Check for empty mask
    if not np.any(binary_bool):
        # Empty mask fallback: return 1x1 empty transparent patch
        empty_arr = np.zeros((1, 1, 4 if return_rgba else 3), dtype=np.uint8)
        empty_pil = Image.fromarray(empty_arr)
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            empty_pil.save(p)
        return empty_arr, empty_pil

    # 4. Compute bounding box coordinates
    rows = np.any(binary_bool, axis=1)
    cols = np.any(binary_bool, axis=0)
    y_min, y_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x_min, x_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    # 5. Apply padding with boundary clamping
    pad = max(0, int(padding))
    crop_y1 = max(0, y_min - pad)
    crop_y2 = min(h, y_max + pad + 1)
    crop_x1 = max(0, x_min - pad)
    crop_x2 = min(w, x_max + pad + 1)

    crop_h = crop_y2 - crop_y1
    crop_w = crop_x2 - crop_x1

    # Slice image and mask within padded bounding box
    img_slice = orig_np[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    mask_slice = binary_bool[crop_y1:crop_y2, crop_x1:crop_x2]

    # 6. Apply background removal
    if return_rgba:
        # RGBA canvas
        alpha_channel = np.zeros((crop_h, crop_w, 1), dtype=np.uint8)
        alpha_channel[mask_slice] = 255
        
        # Food pixels retain RGB, background gets background_fill RGB
        result_rgb = img_slice.copy()
        if len(background_fill) >= 3:
            fill_rgb = np.array(background_fill[:3], dtype=np.uint8)
            result_rgb[~mask_slice] = fill_rgb

        result_arr = np.concatenate([result_rgb, alpha_channel], axis=-1)
        result_pil = Image.fromarray(result_arr, mode="RGBA")
    else:
        # RGB canvas
        fill_rgb = np.array(background_fill[:3] if len(background_fill) >= 3 else (0, 0, 0), dtype=np.uint8)
        result_arr = img_slice.copy()
        result_arr[~mask_slice] = fill_rgb
        result_pil = Image.fromarray(result_arr, mode="RGB")

    # 7. Optional saving and visualization
    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        result_pil.save(out_p)

        if save_visualization:
            # Create a 3-panel visualization: Original with Box, Mask, Isolated Food
            vis_p = out_p.with_name(f"{out_p.stem}_viz.png")
            
            # Panel 1: Original patch
            p1 = img_slice.copy()
            # Panel 2: Mask patch (white on black)
            p2 = np.stack([(mask_slice.astype(np.uint8) * 255)] * 3, axis=-1)
            # Panel 3: Isolated food on solid dark gray background
            p3 = img_slice.copy()
            p3[~mask_slice] = (30, 30, 30)

            combined_vis = np.concatenate([p1, p2, p3], axis=1)
            Image.fromarray(combined_vis).save(vis_p)

    return result_arr, result_pil
