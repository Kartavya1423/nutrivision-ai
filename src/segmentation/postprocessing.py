"""
NutriVision AI — Mask Post-Processing Subsystem
Phase 3: SAM 2 Food Instance Segmentation

Performs safe, non-destructive post-processing on raw segmentation masks:
1. Clipping to image boundaries
2. Removing disconnected tiny noise artifacts using connected-components analysis
3. Filling small internal holes (e.g., texture gaps in rice, grains, or bread)
4. Constraining spurious mask leakage beyond the YOLO bounding box margin
"""

from typing import List, Optional
import cv2
import numpy as np


def clip_mask_to_image(mask: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    """Clips mask dimensions to match target image bounds."""
    h, w = mask.shape[:2]
    if (h, w) == (max_h, max_w):
        return mask
    clipped = np.zeros((max_h, max_w), dtype=mask.dtype)
    sh = min(h, max_h)
    sw = min(w, max_w)
    clipped[:sh, :sw] = mask[:sh, :sw]
    return clipped


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 32,
    keep_primary: bool = True,
) -> np.ndarray:
    """
    Eliminates isolated stray noise islands while keeping the primary food body.

    Args:
        mask: 2D boolean or uint8 mask.
        min_size: Minimum pixel count to retain a disconnected component.
        keep_primary: If True, always retains the largest non-background component.

    Returns:
        Cleaned 2D boolean mask.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return np.zeros_like(mask, dtype=bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=bool)

    # Label 0 is background. Find largest foreground label
    fg_sizes = stats[1:, cv2.CC_STAT_AREA]
    max_label = 1 + int(np.argmax(fg_sizes))

    cleaned = np.zeros_like(mask, dtype=bool)
    for lbl in range(1, num_labels):
        size = stats[lbl, cv2.CC_STAT_AREA]
        if (lbl == max_label and keep_primary) or size >= min_size:
            cleaned[labels == lbl] = True

    return cleaned


def fill_mask_holes(mask: np.ndarray, max_hole_area: int = 150) -> np.ndarray:
    """
    Fills internal pinhole gaps in solid food items (e.g., rice, steak, bread)
    without expanding outer object boundaries.

    Args:
        mask: 2D boolean mask.
        max_hole_area: Maximum enclosed hole area in pixels to fill.

    Returns:
        Hole-filled 2D boolean mask.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return mask.copy()

    h, w = mask_u8.shape[:2]
    # Inverted mask
    inv_mask = (1 - mask_u8).astype(np.uint8)

    # Connected components on background
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_mask, connectivity=8)
    if num_labels <= 1:
        return mask.copy()

    filled = mask.copy()

    # Identify background components that touch image boundary (true outer background)
    boundary_labels = set()
    boundary_labels.update(labels[0, :].tolist())
    boundary_labels.update(labels[h - 1, :].tolist())
    boundary_labels.update(labels[:, 0].tolist())
    boundary_labels.update(labels[:, w - 1].tolist())

    for lbl in range(1, num_labels):
        if lbl in boundary_labels:
            continue  # Truly outer background
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area <= max_hole_area:
            filled[labels == lbl] = True

    return filled


def constrain_mask_to_bbox(
    mask: np.ndarray,
    bbox: List[float],
    margin_ratio: float = 0.10,
) -> np.ndarray:
    """
    Safely constrains a mask to within a margin around the prompt bounding box,
    suppressing runaway leaks into unrelated areas of the scene.

    Args:
        mask: 2D boolean mask.
        bbox: Bounding box [x1, y1, x2, y2].
        margin_ratio: Tolerated expansion margin outside the box (default 10%).

    Returns:
        Constrained boolean mask.
    """
    h, w = mask.shape[:2]
    bx1, by1, bx2, by2 = bbox
    bw = bx2 - bx1
    bh = by2 - by1

    pad_x = bw * margin_ratio
    pad_y = bh * margin_ratio

    cx1 = max(0, int(np.floor(bx1 - pad_x)))
    cy1 = max(0, int(np.floor(by1 - pad_y)))
    cx2 = min(w, int(np.ceil(bx2 + pad_x)))
    cy2 = min(h, int(np.ceil(by2 + pad_y)))

    valid_region = np.zeros((h, w), dtype=bool)
    valid_region[cy1:cy2, cx1:cx2] = True

    return np.logical_and(mask, valid_region)


def postprocess_mask(
    mask: np.ndarray,
    prompt_bbox: Optional[List[float]] = None,
    min_component_size: int = 32,
    fill_holes: bool = True,
    max_hole_size: int = 150,
    constrain_to_box: bool = True,
    box_margin_ratio: float = 0.10,
) -> np.ndarray:
    """
    Applies the full post-processing pipeline to a raw candidate mask.

    Args:
        mask: Raw 2D boolean mask from SAM 2.
        prompt_bbox: Optional prompt bounding box [x1, y1, x2, y2].
        min_component_size: Minimum pixel threshold for isolated components.
        fill_holes: Whether to fill small internal cavities.
        max_hole_size: Maximum pixel area of internal cavities to fill.
        constrain_to_box: Whether to suppress leakage beyond box margin.
        box_margin_ratio: Margin around prompt box.

    Returns:
        Cleaned, contiguous 2D boolean mask.
    """
    if not isinstance(mask, np.ndarray) or not np.any(mask):
        return mask

    processed = mask > 0

    # 1. Constrain to bounding box margin if prompt provided
    if constrain_to_box and prompt_bbox is not None:
        processed = constrain_mask_to_bbox(processed, prompt_bbox, margin_ratio=box_margin_ratio)

    # 2. Eliminate disconnected noise islands
    if min_component_size > 0:
        processed = remove_small_components(processed, min_size=min_component_size, keep_primary=True)

    # 3. Fill small internal holes
    if fill_holes and max_hole_size > 0:
        processed = fill_mask_holes(processed, max_hole_area=max_hole_size)

    return processed
