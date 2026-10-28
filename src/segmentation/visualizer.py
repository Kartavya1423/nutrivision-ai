"""
NutriVision AI — Segmentation Visualization Module
Phase 3: SAM 2 Food Instance Segmentation

Generates multi-panel comparison visualizations:
1. Original RGB input
2. YOLO11 Bounding Boxes
3. SAM 2 Binary Segmentation Masks
4. Color-Coded Mask Overlays
5. Isolated Food-Only RGBA Crops
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.segmentation.segmenter import SegmentationResult
from src.utils.image_ops import load_image


COLOR_PALETTE = [
    (230, 25, 75),    # Red
    (60, 180, 75),    # Green
    (255, 225, 25),   # Yellow
    (0, 130, 200),    # Blue
    (245, 130, 48),   # Orange
    (145, 30, 180),   # Purple
    (70, 240, 240),   # Cyan
    (240, 50, 230),   # Magenta
    (210, 245, 60),   # Lime
    (250, 190, 212),  # Pink
]


def draw_yolo_boxes(
    image: Image.Image,
    bboxes: List[List[float]],
    class_names: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
) -> Image.Image:
    """Draws colored bounding boxes with class and confidence labels on PIL image."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for idx, box in enumerate(bboxes):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        c_name = class_names[idx] if class_names and idx < len(class_names) else f"food_{idx+1}"
        conf = confidences[idx] if confidences and idx < len(confidences) else 1.0

        # Draw box outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label badge
        label = f"{c_name} {conf:.2f}"
        tw = len(label) * 7 + 8
        th = 16
        draw.rectangle([x1, max(0, y1 - th), x1 + tw, max(0, y1)], fill=color)
        draw.text((x1 + 4, max(0, y1 - th) + 1), label, fill="white")

    return canvas


def create_checkered_background(w: int, h: int, grid_size: int = 16) -> Image.Image:
    """Creates a light checkered pattern representing transparency."""
    bg = Image.new("RGB", (w, h), (240, 240, 240))
    draw = ImageDraw.Draw(bg)
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            if ((x // grid_size) + (y // grid_size)) % 2 == 1:
                draw.rectangle([x, y, min(w, x + grid_size), min(h, y + grid_size)], fill=(220, 220, 220))
    return bg


def render_isolated_food_composite(
    image_shape: Tuple[int, int],
    result: SegmentationResult,
    orig_np: np.ndarray,
) -> Image.Image:
    """Renders all segmented food items on a checkered transparent canvas."""
    h, w = image_shape[:2]
    checkered = create_checkered_background(w, h)
    checkered_np = np.array(checkered)

    for f_mask in result.masks:
        if f_mask.area_pixels == 0:
            continue
        m = f_mask.mask
        if m.shape[:2] == (h, w):
            checkered_np[m] = orig_np[m]

    return Image.fromarray(checkered_np)


def create_segmentation_comparison_panel(
    image: Union[str, Path, np.ndarray, Image.Image],
    result: SegmentationResult,
    bboxes: Optional[List[List[float]]] = None,
    class_names: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    panel_height: int = 320,
) -> Image.Image:
    """
    Creates an executive 5-panel visualization comparing:
    1. Input Image | 2. YOLO11 BBoxes | 3. Binary Masks | 4. Overlay | 5. Isolated Crops
    """
    orig_np, _ = load_image(image)
    h, w = orig_np.shape[:2]
    orig_pil = Image.fromarray(orig_np)

    # Calculate proportional panel width
    panel_w = int(w * (panel_height / h))
    size = (panel_w, panel_height)

    # 1. Original Panel
    p1 = orig_pil.resize(size, Image.Resampling.BILINEAR)

    # 2. YOLO Bounding Boxes Panel
    boxes_to_draw = bboxes or [m.prompt_bbox for m in result.masks if m.prompt_bbox]
    c_names = class_names or [m.class_name for m in result.masks]
    confs = confidences or [m.metadata.get("yolo_confidence", 1.0) for m in result.masks]
    p2 = draw_yolo_boxes(orig_pil, boxes_to_draw, c_names, confs).resize(size, Image.Resampling.BILINEAR)

    # 3. SAM 2 Binary / Colored Masks Panel
    mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, f_mask in enumerate(result.masks):
        if f_mask.area_pixels == 0:
            continue
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        m = f_mask.mask
        if m.shape[:2] == (h, w):
            mask_canvas[m] = color
    p3 = Image.fromarray(mask_canvas).resize(size, Image.Resampling.NEAREST)

    # 4. Color-Coded Overlay Panel
    overlay_np = result.render_overlay(orig_np, alpha=0.45)
    p4 = Image.fromarray(overlay_np).resize(size, Image.Resampling.BILINEAR)

    # 5. Isolated Food RGBA on Checkered Canvas
    isolated_food_pil = render_isolated_food_composite((h, w), result, orig_np)
    p5 = isolated_food_pil.resize(size, Image.Resampling.BILINEAR)

    # Assemble horizontally with header labels
    header_h = 32
    total_w = panel_w * 5 + 40  # 10px spacing
    total_h = panel_height + header_h + 20

    master = Image.new("RGB", (total_w, total_h), (25, 27, 33))
    draw = ImageDraw.Draw(master)

    panels = [
        ("1. Input Image", p1),
        ("2. YOLO11 BBoxes", p2),
        ("3. SAM 2 Masks", p3),
        ("4. Food Overlay", p4),
        ("5. Isolated Food (RGBA)", p5),
    ]

    for idx, (title, img) in enumerate(panels):
        x_offset = 10 + idx * (panel_w + 5)
        # Draw header text
        draw.text((x_offset + 5, 8), title, fill=(230, 230, 235))
        # Paste panel
        master.paste(img, (x_offset, header_h + 5))

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        master.save(p, quality=94)

    return master
