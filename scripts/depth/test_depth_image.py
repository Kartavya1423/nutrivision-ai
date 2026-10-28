"""
NutriVision AI — Single-Image Depth Anything V2 Inference Script
Executes real monocular depth estimation on input food images.
Saves raw numpy depth maps, normalized colorized visualizations, and JSON metadata.

Usage:
    python scripts/depth/test_depth_image.py --image data/raw/FoodSeg103/Images/img_dir/test/00000048.jpg
"""

import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.depth.depth_anything import DepthAnythingV2Estimator
from src.utils.image_ops import load_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Depth Anything V2 monocular depth estimation on a single food image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to Depth Anything V2 checkpoint (.pth).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Encoder architecture variant (default: vits).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Model inference input resolution (default: 518).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda' or 'cpu'). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/depth",
        help="Directory where depth outputs and metadata are stored.",
    )
    return parser.parse_args()


def test_depth_image(
    image_path: str,
    checkpoint: str = None,
    encoder: str = "vits",
    input_size: int = 518,
    device: str = None,
    output_dir: str = "outputs/depth",
):
    settings = get_settings()
    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    # Prepare output directories
    base_out = Path(output_dir)
    raw_dir = base_out / "raw"
    vis_dir = base_out / "visualizations"
    meta_dir = base_out / "metadata"

    raw_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================")
    print(f" NutriVision AI — Depth Anything V2 Inference")
    print(f"========================================================")
    print(f"Target Image:      {img_path.resolve()}")
    print(f"Encoder Variant:   {encoder}")
    print(f"Input Resolution:  {input_size}x{input_size}")

    # Initialize Depth Anything V2 Estimator
    estimator = DepthAnythingV2Estimator(
        model_path=checkpoint,
        encoder=encoder,
        device=device,
        input_size=input_size,
        settings=settings,
    )

    if not estimator.is_ready():
        raise RuntimeError(f"Depth model failed to initialize: {estimator.get_model_info()}")

    actual_device = estimator.device
    print(f"Active Device:     {actual_device} (verified)")
    print(f"Checkpoint:        {estimator.model_path}")

    # Load image to verify dimensions
    np_img, _ = load_image(img_path)
    h, w = np_img.shape[:2]
    print(f"Original Size:     {w} x {h} (WxH)")

    # Run inference
    result = estimator.estimate_depth(img_path)
    if not result.is_success:
        raise RuntimeError(f"Depth estimation failed: {result.error_message}")

    stem = img_path.stem
    raw_depth = result.depth_map

    # 1. Save raw float32 depth map array
    raw_save_path = raw_dir / f"{stem}_depth_raw.npy"
    np.save(raw_save_path, raw_depth)

    # 2. Save normalized colormap visualization
    vis_rgb = result.render_colormap()
    vis_save_path = vis_dir / f"{stem}_depth_vis.png"
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(vis_save_path), vis_bgr)

    # 3. Compile metadata
    metadata = {
        "source_image": str(img_path.resolve()),
        "model": "Depth Anything V2",
        "encoder": encoder,
        "checkpoint": str(estimator.model_path.resolve()) if estimator.model_path else None,
        "device": actual_device,
        "image_width": int(w),
        "image_height": int(h),
        "inference_time_ms": round(result.inference_time_ms, 2),
        "depth_type": result.depth_type.value,
        "is_metric_calibrated": result.is_metric_calibrated,
        "calibration_status": result.calibration_status,
        "depth_minimum": round(float(result.min_depth), 4),
        "depth_maximum": round(float(result.max_depth), 4),
        "depth_mean": round(float(result.mean_depth), 4),
        "depth_median": round(float(result.median_depth), 4),
        "percentiles": result.percentiles,
        "raw_depth_file": str(raw_save_path.resolve()),
        "visualization_file": str(vis_save_path.resolve()),
    }

    meta_save_path = meta_dir / f"{stem}_depth_metadata.json"
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"--------------------------------------------------------")
    print(f"Inference Time:    {result.inference_time_ms:.2f} ms")
    print(f"Depth Min / Max:   {result.min_depth:.4f} / {result.max_depth:.4f}")
    print(f"Depth Mean/Median: {result.mean_depth:.4f} / {result.median_depth:.4f}")
    print(f"Calibration State: {result.calibration_status}")
    print(f"Raw Output:        {raw_save_path}")
    print(f"Visualization:     {vis_save_path}")
    print(f"Metadata JSON:     {meta_save_path}")
    print(f"========================================================\n")

    return metadata


if __name__ == "__main__":
    args = parse_args()
    test_depth_image(
        image_path=args.image,
        checkpoint=args.checkpoint,
        encoder=args.encoder,
        input_size=args.input_size,
        device=args.device,
        output_dir=args.output_dir,
    )
