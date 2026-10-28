#!/usr/bin/env python3
"""
NutriVision AI — Single Image Freshness Inference CLI (Phase 6)
Runs food freshness condition assessment on a single food image or crop.

Usage:
    python scripts/freshness/test_freshness_image.py --image <IMAGE_PATH> [--threshold 0.60] [--save_viz]

Outputs:
- JSON predictions under outputs/freshness/predictions/
- Visual overlay under outputs/freshness/predictions/
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.freshness.freshness_detector import EfficientNetFreshnessDetector, FreshnessState


def render_prediction_overlay(
    image_path: Path,
    freshness: str,
    confidence: float,
    probabilities: dict,
    output_path: Path,
):
    """Generates an annotated visual overlay image showing freshness prediction and probabilities."""
    with Image.open(image_path) as orig:
        img = orig.convert("RGB")

    w, h = img.size
    # Color scheme: Green for fresh, Crimson for rotten, Amber for low-confidence/unknown
    if freshness.lower() == "fresh":
        badge_color = (34, 139, 34)  # Forest Green
        status_text = "FRESH"
    elif freshness.lower() in ("rotten", "spoiled"):
        badge_color = (220, 20, 60)  # Crimson
        status_text = "ROTTEN"
    else:
        badge_color = (218, 165, 32)  # Goldenrod
        status_text = "UNCERTAIN"

    # Create banner at bottom
    banner_height = max(70, int(h * 0.22))
    canvas = Image.new("RGB", (w, h + banner_height), (245, 247, 250))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # Banner header background
    draw.rectangle([(0, h), (w, h + banner_height)], fill=(20, 24, 33))

    # Freshness badge
    badge_w = max(110, int(w * 0.35))
    badge_h = max(32, int(banner_height * 0.45))
    badge_x = 12
    badge_y = h + int((banner_height - badge_h) / 2)
    draw.rounded_rectangle([(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)], radius=6, fill=badge_color)

    # Draw text
    draw.text(
        (badge_x + 12, badge_y + 6),
        f"{status_text} ({confidence*100:.1f}%)",
        fill=(255, 255, 255),
    )

    # Probabilities bar
    p_fresh = probabilities.get("fresh", 0.0)
    p_rotten = probabilities.get("rotten", 0.0)
    prob_text = f"P(Fresh): {p_fresh*100:.1f}% | P(Rotten): {p_rotten*100:.1f}%"
    draw.text(
        (badge_x + badge_w + 16, badge_y + 4),
        prob_text,
        fill=(220, 225, 235),
    )
    draw.text(
        (badge_x + badge_w + 16, badge_y + 22),
        "NutriVision AI • Visual Estimate",
        fill=(140, 150, 165),
    )

    canvas.save(output_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description="Run freshness detection on a food image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input food image.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to model checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold (default: 0.60).")
    parser.add_argument("--output_dir", type=str, default="outputs/freshness/predictions", help="Output directory.")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_absolute():
        img_path = PROJECT_ROOT / img_path

    if not img_path.is_file():
        print(f"[Error] Image file not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("      NutriVision AI — Food Freshness Condition Assessment      ")
    print("=" * 68)
    print(f"  Input Image : {img_path.name}")
    print(f"  Confidence  : >= {args.threshold:.2f}")

    # Load detector
    detector = EfficientNetFreshnessDetector(
        model_path=args.checkpoint,
        confidence_threshold=args.threshold,
    )

    if not detector.is_ready():
        print(f"\n[Error] Freshness model is not ready: {detector._error_msg}", file=sys.stderr)
        sys.exit(1)

    info = detector.get_model_info()
    print(f"  Model Type  : {info['model_type']}")
    print(f"  Device      : {info['device']}")
    print("-" * 68)

    # Run inference
    result = detector.assess_freshness(img_path)

    if not result.is_success:
        print(f"\n[Error] Inference failed: {result.error_message}", file=sys.stderr)
        sys.exit(1)

    eval_data = result.evaluation
    print(f"\n[Result]")
    print(f"  Predicted Condition : {eval_data.state.value.upper()}")
    print(f"  Confidence Score    : {eval_data.confidence*100:.2f}%")
    print(f"  Class Probabilities :")
    for cname, prob in eval_data.probabilities.items():
        print(f"    - {cname.capitalize():<8}: {prob*100:.2f}%")
    print(f"  Inference Latency   : {result.inference_time_ms:.1f} ms")
    print(f"  Notes               : {eval_data.notes}")

    # Save JSON result
    stem = img_path.stem
    json_path = out_dir / f"{stem}_freshness.json"
    result_dict = {
        "source_image": str(img_path.relative_to(PROJECT_ROOT)) if str(img_path).startswith(str(PROJECT_ROOT)) else str(img_path),
        "timestamp": datetime.now().isoformat(),
        "model_info": info,
        "freshness_result": result.to_dict(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    # Save visual overlay
    viz_path = out_dir / f"{stem}_prediction.png"
    render_prediction_overlay(
        image_path=img_path,
        freshness=eval_data.state.value,
        confidence=eval_data.confidence or 0.0,
        probabilities=eval_data.probabilities,
        output_path=viz_path,
    )

    print(f"\n[Outputs]")
    print(f"  -> Prediction JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"  -> Visual Overlay : {viz_path.relative_to(PROJECT_ROOT)}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
