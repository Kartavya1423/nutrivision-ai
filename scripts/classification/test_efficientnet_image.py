#!/usr/bin/env python3
"""
NutriVision AI — Single Image Food Classification CLI
Phase 4.5: Runs fine-grained food classification on an input food crop/image.
Loads best.pth, applies canonical 198-class mapping, computes top-1 and top-5
predictions with confidence scores, and saves visual & JSON predictions.

Usage:
    python scripts/classification/test_efficientnet_image.py --image path/to/food.jpg
    python scripts/classification/test_efficientnet_image.py --image path/to/crop.png --top_k 5 --save_json
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.efficientnet_classifier import EfficientNetB4Classifier
from src.config.settings import get_settings


def save_visual_prediction(image_path: Path, result: Any, out_path: Path):
    """Draws top prediction badge on image and saves visualization."""
    try:
        with Image.open(image_path) as im:
            orig = im.convert("RGB")
        w, h = orig.size

        # Create canvas with bottom banner for top-1 and top-5
        banner_h = 70
        canvas = Image.new("RGB", (w, h + banner_h), (15, 23, 42))
        canvas.paste(orig, (0, 0))

        draw = ImageDraw.Draw(canvas)
        out = result.result
        top1_text = f"Top-1: {out.class_name} ({out.confidence*100:.1f}%)"
        draw.text((12, h + 8), top1_text, fill=(52, 211, 153))

        cands = [f"{c['class_name']}: {c['confidence']*100:.0f}%" for c in out.top_k_candidates[1:4]]
        sub_text = "Top-2..4: " + " | ".join(cands)
        draw.text((12, h + 38), sub_text, fill=(148, 163, 184))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, quality=92)
    except Exception as e:
        print(f"Warning: Failed to save visual prediction: {e}")


def main():
    parser = argparse.ArgumentParser(description="Classify food image using EfficientNet-B4")
    parser.add_argument("--image", type=str, required=True, help="Path to input food crop/image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/classification/efficientnet_b4/best.pth",
        help="Path to checkpoint file",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of Top-K candidates to display")
    parser.add_argument("--confidence_threshold", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--save_json", action="store_true", default=True, help="Save result to JSON file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/classification/predictions",
        help="Output directory for predictions",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        cand = PROJECT_ROOT / args.image
        if cand.is_file():
            image_path = cand
        else:
            print(f"[!] Error: Image file not found: {args.image}")
            sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        cand = PROJECT_ROOT / args.checkpoint
        if cand.is_file():
            ckpt_path = cand

    classifier = EfficientNetB4Classifier(
        model_path=ckpt_path,
        confidence_threshold=args.confidence_threshold,
    )

    if not classifier.is_ready():
        print("[!] Classifier is NOT ready:")
        print(f"    Status: {classifier.status.value}")
        print(f"    Reason: {classifier._error_msg}")
        sys.exit(1)

    print(f"[*] Loaded EfficientNet-B4 model from: {classifier.model_path}")
    print(f"[*] Classifying image: {image_path}")

    # Run inference
    result = classifier.classify(image_path, top_k=args.top_k)

    print("\n=======================================================")
    print("Classification Result:")
    print("=======================================================")
    print(f"Status:             {result.status.value}")
    print(f"Confidence Status:  {result.confidence_status}")
    print(f"Inference Latency:  {result.inference_time_ms:.2f} ms")
    if result.breakdown_ms:
        b = result.breakdown_ms
        print(f"  - Preprocess:     {b['preprocess_ms']:.2f} ms")
        print(f"  - Forward Pass:   {b['forward_ms']:.2f} ms")
        print(f"  - Postprocess:    {b['postprocess_ms']:.2f} ms")

    if result.result:
        out = result.result
        print(f"\nTop-1 Prediction:")
        print(f"  food_name:   {out.class_name}")
        print(f"  confidence:  {out.confidence:.4f}")
        print(f"  class_id:    {out.class_id}")

        print(f"\nTop-{len(out.top_k_candidates)} Candidates:")
        for cand in out.top_k_candidates:
            bar = "█" * int(cand["confidence"] * 25)
            print(f"  [{cand['rank']}] {cand['class_name']:<30} {cand['confidence']:.4f} {bar}")

    print("=======================================================\n")

    # Save artifacts
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{image_path.stem}_pred.json"
    res_dict = result.to_dict()
    res_dict["image_path"] = str(image_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)
    print(f"[*] Prediction JSON saved to: {json_path.relative_to(PROJECT_ROOT)}")

    if result.result:
        vis_path = out_dir / f"{image_path.stem}_pred.jpg"
        save_visual_prediction(image_path, result, vis_path)
        print(f"[*] Prediction visual saved to: {vis_path.relative_to(PROJECT_ROOT)}")

    return result.to_dict()


if __name__ == "__main__":
    main()
