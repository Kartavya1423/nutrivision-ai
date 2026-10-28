"""
NutriVision AI — Single Image Food Classification CLI (Phase 4)
Runs fine-grained food classification on an input food image or crop.

Usage:
    python scripts/test_efficientnet_image.py --image path/to/food.jpg
    python scripts/test_efficientnet_image.py --image path/to/crop.png --top_k 5 --save_json
"""

import argparse
import json
from pathlib import Path
import sys
import time

# Project root path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.efficientnet_classifier import EfficientNetB4Classifier
from src.config.settings import get_settings


def main():
    parser = argparse.ArgumentParser(description="Classify food image using EfficientNet-B4")
    parser.add_argument("--image", type=str, required=True, help="Path to input food crop/image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of Top-K candidates to display")
    parser.add_argument("--confidence_threshold", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--save_json", action="store_true", help="Save result to JSON file")
    parser.add_argument("--output_json", type=str, default=None, help="Custom output JSON path")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"[!] Error: Image file not found: {image_path}")
        sys.exit(1)

    # Initialize classifier
    classifier = EfficientNetB4Classifier(
        model_path=args.checkpoint,
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

    # Optional JSON output
    if args.save_json or args.output_json:
        out_path = Path(args.output_json) if args.output_json else image_path.with_suffix(".prediction.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res_dict = result.to_dict()
        res_dict["image_path"] = str(image_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, indent=2)
        print(f"[*] Prediction JSON saved to: {out_path}")


if __name__ == "__main__":
    main()
