"""
NutriVision AI — Visual Explainability (Grad-CAM) CLI Tool (Phase 7)
Usage:
    python scripts/explainability/test_gradcam.py --image data/processed/freshness/test/freshapples/freshapples_0006.png --model food
    python scripts/explainability/test_gradcam.py --image data/processed/freshness/test/freshapples/freshapples_0006.png --model freshness
    python scripts/explainability/test_gradcam.py --image data/processed/freshness/test/rottenapples/rottenapples_0006.png --model both
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.explainability.explainer import NutriVisionExplainer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visual explanations for food classifiers.")
    parser.add_argument("--image", type=str, required=True, help="Path to input food crop image.")
    parser.add_argument("--model", type=str, default="both", choices=["food", "freshness", "both"], help="Target model.")
    parser.add_argument("--target-class", type=str, default=None, help="Optional explicit target class name or index.")
    parser.add_argument("--colormap", type=str, default="JET", choices=["JET", "VIRIDIS", "TURBO", "INFERNO"], help="Colormap.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap blend transparency (0.0 to 1.0).")
    parser.add_argument("--output-dir", type=str, default="outputs/explainability/gradcam", help="Output directory.")
    return parser.parse_args()


def run_explanation(explainer, model_type, image_path, target_class, colormap, alpha, output_base):
    print(f"\n--- Running Grad-CAM for {model_type.upper()} ---")
    img_name = Path(image_path).stem

    if model_type == "food":
        sub_dir = output_base / "food_classification"
        res = explainer.explain_food_classification(
            image=image_path,
            target_class=target_class,
            colormap=colormap,
            alpha=alpha,
        )
    else:
        sub_dir = output_base / "freshness"
        res = explainer.explain_freshness(
            image=image_path,
            target_class=target_class,
            colormap=colormap,
            alpha=alpha,
        )

    if res.get("status") == "AVAILABLE":
        print(f" Status:           SUCCESS (AVAILABLE)")
        print(f" Target Layer:     {res['metadata']['target_layer']}")
        print(f" Predicted Class:  {res['predicted_class_name']} (conf: {res['confidence'] * 100:.2f}%)")
        print(f" Target Class:     {res['target_class_name']} (conf: {res['target_confidence'] * 100:.2f}%)")

        saved = explainer.save_artifacts(
            explanation_result=res,
            output_dir=sub_dir,
            file_prefix=f"{img_name}_{model_type}",
        )
        print(f" Artifacts Saved:")
        for k, v in saved.items():
            print(f"   • {k}: {v}")
        return True
    else:
        print(f" Status:           FAILED ({res.get('status')})")
        print(f" Error Code:       {res.get('error_code')}")
        print(f" Message:          {res.get('message')}")
        return False


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Error: Image not found at '{image_path}'")
        return 1

    output_base = root_dir / args.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"========================================================")
    print(f" NutriVision AI — Grad-CAM Explainability (Phase 7)")
    print(f"========================================================")
    print(f" Image:     {image_path.name}")
    print(f" Model:     {args.model}")
    print(f" Colormap:  {args.colormap} (alpha: {args.alpha})")
    print(f"--------------------------------------------------------")

    explainer = NutriVisionExplainer(colormap=args.colormap, alpha=args.alpha)

    success = True
    if args.model in ["food", "both"]:
        ok = run_explanation(explainer, "food", image_path, args.target_class, args.colormap, args.alpha, output_base)
        success = success and ok

    if args.model in ["freshness", "both"]:
        ok = run_explanation(explainer, "freshness", image_path, args.target_class, args.colormap, args.alpha, output_base)
        success = success and ok

    print(f"========================================================\n")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
