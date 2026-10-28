"""NutriVision AI — Production YOLO11 GPU Retraining Pipeline.

Configures and executes full-scale GPU training on 17,118 images across 198
unified food categories using Ultralytics YOLO11s / YOLO11n with reproducible
seeds, AdamW optimization, cosine learning rate scheduling, and held-out test evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
import yaml

# Authoritative project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.utils.torch_patch
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NutriVision AI YOLO11 GPU Retraining")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Pretrained base model (e.g. yolo11s.pt or yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=60, help="Total training epochs (default: 60)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution (default: 640)")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device identifier: 0 or cpu")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience (default: 12)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def verify_taxonomy(data_yaml_path: Path) -> list[str]:
    """Verify that data.yaml contains 198 unified classes in correct order."""
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names", {})
    if len(names) != 198:
        raise ValueError(f"Taxonomy mismatch: expected 198 classes, found {len(names)}")

    if names.get(27) != "banana":
        raise ValueError(f"Taxonomy error: class 27 must be 'banana', found '{names.get(27)}'")

    if names.get(23) != "apple":
        raise ValueError(f"Taxonomy error: class 23 must be 'apple', found '{names.get(23)}'")

    print(f"[PASS] Pre-flight taxonomy check verified: 198 unified classes aligned.")
    return [names[i] for i in range(198)]


def run_training():
    args = parse_args()
    print("=" * 80)
    print("         NutriVision AI — Production YOLO11 GPU Retraining Pipeline         ")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device} (CUDA Available: {torch.cuda.is_available()})")
    if torch.cuda.is_available() and args.device != "cpu":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_yaml = PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "data.yaml"
    class_names = verify_taxonomy(data_yaml)

    from ultralytics import YOLO

    output_dir = PROJECT_ROOT / "models" / "detection" / "yolo11" / "production"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Initializing pretrained base model: {args.model}...")
    model = YOLO(args.model)

    training_params = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": True,
        "box": 7.5,
        "cls": 1.2,
        "dfl": 1.5,
        "label_smoothing": 0.05,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "mosaic": 1.0,
        "mixup": 0.15,
        "project": str(output_dir),
        "name": "gpu_training_run",
        "save": True,
        "exist_ok": True,
        "verbose": True
    }

    print(f"\n[2/4] Starting training for {args.epochs} epochs...")
    t0 = time.time()
    results = model.train(**training_params)
    train_duration = time.time() - t0
    print(f"\n[3/4] Training completed in {train_duration:.2f}s ({train_duration/60:.2f} min).")

    # Locate generated best.pt
    run_weights_best = output_dir / "gpu_training_run" / "weights" / "best.pt"
    run_weights_last = output_dir / "gpu_training_run" / "weights" / "last.pt"

    prod_best = output_dir / "best.pt"
    prod_last = output_dir / "last.pt"

    if run_weights_best.exists():
        shutil.copy2(run_weights_best, prod_best)
        print(f"Installed production checkpoint: {prod_best}")
    if run_weights_last.exists():
        shutil.copy2(run_weights_last, prod_last)

    # Evaluate on held-out test split
    print("\n[4/4] Evaluating best checkpoint on held-out test split (3,135 images)...")
    eval_model = YOLO(str(prod_best))
    test_metrics = eval_model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, device=args.device)

    print("\n" + "=" * 80)
    print("                     TRAINING & EVALUATION SUMMARY                          ")
    print("=" * 80)
    print(f"Validation mAP@50:     {test_metrics.box.map50:.4f} ({test_metrics.box.map50*100:.2f}%)")
    print(f"Validation mAP@50-95:  {test_metrics.box.map:.4f} ({test_metrics.box.map*100:.2f}%)")
    print(f"Validation Precision:  {test_metrics.box.p:.4f} ({test_metrics.box.p*100:.2f}%)")
    print(f"Validation Recall:     {test_metrics.box.r:.4f} ({test_metrics.box.r*100:.2f}%)")

    # Export report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_architecture": args.model,
        "epochs_trained": args.epochs,
        "duration_seconds": round(train_duration, 2),
        "device": args.device,
        "batch_size": args.batch,
        "image_size": args.imgsz,
        "optimizer": "AdamW",
        "held_out_test_metrics": {
            "mAP50": round(float(test_metrics.box.map50), 4),
            "mAP50_95": round(float(test_metrics.box.map), 4),
            "precision": round(float(test_metrics.box.p), 4),
            "recall": round(float(test_metrics.box.r), 4)
        }
    }
    report_path = output_dir / "training_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation metrics to: {report_path}")


if __name__ == "__main__":
    run_training()
