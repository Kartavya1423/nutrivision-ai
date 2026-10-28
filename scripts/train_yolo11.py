#!/usr/bin/env python3
"""
NutriVision AI — YOLO11 Food Detection Training Script
Phase 2: YOLO11 Food Detection Pipeline

Validates YOLO dataset integrity, inspects hardware (CPU/CUDA),
loads YOLO11 (nano baseline), executes model training,
and persists checkpoints and training metrics to models/detection/yolo11/.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings


def detect_hardware() -> Dict[str, Any]:
    """Detects CPU and GPU acceleration capabilities."""
    import torch
    cuda_available = torch.cuda.is_available()
    info = {
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "CPU",
        "pytorch_version": torch.__version__,
        "recommended_device": "cuda:0" if cuda_available else "cpu",
    }
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
    return info


def validate_yolo_dataset(yaml_path: Path) -> Dict[str, Any]:
    """Validates presence and integrity of dataset.yaml and partition images."""
    if not yaml_path.is_file():
        raise FileNotFoundError(f"dataset.yaml not found at: {yaml_path}")

    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base_path = Path(data.get("path", yaml_path.parent))
    train_dir = base_path / data.get("train", "images/train")
    val_dir = base_path / data.get("val", "images/val")
    test_dir = base_path / data.get("test", "images/test")

    train_imgs = list(train_dir.glob("*.*")) if train_dir.is_dir() else []
    val_imgs = list(val_dir.glob("*.*")) if val_dir.is_dir() else []
    test_imgs = list(test_dir.glob("*.*")) if test_dir.is_dir() else []

    num_classes = len(data.get("names", {}))

    stats = {
        "yaml_path": str(yaml_path),
        "num_classes": num_classes,
        "classes": {int(k): str(v) for k, v in data.get("names", {}).items()},
        "train_images": len(train_imgs),
        "val_images": len(val_imgs),
        "test_images": len(test_imgs),
    }

    if stats["train_images"] == 0:
        raise ValueError(f"No training images found in: {train_dir}")
    if stats["val_images"] == 0:
        raise ValueError(f"No validation images found in: {val_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="NutriVision AI — Train YOLO11 Food Detector")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Initial YOLO11 checkpoint")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image dimension")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda:0)")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 recommended for CPU)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="models/detection/yolo11", help="Output directory")
    parser.add_argument("--name", type=str, default="phase2_fixed_run", help="Experiment name")
    parser.add_argument("--fraction", type=float, default=0.08, help="Fraction of dataset to train on")
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer (auto, SGD, AdamW)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="Warmup epochs")
    args = parser.parse_args()

    print("=" * 76)
    print("         NutriVision AI — YOLO11 Food Detector Training Engine           ")
    print("=" * 76)

    # 1. HARDWARE DETECTION
    hw = detect_hardware()
    target_device = args.device or hw["recommended_device"]
    print(f"\n[Step 1/5] Hardware Environment:")
    print(f"  -> PyTorch Version   : {hw['pytorch_version']}")
    print(f"  -> Acceleration      : {'CUDA (' + hw['device_name'] + ')' if hw['cuda_available'] else 'CPU Fallback'}")
    print(f"  -> Target Device     : {target_device}")

    # Set thread count on CPU for optimal throughput
    if not hw["cuda_available"]:
        import torch
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"  -> CPU Threads       : {num_threads}")

    # 2. DATASET VALIDATION
    dataset_yaml = PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "dataset.yaml"
    print(f"\n[Step 2/5] Validating YOLO Detection Dataset at {dataset_yaml.relative_to(PROJECT_ROOT)}...")
    ds_stats = validate_yolo_dataset(dataset_yaml)
    print(f"  -> Total Classes     : {ds_stats['num_classes']}")
    print(f"  -> Train Images      : {ds_stats['train_images']}")
    print(f"  -> Val Images        : {ds_stats['val_images']}")
    print(f"  -> Test Images       : {ds_stats['test_images']}")

    # 3. INITIALIZE ULTRALYTICS YOLO11
    print(f"\n[Step 3/5] Loading YOLO11 architecture ({args.model})...")
    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"  -> Model initialized: {model.model_name if hasattr(model, 'model_name') else args.model}")

    # 4. EXECUTE TRAINING
    frac_pct = int(args.fraction * 100) if args.fraction <= 1.0 else 100
    print(f"\n[Step 4/5] Initiating training (Run: {args.name}, Epochs: {args.epochs}, Batch: {args.batch_size}, ImageSize: {args.imgsz}, Sampling: {frac_pct}%)...")
    output_proj = PROJECT_ROOT / args.project
    output_proj.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train_results = model.train(
        data=str(dataset_yaml.resolve()),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        fraction=args.fraction,
        device=target_device,
        workers=args.workers,
        patience=args.patience,
        project=str(output_proj),
        name=args.name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
        seed=42,
        deterministic=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        warmup_epochs=args.warmup_epochs,
    )
    train_duration = round(time.time() - t0, 2)
    print(f"  -> Training completed in {train_duration} seconds.")

    # 5. LOCATE & LINK BEST/LAST CHECKPOINTS
    run_dir = output_proj / args.name
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    canonical_best = output_proj / "best.pt"
    canonical_last = output_proj / "last.pt"

    # Backup existing baseline checkpoint before updating canonical destination
    if canonical_best.is_file():
        backup_best = output_proj / "baseline_best_backup.pt"
        if not backup_best.is_file():
            import shutil
            shutil.copy2(canonical_best, backup_best)
            print(f"  -> Backed up previous baseline to: {backup_best.relative_to(PROJECT_ROOT)}")

    if best_pt.is_file():
        import shutil
        shutil.copy2(best_pt, canonical_best)
        print(f"  -> Saved best checkpoint : {canonical_best.relative_to(PROJECT_ROOT)}")
    elif last_pt.is_file():
        import shutil
        shutil.copy2(last_pt, canonical_best)
        print(f"  -> Saved last checkpoint : {canonical_best.relative_to(PROJECT_ROOT)}")

    if last_pt.is_file():
        import shutil
        shutil.copy2(last_pt, canonical_last)

    # Save training configuration and outcome summary
    summary = {
        "model": args.model,
        "epochs_completed": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.imgsz,
        "fraction": args.fraction,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "warmup_epochs": args.warmup_epochs,
        "device": target_device,
        "hardware": hw,
        "training_duration_seconds": train_duration,
        "run_directory": str(run_dir.relative_to(PROJECT_ROOT)),
        "best_checkpoint": str(canonical_best.relative_to(PROJECT_ROOT)) if canonical_best.is_file() else None,
        "run_best_checkpoint": str(best_pt.relative_to(PROJECT_ROOT)) if best_pt.is_file() else None,
    }

    config_json_path = run_dir / "training_config.json"
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also update top-level config for backward compatibility
    with open(output_proj / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save class names JSON
    with open(output_proj / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(ds_stats.get("classes", {i: f"class_{i}" for i in range(ds_stats["num_classes"])}), f, indent=2)
    with open(run_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(ds_stats.get("classes", {i: f"class_{i}" for i in range(ds_stats["num_classes"])}), f, indent=2)

    print(f"  -> Saved training config  : {config_json_path.relative_to(PROJECT_ROOT)}")
    print(f"  -> Saved class names      : {(output_proj / 'class_names.json').relative_to(PROJECT_ROOT)}")
    print("\n" + "=" * 76)
    print("         [SUCCESS] YOLO11 TRAINING COMPLETED!                           ")
    print("=" * 76)
    print(f"  Run Directory    : {run_dir.relative_to(PROJECT_ROOT)}")
    print(f"  Model Checkpoint : {canonical_best.relative_to(PROJECT_ROOT)}")
    print(f"  Training Time    : {train_duration}s")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
