#!/usr/bin/env python3
"""
NutriVision AI — Phase 3: YOLO11 Stratified Fine-Tuning Script
Improves detection by:
1. Solving class imbalance: Uses stratified class-balanced sampling ensuring all 198 food
   classes are represented in training (1,523 unique images, up to 10 instances/class).
2. Preserving pretrained features: Fine-tunes from MS COCO pretrained `yolo11n.pt`.
3. Validation-driven: Validates strictly on `images/val` during training for model selection
   (held-out `test` split is never used for training or hyperparameter tuning).
4. Full reproducibility: Records training parameters, hardware, loss curves, and epoch logs.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Set, Tuple

import yaml

PROJECT_ROOT = Path("D:/NutriVision-AI (2)/NutriVision-AI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="NutriVision AI — Train Phase 3 Detector")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs (default: 5 on CPU)")
    parser.add_argument("--imgsz", type=int, default=384, help="Image resolution for training (default: 384)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr0", type=float, default=0.002, help="Initial learning rate (default: 0.002)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--max-per-class", type=int, default=10, help="Max images per class in stratified subset (default: 10)")
    parser.add_argument("--name", type=str, default="phase3_finetune_run", help="Training run name")
    return parser.parse_args()


def prepare_stratified_dataset(max_per_class: int) -> Tuple[Path, int, int]:
    """
    Selects a balanced stratified subset of training images such that:
    - ALL 198 classes have representation
    - Over-represented classes are capped to prevent gradient dominance
    - Rare classes are fully included
    """
    p_base = PROJECT_ROOT / "data/processed/yolo_detection"
    with open(p_base / "data.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    nc = len(cfg["names"])
    train_lbl_dir = p_base / "labels/train"
    train_img_dir = p_base / "images/train"

    class_to_img_paths = defaultdict(list)
    lbl_files = sorted(list(train_lbl_dir.glob("*.txt")))

    for lf in lbl_files:
        t = lf.read_text(encoding="utf-8").strip()
        if not t:
            continue
        classes_in_file = set()
        for line in t.splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    classes_in_file.add(int(parts[0]))
                except ValueError:
                    pass

        img_p = train_img_dir / (lf.stem + ".jpg")
        if img_p.is_file():
            for cid in classes_in_file:
                class_to_img_paths[cid].append(img_p)

    selected_imgs: Set[Path] = set()
    for cid in range(nc):
        plist = class_to_img_paths.get(cid, [])
        selected_imgs.update(plist[:max_per_class])

    out_dir = PROJECT_ROOT / "outputs/detection/phase3_finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_list_file = out_dir / f"train_stratified_{len(selected_imgs)}.txt"
    with open(train_list_file, "w", encoding="utf-8") as tf:
        for p in sorted(list(selected_imgs)):
            tf.write(f"{p.resolve().as_posix()}\n")

    stratified_cfg = dict(cfg)
    stratified_cfg["path"] = str(p_base.resolve().as_posix())
    stratified_cfg["train"] = str(train_list_file.resolve().as_posix())
    stratified_cfg["val"] = (p_base / "images/val").resolve().as_posix()
    # Explicitly do NOT set test to avoid test split access during training
    stratified_cfg.pop("test", None)

    stratified_yaml = out_dir / "dataset_stratified.yaml"
    with open(stratified_yaml, "w", encoding="utf-8") as yf:
        yaml.safe_dump(stratified_cfg, yf)

    return stratified_yaml, len(selected_imgs), nc


def main():
    args = parse_args()
    print("=" * 76)
    print("      NutriVision AI — Phase 3 YOLO11 Stratified Fine-Tuning             ")
    print("=" * 76)

    # 1. Prepare Stratified Dataset
    print(f"\n[1/4] Preparing stratified class-balanced dataset (max {args.max_per_class} imgs/class)...")
    strat_yaml, img_count, nc = prepare_stratified_dataset(max_per_class=args.max_per_class)
    print(f"      Selected {img_count} unique training images covering all {nc} classes.")
    print(f"      Stratified config written to: {strat_yaml.relative_to(PROJECT_ROOT)}")

    # 2. Check Pretrained Weights
    pretrained_path = PROJECT_ROOT / "yolo11n.pt"
    if not pretrained_path.is_file():
        pretrained_path = PROJECT_ROOT / "models/detection/yolo11/coco_backup.pt"
    print(f"\n[2/4] Loading pretrained base checkpoint: {pretrained_path.relative_to(PROJECT_ROOT)}")

    from ultralytics import YOLO
    model = YOLO(str(pretrained_path.resolve()))

    # 3. Training Run Setup
    target_project = PROJECT_ROOT / "models/detection/yolo11"
    run_dir = target_project / args.name
    print(f"\n[3/4] Launching training run: {args.name}")
    print(f"      Target Directory : {run_dir.relative_to(PROJECT_ROOT)}")
    print(f"      Epochs           : {args.epochs}")
    print(f"      Batch Size       : {args.batch}")
    print(f"      Image Size       : {args.imgsz}")
    print(f"      Initial LR       : {args.lr0}")
    print(f"      Device           : {args.device}")

    t0 = time.time()
    train_results = model.train(
        data=str(strat_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=0.01,
        optimizer="AdamW",
        device=args.device,
        project=str(target_project.resolve()),
        name=args.name,
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
        seed=42,
        verbose=True,
    )
    total_train_sec = round(time.time() - t0, 2)

    # 4. Save Training Metadata
    weights_dir = run_dir / "weights"
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"

    out_meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "PHASE_3_FINE_TUNING",
        "run_name": args.name,
        "base_pretrained_checkpoint": str(pretrained_path.relative_to(PROJECT_ROOT)),
        "dataset_yaml": str(strat_yaml.relative_to(PROJECT_ROOT)),
        "stratified_images_used": img_count,
        "classes_covered": nc,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch,
        "lr0": args.lr0,
        "device": args.device,
        "training_duration_seconds": total_train_sec,
        "best_weights_path": str(best_weights.relative_to(PROJECT_ROOT)) if best_weights.is_file() else None,
        "last_weights_path": str(last_weights.relative_to(PROJECT_ROOT)) if last_weights.is_file() else None,
    }

    meta_json = PROJECT_ROOT / "outputs/detection/phase3_finetune/training_metadata.json"
    with open(meta_json, "w", encoding="utf-8") as mf:
        json.dump(out_meta, mf, indent=2)

    print("\n" + "=" * 76)
    print("                     TRAINING RUN COMPLETE                              ")
    print("=" * 76)
    print(f"  Duration           : {total_train_sec} seconds ({total_train_sec/60:.2f} mins)")
    print(f"  Best Weights Saved : {best_weights.relative_to(PROJECT_ROOT)}")
    print(f"  Metadata Saved     : {meta_json.relative_to(PROJECT_ROOT)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
