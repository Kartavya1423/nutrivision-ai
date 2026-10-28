"""
NutriVision AI — MobileNetV2 Food Classifier (Research Paper Architecture)
Based on: "Vision-Based Approach for Food Calories And Weight Estimation from Images"
Author: Sonu Kumar (Centurion University of Technology and Management)

Implements:
- MobileNetV2 ImageNet pretrained feature extractor
- Global Average Pooling -> Dense(128, ReLU) -> Dropout -> Dense(50, Softmax)
- Real-time logging to logs/top50_training.log
- Raw and ripe produce augmentations (tomato, mango, etc.)
- Held-out test evaluation tracking Accuracy, Top-5, and per-class metrics
"""

import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Setup logger for real-time progress
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "top50_training.log"

logger = logging.getLogger("top50_training")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(stream_handler)


PRIMARY_50_CLASSES = [
    # Fruits (12)
    "apple", "banana", "orange", "mango", "tomato", "strawberry", "grape", "watermelon",
    "pineapple", "lemon", "peach", "pear",
    # Vegetables (16)
    "potato", "carrot", "broccoli", "cucumber", "onion", "garlic", "corn", "mushroom",
    "cabbage", "cauliflower", "spinach", "lettuce", "chili", "eggplant", "zucchini", "asparagus",
    # Staples, Grains & Prepared Meals (14)
    "rice", "fried_rice", "pizza", "hamburger", "bread", "french_fries", "curry",
    "soup", "salad", "sushi", "dumpling", "pancake", "noodle", "pasta_noodle",
    # Proteins, Dairy, Snacks & Desserts (8)
    "egg", "chicken_duck", "beef", "fish", "cheese_butter", "cake", "chocolate", "ice_cream"
]


def augment_produce_variations(img: Image.Image, is_produce: bool = False) -> List[Image.Image]:
    """Generates variations including raw/green and ripe shifts for produce."""
    variations = [img]
    # Flips & Rotations
    variations.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    variations.append(img.rotate(15, expand=False))
    variations.append(img.rotate(-15, expand=False))
    # Lighting
    enhancer_b = ImageEnhance.Brightness(img)
    variations.append(enhancer_b.enhance(1.15))
    variations.append(enhancer_b.enhance(0.85))

    if is_produce:
        arr = np.array(img).astype(np.float32)
        # 1. Raw / Unripe green shift (boost green, reduce red)
        raw_arr = arr.copy()
        raw_arr[:, :, 1] = np.clip(raw_arr[:, :, 1] * 1.30, 0, 255)
        raw_arr[:, :, 0] = np.clip(raw_arr[:, :, 0] * 0.75, 0, 255)
        variations.append(Image.fromarray(raw_arr.astype(np.uint8)))

        # 2. Ripe / Rich color shift (boost red/yellow)
        ripe_arr = arr.copy()
        ripe_arr[:, :, 0] = np.clip(ripe_arr[:, :, 0] * 1.25, 0, 255)
        ripe_arr[:, :, 1] = np.clip(ripe_arr[:, :, 1] * 1.05, 0, 255)
        variations.append(Image.fromarray(ripe_arr.astype(np.uint8)))

    return variations


def extract_mobilenet_features(
    split_dir: Path,
    target_classes: List[str],
    backbone: nn.Module,
    pool: nn.Module,
    transform: T.Compose,
    batch_size: int = 32,
    augment: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts 1280-dim embeddings from MobileNetV2."""
    backbone.eval()
    all_features = []
    all_labels = []

    produce_set = {"tomato", "mango", "apple", "banana", "orange", "lemon", "pear", "peach", "strawberry"}
    class_to_idx = {name: i for i, name in enumerate(target_classes)}

    img_records = []
    for cname in target_classes:
        cdir = split_dir / cname
        if not cdir.is_dir():
            continue
        files = sorted(list(cdir.glob("*.jpg")) + list(cdir.glob("*.png")))
        cid = class_to_idx[cname]
        for f in files:
            img_records.append((f, cid, cname in produce_set))

    logger.info(f"  -> Processing {len(img_records)} images from {split_dir.name} (augment={augment})...")
    batch_tensors = []
    batch_labels = []

    t0 = time.time()
    for idx, (fpath, cid, is_prod) in enumerate(img_records):
        try:
            im = Image.open(fpath).convert("RGB")
            imgs = augment_produce_variations(im, is_produce=is_prod) if augment else [im]

            for im_var in imgs:
                batch_tensors.append(transform(im_var))
                batch_labels.append(cid)

                if len(batch_tensors) >= batch_size:
                    batch_stack = torch.stack(batch_tensors)
                    with torch.no_grad():
                        feats = pool(backbone(batch_stack)).flatten(1)
                    all_features.append(feats)
                    all_labels.extend(batch_labels)
                    batch_tensors = []
                    batch_labels = []
        except Exception:
            continue

    if batch_tensors:
        batch_stack = torch.stack(batch_tensors)
        with torch.no_grad():
            feats = pool(backbone(batch_stack)).flatten(1)
        all_features.append(feats)
        all_labels.extend(batch_labels)

    feat_tensor = torch.cat(all_features, dim=0)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)
    elapsed = time.time() - t0
    logger.info(f"  -> Extracted {feat_tensor.shape[0]} feature vectors of dim {feat_tensor.shape[1]} in {elapsed:.1f}s")
    return feat_tensor, label_tensor


def main():
    logger.info("=" * 76)
    logger.info("  NutriVision AI — MobileNetV2 50-Class Food Recognition Training   ")
    logger.info("  Based on Centurion University Research Paper by Sonu Kumar        ")
    logger.info("=" * 76)

    target_classes = PRIMARY_50_CLASSES
    num_classes = len(target_classes)
    class_to_idx = {name: i for i, name in enumerate(target_classes)}
    idx_to_class = {i: name for i, name in enumerate(target_classes)}

    logger.info(f"Target classes count: {num_classes}")
    logger.info(f"Key produce: apple={class_to_idx['apple']}, mango={class_to_idx['mango']}, tomato={class_to_idx['tomato']}")

    # 1. Initialize MobileNetV2 backbone
    logger.info("\n[*] Loading official ImageNet pretrained MobileNetV2 backbone...")
    weights = models.MobileNet_V2_Weights.DEFAULT
    mobilenet = models.mobilenet_v2(weights=weights)
    mobilenet.eval()

    backbone = mobilenet.features
    pool = nn.AdaptiveAvgPool2d((1, 1))

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = PROJECT_ROOT / "data" / "processed" / "classification"
    cache_dir = PROJECT_ROOT / "models" / "classification" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_cache = cache_dir / "mobilenet_train_top50.pt"
    val_cache = cache_dir / "mobilenet_val_top50.pt"
    test_cache = cache_dir / "mobilenet_test_top50.pt"

    if train_cache.is_file() and val_cache.is_file() and test_cache.is_file():
        logger.info("\n[*] Loading cached MobileNetV2 embeddings from disk...")
        train_d = torch.load(train_cache)
        val_d = torch.load(val_cache)
        test_d = torch.load(test_cache)
        X_train, y_train = train_d["features"], train_d["labels"]
        X_val, y_val = val_d["features"], val_d["labels"]
        X_test, y_test = test_d["features"], test_d["labels"]
    else:
        logger.info("\n[*] Extracting feature embeddings for 50 primary food classes...")
        X_train, y_train = extract_mobilenet_features(
            data_dir / "train", target_classes, backbone, pool, transform, batch_size=32, augment=True
        )
        X_val, y_val = extract_mobilenet_features(
            data_dir / "val", target_classes, backbone, pool, transform, batch_size=32, augment=False
        )
        X_test, y_test = extract_mobilenet_features(
            data_dir / "test", target_classes, backbone, pool, transform, batch_size=32, augment=False
        )

        torch.save({"features": X_train, "labels": y_train}, train_cache)
        torch.save({"features": X_val, "labels": y_val}, val_cache)
        torch.save({"features": X_test, "labels": y_test}, test_cache)
        logger.info("[*] Feature extraction complete and cached.")

    # 2. Build MobileNetV2 Classifier Head as specified in Research Paper (Section III-C)
    # Architecture: Global Average Pooling (1280) -> Dense(128, ReLU) -> Dropout -> Output(50)
    in_features = 1280
    head = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )

    class_counts = torch.bincount(y_train, minlength=num_classes).float()
    weights = torch.where(class_counts > 0, 1.0 / (class_counts ** 0.5), torch.tensor(1.0))
    weights = weights / weights.mean()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.02)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    epochs = 40
    best_val_acc = 0.0
    best_head_state = None

    logger.info(f"\n[*] Training MobileNetV2 classifier head for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            out = head(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            bs = by.size(0)
            total_loss += loss.item() * bs
            total_correct += (out.argmax(dim=1) == by).sum().item()
            total_samples += bs

        scheduler.step()
        train_acc = 100.0 * total_correct / total_samples

        # Validation
        head.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for bx, by in val_loader:
                out = head(bx)
                loss = criterion(out, by)
                bs = by.size(0)
                val_loss += loss.item() * bs
                val_correct += (out.argmax(dim=1) == by).sum().item()
                val_samples += bs

        val_acc = 100.0 * val_correct / val_samples
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            logger.info(
                f"Epoch [{epoch:02d}/{epochs}] "
                f"Train Loss: {total_loss/total_samples:.4f} | Train Acc: {train_acc:5.1f}% || "
                f"Val Loss: {val_loss/val_samples:.4f} | Val Acc: {val_acc:5.1f}%"
            )

    logger.info(f"\n[*] Training Finished. Best Validation Accuracy: {best_val_acc:.2f}%")

    # 3. Held-Out Test Evaluation
    head.load_state_dict(best_head_state)
    head.eval()
    test_correct = 0
    test_top5 = 0
    test_samples = 0
    per_class_cor = {c: 0 for c in range(num_classes)}
    per_class_tot = {c: 0 for c in range(num_classes)}

    with torch.no_grad():
        for bx, by in test_loader:
            out = head(bx)
            bs = by.size(0)
            test_samples += bs

            _, pred5 = out.topk(min(5, num_classes), dim=1)
            pred1 = pred5[:, 0]
            test_correct += (pred1 == by).sum().item()
            test_top5 += (pred5 == by.unsqueeze(1)).sum().item()

            for i in range(bs):
                lbl = by[i].item()
                per_class_tot[lbl] += 1
                if pred1[i].item() == lbl:
                    per_class_cor[lbl] += 1

    test_acc = 100.0 * test_correct / test_samples
    test_top5_acc = 100.0 * test_top5 / test_samples

    logger.info("\n" + "=" * 60)
    logger.info("      HELD-OUT TEST SET EVALUATION REPORT (50 CLASSES)      ")
    logger.info("=" * 60)
    logger.info(f"Overall Test Accuracy:      {test_acc:.2f}%")
    logger.info(f"Overall Test Top-5 Accuracy: {test_top5_acc:.2f}%")
    logger.info("-" * 60)
    logger.info("Per-Class Accuracies for Target Produce & Key Dishes:")
    for food in ["tomato", "mango", "apple", "banana", "orange", "pizza", "rice", "bread", "potato", "salad"]:
        cid = class_to_idx.get(food)
        if cid is not None:
            cor = per_class_cor[cid]
            tot = per_class_tot[cid]
            acc = 100.0 * cor / tot if tot > 0 else 0.0
            logger.info(f"  {food.capitalize():<15}: {acc:5.1f}% ({cor}/{tot} correct)")

    # 4. Serialize Model
    mobilenet.classifier = head
    checkpoint_payload = {
        "architecture": "MobileNetV2",
        "num_classes": num_classes,
        "class_names": target_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "model_state_dict": mobilenet.state_dict(),
        "image_size": 224,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "val_metrics": {
            "val_accuracy": best_val_acc,
            "test_accuracy": test_acc,
            "test_top5": test_top5_acc,
        },
    }

    out_path = PROJECT_ROOT / "models" / "classification" / "mobilenet_v2_food50.pth"
    torch.save(checkpoint_payload, out_path)
    logger.info(f"\n[SUCCESS] Saved 50-Class MobileNetV2 checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
