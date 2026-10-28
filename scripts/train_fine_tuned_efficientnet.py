"""
NutriVision AI — Enhanced Pretrained EfficientNet-B4 Training & Evaluation Script
Trains fine-grained food classifier on 198 canonical food classes using official ImageNet features.
Includes specific augmentations for raw and ripe produce (mango, tomato, etc.).
"""

import csv
import json
import os
from pathlib import Path
import random
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


def load_canonical_mapping(mapping_path: Path) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    class_names = []
    class_to_idx = {}
    idx_to_class = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            cname = row["class_name"].strip()
            class_names.append(cname)
            class_to_idx[cname] = cid
            idx_to_class[cid] = cname
    return class_names, class_to_idx, idx_to_class


def augment_image_variations(img: Image.Image, is_produce: bool = False) -> List[Image.Image]:
    """Generates variations including raw/green and ripe color shifts for produce."""
    variations = [img]
    
    # Horizontal flip
    variations.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    
    # Slight rotation
    variations.append(img.rotate(15, expand=False))
    variations.append(img.rotate(-15, expand=False))
    
    # Brightness / Contrast variations
    enhancer_b = ImageEnhance.Brightness(img)
    variations.append(enhancer_b.enhance(1.15))
    variations.append(enhancer_b.enhance(0.85))

    if is_produce:
        # Green / Raw shift variation
        arr = np.array(img).astype(np.float32)
        # Shift toward green for raw state
        raw_arr = arr.copy()
        raw_arr[:, :, 1] = np.clip(raw_arr[:, :, 1] * 1.25, 0, 255) # boost green
        raw_arr[:, :, 0] = np.clip(raw_arr[:, :, 0] * 0.80, 0, 255) # reduce red
        variations.append(Image.fromarray(raw_arr.astype(np.uint8)))

        # Saturated / Ripe shift variation
        ripe_arr = arr.copy()
        ripe_arr[:, :, 0] = np.clip(ripe_arr[:, :, 0] * 1.20, 0, 255) # boost red/yellow
        ripe_arr[:, :, 1] = np.clip(ripe_arr[:, :, 1] * 1.05, 0, 255)
        variations.append(Image.fromarray(ripe_arr.astype(np.uint8)))

    return variations


def extract_features_from_split(
    split_dir: Path,
    class_to_idx: Dict[str, int],
    backbone: nn.Module,
    pool: nn.Module,
    transform: T.Compose,
    batch_size: int = 32,
    augment: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts 1792-dim features for all images in the split directory."""
    backbone.eval()
    all_features = []
    all_labels = []

    produce_classes = {"tomato", "mango", "apple", "banana", "orange", "lemon", "pear", "peach"}

    img_records = []
    for cname, cid in class_to_idx.items():
        cdir = split_dir / cname
        if not cdir.is_dir():
            continue
        files = sorted(list(cdir.glob("*.jpg")) + list(cdir.glob("*.png")))
        for f in files:
            img_records.append((f, cid, cname in produce_classes))

    print(f"  -> Extracting features from {len(img_records)} image records (augment={augment})...")
    
    t0 = time.time()
    batch_tensors = []
    batch_labels = []

    for idx, (fpath, cid, is_prod) in enumerate(img_records):
        try:
            im = Image.open(fpath).convert("RGB")
            if augment:
                imgs_to_process = augment_image_variations(im, is_produce=is_prod)
            else:
                imgs_to_process = [im]

            for im_var in imgs_to_process:
                t = transform(im_var)
                batch_tensors.append(t)
                batch_labels.append(cid)

                if len(batch_tensors) >= batch_size:
                    batch_stack = torch.stack(batch_tensors)
                    with torch.no_grad():
                        feats = pool(backbone(batch_stack)).flatten(1)
                    all_features.append(feats)
                    all_labels.extend(batch_labels)
                    batch_tensors = []
                    batch_labels = []

        except Exception as e:
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
    print(f"  -> Extracted {feat_tensor.shape[0]} feature vectors of dim {feat_tensor.shape[1]} in {elapsed:.1f}s")
    return feat_tensor, label_tensor


def main():
    print("=" * 76)
    print("   NutriVision AI — Training Pretrained EfficientNet-B4 Food Classifier   ")
    print("=" * 76)

    # 1. Load canonical class mapping
    mapping_path = PROJECT_ROOT / "data" / "processed" / "classification" / "classification_class_mapping.csv"
    class_names, class_to_idx, idx_to_class = load_canonical_mapping(mapping_path)
    num_classes = len(class_names)
    print(f"[*] Loaded {num_classes} canonical food classes.")
    print(f"    Sample: apple={class_to_idx.get('apple')}, mango={class_to_idx.get('mango')}, tomato={class_to_idx.get('tomato')}")

    # 2. Build EfficientNet-B4 backbone
    print("\n[*] Initializing official pretrained EfficientNet-B4 backbone...")
    weights = models.EfficientNet_B4_Weights.DEFAULT
    full_model = models.efficientnet_b4(weights=weights)
    full_model.eval()

    backbone = full_model.features
    pool = full_model.avgpool

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Extract features for train, val, and test splits
    data_dir = PROJECT_ROOT / "data" / "processed" / "classification"
    cache_dir = PROJECT_ROOT / "models" / "classification" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_cache = cache_dir / "train_features.pt"
    val_cache = cache_dir / "val_features.pt"
    test_cache = cache_dir / "test_features.pt"

    if train_cache.is_file() and val_cache.is_file() and test_cache.is_file():
        print("\n[*] Loading cached features from disk...")
        train_data = torch.load(train_cache)
        val_data = torch.load(val_cache)
        test_data = torch.load(test_cache)
        X_train, y_train = train_data["features"], train_data["labels"]
        X_val, y_val = val_data["features"], val_data["labels"]
        X_test, y_test = test_data["features"], test_data["labels"]
    else:
        print("\n[*] Extracting training features (with raw/ripe produce augmentations)...")
        X_train, y_train = extract_features_from_split(
            data_dir / "train", class_to_idx, backbone, pool, transform, batch_size=32, augment=True
        )
        print("\n[*] Extracting validation features...")
        X_val, y_val = extract_features_from_split(
            data_dir / "val", class_to_idx, backbone, pool, transform, batch_size=32, augment=False
        )
        print("\n[*] Extracting held-out test features...")
        X_test, y_test = extract_features_from_split(
            data_dir / "test", class_to_idx, backbone, pool, transform, batch_size=32, augment=False
        )

        torch.save({"features": X_train, "labels": y_train}, train_cache)
        torch.save({"features": X_val, "labels": y_val}, val_cache)
        torch.save({"features": X_test, "labels": y_test}, test_cache)
        print("[*] Cached feature embeddings successfully.")

    # 4. Train classification head
    print(f"\n[*] Training classification head for {num_classes} classes...")
    in_features = 1792
    
    # Compute inverse class frequencies for balancing
    class_counts = torch.bincount(y_train, minlength=num_classes).float()
    weights = torch.where(class_counts > 0, 1.0 / (class_counts ** 0.5), torch.tensor(1.0))
    weights = weights / weights.mean()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=64, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=64, shuffle=False
    )

    # Multi-layer classification head with dropout
    head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    best_val_top1 = 0.0
    best_head_state = None
    epochs = 40

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_correct1 = 0
        total_correct5 = 0
        total_samples = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = head(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            bs = batch_y.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            _, pred5 = outputs.topk(5, dim=1, largest=True, sorted=True)
            pred1 = pred5[:, 0]
            total_correct1 += (pred1 == batch_y).sum().item()
            total_correct5 += (pred5 == batch_y.unsqueeze(1)).sum().item()

        scheduler.step()
        train_loss = total_loss / total_samples
        train_top1 = 100.0 * total_correct1 / total_samples
        train_top5 = 100.0 * total_correct5 / total_samples

        # Validation
        head.eval()
        val_loss = 0.0
        val_correct1 = 0
        val_correct5 = 0
        val_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = head(batch_x)
                loss = criterion(outputs, batch_y)
                bs = batch_y.size(0)
                val_loss += loss.item() * bs
                val_samples += bs

                _, pred5 = outputs.topk(5, dim=1, largest=True, sorted=True)
                val_correct1 += (pred5[:, 0] == batch_y).sum().item()
                val_correct5 += (pred5 == batch_y.unsqueeze(1)).sum().item()

        val_top1 = 100.0 * val_correct1 / val_samples
        val_top5 = 100.0 * val_correct5 / val_samples

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Top-1: {train_top1:5.1f}% | Top-5: {train_top5:5.1f}% || "
                f"Val Loss: {val_loss / val_samples:.4f} | Top-1: {val_top1:5.1f}% | Top-5: {val_top5:5.1f}%"
            )

    print(f"\n[*] Training Complete. Best Validation Top-1: {best_val_top1:.2f}%")

    # 5. Evaluate on Held-Out Test Set
    head.load_state_dict(best_head_state)
    head.eval()
    test_correct1 = 0
    test_correct5 = 0
    test_samples = 0
    per_class_correct = {c: 0 for c in range(num_classes)}
    per_class_total = {c: 0 for c in range(num_classes)}

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = head(batch_x)
            bs = batch_y.size(0)
            test_samples += bs

            _, pred5 = outputs.topk(5, dim=1, largest=True, sorted=True)
            test_correct1 += (pred5[:, 0] == batch_y).sum().item()
            test_correct5 += (pred5 == batch_y.unsqueeze(1)).sum().item()

            for i in range(bs):
                lbl = batch_y[i].item()
                per_class_total[lbl] += 1
                if pred5[i, 0].item() == lbl:
                    per_class_correct[lbl] += 1

    test_top1 = 100.0 * test_correct1 / test_samples
    test_top5 = 100.0 * test_correct5 / test_samples
    print("\n" + "=" * 50)
    print("      HELD-OUT TEST SET EVALUATION RESULTS       ")
    print("=" * 50)
    print(f"Overall Test Top-1 Accuracy: {test_top1:.2f}%")
    print(f"Overall Test Top-5 Accuracy: {test_top5:.2f}%")

    # Specific inspection for key produce
    for food in ["tomato", "mango", "apple", "banana", "orange"]:
        cid = class_to_idx.get(food)
        if cid is not None:
            tot = per_class_total.get(cid, 0)
            cor = per_class_correct.get(cid, 0)
            acc = 100.0 * cor / tot if tot > 0 else 0.0
            print(f"  -> {food.capitalize():<10} (class {cid}): {acc:5.1f}% ({cor}/{tot} correct)")

    # 6. Assemble complete model and save checkpoints
    print("\n[*] Packaging trained weights into EfficientNet-B4 architecture...")
    full_model.classifier = head

    checkpoint_payload = {
        "epoch": epochs,
        "model_state_dict": full_model.state_dict(),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": num_classes,
        "image_size": 224,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "training_config": {
            "backbone": "efficientnet_b4",
            "pretrained": True,
            "epochs": epochs,
            "optimizer": "AdamW",
            "lr": 1e-3,
        },
        "val_metrics": {
            "val_top1": best_val_top1,
            "test_top1": test_top1,
            "test_top5": test_top5,
        },
    }

    target_paths = [
        PROJECT_ROOT / "models" / "classification" / "efficientnet_b4" / "best.pth",
        PROJECT_ROOT / "models" / "classification" / "efficientnet_b4_food.pth",
    ]

    for p in target_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload, p)
        print(f"  -> Saved checkpoint to: {p}")

    print("\n[SUCCESS] Model training and serialization complete!")


if __name__ == "__main__":
    main()
