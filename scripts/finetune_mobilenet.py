import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

class FoodDataset(Dataset):
    def __init__(self, root_dir: Path, classes: List[str], transform=None, is_train=False):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        
        for cname in classes:
            cdir = root_dir / cname
            if not cdir.is_dir():
                continue
            cid = self.class_to_idx[cname]
            files = sorted(list(cdir.glob("*.jpg")) + list(cdir.glob("*.png")))
            for f in files:
                self.samples.append((f, cid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    print("=" * 70)
    print("  Fine-Tuning MobileNetV2 (Backbone + Head) on Food Dataset")
    print("=" * 70)

    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_dir = PROJECT_ROOT / "data" / "processed" / "classification"
    train_ds = FoodDataset(data_dir / "train", PRIMARY_50_CLASSES, transform=train_tf, is_train=True)
    val_ds = FoodDataset(data_dir / "val", PRIMARY_50_CLASSES, transform=val_tf)
    test_ds = FoodDataset(data_dir / "test", PRIMARY_50_CLASSES, transform=val_tf)

    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test across 50 classes")

    # Balanced class weights for balanced learning
    class_counts = [0] * len(PRIMARY_50_CLASSES)
    for _, cid in train_ds.samples:
        class_counts[cid] += 1
    total_samples = len(train_ds)
    weights = [total_samples / max(1, c * len(PRIMARY_50_CLASSES)) for c in class_counts]
    class_weights_t = torch.tensor(weights, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load MobileNetV2
    weights_pre = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights_pre)

    # Unfreeze last 5 feature blocks (13, 14, 15, 16, 17, 18)
    for p in model.features.parameters():
        p.requires_grad = False
    for i in range(13, len(model.features)):
        for p in model.features[i].parameters():
            p.requires_grad = True

    # Custom classification head as in research paper
    model.classifier = nn.Sequential(
        nn.Linear(1280, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, len(PRIMARY_50_CLASSES)),
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
    optimizer = torch.optim.AdamW([
        {"params": model.features[13:].parameters(), "lr": 1.5e-4},
        {"params": model.classifier.parameters(), "lr": 1.2e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)

    epochs = 15
    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_cor, train_tot = 0.0, 0, 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            bs = by.size(0)
            train_loss += loss.item() * bs
            train_cor += (out.argmax(dim=1) == by).sum().item()
            train_tot += bs

        # Val
        model.eval()
        val_loss, val_cor, val_tot = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                loss = criterion(out, by)
                bs = by.size(0)
                val_loss += loss.item() * bs
                val_cor += (out.argmax(dim=1) == by).sum().item()
                val_tot += bs

        val_acc = 100.0 * val_cor / val_tot
        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{epochs}] ({elapsed:.1f}s) "
              f"Train Acc: {100.0*train_cor/train_tot:5.1f}% | "
              f"Val Acc: {val_acc:5.1f}% (loss: {val_loss/val_tot:.3f})")

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    test_cor, test_tot = 0, 0
    per_class_cor = {i: 0 for i in range(len(PRIMARY_50_CLASSES))}
    per_class_tot = {i: 0 for i in range(len(PRIMARY_50_CLASSES))}

    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx)
            preds = out.argmax(dim=1)
            test_cor += (preds == by).sum().item()
            test_tot += by.size(0)
            for i in range(by.size(0)):
                lbl = by[i].item()
                per_class_tot[lbl] += 1
                if preds[i].item() == lbl:
                    per_class_cor[lbl] += 1

    print("\n" + "=" * 60)
    print(f"Held-Out Test Set Accuracy: {100.0 * test_cor / test_tot:.2f}%")
    print("Key Produce Performance:")
    c2i = {c: i for i, c in enumerate(PRIMARY_50_CLASSES)}
    for f in ["tomato", "mango", "apple", "banana", "orange"]:
        idx = c2i[f]
        cor, tot = per_class_cor[idx], per_class_tot[idx]
        pct = 100.0 * cor / tot if tot > 0 else 0
        print(f"  {f.capitalize():<10}: {pct:5.1f}% ({cor}/{tot})")

    out_path = PROJECT_ROOT / "models" / "classification" / "mobilenet_v2_food50.pth"
    torch.save({
        "architecture": "MobileNetV2",
        "num_classes": len(PRIMARY_50_CLASSES),
        "class_names": PRIMARY_50_CLASSES,
        "class_to_idx": c2i,
        "idx_to_class": {i: c for i, c in enumerate(PRIMARY_50_CLASSES)},
        "model_state_dict": best_state,
        "image_size": 224,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }, out_path)
    print(f"\n[SUCCESS] Saved fine-tuned checkpoint to: {out_path}")

if __name__ == "__main__":
    main()
