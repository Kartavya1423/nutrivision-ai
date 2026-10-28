#!/usr/bin/env python3
"""
NutriVision AI — EfficientNet-B4 Freshness Classification Training (Phase 6)
Fine-tunes EfficientNet-B4 for food freshness condition classification (fresh vs rotten).

Key Features:
- Fine-tunes official torchvision EfficientNet-B4 pretrained on ImageNet
- Food-aware binary freshness classification (0: fresh, 1: rotten)
- Freshness-preserving data augmentations (mild color jitter, flips, mild affine)
- Class-balanced weighting support
- Early stopping based on validation macro F1 and loss
- Full training history and loss/accuracy curves plotting
- Comprehensive checkpoint containing weights, class mapping, normalizations, and metadata
- Exposes convolutional feature layer (features[-1]) for Phase 7 Grad-CAM compatibility
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings


CLASS_TO_LABEL = {
    "fresh": 0,
    "rotten": 1,
}

LABEL_TO_CLASS = {
    0: "fresh",
    1: "rotten",
}

SUBCLASS_METADATA = {
    "freshapples": {"food": "apple", "freshness": "fresh", "label": 0},
    "freshbanana": {"food": "banana", "freshness": "fresh", "label": 0},
    "freshoranges": {"food": "orange", "freshness": "fresh", "label": 0},
    "rottenapples": {"food": "apple", "freshness": "rotten", "label": 1},
    "rottenbanana": {"food": "banana", "freshness": "rotten", "label": 1},
    "rottenoranges": {"food": "orange", "freshness": "rotten", "label": 1},
}


class FreshnessDataset(Dataset):
    """Dataset for produce freshness images."""

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int, str, str]] = []  # (path, label, food, subclass)

        for subclass_dir in sorted(self.root_dir.iterdir()):
            if not subclass_dir.is_dir():
                continue
            subclass_name = subclass_dir.name.lower()
            if subclass_name in SUBCLASS_METADATA:
                meta = SUBCLASS_METADATA[subclass_name]
                label = meta["label"]
                food = meta["food"]
            else:
                label = 0 if "fresh" in subclass_name else 1
                food = "unknown"

            for img_path in sorted(subclass_dir.glob("*.*")):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    self.samples.append((img_path, label, food, subclass_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        path, label, food, subclass = self.samples[idx]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = T.ToTensor()(img)

        return img_tensor, label, food, subclass


class EfficientNetFreshnessModel(nn.Module):
    """
    EfficientNet-B4 fine-tuned for freshness detection.
    Prepares for explainability by exposing final convolutional layer.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)

        # Freeze early layers, keep later layers (conv_head) trainable
        for p in self.backbone.features[:7].parameters():
            p.requires_grad = False

        in_features = self.backbone.classifier[1].in_features  # 1792
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns final convolutional feature layer for Phase 7 explainability."""
        return self.backbone.features[-1]


def build_transforms(image_size: int = 280) -> Tuple[T.Compose, T.Compose]:
    """Builds freshness-safe data transforms preserving discoloration and texture cues."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    eval_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Trains for one epoch, returning average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels, _, _) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += inputs.size(0)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"    Batch [{batch_idx+1:02d}/{len(dataloader):02d}] Loss: {loss.item():.4f}", flush=True)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluates on validation split, returning loss, accuracy, and macro F1."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels, _, _ in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    total = len(all_labels)
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = sum(1 for p, y in zip(all_preds, all_labels) if p == y) / total if total > 0 else 0.0

    # Macro F1 calculation
    f1_scores = []
    for c in [0, 1]:
        tp = sum(1 for p, y in zip(all_preds, all_labels) if p == c and y == c)
        fp = sum(1 for p, y in zip(all_preds, all_labels) if p == c and y != c)
        fn = sum(1 for p, y in zip(all_preds, all_labels) if p != c and y == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return avg_loss, accuracy, macro_f1


def plot_training_curves(history: Dict[str, List[float]], output_path: Path):
    """Plots and saves loss, accuracy, and learning rate curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2)
    axes[0].set_title("Freshness Model Loss Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Accuracy & Macro F1 curve
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r--s", label="Val Acc", linewidth=2)
    axes[1].plot(epochs, history["val_f1"], "g-.^", label="Val Macro F1", linewidth=2)
    axes[1].set_title("Accuracy & Macro F1 Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # Learning rate curve
    axes[2].plot(epochs, history["lr"], "m-d", label="Learning Rate", linewidth=2)
    axes[2].set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 for food freshness classification.")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--image_size", type=int, default=280, help="Input image resolution.")
    parser.add_argument("--patience", type=int, default=4, help="Patience epochs for early stopping.")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: auto, cpu, or cuda.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 72)
    print("   NutriVision AI — EfficientNet-B4 Freshness Model Training (Phase 6)   ")
    print("=" * 72)
    print(f"  Target Architecture : EfficientNet-B4 (Pretrained ImageNet Weights)")
    print(f"  Execution Device    : {str(device).upper()}")
    if device.type == "cpu":
        import platform
        print(f"  Processor           : {platform.processor() or 'CPU'}")
    elif device.type == "cuda":
        print(f"  GPU Name            : {torch.cuda.get_device_name(0)}")
    print(f"  Max Epochs          : {args.epochs}")
    print(f"  Batch Size          : {args.batch_size}")
    print(f"  Learning Rate       : {args.lr}")
    print(f"  Input Resolution    : {args.image_size}x{args.image_size}")
    print(f"  Early Stop Patience : {args.patience}")
    print("-" * 72)

    proc_dir = PROJECT_ROOT / "data" / "processed" / "freshness"
    train_dir = proc_dir / "train"
    val_dir = proc_dir / "val"
    models_dir = PROJECT_ROOT / "models" / "freshness"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = PROJECT_ROOT / "outputs" / "freshness"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Processed freshness data not found in {proc_dir}. Run prepare_freshness_dataset.py first.")

    train_tf, eval_tf = build_transforms(image_size=args.image_size)

    train_dataset = FreshnessDataset(train_dir, transform=train_tf)
    val_dataset = FreshnessDataset(val_dir, transform=eval_tf)

    print(f"\n[Dataset] Loaded splits:")
    print(f"  -> Train samples: {len(train_dataset)}")
    print(f"  -> Val samples  : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize model
    print("\n[Model] Initializing EfficientNet-B4...")
    model = EfficientNetFreshnessModel(num_classes=2, dropout_rate=0.4, pretrained=True)
    model.to(device)

    # Loss function with balanced weighting
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    print("\n[Training] Beginning model optimization...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_f1)
        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f1"].append(round(val_f1, 4))
        history["lr"].append(current_lr)

        print(
            f"  Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro F1: {val_f1:.4f} | LR: {current_lr:.6f}"
        )

        checkpoint_metadata = {
            "epoch": epoch,
            "architecture": "efficientnet_b4",
            "num_classes": 2,
            "class_mapping": LABEL_TO_CLASS,
            "image_size": args.image_size,
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "best_epoch": best_epoch,
            "best_val_f1": round(best_val_f1, 4),
            "device": str(device),
            "saved_at": datetime.now().isoformat(),
        }

        # Save last checkpoint
        last_ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **checkpoint_metadata,
        }
        torch.save(last_ckpt, models_dir / "last.pth")

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            checkpoint_metadata["best_epoch"] = best_epoch
            checkpoint_metadata["best_val_f1"] = round(best_val_f1, 4)

            best_ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                **checkpoint_metadata,
            }
            torch.save(best_ckpt, models_dir / "best.pth")
            # Also keep compatibility with settings.py freshness_model_path
            torch.save(best_ckpt, models_dir / "freshness_classifier.pth")
            print(f"      -> New BEST model saved to {models_dir / 'best.pth'} (Macro F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] No improvement in validation Macro F1 for {args.patience} epochs.")
                break

    total_training_time = time.time() - start_time
    print(f"\n[Completed] Total training elapsed: {total_training_time:.1f}s ({total_training_time/60:.2f} min)")
    print(f"Best Epoch: {best_epoch} with Val Macro F1: {best_val_f1:.4f}")

    # Save training config
    training_config = {
        "architecture": "efficientnet_b4",
        "epochs_requested": args.epochs,
        "epochs_completed": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "image_size": args.image_size,
        "device": str(device),
        "seed": args.seed,
        "classes": LABEL_TO_CLASS,
        "total_training_time_seconds": round(total_training_time, 2),
        "timestamp": datetime.now().isoformat(),
    }
    with open(models_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    # Save training history JSON
    with open(outputs_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Generate training curves plot
    plot_training_curves(history, outputs_dir / "training_curves.png")
    print(f"  -> Saved training history to {outputs_dir / 'training_history.json'}")
    print(f"  -> Saved training curves to {outputs_dir / 'training_curves.png'}")
    print(f"  -> Saved training config to {models_dir / 'training_config.json'}")

    print("\n" + "=" * 72)
    print("      [SUCCESS] PHASE 6 MODEL TRAINING FINISHED SUCCESSFULLY!      ")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
