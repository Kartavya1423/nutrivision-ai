"""
NutriVision AI — EfficientNet-B4 Training Script (Phase 4)
Trains EfficientNet-B4 for fine-grained food classification on isolated food crops.

Features:
- Configurable hyperparameters (epochs, batch size, lr, image size, workers)
- Synchronized class mapping across train/val/test splits (198 canonical classes)
- Transfer learning with official pretrained ImageNet weights
- Configurable backbone freeze/unfreeze strategy
- Inverse class frequency weighting support
- Loss & Top-1 / Top-5 Accuracy tracking
- Early stopping & best checkpoint serialization
- Training curves plotting (loss & accuracy)
- Strict Windows and CPU/CUDA auto-detection
"""

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T

# Project root path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings


def safe_food_crop_loader(path: str) -> Image.Image:
    """Loads image and composites RGBA transparent crops onto a clean neutral background."""
    with open(path, "rb") as f:
        img = Image.open(f)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            return bg
        return img.convert("RGB")


class FoodClassificationDataset(Dataset):
    """
    Classification dataset for food crops.
    Loads samples from split manifest or directory structure using canonical class mapping.
    """

    def __init__(
        self,
        split_dir: Path,
        class_to_idx: Dict[str, int],
        transform: Optional[T.Compose] = None,
    ):
        self.split_dir = Path(split_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        # Find all valid crop files belonging to canonical classes
        for class_name, class_idx in self.class_to_idx.items():
            class_folder = self.split_dir / class_name
            if not class_folder.is_dir():
                continue
            for img_file in class_folder.glob("*.png"):
                self.samples.append((img_file, class_idx))
            for img_file in class_folder.glob("*.jpg"):
                self.samples.append((img_file, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = safe_food_crop_loader(str(img_path))
        if self.transform:
            image = self.transform(image)
        return image, label


def load_canonical_class_mapping(mapping_csv: Path) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Loads canonical food classes and index mapping from classification_class_mapping.csv."""
    class_names = []
    class_to_idx = {}
    idx_to_class = {}

    with open(mapping_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            cname = row["class_name"]
            class_names.append(cname)
            class_to_idx[cname] = cid
            idx_to_class[cid] = cname

    return class_names, class_to_idx, idx_to_class


def compute_class_weights(dataset: FoodClassificationDataset, num_classes: int) -> torch.Tensor:
    """Computes inverse class frequency weights for balanced CrossEntropyLoss."""
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, label in dataset.samples:
        counts[label] += 1.0

    total = len(dataset)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 1.0

    # Clamp weights to prevent extreme gradient swings and normalize mean to 1.0
    weights = np.clip(weights, 0.2, 5.0)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def get_data_transforms(image_size: int = 224) -> Tuple[T.Compose, T.Compose]:
    """Returns train and val/test transforms with standard ImageNet normalization."""
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    eval_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    return train_transform, eval_transform


def build_efficientnet_b4(
    num_classes: int,
    pretrained_weights_path: Optional[Path] = None,
    freeze_backbone: bool = True,
    dropout: float = 0.4,
) -> nn.Module:
    """
    Builds EfficientNet-B4 model with custom classification head.
    Loads pretrained ImageNet weights if provided.
    """
    model = models.efficientnet_b4(weights=None)

    # Load pretrained ImageNet weights if available
    if pretrained_weights_path and pretrained_weights_path.is_file():
        print(f"[*] Loading pretrained ImageNet weights from: {pretrained_weights_path}")
        state_dict = torch.load(pretrained_weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
    else:
        print("[!] Pretrained weights not specified or not found. Training from scratch.")

    # Freeze feature extraction backbone if requested
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        print("[*] Backbone frozen: feature extraction mode enabled.")
    else:
        print("[*] Full model unfreeze: fine-tuning all layers.")

    # Replace classifier head: EfficientNet-B4 feature dimension is 1792
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )

    return model


def compute_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[float]:
    """Computes Top-K accuracy for evaluation."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, float]:
    """Runs one training epoch."""
    model.train()
    running_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    t_start = time.time()
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = targets.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        top1, top5 = compute_accuracy(outputs, targets, topk=(1, 5))
        running_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_samples += batch_size

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(loader):
            cur_loss = running_loss / total_samples
            cur_top1 = total_top1 / total_samples
            cur_top5 = total_top5 / total_samples
            elapsed = time.time() - t_start
            print(
                f"  Epoch [{epoch}] Batch [{batch_idx+1}/{len(loader)}] "
                f"Loss: {cur_loss:.4f} | Top-1: {cur_top1:.2f}% | Top-5: {cur_top5:.2f}% | "
                f"Time: {elapsed:.1f}s"
            )

    epoch_loss = running_loss / max(1, total_samples)
    epoch_top1 = total_top1 / max(1, total_samples)
    epoch_top5 = total_top5 / max(1, total_samples)
    return epoch_loss, epoch_top1, epoch_top5


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluates model performance on validation or test split."""
    model.eval()
    running_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = targets.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            top1, top5 = compute_accuracy(outputs, targets, topk=(1, 5))
            running_loss += loss.item() * batch_size
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    epoch_top1 = total_top1 / max(1, total_samples)
    epoch_top5 = total_top5 / max(1, total_samples)
    return epoch_loss, epoch_top1, epoch_top5


def plot_training_curves(history: List[Dict[str, Any]], save_path: Path):
    """Generates training and validation loss and accuracy curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        train_top1 = [h["train_top1"] for h in history]
        val_top1 = [h["val_top1"] for h in history]
        train_top5 = [h["train_top5"] for h in history]
        val_top5 = [h["val_top5"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="#2563eb", linewidth=2)
        ax1.plot(epochs, val_loss, "s--", label="Val Loss", color="#dc2626", linewidth=2)
        ax1.set_title("EfficientNet-B4 Loss Progression", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=11)

        # Accuracy curve
        ax2.plot(epochs, train_top1, "o-", label="Train Top-1", color="#2563eb", linewidth=2)
        ax2.plot(epochs, val_top1, "s--", label="Val Top-1", color="#dc2626", linewidth=2)
        ax2.plot(epochs, train_top5, "^-.", label="Train Top-5", color="#16a34a", linewidth=1.5)
        ax2.plot(epochs, val_top5, "v:", label="Val Top-5", color="#ea580c", linewidth=1.5)
        ax2.set_title("EfficientNet-B4 Accuracy Progression", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"[*] Training curves saved to: {save_path}")
    except Exception as e:
        print(f"[!] Warning: Failed to render training curves: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 on food crops")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--image_size", type=int, default=224, help="Input image dimension")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze backbone features")
    parser.add_argument("--unfreeze_all", action="store_true", default=False, help="Unfreeze entire model")
    parser.add_argument("--use_class_weights", action="store_true", default=True, help="Use inverse frequency weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_name", type=str, default="baseline_run", help="Run directory name")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Resume training from an existing checkpoint")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 recommended on Windows)")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = get_settings()
    device = torch.device(cfg.resolve_device())
    print(f"[*] Running on device: {device}")

    # Paths
    class_mapping_csv = PROJECT_ROOT / "data" / "processed" / "classification" / "classification_class_mapping.csv"
    train_dir = PROJECT_ROOT / "data" / "processed" / "classification" / "train"
    val_dir = PROJECT_ROOT / "data" / "processed" / "classification" / "val"
    test_dir = PROJECT_ROOT / "data" / "processed" / "classification" / "test"

    output_dir = PROJECT_ROOT / "models" / "classification" / "efficientnet_b4" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_ckpt = Path("C:/Users/mdahs/.cache/torch/hub/checkpoints/efficientnet_b4_rwightman-23ab8bcd.pth")
    if not cache_ckpt.is_file():
        cache_ckpt = None

    # Load canonical mapping
    class_names, class_to_idx, idx_to_class = load_canonical_class_mapping(class_mapping_csv)
    num_classes = len(class_names)
    print(f"[*] Loaded canonical taxonomy: {num_classes} classes.")

    # Datasets and Loaders
    train_transform, eval_transform = get_data_transforms(image_size=args.image_size)

    train_ds = FoodClassificationDataset(train_dir, class_to_idx, transform=train_transform)
    val_ds = FoodClassificationDataset(val_dir, class_to_idx, transform=eval_transform)
    test_ds = FoodClassificationDataset(test_dir, class_to_idx, transform=eval_transform)

    print(f"[*] Dataset splits loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    # Model
    freeze_bb = not args.unfreeze_all
    model = build_efficientnet_b4(
        num_classes=num_classes,
        pretrained_weights_path=cache_ckpt,
        freeze_backbone=freeze_bb,
    )
    model = model.to(device)

    # Loss Criterion
    if args.use_class_weights:
        class_weights = compute_class_weights(train_ds, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        print("[*] Enabled class-weighted CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        print("[*] Standard CrossEntropyLoss active.")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    # Training Loop Setup
    best_val_loss = float("inf")
    best_val_top1 = 0.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    start_epoch = 1

    if args.resume_checkpoint and Path(args.resume_checkpoint).is_file():
        print(f"[*] Resuming model weights from: {args.resume_checkpoint}")
        resume_data = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(resume_data["model_state_dict"])
        start_epoch = resume_data.get("epoch", 0) + 1
        best_val_loss = resume_data.get("val_metrics", {}).get("val_loss", float("inf"))
        best_val_top1 = resume_data.get("val_metrics", {}).get("val_top1", 0.0)
        best_epoch = resume_data.get("epoch", 0)

        prev_hist = Path(args.resume_checkpoint).parent / "training_history.json"
        if prev_hist.is_file():
            try:
                with open(prev_hist, "r", encoding="utf-8") as f:
                    history = json.load(f)
                print(f"[*] Loaded {len(history)} previous epoch history records.")
            except Exception:
                pass

    total_target_epochs = max(args.epochs, start_epoch + args.epochs - 1) if args.resume_checkpoint else args.epochs

    print(f"\n=======================================================")
    print(f"Starting EfficientNet-B4 Training (Epochs {start_epoch} to {total_target_epochs})")
    print(f"=======================================================\n")

    for epoch in range(start_epoch, total_target_epochs + 1):
        t0 = time.time()

        train_loss, train_top1, train_top5 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        val_loss, val_top1, val_top5 = evaluate_split(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        epoch_time = time.time() - t0

        print(
            f"--> Epoch [{epoch}/{args.epochs}] Complete in {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f}, Top-1: {train_top1:.2f}%, Top-5: {train_top5:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Top-1: {val_top1:.2f}%, Top-5: {val_top5:.2f}%"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_top1": round(train_top1, 2),
            "train_top5": round(train_top5, 2),
            "val_loss": round(val_loss, 4),
            "val_top1": round(val_top1, 2),
            "val_top5": round(val_top5, 2),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "epoch_time_s": round(epoch_time, 2),
        }
        history.append(epoch_record)

        # Checkpoint serialization
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "num_classes": num_classes,
            "image_size": args.image_size,
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "training_config": vars(args),
            "val_metrics": {
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top5": val_top5,
            },
            "random_seed": args.seed,
        }

        # Save last checkpoint
        torch.save(ckpt_data, output_dir / "last.pth")
        torch.save(ckpt_data, PROJECT_ROOT / "models" / "classification" / "efficientnet_b4" / "last.pth")

        # Save best checkpoint
        if val_loss < best_val_loss or (val_loss == best_val_loss and val_top1 > best_val_top1):
            best_val_loss = val_loss
            best_val_top1 = val_top1
            best_epoch = epoch
            torch.save(ckpt_data, output_dir / "best.pth")

            # Also mirror best checkpoint to canonical model locations
            canon_dest1 = PROJECT_ROOT / "models" / "classification" / "efficientnet_b4" / "best.pth"
            canon_dest2 = PROJECT_ROOT / "models" / "classification" / "efficientnet_b4_food.pth"
            canon_dest1.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt_data, canon_dest1)
            torch.save(ckpt_data, canon_dest2)
            print(f"  [+] New BEST checkpoint saved at Epoch {epoch} (Val Loss: {val_loss:.4f}, Top-1: {val_top1:.2f}%)")

    # Save training history
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(PROJECT_ROOT / "models" / "classification" / "efficientnet_b4" / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(output_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    plot_training_curves(history, output_dir / "training_curves.png")
    plot_training_curves(history, PROJECT_ROOT / "models" / "classification" / "efficientnet_b4" / "training_curves.png")

    print("\n=======================================================")
    print(f"Training Complete! Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Best Top-1: {best_val_top1:.2f}%")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
