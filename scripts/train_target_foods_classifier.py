"""
NutriVision AI — Phase 5: Train Target Food Classifier
Trains a specialized EfficientNet-B4 classifier on the approved target_foods_v1 dataset.

Safety & Constraints:
- Production apple model (models/classification/efficientnet_b4_food.pth) is strictly preserved.
- Output directory: models/classification/target_foods_v1/
- Only train split used for optimization; validation split used for model selection and early stopping.
- Test split (data/processed/target_foods_v1/test) is strictly quarantined and untouched.
- Deterministic: Seed 42 for complete reproducibility.
- Two-stage training protocol: Head warmup (Stage 1) -> Top-block fine-tuning (Stage 2).
- Dynamic training augmentation with class-weighted Cross-Entropy loss.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# 1. Reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

PROJECT_ROOT = Path(r"D:\NutriVision-AI (2)\NutriVision-AI")
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "target_foods_v1"
OUTPUT_DIR = PROJECT_ROOT / "models" / "classification" / "target_foods_v1"
BACKBONE_CKPT = PROJECT_ROOT / "models" / "classification" / "efficientnet_b4_food.pth"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
# TEST_DIR is strictly NOT accessed

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
STAGE1_EPOCHS = 5      # Head warmup
STAGE2_EPOCHS = 15     # Top-block fine-tuning
PATIENCE = 5           # Early stopping patience on val_macro_f1
LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def compute_macro_metrics(y_true, y_pred, num_classes):
    """Compute per-class and macro-averaged Precision, Recall, and F1."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    per_class = {}
    f1_list = []
    
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)
        
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(support),
        }
        f1_list.append(f1)
        
    macro_f1 = float(np.mean(f1_list))
    accuracy = float(np.mean(y_true == y_pred))
    return accuracy, macro_f1, per_class

def main():
    print("=" * 75)
    print("NutriVision AI — Phase 5: Target Food Classifier Training")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Device: {DEVICE} (PyTorch: {torch.__version__})")
    print(f"Output Checkpoints: {OUTPUT_DIR}")
    print("=" * 75)

    # 3. Data Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    # 4. Load Datasets
    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=val_transform)

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_names)

    print(f"\nLoaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    print(f"Number of classes: {num_classes}")
    print(f"Class Names: {class_names}")

    # Verify val dataset has matching classes
    assert val_dataset.classes == class_names, "Validation classes do not match training classes!"

    # 5. Compute Class Weights for Loss (strictly from training split)
    class_counts = Counter(train_dataset.targets)
    total_samples = len(train_dataset)
    weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print("\nTraining Class Distribution & Loss Weights:")
    for i, cname in enumerate(class_names):
        print(f"  [{i:2d}] {cname:15s}: {class_counts[i]:3d} samples | Weight: {weights[i]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=LABEL_SMOOTHING)

    # 6. Data Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, # Windows safe
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 7. Model Construction & Transfer Learning
    print("\nConstructing EfficientNet-B4 Architecture...")
    model = models.efficientnet_b4(weights=None)
    
    # Load food visual feature backbone from production apple checkpoint if available
    if BACKBONE_CKPT.is_file():
        print(f"Loading pretrained food backbone weights from: {BACKBONE_CKPT.name}")
        ckpt = torch.load(BACKBONE_CKPT, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        # Filter out old 198-class classifier head weights
        backbone_sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
        missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
        print(f"Backbone weights loaded cleanly ({len(backbone_sd)} layers transferred).")
    else:
        print("Pretrained checkpoint not found; loading ImageNet default weights...")
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

    # Replace classifier head with dedicated 21-class head
    in_features = model.classifier[1].in_features # 1792
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    model.to(DEVICE)

    # 8. Training Configuration Setup
    config = {
        "model_architecture": "efficientnet_b4",
        "num_classes": num_classes,
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "stage1_epochs": STAGE1_EPOCHS,
        "stage2_epochs": STAGE2_EPOCHS,
        "patience": PATIENCE,
        "label_smoothing": LABEL_SMOOTHING,
        "weight_decay": WEIGHT_DECAY,
        "random_seed": RANDOM_SEED,
        "device": DEVICE,
        "dataset": "target_foods_v1",
        "backbone_initialization": str(BACKBONE_CKPT.name),
    }

    with open(OUTPUT_DIR / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(OUTPUT_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    # 9. Training Loop Variables
    best_val_f1 = -1.0
    best_epoch = 0
    best_metrics = {}
    epochs_no_improve = 0
    history = []

    # Helper: set layer trainability
    def set_stage(stage_num):
        if stage_num == 1:
            print("\n>>> STAGE 1: Head-Only Warmup (Backbone Frozen) <<<")
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
            opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
            sched = None
            return opt, sched
        else:
            print("\n>>> STAGE 2: Top-Block Fine-Tuning (Features.8 Unfrozen) <<<")
            # Keep features 0 to 7 frozen
            for i in range(8):
                for param in model.features[i].parameters():
                    param.requires_grad = False
            # Unfreeze top conv block (features.8) and classifier head
            for param in model.features[8].parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True

            params = [
                {"params": model.features[8].parameters(), "lr": 5e-5},
                {"params": model.classifier.parameters(), "lr": 1e-4},
            ]
            opt = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS, eta_min=1e-5)
            return opt, sched

    optimizer, scheduler = set_stage(1)

    print("\nStarting Training Execution...")
    start_train_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Switch to Stage 2 at epoch 6
        if epoch == STAGE1_EPOCHS + 1:
            optimizer, scheduler = set_stage(2)

        # A. Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        if scheduler:
            scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # B. Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds_all = []
        val_labels_all = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)

                val_preds_all.extend(preds.cpu().numpy().tolist())
                val_labels_all.extend(labels.cpu().numpy().tolist())

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc, epoch_val_f1, per_class_report = compute_macro_metrics(
            val_labels_all, val_preds_all, num_classes
        )

        epoch_duration = time.time() - epoch_start
        overfitting_gap = epoch_train_acc - epoch_val_acc

        print(
            f"Epoch [{epoch:2d}/{NUM_EPOCHS:2d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}% | "
            f"Val Macro-F1: {epoch_val_f1*100:.2f}% | "
            f"Gap: {overfitting_gap*100:+.1f}%"
        )

        # Record history record
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_accuracy": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_accuracy": round(epoch_val_acc, 4),
            "val_macro_f1": round(epoch_val_f1, 4),
            "overfitting_gap": round(overfitting_gap, 4),
            "duration_s": round(epoch_duration, 1),
            "stage": 1 if epoch <= STAGE1_EPOCHS else 2,
        }
        history.append(epoch_record)

        # Save last checkpoint
        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "num_classes": num_classes,
            "image_size": IMAGE_SIZE,
            "normalization": {"mean": NORM_MEAN, "std": NORM_STD},
            "metrics": epoch_record,
            "training_config": config,
        }
        torch.save(last_ckpt, OUTPUT_DIR / "last.pth")

        # Check for improvement on Val Macro-F1
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            best_epoch = epoch
            best_metrics = {
                "epoch": epoch,
                "val_accuracy": round(epoch_val_acc, 4),
                "val_macro_f1": round(epoch_val_f1, 4),
                "val_loss": round(epoch_val_loss, 4),
                "train_accuracy": round(epoch_train_acc, 4),
                "train_loss": round(epoch_train_loss, 4),
                "per_class": {class_names[c]: per_class_report[c] for c in range(num_classes)},
            }
            epochs_no_improve = 0

            # Save best checkpoint separately
            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "num_classes": num_classes,
                "image_size": IMAGE_SIZE,
                "normalization": {"mean": NORM_MEAN, "std": NORM_STD},
                "val_metrics": best_metrics,
                "training_config": config,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(best_ckpt, OUTPUT_DIR / "best.pth")
            print(f"  --> [NEW BEST CHECKPOINT] Saved to best.pth (Val Macro-F1: {best_val_f1*100:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE and epoch > STAGE1_EPOCHS:
                print(f"\n[EARLY STOPPING TRIGGERED] No improvement for {PATIENCE} epochs.")
                break

    total_training_time = time.time() - start_train_time
    print("\n" + "=" * 75)
    print(f"Training Complete in {total_training_time/60:.2f} minutes.")
    print(f"Best Checkpoint: Epoch {best_epoch} with Val Macro-F1: {best_val_f1*100:.2f}%")
    print(f"Saved Checkpoint Path: {OUTPUT_DIR / 'best.pth'}")
    print("=" * 75)

    # 10. Save Detailed History and Per-Class Final Report
    history_file = OUTPUT_DIR / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "best_summary": best_metrics,
            "total_time_seconds": round(total_training_time, 2),
            "history": history,
        }, f, indent=2)

    # Print final per-class table for best checkpoint
    print("\n=== PER-CLASS VALIDATION PERFORMANCE (BEST CHECKPOINT) ===")
    print(f"{'Class':16s} | {'Support':7s} | {'Precision':10s} | {'Recall':8s} | {'F1-Score':8s}")
    print("-" * 60)
    for cname in class_names:
        stats = best_metrics["per_class"][cname]
        print(
            f"{cname:16s} | {stats['support']:7d} | "
            f"{stats['precision']*100:9.1f}% | "
            f"{stats['recall']*100:7.1f}% | "
            f"{stats['f1']*100:7.1f}%"
        )
    print("-" * 60)
    print(f"{'MACRO AVERAGE':16s} | {len(val_dataset):7d} | {'--':10s} | {'--':8s} | {best_val_f1*100:7.1f}%")

if __name__ == "__main__":
    main()
