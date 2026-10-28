"""
NutriVision AI — Train 6-Class Fruit Classifier
Trains an EfficientNet-B4 classifier on the updated fruit dataset:
- Apple_Good
- Banana_Good
- Guava_Good
- Lime_Good
- Orange_Good
- Pomegranate_Good

Features:
- Reproducible random seed (42)
- GPU detection with graceful CPU fallback
- Stratified train/val/test splitting with persistent split manifest
- Class-weighted Cross-Entropy loss to handle class imbalance
- AdamW optimizer + Cosine Annealing learning rate schedule
- Early stopping based on validation macro-F1
- Saves best checkpoint with comprehensive metadata to models/fruit_classifier/best_model.pth
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.data.fruit_dataset_loader import (
    FRUIT_CLASSES,
    FruitDatasetValidator,
    build_fruit_dataloaders,
    create_stratified_fruit_splits,
    parse_fruit_class,
    NORM_MEAN,
    NORM_STD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_fruit_classifier")


def compute_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Tuple[float, float, Dict[str, Any]]:
    """Computes overall accuracy, macro-F1, and per-class precision, recall, and F1."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    total = len(y_true_arr)

    if total == 0:
        return 0.0, 0.0, {}

    accuracy = float(np.mean(y_true_arr == y_pred_arr))
    per_class = {}
    f1_list = []

    for idx, cname in enumerate(class_names):
        tp = int(np.sum((y_true_arr == idx) & (y_pred_arr == idx)))
        fp = int(np.sum((y_true_arr != idx) & (y_pred_arr == idx)))
        fn = int(np.sum((y_true_arr == idx) & (y_pred_arr != idx)))
        support = int(np.sum(y_true_arr == idx))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[cname] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    return round(accuracy, 4), round(macro_f1, 4), per_class


def build_model(num_classes: int = 6, pretrained_backbone_path: Optional[Path] = None, device: str = "cpu") -> nn.Module:
    """
    Initializes EfficientNet-B4 with a 6-class linear classification head.
    Loads pretrained feature weights if available.
    """
    base_model = models.efficientnet_b4(weights=None)
    in_features = base_model.classifier[1].in_features

    # If pretrained backbone exists, initialize feature extractors
    if pretrained_backbone_path and pretrained_backbone_path.is_file():
        try:
            logger.info(f"Loading pretrained backbone weights from: {pretrained_backbone_path}")
            ckpt = torch.load(pretrained_backbone_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            # Filter out final classifier head weights
            filtered = {k: v for k, v in state_dict.items() if not k.startswith("classifier.1")}
            base_model.load_state_dict(filtered, strict=False)
            logger.info("Successfully transferred feature extractor weights.")
        except Exception as exc:
            logger.warning(f"Could not load pretrained weights ({exc}). Initializing standard backbone.")

    # Custom classification head for 6 fruit classes
    base_model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return base_model.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    class_names: List[str],
    device: str,
) -> Tuple[float, float, float, Dict[str, Any], np.ndarray]:
    """Evaluates the model on validation or test DataLoader."""
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            total += targets.size(0)

    val_loss = running_loss / max(1, total)
    acc, macro_f1, per_class = compute_metrics(all_targets, all_preds, class_names)

    # Confusion Matrix
    n_c = len(class_names)
    cm = np.zeros((n_c, n_c), dtype=int)
    for t, p in zip(all_targets, all_preds):
        if 0 <= t < n_c and 0 <= p < n_c:
            cm[t, p] += 1

    return val_loss, acc, macro_f1, per_class, cm


def main():
    parser = argparse.ArgumentParser(description="Train 6-Class Fruit Classifier on Updated Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Input image dimension")
    parser.add_argument("--device", type=str, default="auto", help="Hardware device ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible random seed")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--max-samples-per-class", type=int, default=None, help="Cap samples per class for quick training runs")
    parser.add_argument("--output-dir", type=str, default="models/fruit_classifier", help="Target output checkpoint directory")
    args = parser.parse_args()

    settings = get_settings()
    root_dir = settings.root_dir

    # 1. Device resolution
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Target execution device: {device} (CUDA available: {torch.cuda.is_available()})")

    # 2. Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 3. Validate dataset directory
    dataset_dir = settings.datasets.updated_fruit_dataset_dir.resolve()
    logger.info(f"Loading dataset from: {dataset_dir}")
    validator = FruitDatasetValidator(dataset_path=dataset_dir, settings=settings)
    val_report = validator.validate_dataset(quick_verify=True)

    if not val_report.is_valid:
        logger.error(f"Dataset validation failed: {val_report.errors}")
        sys.exit(1)

    logger.info(f"Discovered {len(val_report.classes)} classes: {val_report.classes}")
    logger.info(f"Total valid images: {val_report.total_valid_images}")
    for cname, s in val_report.class_summaries.items():
        logger.info(f"  {cname}: {s.valid_count} images")

    # 4. Prepare / Load Stratified Splits
    split_dir = root_dir / "data" / "splits"
    split_file = split_dir / "updated_fruit_dataset_split.json"
    logger.info(f"Generating stratified splits (Train 70%, Val 15%, Test 15%) at: {split_file}")
    
    train_loader, val_loader, test_loader, split_manifest = build_fruit_dataloaders(
        dataset_dir=dataset_dir,
        split_path=split_file,
        image_size=args.image_size,
        batch_size=args.batch_size,
        random_seed=args.seed,
        max_samples_per_class=args.max_samples_per_class,
    )

    logger.info(
        f"DataLoaders ready — Train: {len(train_loader.dataset)} images, "
        f"Val: {len(val_loader.dataset)} images, "
        f"Test: {len(test_loader.dataset)} images"
    )

    # 5. Compute class weights to counter imbalance
    class_names = val_report.classes
    counts = [split_manifest["per_class_counts"][c]["train"] for c in class_names]
    total_train = sum(counts)
    class_weights = [total_train / (len(class_names) * max(1, c)) for c in counts]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info(f"Class weights (inverted frequency): {dict(zip(class_names, [round(w, 3) for w in class_weights]))}")

    # 6. Initialize EfficientNet-B4 model
    pretrained_cand = root_dir / "models" / "classification" / "efficientnet_b4_food.pth"
    model = build_model(
        num_classes=len(class_names),
        pretrained_backbone_path=pretrained_cand,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 7. Training Loop
    output_dir = root_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best_model.pth"
    # Secondary path for classification/fruit_classifier/ convention
    secondary_checkpoint_path = root_dir / "models" / "classification" / "fruit_classifier" / "best_model.pth"
    secondary_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_val_metrics: Dict[str, Any] = {}
    patience_counter = 0
    training_history: List[Dict[str, Any]] = []

    start_train_time = time.time()
    logger.info(f"Starting training for {args.epochs} epochs on {device}...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_per_class, _ = evaluate(model, val_loader, criterion, class_names, device)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}%"
        )

        history_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "elapsed_seconds": round(elapsed, 2),
        }
        training_history.append(history_entry)

        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_metrics = {
                "epoch": epoch,
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "val_macro_f1": round(val_f1, 4),
                "per_class": val_per_class,
            }
            patience_counter = 0

            # Save best checkpoint with exhaustive metadata
            fruit_name_map = {}
            dataset_label_map = {}
            for c in class_names:
                fn, dl = parse_fruit_class(c)
                fruit_name_map[c] = fn
                dataset_label_map[c] = dl

            checkpoint_data = {
                "architecture": "efficientnet_b4",
                "model_state_dict": model.state_dict(),
                "num_classes": len(class_names),
                "class_names": class_names,
                "class_to_idx": val_report.class_to_idx,
                "idx_to_class": val_report.idx_to_class,
                "fruit_name_map": fruit_name_map,
                "dataset_label_map": dataset_label_map,
                "image_size": args.image_size,
                "normalization": {"mean": NORM_MEAN, "std": NORM_STD},
                "training_config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "optimizer": "AdamW",
                    "scheduler": "CosineAnnealingLR",
                    "device": device,
                    "seed": args.seed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "val_metrics": best_val_metrics,
                "dataset_info": {
                    "total_valid_images": val_report.total_valid_images,
                    "train_count": len(train_loader.dataset),
                    "val_count": len(val_loader.dataset),
                    "test_count": len(test_loader.dataset),
                },
                "framework_version": torch.__version__,
            }

            torch.save(checkpoint_data, best_checkpoint_path)
            torch.save(checkpoint_data, secondary_checkpoint_path)
            logger.info(f"--> Saved new best checkpoint (Val F1: {val_f1*100:.2f}%) to: {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (patience={args.patience}).")
                break

    total_training_time = time.time() - start_train_time
    logger.info(f"Training completed in {total_training_time:.1f}s. Best Val F1: {best_val_f1*100:.2f}%")

    # 8. Evaluation on Genuinely Held-Out Test Set
    logger.info("Evaluating best model on strictly held-out Test set...")
    best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc, test_f1, test_per_class, test_cm = evaluate(
        model, test_loader, criterion, class_names, device
    )

    logger.info("=" * 65)
    logger.info(f"FINAL TEST SET EVALUATION:")
    logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}% | Test Macro-F1: {test_f1*100:.2f}%")
    logger.info("-" * 65)
    for cname, metrics in test_per_class.items():
        logger.info(f"  {cname:<20}: Prec: {metrics['precision']*100:.1f}%, Rec: {metrics['recall']*100:.1f}%, F1: {metrics['f1']*100:.1f}% (N={metrics['support']})")
    logger.info("=" * 65)

    # Update checkpoint with test metrics
    best_ckpt["test_metrics"] = {
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "per_class": test_per_class,
        "confusion_matrix": test_cm.tolist(),
    }
    best_ckpt["training_history"] = training_history
    torch.save(best_ckpt, best_checkpoint_path)
    torch.save(best_ckpt, secondary_checkpoint_path)

    # Save summary report JSON
    eval_report_path = output_dir / "evaluation_report.json"
    with open(eval_report_path, "w", encoding="utf-8") as fp:
        json.dump({
            "model": "EfficientNet-B4",
            "task": "Fruit Classification (6 Classes)",
            "dataset": "data/raw/updated dataset",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "best_epoch": best_val_metrics.get("epoch"),
            "val_metrics": best_val_metrics,
            "test_metrics": best_ckpt["test_metrics"],
            "classes": class_names,
            "training_history": training_history,
        }, fp, indent=2)

    logger.info(f"Evaluation report saved to: {eval_report_path}")
    print(f"\nTraining and test evaluation complete. Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
