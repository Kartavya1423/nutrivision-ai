"""NutriVision AI - Fruit Classifier Evaluation Script
Evaluates the trained 6-class fruit classifier checkpoint on a held-out test split.
Outputs classification report, confusion matrix, per-class metrics, and saves evaluation summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Project root path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.classification.fruit_classifier import FRUIT_CLASSES, FruitClassifier
from src.config.settings import settings
from src.data.fruit_dataset_loader import (
    build_fruit_dataloaders,
    get_fruit_transforms,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("evaluate_fruit_classifier")


def evaluate_fruit_checkpoint(
    checkpoint_path: Path | str,
    dataset_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    batch_size: int = 16,
    device_name: str | None = None,
    max_samples_per_class: int | None = None,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Please train the model first using 'scripts/train_fruit_classifier.py'."
        )

    output_dir = Path(output_dir) if output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Using evaluation device: {device}")

    # Load Model
    logger.info(f"Loading fruit classifier from {checkpoint_path}")
    classifier = FruitClassifier.from_checkpoint(checkpoint_path, device=device)
    classifier.eval()

    # Load Dataset Splits
    dataset_path = Path(dataset_dir) if dataset_dir else Path(settings.datasets.updated_fruit_dataset_dir)
    if not dataset_path.is_absolute():
        dataset_path = ROOT_DIR / dataset_path

    logger.info(f"Loading test split from {dataset_path}")
    split_file = ROOT_DIR / "data" / "splits" / "updated_fruit_dataset_split.json"
    _, _, test_loader, split_manifest = build_fruit_dataloaders(
        dataset_dir=dataset_path,
        split_path=split_file if split_file.is_file() else None,
        image_size=classifier.image_size,
        batch_size=batch_size,
        max_samples_per_class=max_samples_per_class,
    )

    test_dataset = test_loader.dataset
    logger.info(f"Evaluating {len(test_dataset)} test samples...")
    y_true: list[int] = []
    y_pred: list[int] = []
    y_conf: list[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = classifier.model(images)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_conf.extend(confs.cpu().numpy().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    accuracy = float(np.mean(y_true_np == y_pred_np))

    target_names = [classifier.idx_to_class[i] for i in range(len(classifier.idx_to_class))]
    report_dict = classification_report(
        y_true_np,
        y_pred_np,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    conf_matrix = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(target_names))))

    results = {
        "checkpoint": str(checkpoint_path),
        "test_samples": len(test_dataset),
        "test_accuracy": round(accuracy, 4),
        "mean_confidence": round(float(np.mean(y_conf)), 4),
        "macro_avg": report_dict.get("macro avg", {}),
        "weighted_avg": report_dict.get("weighted avg", {}),
        "per_class": {
            cls_name: {
                "precision": round(report_dict[cls_name]["precision"], 4),
                "recall": round(report_dict[cls_name]["recall"], 4),
                "f1-score": round(report_dict[cls_name]["f1-score"], 4),
                "support": report_dict[cls_name]["support"],
            }
            for cls_name in target_names
            if cls_name in report_dict
        },
        "confusion_matrix": conf_matrix.tolist(),
        "class_names": target_names,
        "freshness_note": (
            "All classes in this dataset contain '_Good' suffixes. "
            "Evaluated metrics reflect botanical fruit recognition capability, "
            "not rotten/spoiled freshness discrimination."
        ),
    }

    # Save results to json
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("FRUIT CLASSIFIER EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"Macro F1: {report_dict.get('macro avg', {}).get('f1-score', 0.0):.4f}")
    logger.info(f"Detailed classification report saved to: {results_file}")
    logger.info("\n" + classification_report(y_true_np, y_pred_np, target_names=target_names, zero_division=0))
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    logger.info("=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NutriVision AI Fruit Classifier Checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(settings.model_paths.fruit_classifier_path),
        help="Path to fruit classifier checkpoint (.pth)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(settings.datasets.updated_fruit_dataset_dir),
        help="Path to dataset root folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Cap samples per class during evaluation (useful for quick validation)",
    )

    args = parser.parse_args()

    evaluate_fruit_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
        max_samples_per_class=args.max_samples_per_class,
    )


if __name__ == "__main__":
    main()
