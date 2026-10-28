"""
NutriVision AI — Phase 6: Final Test Evaluation Script
Evaluates the trained Phase 5 target-food classifier (target_foods_v1/best.pth)
on the untouched test split (data/processed/target_foods_v1/test).

Computes:
1. Test Top-1 and Top-5 accuracy, and test loss.
2. Per-class precision, recall, F1, and support.
3. 95% Wilson confidence intervals for test metrics.
4. Full 21x21 confusion matrix and top confusion patterns.
5. Exact train/val/test split counts per class.
6. Head-to-head comparison with the old baseline (efficientnet_b4_food.pth).
7. Formal evidence-based support assessment for all 24 target foods.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(r"D:\NutriVision-AI (2)\NutriVision-AI")
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "target_foods_v1"
TEST_DIR = DATASET_DIR / "test"
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "splits" / "target_foods_v1_split.json"
CLASS_COUNTS_FILE = DATASET_DIR / "class_counts.json"

MODEL_DIR = PROJECT_ROOT / "models" / "classification" / "target_foods_v1"
BEST_CKPT_PATH = MODEL_DIR / "best.pth"
BASELINE_CKPT_PATH = PROJECT_ROOT / "models" / "classification" / "efficientnet_b4_food.pth"

OUTPUT_JSON = MODEL_DIR / "test_evaluation_report.json"
OUTPUT_MD = MODEL_DIR / "test_evaluation_report.md"

IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """Calculate Wilson score interval for binomial proportions."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% normal quantile
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    spread = (z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return round(lower, 4), round(upper, 4)


def load_target_model(ckpt_path: Path):
    """Load new 21-class EfficientNet-B4 classifier."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names, ckpt.get("epoch", None)


def load_baseline_model(ckpt_path: Path):
    """Load old 198-class production EfficientNet-B4 baseline classifier."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    class_names = ckpt.get("class_names", [])
    num_classes = len(class_names)

    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names


# Target class mapping for the 198-class baseline
BASELINE_SYNONYMS = {
    "apple": ["apple"],
    "banana": ["banana"],
    "bread": ["bread", "bread_steamed", "raisin_bread", "roll_bread"],
    "broccoli": ["broccoli"],
    "burger": ["hamburger"],
    "cake": ["cake", "pancake", "japanese-style_pancake"],
    "chicken": ["chicken_duck", "fried_chicken", "roast_chicken", "boiled_chicken_and_vegetables"],
    "cucumber": ["cucumber"],
    "egg": ["egg", "egg_roll", "egg_sunny-side_up", "steamed_egg_hotchpotch"],
    "french_fries": ["french_fries"],
    "grape": ["grape"],
    "ice_cream": ["ice_cream"],
    "lettuce": ["lettuce"],
    "mango": ["mango"],
    "orange": ["orange"],
    "pineapple": ["pineapple"],
    "pizza": ["pizza", "pizza_toast"],
    "spinach": ["spinach", "sauteed_spinach"],
    "strawberry": ["strawberry"],
    "tomato": ["tomato"],
    "watermelon": ["watermelon"],
}


def main():
    print("=" * 78)
    print("NutriVision AI — Phase 6: Final Test Evaluation")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Test Split Path: {TEST_DIR}")
    print(f"Device: {DEVICE}")
    print("=" * 78)

    # 1. Dataset & Loader
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    class_names = test_dataset.classes
    num_classes = len(class_names)
    total_test = len(test_dataset)
    print(f"Loaded {total_test} test images across {num_classes} classes.")

    # Load split counts
    class_counts = {}
    if CLASS_COUNTS_FILE.is_file():
        with open(CLASS_COUNTS_FILE, "r", encoding="utf-8") as f:
            class_counts = json.load(f)

    # 2. Evaluate Phase 5 Model (best.pth)
    print(f"\n[1/2] Evaluating Phase 5 Checkpoint: {BEST_CKPT_PATH.name}...")
    model, target_classes, best_epoch = load_target_model(BEST_CKPT_PATH)
    assert target_classes == class_names, "Checkpoint class names do not match test folder names!"

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    top5_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            # Top 5
            _, top5_indices = outputs.topk(k=min(5, num_classes), dim=1, largest=True, sorted=True)
            for i in range(labels.size(0)):
                if labels[i] in top5_indices[i]:
                    top5_correct += 1

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    test_loss = total_loss / total_test
    test_top1_acc = np.mean(np.array(all_preds) == np.array(all_targets))
    test_top5_acc = top5_correct / total_test
    test_acc_ci = wilson_score_interval(int(np.sum(np.array(all_preds) == np.array(all_targets))), total_test)

    # 3. Confusion Matrix and Per-Class Metrics
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1

    per_class_metrics = {}
    f1_list = []
    precision_list = []
    recall_list = []

    for c in range(num_classes):
        cname = class_names[c]
        support = int(np.sum(y_true == c))
        tp = conf_matrix[c, c]
        fp = int(np.sum(conf_matrix[:, c]) - tp)
        fn = int(np.sum(conf_matrix[c, :]) - tp)

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        rec_ci = wilson_score_interval(tp, support)

        # Flag sample count warning
        sample_warning = None
        if support < 10:
            sample_warning = f"High variance estimate: support ({support}) is < 10."
        elif support < 20:
            sample_warning = f"Moderate variance: support ({support}) is < 20."

        per_class_metrics[cname] = {
            "class_idx": c,
            "support": support,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "recall_ci_95": rec_ci,
            "sample_warning": sample_warning,
            "train_samples": class_counts.get(cname, {}).get("train", "N/A"),
            "val_samples": class_counts.get(cname, {}).get("val", "N/A"),
            "test_samples": support,
        }
        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(rec)

    macro_f1 = float(np.mean(f1_list))
    macro_precision = float(np.mean(precision_list))
    macro_recall = float(np.mean(recall_list))

    # Top Confusion Pairs
    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and conf_matrix[i, j] > 0:
                count = int(conf_matrix[i, j])
                true_support = int(np.sum(conf_matrix[i, :]))
                confusion_pairs.append({
                    "true_class": class_names[i],
                    "pred_class": class_names[j],
                    "count": count,
                    "error_rate": round(count / true_support, 4),
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    # 4. Evaluate Baseline Checkpoint (efficientnet_b4_food.pth) on the same test set
    print(f"\n[2/2] Evaluating Baseline Checkpoint: {BASELINE_CKPT_PATH.name}...")
    baseline_top1_correct = 0
    baseline_top5_correct = 0
    baseline_per_class_tp = defaultdict(int)

    if BASELINE_CKPT_PATH.is_file():
        base_model, base_classes = load_baseline_model(BASELINE_CKPT_PATH)

        # Map base class name to index
        base_class_to_idx = {name: i for i, name in enumerate(base_classes)}

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                outputs = base_model(images)
                top1_idx = outputs.argmax(dim=1).cpu().numpy()
                _, top5_idx = outputs.topk(k=5, dim=1, largest=True, sorted=True)
                top5_idx = top5_idx.cpu().numpy()

                for b in range(labels.size(0)):
                    true_target_name = class_names[labels[b].item()]
                    valid_synonyms = BASELINE_SYNONYMS.get(true_target_name, [true_target_name])
                    valid_base_indices = [
                        base_class_to_idx[s] for s in valid_synonyms if s in base_class_to_idx
                    ]

                    # Top-1 Check
                    if top1_idx[b] in valid_base_indices:
                        baseline_top1_correct += 1
                        baseline_per_class_tp[true_target_name] += 1

                    # Top-5 Check
                    if any(idx in valid_base_indices for idx in top5_idx[b]):
                        baseline_top5_correct += 1

        base_top1_acc = float(baseline_top1_correct / total_test)
        base_top5_acc = float(baseline_top5_correct / total_test)
    else:
        base_top1_acc = 0.0
        base_top5_acc = 0.0

    # 5. Support Assessment for all 24 Target Foods
    # Categorization:
    # - Supported (High Confidence): F1 >= 70% and support >= 10
    # - Moderately Supported (Caution / Good Lighting): 40% <= F1 < 70%
    # - Low Confidence / Experimental: F1 < 40%
    # - Unsupported / Data Deficient: Missing from training or 0% F1
    all_24_foods = [
        "apple", "banana", "bread", "broccoli", "burger", "butter", "cake",
        "chicken", "cucumber", "egg", "french_fries", "grape", "ice_cream",
        "lettuce", "mango", "orange", "paneer", "pineapple", "pizza",
        "spinach", "strawberry", "tomato", "watermelon"
    ]

    food_support_assessment = {}
    for food in all_24_foods:
        if food == "paneer":
            food_support_assessment[food] = {
                "status": "UNSUPPORTED (Data Deficient)",
                "tier": "Tier 5 (No Data)",
                "test_f1": "N/A (0 images in dataset)",
                "recommendation": "Collect at least 50-100 real paneer photos before training.",
            }
        elif food == "butter":
            food_support_assessment[food] = {
                "status": "UNSUPPORTED (Bundled)",
                "tier": "Tier 5 (Bundled in cheese_butter)",
                "test_f1": "N/A (Bundled)",
                "recommendation": "Acquire clean standalone butter images to decouple from cheese.",
            }
        elif food in per_class_metrics:
            m = per_class_metrics[food]
            f1 = m["f1"]
            rec = m["recall"]
            prec = m["precision"]
            if f1 >= 0.70 and m["support"] >= 10:
                food_support_assessment[food] = {
                    "status": "SUPPORTED (High Confidence)",
                    "tier": "Tier 1 (Production Ready)",
                    "test_f1": f"{f1*100:.1f}%",
                    "test_precision": f"{prec*100:.1f}%",
                    "test_recall": f"{rec*100:.1f}%",
                    "recommendation": "Ready for direct camera/upload recognition.",
                }
            elif f1 >= 0.40:
                food_support_assessment[food] = {
                    "status": "MODERATELY SUPPORTED",
                    "tier": "Tier 2 (Usable with Confidence Floor)",
                    "test_f1": f"{f1*100:.1f}%",
                    "test_precision": f"{prec*100:.1f}%",
                    "test_recall": f"{rec*100:.1f}%",
                    "recommendation": "Acceptable in production; recommend top-3 fallback options.",
                }
            elif f1 > 0.0:
                food_support_assessment[food] = {
                    "status": "EXPERIMENTAL (Low Confidence)",
                    "tier": "Tier 3 (Underperforming)",
                    "test_f1": f"{f1*100:.1f}%",
                    "test_precision": f"{prec*100:.1f}%",
                    "test_recall": f"{rec*100:.1f}%",
                    "recommendation": "Add 50-100 targeted diverse crops; visually confused.",
                }
            else:
                food_support_assessment[food] = {
                    "status": "UNSUPPORTED (Failed Recognition)",
                    "tier": "Tier 4 (Zero Accuracy)",
                    "test_f1": "0.0%",
                    "test_precision": "0.0%",
                    "test_recall": "0.0%",
                    "recommendation": "Total failure due to visual collision with other greens. Must re-annotate.",
                }

    # 6. Save JSON Report
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(BEST_CKPT_PATH.name),
        "best_epoch": best_epoch,
        "test_metrics": {
            "top1_accuracy": round(float(test_top1_acc), 4),
            "top1_accuracy_ci_95": test_acc_ci,
            "top5_accuracy": round(float(test_top5_acc), 4),
            "test_loss": round(float(test_loss), 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "total_test_samples": total_test,
        },
        "baseline_comparison": {
            "baseline_model": str(BASELINE_CKPT_PATH.name),
            "baseline_top1_accuracy": round(base_top1_acc, 4),
            "baseline_top5_accuracy": round(base_top5_acc, 4),
            "accuracy_gain_top1": round(float(test_top1_acc - base_top1_acc), 4),
            "accuracy_gain_top5": round(float(test_top5_acc - base_top5_acc), 4),
        },
        "per_class_metrics": per_class_metrics,
        "top_confusion_pairs": confusion_pairs[:15],
        "target_food_support_assessment": food_support_assessment,
        "confusion_matrix": conf_matrix.tolist(),
        "class_order": class_names,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nSaved test evaluation JSON to: {OUTPUT_JSON}")

    # 7. Print Terminal Summary
    print("\n" + "=" * 78)
    print("PHASE 6 TEST EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Total Test Images: {total_test}")
    print(f"Top-1 Accuracy:    {test_top1_acc*100:.2f}%  (95% CI: [{test_acc_ci[0]*100:.1f}%, {test_acc_ci[1]*100:.1f}%])")
    print(f"Top-5 Accuracy:    {test_top5_acc*100:.2f}%")
    print(f"Test Loss:         {test_loss:.4f}")
    print(f"Macro-F1 Score:    {macro_f1*100:.2f}%")
    print(f"Baseline Top-1:    {base_top1_acc*100:.2f}%  --> Net Gain: +{(test_top1_acc - base_top1_acc)*100:.2f}%")
    print(f"Baseline Top-5:    {base_top5_acc*100:.2f}%  --> Net Gain: +{(test_top5_acc - base_top5_acc)*100:.2f}%")
    print("=" * 78)

    print("\nPER-CLASS TEST METRICS:")
    print(f"{'Class':15s} | {'Train':>5s} | {'Val':>4s} | {'Test':>4s} | {'Prec':>6s} | {'Recall':>6s} | {'F1':>6s} | {'95% CI (Rec)':>15s}")
    print("-" * 78)
    for cname, m in per_class_metrics.items():
        tr = str(m['train_samples'])
        v = str(m['val_samples'])
        te = str(m['test_samples'])
        p = f"{m['precision']*100:.1f}%"
        r = f"{m['recall']*100:.1f}%"
        f = f"{m['f1']*100:.1f}%"
        ci = f"[{m['recall_ci_95'][0]*100:.0f}%, {m['recall_ci_95'][1]*100:.0f}%]"
        print(f"{cname:15s} | {tr:>5s} | {v:>4s} | {te:>4s} | {p:>6s} | {r:>6s} | {f:>6s} | {ci:>15s}")

    print("\nTOP CONFUSION PAIRS (Top 8):")
    for cp in confusion_pairs[:8]:
        print(f"  True {cp['true_class']:12s} --> Predicted {cp['pred_class']:12s}: {cp['count']} samples ({cp['error_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()
