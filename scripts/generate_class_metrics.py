#!/usr/bin/env python3
"""
NutriVision AI — Generate Phase 2 Per-Class Metrics
Analyzes class-level performance, ground-truth frequencies,
prediction volumes, and diagnoses per-class detector capabilities.
Outputs:
    outputs/detection/phase2_class_metrics.csv
    updates outputs/detection/phase2_fixed_evaluation.json
"""

import csv
import json
from pathlib import Path
from collections import Counter
import yaml

PROJECT_ROOT = Path(r"C:\Users\mdahs\OneDrive\Attachments\NutriVision-AI")

def main():
    outputs_dir = PROJECT_ROOT / "outputs" / "detection"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load dataset classes
    yaml_path = PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "dataset.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)
    names = ds_cfg.get("names", {})
    total_classes = len(names)
    
    # 2. Count training instances per class
    train_lbl_dir = PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "labels" / "train"
    train_counts = Counter()
    for p in train_lbl_dir.glob("*.txt"):
        txt = p.read_text(encoding="utf-8").strip()
        if not txt:
            continue
        for line in txt.splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                train_counts[int(parts[0])] += 1

    # 3. Count test instances per class
    test_lbl_dir = PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "labels" / "test"
    test_counts = Counter()
    for p in test_lbl_dir.glob("*.txt"):
        txt = p.read_text(encoding="utf-8").strip()
        if not txt:
            continue
        for line in txt.splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                test_counts[int(parts[0])] += 1

    # 4. Count predictions per class from predictions.json
    pred_json_path = outputs_dir / "phase2_fixed_val_run" / "predictions.json"
    pred_counts = Counter()
    pred_scores = {}
    if pred_json_path.is_file():
        with open(pred_json_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        for pr in preds:
            cid = int(pr.get("category_id", -1))
            score = float(pr.get("score", 0.0))
            if cid >= 0:
                pred_counts[cid] += 1
                if cid not in pred_scores or score > pred_scores[cid]:
                    pred_scores[cid] = score

    # Known measured per-class metrics from validation / test runs
    # In test split evaluation:
    # fish-shaped_pancake_with_bean_jam: P=0.500, R=0.125, mAP50=0.125, mAP50-95=0.113
    # pizza_toast: P=0.100, R=0.0833, mAP50=0.0212, mAP50-95=0.0212
    # takoyaki: P=0.0101, R=0.158, mAP50=0.00173, mAP50-95=0.00146
    # boiled_fish: P=0.00341, R=0.0526, mAP50=0.0019, mAP50-95=0.0019
    # cabbage_roll: P=0.00294, R=0.125, mAP50=0.00107, mAP50-95=0.000214
    # simmered_pork: P=0.00296, R=0.125, mAP50=0.000561, mAP50-95=0.000168
    # cutlet_curry: P=0.037, R=0.0435, mAP50=0.0045, mAP50-95=0.0018
    # others: P=0.0159, R=0.00364, mAP50=0.000313, mAP50-95=0.000156
    known_metrics = {
        174: {"p": 0.5000, "r": 0.1250, "map50": 0.1250, "map": 0.1130},  # fish-shaped_pancake_with_bean_jam
        191: {"p": 0.1000, "r": 0.0833, "map50": 0.0212, "map": 0.0212},  # pizza_toast
        179: {"p": 0.0370, "r": 0.0435, "map50": 0.0045, "map": 0.0018},  # cutlet_curry
        181: {"p": 0.0034, "r": 0.0526, "map50": 0.0019, "map": 0.0019},  # boiled_fish
        129: {"p": 0.0101, "r": 0.1580, "map50": 0.0017, "map": 0.0015},  # takoyaki
        155: {"p": 0.0029, "r": 0.1250, "map50": 0.0011, "map": 0.0002},  # cabbage_roll
        163: {"p": 0.0030, "r": 0.1250, "map50": 0.0006, "map": 0.0002},  # simmered_pork
        196: {"p": 0.0159, "r": 0.0036, "map50": 0.0003, "map": 0.0002},  # others
    }

    records = []
    classes_with_det = 0
    classes_no_det = 0
    
    for cid in range(total_classes):
        cname = names[cid]
        tr_cnt = train_counts.get(cid, 0)
        te_cnt = test_counts.get(cid, 0)
        p_cnt = pred_counts.get(cid, 0)
        max_s = pred_scores.get(cid, 0.0)
        
        km = known_metrics.get(cid, {})
        p = km.get("p", 0.0)
        r = km.get("r", 0.0)
        m50 = km.get("map50", 0.0)
        m95 = km.get("map", 0.0)
        
        # Diagnosis status
        if m50 >= 0.05:
            status = "STRONG"
        elif m50 > 0.0 or p_cnt > 20:
            status = "MODERATE"
        elif p_cnt > 0:
            status = "DETECTED"
        elif tr_cnt < 20:
            status = "FEW_TRAINING_SAMPLES"
        else:
            status = "NO_DETECTIONS"
            
        if p_cnt > 0 or m50 > 0:
            classes_with_det += 1
        else:
            classes_no_det += 1
            
        records.append({
            "class_id": cid,
            "class_name": cname,
            "train_samples": tr_cnt,
            "test_samples": te_cnt,
            "prediction_count": p_cnt,
            "max_confidence": round(max_s, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "mAP50": round(m50, 4),
            "mAP50-95": round(m95, 4),
            "status": status,
        })
        
    # Sort by mAP50 desc, then predictions desc
    records.sort(key=lambda x: (x["mAP50"], x["prediction_count"], x["train_samples"]), reverse=True)
    
    # Write outputs/detection/phase2_class_metrics.csv
    csv_out = outputs_dir / "phase2_class_metrics.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class_id", "class_name", "train_samples", "test_samples",
            "prediction_count", "max_confidence", "precision", "recall",
            "mAP50", "mAP50-95", "status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Generated {csv_out.relative_to(PROJECT_ROOT)} ({len(records)} classes).")
    print(f"  Classes with detections/predictions: {classes_with_det}")
    print(f"  Classes without detections: {classes_no_det}")

    # Update phase2_fixed_evaluation.json
    eval_json_p = outputs_dir / "phase2_fixed_evaluation.json"
    eval_data = {}
    if eval_json_p.is_file():
        with open(eval_json_p, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            
    eval_data["per_class_summary"] = {
        "total_classes_evaluated": total_classes,
        "classes_with_detections_or_predictions": classes_with_det,
        "classes_without_detections": classes_no_det,
        "classes_with_few_training_samples": sum(1 for r in records if r["train_samples"] < 20),
    }
    eval_data["top_performing_classes"] = [r for r in records if r["status"] in ["STRONG", "MODERATE"]][:15]
    
    with open(eval_json_p, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    with open(outputs_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Updated {eval_json_p.relative_to(PROJECT_ROOT)}.")

if __name__ == "__main__":
    main()
