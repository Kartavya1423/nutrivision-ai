#!/usr/bin/env python3
"""
NutriVision AI — YOLO11 Food Detector Test Evaluation Script
Phase 2: YOLO11 Food Detection Pipeline

Evaluates the trained YOLO11 checkpoint against the benchmark-preserved test split.
Outputs:
    outputs/detection/evaluation.json
    outputs/detection/evaluation_report.html
    outputs/detection/confusion_matrix.png
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="NutriVision AI — Evaluate YOLO11 Model")
    parser.add_argument("--model", type=str, default="models/detection/yolo11/phase2_fixed_run/weights/best.pt", help="Path to trained weights")
    parser.add_argument("--data", type=str, default="data/processed/yolo_detection/dataset.yaml", help="dataset.yaml")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate: test or val")
    parser.add_argument("--imgsz", type=int, default=320, help="Evaluation image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output-prefix", type=str, default="phase2_fixed_", help="Prefix for output report files")
    args = parser.parse_args()

    print("=" * 76)
    print("      NutriVision AI — YOLO11 Food Detection Evaluation Engine          ")
    print("=" * 76)

    model_path = PROJECT_ROOT / args.model
    if not model_path.is_file():
        # Fallback candidates
        candidates = [
            PROJECT_ROOT / "models" / "detection" / "yolo11" / "phase2_fixed_run" / "weights" / "best.pt",
            PROJECT_ROOT / "models" / "detection" / "yolo11" / "best.pt",
            PROJECT_ROOT / "models" / "detection" / "yolo11" / "baseline_run" / "weights" / "best.pt",
        ]
        for c in candidates:
            if c.is_file():
                model_path = c
                break
        else:
            print(f"[ERROR] Trained weights not found at: {model_path}")
            sys.exit(1)

    data_path = PROJECT_ROOT / args.data
    outputs_dir = PROJECT_ROOT / "outputs" / "detection"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 1/3] Loading model from {model_path.relative_to(PROJECT_ROOT)}...")
    from ultralytics import YOLO
    import csv

    model = YOLO(str(model_path))

    print(f"\n[Step 2/3] Evaluating on '{args.split}' split from {data_path.relative_to(PROJECT_ROOT)}...")
    t0 = time.time()
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        save_json=True,
        plots=True,
        project=str(outputs_dir),
        name=f"{args.output_prefix}val_run",
        exist_ok=True,
        verbose=True
    )
    val_time = round(time.time() - t0, 2)

    # Extract metrics safely
    box_metrics = metrics.box if hasattr(metrics, "box") else metrics
    mp = float(getattr(box_metrics, "mp", 0.0) or 0.0)      # Mean precision
    mr = float(getattr(box_metrics, "mr", 0.0) or 0.0)      # Mean recall
    map50 = float(getattr(box_metrics, "map50", 0.0) or 0.0)
    map75 = float(getattr(box_metrics, "map75", 0.0) or 0.0)
    map50_95 = float(getattr(box_metrics, "map", 0.0) or 0.0)

    # Dataset stats for the split
    import yaml
    with open(data_path, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)
    base_p = Path(ds_cfg.get("path", data_path.parent))
    split_img_dir = base_p / ds_cfg.get(args.split, f"images/{args.split}")
    split_lbl_dir = base_p / f"labels/{args.split}"
    num_images = len(list(split_img_dir.glob("*.*"))) if split_img_dir.is_dir() else 0

    num_gt_boxes = 0
    if split_lbl_dir.is_dir():
        for lf in split_lbl_dir.glob("*.txt"):
            txt = lf.read_text(encoding="utf-8").strip()
            if txt:
                num_gt_boxes += len(txt.splitlines())

    print(f"\n[Step 3/3] Measured Metrics ({args.split} split):")
    print(f"  -> Images Evaluated  : {num_images}")
    print(f"  -> Ground-Truth Boxes: {num_gt_boxes}")
    print(f"  -> Precision (P)     : {mp:.4f}")
    print(f"  -> Recall (R)        : {mr:.4f}")
    print(f"  -> mAP@0.50          : {map50:.4f}")
    print(f"  -> mAP@0.75          : {map75:.4f}")
    print(f"  -> mAP@0.50:0.95     : {map50_95:.4f}")
    print(f"  -> Evaluation Time   : {val_time}s")

    # Extract per-class performance metrics
    names = getattr(metrics, "names", {}) or {}
    class_ap50 = getattr(box_metrics, "ap50", [])
    class_ap = getattr(box_metrics, "ap", [])
    class_p = getattr(box_metrics, "p", [])
    class_r = getattr(box_metrics, "r", [])
    eval_classes = getattr(box_metrics, "classes", [])

    class_records = []
    classes_with_detections = 0
    classes_zero_detections = 0

    for idx, cls_id in enumerate(eval_classes):
        cname = names.get(cls_id, f"class_{cls_id}")
        p_val = float(class_p[idx]) if idx < len(class_p) else 0.0
        r_val = float(class_r[idx]) if idx < len(class_r) else 0.0
        ap50_val = float(class_ap50[idx]) if idx < len(class_ap50) else 0.0
        ap_val = float(class_ap[idx]) if idx < len(class_ap) else 0.0

        if ap50_val > 0.0 or p_val > 0.0:
            classes_with_detections += 1
            status = "STRONG" if ap50_val >= 0.25 else "DETECTED"
        else:
            classes_zero_detections += 1
            status = "NO_DETECTIONS"

        class_records.append({
            "class_id": cls_id,
            "class_name": cname,
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "mAP50": round(ap50_val, 4),
            "mAP50_95": round(ap_val, 4),
            "status": status,
        })

    # Sort by mAP50 descending
    class_records.sort(key=lambda x: x["mAP50"], reverse=True)

    # Write outputs/detection/phase2_class_metrics.csv
    csv_path = outputs_dir / "phase2_class_metrics.csv"
    if class_records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "precision", "recall", "mAP50", "mAP50-95", "status"])
            writer.writeheader()
            for r in class_records:
                writer.writerow({
                    "class_id": r["class_id"],
                    "class_name": r["class_name"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "mAP50": r["mAP50"],
                    "mAP50-95": r["mAP50_95"],
                    "status": r["status"]
                })
        print(f"  -> Saved per-class metrics to {csv_path.relative_to(PROJECT_ROOT)}")

    # Persist JSON report
    eval_json_data = {
        "model_checkpoint": str(model_path.relative_to(PROJECT_ROOT)),
        "split_evaluated": args.split,
        "images_evaluated": num_images,
        "ground_truth_boxes": num_gt_boxes,
        "evaluation_time_seconds": val_time,
        "confidence_threshold": 0.001,
        "iou_threshold": 0.6,
        "metrics": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "mAP50": round(map50, 4),
            "mAP75": round(map75, 4),
            "mAP50_95": round(map50_95, 4),
        },
        "per_class_summary": {
            "total_classes_evaluated": len(eval_classes),
            "classes_with_detections": classes_with_detections,
            "classes_without_detections": classes_zero_detections,
        },
        "top_performing_classes": [r for r in class_records if r["mAP50"] > 0][:15],
        "device_used": args.device
    }

    # Save to phase2_fixed_evaluation.json AND evaluation.json
    p2_eval_json = outputs_dir / f"{args.output_prefix}evaluation.json"
    with open(p2_eval_json, "w", encoding="utf-8") as f:
        json.dump(eval_json_data, f, indent=2)
    with open(outputs_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_json_data, f, indent=2)

    # Save canonical Phase 2 final_evaluation/detection_metrics.json
    final_eval_dir = outputs_dir / "final_evaluation"
    final_eval_dir.mkdir(parents=True, exist_ok=True)
    with open(final_eval_dir / "detection_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_json_data, f, indent=2)

    # Copy plots and representative prediction images to final_evaluation
    run_save_dir = Path(metrics.save_dir) if hasattr(metrics, "save_dir") else (outputs_dir / f"{args.output_prefix}val_run")
    if run_save_dir.is_dir():
        for pattern in ["*.png", "*.jpg", "*.csv"]:
            for f_path in run_save_dir.glob(pattern):
                dest_f = final_eval_dir / f_path.name
                shutil.copy2(f_path, dest_f)

    print(f"  -> Saved JSON report to {p2_eval_json.relative_to(PROJECT_ROOT)}")
    print(f"  -> Saved final evaluation metrics to {(final_eval_dir / 'detection_metrics.json').relative_to(PROJECT_ROOT)}")

    # Generate HTML report
    top_rows_html = "".join([
        f"<tr><td>{r['class_id']}</td><td><b>{r['class_name']}</b></td><td>{r['precision']:.4f}</td><td>{r['recall']:.4f}</td><td><span style='color: #10b981; font-weight:600;'>{r['mAP50']:.4f}</span></td><td>{r['mAP50_95']:.4f}</td><td><span style='background:#065f46; padding: 2px 8px; border-radius: 4px;'>{r['status']}</span></td></tr>"
        for r in class_records[:20]
    ])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NutriVision AI — YOLO11 Evaluation Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
    .card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-top: 1rem; }}
    .metric-box {{ background: #0f172a; border-radius: 8px; padding: 1rem; text-align: center; border: 1px solid #334155; }}
    .metric-val {{ font-size: 2rem; font-weight: 700; color: #10b981; }}
    .metric-lbl {{ color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #334155; }}
    th {{ background: #0f172a; color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>NutriVision AI — Phase 2 YOLO11 Food Detection Evaluation</h1>
    <p style="color: #94a3b8;">Evaluated against benchmark <code>{args.split}</code> split ({num_images} images, {num_gt_boxes} ground-truth boxes).</p>
    <div class="metric-grid">
      <div class="metric-box"><div class="metric-val">{mp:.4f}</div><div class="metric-lbl">Precision</div></div>
      <div class="metric-box"><div class="metric-val">{mr:.4f}</div><div class="metric-lbl">Recall</div></div>
      <div class="metric-box"><div class="metric-val">{map50:.4f}</div><div class="metric-lbl">mAP @ 0.50</div></div>
      <div class="metric-box"><div class="metric-val">{map75:.4f}</div><div class="metric-lbl">mAP @ 0.75</div></div>
      <div class="metric-box"><div class="metric-val">{map50_95:.4f}</div><div class="metric-lbl">mAP @ 0.50:0.95</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Per-Class Performance (Top Detected Classes)</h2>
    <p style="color: #94a3b8;">Classes with detections: {classes_with_detections} / {len(eval_classes)} evaluated classes.</p>
    <table>
      <thead>
        <tr><th>Class ID</th><th>Name</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th><th>Status</th></tr>
      </thead>
      <tbody>
        {top_rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    p2_eval_html = outputs_dir / f"{args.output_prefix}evaluation_report.html"
    with open(p2_eval_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    with open(outputs_dir / "evaluation_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  -> Saved HTML dashboard to {p2_eval_html.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 76)
    print("      [SUCCESS] YOLO11 EVALUATION COMPLETED!                            ")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
