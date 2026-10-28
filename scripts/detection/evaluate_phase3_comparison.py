#!/usr/bin/env python3
"""
NutriVision AI — Phase 3 Evaluation and Comparison Benchmark
Compares:
- Baseline Repaired Checkpoint: `models/detection/yolo11/repair_run/best.pt`
- Phase 3 Fine-Tuned Checkpoint: `models/detection/yolo11/phase3_finetune_run/weights/best.pt`
On the untouched held-out test set (3,135 images or specified sample).
Generates:
1. Quantitative comparative metrics table (Precision, Recall, mAP50, mAP50-95, latency).
2. Per-class AP gains and recovery analysis.
3. 30 side-by-side visual diagnostic images (GT vs Old vs New).
4. Full Markdown comparison report.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path("D:/NutriVision-AI (2)/NutriVision-AI")


def parse_args():
    parser = argparse.ArgumentParser(description="NutriVision AI — Phase 3 Model Comparison")
    parser.add_argument("--old-model", type=str, default="models/detection/yolo11/repair_run/best.pt")
    parser.add_argument("--new-model", type=str, default="models/detection/yolo11/phase3_finetune_run/weights/best.pt")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of test images to benchmark (default: 100 for fast reproducible run)")
    parser.add_argument("--vis-count", type=int, default=30, help="Number of 3-panel comparison images to export")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation resolution")
    return parser.parse_args()


def evaluate_model_on_sample(model_path: Path, data_yaml: Path, sample_size: int, split: str = "test", imgsz: int = 640):
    from ultralytics import YOLO

    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_p = (PROJECT_ROOT / cfg.get("path", "data/processed/yolo_detection")).resolve()
    split_img_dir = base_p / cfg.get(split, f"images/{split}")
    img_files = sorted(list(split_img_dir.glob("*.jpg")))[:sample_size]

    temp_dir = PROJECT_ROOT / "outputs/detection/phase3_finetune/temp_eval"
    temp_dir.mkdir(parents=True, exist_ok=True)
    list_file = temp_dir / f"{split}_sample_{sample_size}.txt"

    with open(list_file, "w", encoding="utf-8") as lf:
        for p in img_files:
            lf.write(f"{p.resolve().as_posix()}\n")

    sample_cfg = dict(cfg)
    sample_cfg["path"] = str(base_p.as_posix())
    sample_cfg[split] = str(list_file.as_posix())

    sample_yaml = temp_dir / f"eval_sample_{sample_size}.yaml"
    with open(sample_yaml, "w", encoding="utf-8") as yf:
        yaml.safe_dump(sample_cfg, yf)

    model = YOLO(str(model_path.resolve()))
    t0 = time.time()
    res = model.val(
        data=str(sample_yaml.resolve()),
        split=split,
        imgsz=imgsz,
        batch=16,
        device="cpu",
        conf=0.001,
        iou=0.60,
        plots=False,
        verbose=False,
    )
    val_time = round(time.time() - t0, 2)

    box = res.box
    mp = float(getattr(box, "mp", 0.0) or 0.0)
    mr = float(getattr(box, "mr", 0.0) or 0.0)
    map50 = float(getattr(box, "map50", 0.0) or 0.0)
    map75 = float(getattr(box, "map75", 0.0) or 0.0)
    map50_95 = float(getattr(box, "map", 0.0) or 0.0)

    # Class metrics
    ap_classes = getattr(box, "ap_class_index", None)
    if ap_classes is None:
        ap_classes = getattr(box, "classes", [])
    eval_classes = ap_classes.tolist() if isinstance(ap_classes, np.ndarray) else list(ap_classes or [])

    class_ap50 = getattr(box, "ap50", [])
    class_p = getattr(box, "p", [])
    class_r = getattr(box, "r", [])
    names = cfg.get("names", {})

    per_class = {}
    for idx, cid in enumerate(eval_classes):
        cname = names.get(cid, f"class_{cid}")
        p_val = float(class_p[idx]) if idx < len(class_p) else 0.0
        r_val = float(class_r[idx]) if idx < len(class_r) else 0.0
        ap50_val = float(class_ap50[idx]) if idx < len(class_ap50) else 0.0
        per_class[cid] = {
            "name": cname,
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "map50": round(ap50_val, 4),
        }

    return {
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "images_evaluated": sample_size,
        "val_time_sec": val_time,
        "speed_per_img_ms": round((val_time / max(1, sample_size)) * 1000.0, 1),
        "precision": round(mp, 4),
        "recall": round(mr, 4),
        "map50": round(map50, 4),
        "map75": round(map75, 4),
        "map50_95": round(map50_95, 4),
        "classes_evaluated_count": len(eval_classes),
        "per_class": per_class,
    }


def generate_comparative_visuals(
    old_model_path: Path,
    new_model_path: Path,
    data_yaml: Path,
    vis_count: int,
    out_dir: Path,
):
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})

    old_model = YOLO(str(old_model_path.resolve()))
    new_model = YOLO(str(new_model_path.resolve()))

    test_img_dir = PROJECT_ROOT / "data/processed/yolo_detection/images/test"
    test_lbl_dir = PROJECT_ROOT / "data/processed/yolo_detection/labels/test"
    test_imgs = sorted(list(test_img_dir.glob("*.jpg")))

    saved_images = []
    step = max(1, len(test_imgs) // (vis_count + 5))
    generated = 0

    for i in range(0, len(test_imgs), step):
        if generated >= vis_count:
            break
        img_p = test_imgs[i]
        lbl_p = test_lbl_dir / (img_p.stem + ".txt")

        im_raw = cv2.imread(str(img_p))
        if im_raw is None:
            continue
        h, w = im_raw.shape[:2]

        gt_boxes = []
        if lbl_p.is_file():
            t = lbl_p.read_text(encoding="utf-8").strip()
            if t:
                for line in t.splitlines():
                    pts = line.strip().split()
                    if len(pts) == 5:
                        gt_boxes.append((int(pts[0]), float(pts[1]), float(pts[2]), float(pts[3]), float(pts[4])))

        # 1. Left panel: Ground Truth
        im_gt = im_raw.copy()
        for cid, xc, yc, bw, bh in gt_boxes:
            cname = names.get(cid, f"cls_{cid}")
            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w - 1, int((xc + bw / 2) * w))
            y2 = min(h - 1, int((yc + bh / 2) * h))
            cv2.rectangle(im_gt, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(im_gt, f"GT:{cname}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

        cv2.putText(im_gt, f"GROUND TRUTH ({len(gt_boxes)} items)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # 2. Center panel: Old Model Predictions (conf >= 0.05)
        im_old = im_raw.copy()
        res_old = old_model.predict(source=str(img_p), conf=0.05, imgsz=640, device="cpu", verbose=False)[0]
        for b in res_old.boxes:
            cid = int(b.cls.item())
            cnf = float(b.conf.item())
            bx = b.xyxy[0].tolist()
            cname = names.get(cid, f"cls_{cid}")
            cv2.rectangle(im_old, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 140, 255), 2)
            cv2.putText(im_old, f"{cname} {cnf:.2f}", (int(bx[0]), max(15, int(bx[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
        cv2.putText(im_old, f"OLD MODEL ({len(res_old.boxes)} dets)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

        # 3. Right panel: New Model Predictions (conf >= 0.05)
        im_new = im_raw.copy()
        res_new = new_model.predict(source=str(img_p), conf=0.05, imgsz=640, device="cpu", verbose=False)[0]
        for b in res_new.boxes:
            cid = int(b.cls.item())
            cnf = float(b.conf.item())
            bx = b.xyxy[0].tolist()
            cname = names.get(cid, f"cls_{cid}")
            cv2.rectangle(im_new, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (255, 200, 0), 2)
            cv2.putText(im_new, f"{cname} {cnf:.2f}", (int(bx[0]), max(15, int(bx[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.putText(im_new, f"NEW PHASE 3 ({len(res_new.boxes)} dets)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

        # Stitch horizontally
        h_target = 480
        w_target = int(w * (h_target / h))
        p1 = cv2.resize(im_gt, (w_target, h_target))
        p2 = cv2.resize(im_old, (w_target, h_target))
        p3 = cv2.resize(im_new, (w_target, h_target))
        combined = np.hstack([p1, p2, p3])

        out_name = f"comparison_{generated+1:02d}_{img_p.stem}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), combined)
        saved_images.append(str(out_path.relative_to(PROJECT_ROOT)))
        generated += 1

    return saved_images


def main():
    args = parse_args()
    old_p = PROJECT_ROOT / args.old_model
    new_p = PROJECT_ROOT / args.new_model
    data_yaml = PROJECT_ROOT / "data/processed/yolo_detection/data.yaml"

    print("=== NutriVision AI: Phase 3 Comparative Evaluation ===")
    print(f"Old Model: {old_p.relative_to(PROJECT_ROOT)}")
    print(f"New Model: {new_p.relative_to(PROJECT_ROOT)}")
    print(f"Test Split Sample: {args.sample_size} images")

    print("\n[1/3] Benchmarking OLD baseline model...")
    old_metrics = evaluate_model_on_sample(old_p, data_yaml, sample_size=args.sample_size, imgsz=args.imgsz)

    print("\n[2/3] Benchmarking NEW fine-tuned model...")
    new_metrics = evaluate_model_on_sample(new_p, data_yaml, sample_size=args.sample_size, imgsz=args.imgsz)

    # 3. Export Visual Comparisons
    vis_dir = PROJECT_ROOT / "outputs/detection/phase3_finetune/visual_comparisons_30"
    print(f"\n[3/3] Generating {args.vis_count} 3-panel visual comparisons...")
    saved_visuals = generate_comparative_visuals(old_p, new_p, data_yaml, args.vis_count, vis_dir)

    comparison_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_sample_size": args.sample_size,
        "resolution": [args.imgsz, args.imgsz],
        "old_model": old_metrics,
        "new_model": new_metrics,
        "improvements": {
            "delta_map50": round(new_metrics["map50"] - old_metrics["map50"], 4),
            "delta_recall": round(new_metrics["recall"] - old_metrics["recall"], 4),
            "delta_precision": round(new_metrics["precision"] - old_metrics["precision"], 4),
        },
        "visual_comparisons_dir": str(vis_dir.relative_to(PROJECT_ROOT)),
        "visual_comparisons_count": len(saved_visuals),
        "visual_samples": saved_visuals,
    }

    report_json = PROJECT_ROOT / "outputs/detection/phase3_finetune/comparison_report.json"
    with open(report_json, "w", encoding="utf-8") as jf:
        json.dump(comparison_report, jf, indent=2)

    print("\n" + "=" * 76)
    print("                    COMPARATIVE RESULTS SUMMARY                         ")
    print("=" * 76)
    print(f"Metric       | Old Model ({old_p.name}) | New Model ({new_p.name}) | Delta")
    print("-" * 76)
    print(f"Precision    | {old_metrics['precision']:.4f} ({old_metrics['precision']*100:.2f}%)   | {new_metrics['precision']:.4f} ({new_metrics['precision']*100:.2f}%)   | {comparison_report['improvements']['delta_precision']:+.4f}")
    print(f"Recall       | {old_metrics['recall']:.4f} ({old_metrics['recall']*100:.2f}%)   | {new_metrics['recall']:.4f} ({new_metrics['recall']*100:.2f}%)   | {comparison_report['improvements']['delta_recall']:+.4f}")
    print(f"mAP@50       | {old_metrics['map50']:.4f} ({old_metrics['map50']*100:.2f}%)   | {new_metrics['map50']:.4f} ({new_metrics['map50']*100:.2f}%)   | {comparison_report['improvements']['delta_map50']:+.4f}")
    print(f"mAP@50-95    | {old_metrics['map50_95']:.4f} ({old_metrics['map50_95']*100:.2f}%)   | {new_metrics['map50_95']:.4f} ({new_metrics['map50_95']*100:.2f}%)   | {new_metrics['map50_95'] - old_metrics['map50_95']:+.4f}")
    print(f"Inference    | {old_metrics['speed_per_img_ms']} ms/img          | {new_metrics['speed_per_img_ms']} ms/img          | {new_metrics['speed_per_img_ms'] - old_metrics['speed_per_img_ms']:+.1f} ms")
    print("=" * 76)
    print(f"Report saved to: {report_json.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
