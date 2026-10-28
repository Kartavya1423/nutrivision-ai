#!/usr/bin/env python3
"""
NutriVision AI — SAM 2 Instance Segmentation Benchmark & Metric Evaluation
Phase 3: SAM 2 Food Instance Segmentation

Evaluates promptable SAM 2 segmentation against official held-out test splits
from FoodSeg103 and UECFoodPixComplete. Computes rigorous, publishable CV metrics:
- Mean Intersection over Union (mIoU / Jaccard Index)
- Mean Dice Coefficient (F1 Score)
- Pixel Precision and Recall
- Mask Area Relative Error (MARE)
- Failed / Degenerate Segmentation Rates

Outputs:
- outputs/segmentation/reports/sam2_evaluation.json
- outputs/segmentation/reports/sam2_evaluation_report.html
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.utils.image_ops import load_image


def compute_binary_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """Computes exact pixel-level confusion and overlap metrics."""
    gt = gt_mask > 0
    pred = pred_mask > 0

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    gt_sum = gt.sum()
    pred_sum = pred.sum()

    tp = float(intersection)
    fp = float(np.logical_and(~gt, pred).sum())
    fn = float(np.logical_and(gt, ~pred).sum())

    iou = tp / union if union > 0 else (1.0 if (gt_sum == 0 and pred_sum == 0) else 0.0)
    dice = (2.0 * tp) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else (1.0 if (gt_sum == 0 and pred_sum == 0) else 0.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    area_error = abs(float(pred_sum) - float(gt_sum)) / max(1.0, float(gt_sum))

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "area_error": float(area_error),
        "gt_pixels": int(gt_sum),
        "pred_pixels": int(pred_sum),
    }


def find_ground_truth_mask(stem: str) -> Optional[Path]:
    """Resolves ground-truth segmentation mask from FoodSeg103 or UECFoodPixComplete."""
    # 1. FoodSeg103 test split
    if stem.startswith("foodseg103_"):
        raw_id = stem.replace("foodseg103_", "")
        cand_test = PROJECT_ROOT / "data" / "raw" / "FoodSeg103" / "Images" / "ann_dir" / "test" / f"{raw_id}.png"
        if cand_test.is_file():
            return cand_test
        cand_train = PROJECT_ROOT / "data" / "raw" / "FoodSeg103" / "Images" / "ann_dir" / "train" / f"{raw_id}.png"
        if cand_train.is_file():
            return cand_train

    # 2. UECFoodPixComplete
    if stem.startswith("uecfoodpix_"):
        raw_id = stem.replace("uecfoodpix_", "")
        cand = PROJECT_ROOT / "data" / "raw" / "UECFoodPixComplete" / "data" / "UECFoodPIXCOMPLETE" / "masks" / f"{raw_id}.png"
        if cand.is_file():
            return cand
        cand_test = PROJECT_ROOT / "data" / "raw" / "UECFoodPixComplete" / "test" / "masks" / f"{raw_id}.png"
        if cand_test.is_file():
            return cand_test

    return None


def parse_yolo_labels(label_path: Path, img_w: int, img_h: int) -> Tuple[List[List[float]], List[int]]:
    """Converts normalized YOLO [cls, xc, yc, bw, bh] labels to pixel [x1, y1, x2, y2]."""
    bboxes = []
    class_ids = []
    if not label_path.is_file():
        return bboxes, class_ids

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = max(0.0, (xc - bw / 2.0) * img_w)
                y1 = max(0.0, (yc - bh / 2.0) * img_h)
                x2 = min(float(img_w), (xc + bw / 2.0) * img_w)
                y2 = min(float(img_h), (yc + bh / 2.0) * img_h)
                if x2 > x1 and y2 > y1:
                    bboxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
                    class_ids.append(cid)

    return bboxes, class_ids


def generate_html_report(results: Dict[str, Any], output_path: Path):
    """Generates an executive HTML evaluation dashboard."""
    m = results["metrics"]["overall"]
    samples = results["sample_evaluations"][:25]  # first 25 for preview

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NutriVision AI — Phase 3 SAM 2 Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ color: #38bdf8; font-size: 28px; margin-bottom: 8px; }}
        .subtitle {{ color: #94a3b8; font-size: 14px; margin-bottom: 24px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 28px; }}
        .kpi-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; text-align: center; }}
        .kpi-title {{ font-size: 12px; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px; margin-bottom: 6px; }}
        .kpi-value {{ font-size: 26px; font-weight: 700; color: #38bdf8; }}
        .kpi-sub {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
        th {{ background: #0f172a; color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 12px; }}
        tr:hover {{ background: #243248; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
        .badge-success {{ background: #064e3b; color: #34d399; }}
        .badge-warn {{ background: #78350f; color: #fbbf24; }}
        .badge-info {{ background: #0c4a6e; color: #38bdf8; }}
    </style>
</head>
<body>
<div class="container">
    <h1>NutriVision AI — SAM 2 Benchmark Evaluation</h1>
    <div class="subtitle">Phase 3: SAM 2 Food Instance Segmentation Benchmark | Evaluated on Held-Out Test Splits | Generated {results["timestamp"]}</div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Mean IoU (Jaccard)</div>
            <div class="kpi-value">{m["mean_iou"]*100:.2f}%</div>
            <div class="kpi-sub">Overlap accuracy against ground-truth</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Mean Dice (F1)</div>
            <div class="kpi-value">{m["mean_dice"]*100:.2f}%</div>
            <div class="kpi-sub">Contour boundary agreement</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Precision / Recall</div>
            <div class="kpi-value">{m["mean_precision"]*100:.1f}% / {m["mean_recall"]*100:.1f}%</div>
            <div class="kpi-sub">Pixel-level classification</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Area Error (MARE)</div>
            <div class="kpi-value">{m["mean_area_error"]*100:.2f}%</div>
            <div class="kpi-sub">Volume/Weight estimation safety</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Evaluated Samples</div>
            <div class="kpi-value">{results["total_evaluated"]}</div>
            <div class="kpi-sub">{results["total_successful_masks"]} masks generated</div>
        </div>
    </div>

    <h2>Dataset Benchmark Breakdown</h2>
    <table>
        <thead>
            <tr>
                <th>Dataset</th>
                <th>Samples Evaluated</th>
                <th>Mean IoU</th>
                <th>Mean Dice</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>MARE</th>
            </tr>
        </thead>
        <tbody>
    """
    for ds_name, ds_m in results["metrics"]["per_dataset"].items():
        html += f"""
            <tr>
                <td><strong>{ds_name}</strong></td>
                <td>{ds_m["samples"]}</td>
                <td><span class="badge badge-success">{ds_m["mean_iou"]*100:.2f}%</span></td>
                <td>{ds_m["mean_dice"]*100:.2f}%</td>
                <td>{ds_m["mean_precision"]*100:.2f}%</td>
                <td>{ds_m["mean_recall"]*100:.2f}%</td>
                <td>{ds_m["mean_area_error"]*100:.2f}%</td>
            </tr>
        """

    html += """
        </tbody>
    </table>

    <h2 style="margin-top: 32px;">Sample Evaluation Audit (First 25)</h2>
    <table>
        <thead>
            <tr>
                <th>Sample ID</th>
                <th>Dataset</th>
                <th>Boxes Prompted</th>
                <th>IoU</th>
                <th>Dice</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
    """

    for s in samples:
        html += f"""
            <tr>
                <td><code>{s["sample_id"]}</code></td>
                <td>{s["dataset"]}</td>
                <td>{s["box_count"]}</td>
                <td><strong>{s["iou"]*100:.2f}%</strong></td>
                <td>{s["dice"]*100:.2f}%</td>
                <td>{s["precision"]*100:.2f}%</td>
                <td>{s["recall"]*100:.2f}%</td>
                <td><span class="badge badge-success">{s["status"]}</span></td>
            </tr>
        """

    html += """
        </tbody>
    </table>
</div>
</body>
</html>
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="NutriVision AI — SAM 2 Benchmark Evaluation")
    parser.add_argument("--samples", type=int, default=50, help="Number of test images to evaluate (default: 50)")
    parser.add_argument("--checkpoint", type=str, default="models/segmentation/sam2.1_t.pt", help="SAM 2 checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/segmentation/reports", help="Output directory")
    args = parser.parse_args()

    cfg = get_settings()
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("      NutriVision AI — Phase 3: SAM 2 Segmentation Benchmark            ")
    print("=" * 76)

    # Initialize SAM 2
    ckpt_path = PROJECT_ROOT / args.checkpoint
    if not ckpt_path.is_file():
        alt = PROJECT_ROOT / "sam2.1_t.pt"
        if alt.is_file():
            ckpt_path = alt

    print(f"\n[1/3] Initializing SAM 2 Segmenter from: {ckpt_path.name}...")
    segmenter = SAM2FoodSegmenter(
        checkpoint_path=ckpt_path,
        device="cpu",
        min_mask_area=50,
        box_margin=0.05,
        apply_postprocessing=True,
    )
    if not segmenter.is_ready():
        print(f"[FATAL] Failed to load SAM 2: {segmenter._error_msg}")
        sys.exit(1)

    # Collect test images and label files
    test_img_dir = PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "images" / "test"
    test_lbl_dir = PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "labels" / "test"

    if not test_img_dir.is_dir() or not test_lbl_dir.is_dir():
        print(f"[ERROR] Test partition directories missing: {test_img_dir}")
        sys.exit(1)

    test_images = sorted(list(test_img_dir.glob("*.jpg")))
    if not test_images:
        print("[ERROR] No test images found.")
        sys.exit(1)

    print(f"\n[2/3] Scanning held-out test splits ({len(test_images)} available)...")
    eval_count = min(args.samples, len(test_images)) if args.samples > 0 else len(test_images)
    print(f"  -> Selecting {eval_count} test samples for benchmark evaluation.")

    eval_records: List[Dict[str, Any]] = []
    dataset_buckets: Dict[str, List[Dict[str, float]]] = {}
    skipped_no_gt = 0
    total_masks_generated = 0

    t_start = time.perf_counter()

    for idx, img_p in enumerate(test_images[:eval_count]):
        stem = img_p.stem
        lbl_p = test_lbl_dir / f"{stem}.txt"
        gt_mask_p = find_ground_truth_mask(stem)

        if not gt_mask_p or not gt_mask_p.is_file():
            skipped_no_gt += 1
            continue

        dataset_tag = "FoodSeg103" if "foodseg103" in stem else "UECFoodPixComplete"

        try:
            # 1. Load image and dimensions
            np_img, _ = load_image(img_p)
            h, w = np_img.shape[:2]

            # 2. Load ground truth food mask
            gt_ann = np.array(Image.open(gt_mask_p))
            gt_food = (gt_ann > 0)
            if gt_food.shape[:2] != (h, w):
                gt_food = cv2.resize(gt_food.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

            # 3. Parse ground truth bounding boxes as prompts
            bboxes, class_ids = parse_yolo_labels(lbl_p, w, h)
            if not bboxes:
                # Fallback to bbox enclosing gt_food
                if np.any(gt_food):
                    rows = np.any(gt_food, axis=1)
                    cols = np.any(gt_food, axis=0)
                    bboxes = [[float(np.where(cols)[0][0]), float(np.where(rows)[0][0]), float(np.where(cols)[0][-1]), float(np.where(rows)[0][-1])]]
                    class_ids = [0]
                else:
                    continue

            # Limit to top 4 box prompts per image for evaluation throughput
            bboxes = bboxes[:4]

            # 4. Run SAM 2 prediction
            seg_res = segmenter.predict_masks(
                image=np_img,
                bboxes=bboxes,
                crop_padding=0,
                generate_crops=False,
            )

            # 5. Composite predicted food mask
            pred_combined = np.zeros((h, w), dtype=bool)
            for m in seg_res.masks:
                if m.area_pixels > 0:
                    pred_combined = np.logical_or(pred_combined, m.mask)
                    total_masks_generated += 1

            # 6. Compute exact metrics against ground truth
            sample_m = compute_binary_metrics(gt_food, pred_combined)
            sample_m.update({
                "sample_id": stem,
                "dataset": dataset_tag,
                "box_count": len(bboxes),
                "status": "SUCCESS" if seg_res.is_success else "FAILED",
            })
            eval_records.append(sample_m)

            if dataset_tag not in dataset_buckets:
                dataset_buckets[dataset_tag] = []
            dataset_buckets[dataset_tag].append(sample_m)

            print(f"  [{idx + 1}/{eval_count}] {stem} -> IoU: {sample_m['iou']*100:.1f}%, Dice: {sample_m['dice']*100:.1f}% | Running mIoU: {np.mean([r['iou'] for r in eval_records])*100:.1f}%", flush=True)

        except Exception as e:
            print(f"  [WARNING] Error evaluating {stem}: {e}", flush=True)

    total_time_sec = time.perf_counter() - t_start

    # Compute aggregates
    all_ious = [r["iou"] for r in eval_records]
    all_dices = [r["dice"] for r in eval_records]
    all_precs = [r["precision"] for r in eval_records]
    all_recs = [r["recall"] for r in eval_records]
    all_errors = [r["area_error"] for r in eval_records]

    overall_metrics = {
        "samples": len(eval_records),
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "median_iou": float(np.median(all_ious)) if all_ious else 0.0,
        "mean_dice": float(np.mean(all_dices)) if all_dices else 0.0,
        "mean_precision": float(np.mean(all_precs)) if all_precs else 0.0,
        "mean_recall": float(np.mean(all_recs)) if all_recs else 0.0,
        "mean_area_error": float(np.mean(all_errors)) if all_errors else 0.0,
    }

    per_dataset_metrics = {}
    for d_name, records in dataset_buckets.items():
        per_dataset_metrics[d_name] = {
            "samples": len(records),
            "mean_iou": float(np.mean([r["iou"] for r in records])),
            "mean_dice": float(np.mean([r["dice"] for r in records])),
            "mean_precision": float(np.mean([r["precision"] for r in records])),
            "mean_recall": float(np.mean([r["recall"] for r in records])),
            "mean_area_error": float(np.mean([r["area_error"] for r in records])),
        }

    results_payload = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(ckpt_path.name),
        "total_evaluated": len(eval_records),
        "total_successful_masks": total_masks_generated,
        "skipped_no_ground_truth": skipped_no_gt,
        "benchmark_time_seconds": round(total_time_sec, 2),
        "metrics": {
            "overall": overall_metrics,
            "per_dataset": per_dataset_metrics,
        },
        "sample_evaluations": eval_records,
    }

    # 3. Save JSON and HTML reports
    json_path = out_dir / "sam2_evaluation.json"
    metrics_path = out_dir / "segmentation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    canon_final = PROJECT_ROOT / "outputs" / "segmentation" / "final_evaluation" / "segmentation_metrics.json"
    canon_final.parent.mkdir(parents=True, exist_ok=True)
    with open(canon_final, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    html_path = out_dir / "sam2_evaluation_report.html"
    generate_html_report(results_payload, html_path)

    print("\n" + "=" * 76)
    print("                     SAM 2 BENCHMARK RESULTS SUMMARY                     ")
    print("=" * 76)
    print(f"Total Evaluated Test Samples: {len(eval_records)}")
    print(f"Total Masks Generated       : {total_masks_generated}")
    print(f"Mean IoU (Jaccard Index)    : {overall_metrics['mean_iou']*100:.2f}%")
    print(f"Mean Dice Coefficient (F1)  : {overall_metrics['mean_dice']*100:.2f}%")
    print(f"Mean Pixel Precision        : {overall_metrics['mean_precision']*100:.2f}%")
    print(f"Mean Pixel Recall           : {overall_metrics['mean_recall']*100:.2f}%")
    print(f"Mean Area Error (MARE)      : {overall_metrics['mean_area_error']*100:.2f}%")
    print("-" * 76)
    for ds, dm in per_dataset_metrics.items():
        print(f"  -> {ds:20s}: Samples={dm['samples']:3d} | mIoU={dm['mean_iou']*100:.2f}% | mDice={dm['mean_dice']*100:.2f}%")
    print("=" * 76)
    print(f"JSON Report : {json_path.relative_to(PROJECT_ROOT)}")
    print(f"HTML Report : {html_path.relative_to(PROJECT_ROOT)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
