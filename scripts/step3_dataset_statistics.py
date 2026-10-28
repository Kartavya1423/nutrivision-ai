#!/usr/bin/env python3
"""
NutriVision AI — Step 3 YOLO Dataset Complete Statistics Generator
Computes exhaustive dataset metrics, bounding-box geometry distributions,
object-size categorization, and class-level frequencies across splits.
Outputs:
  outputs/detection/dataset_statistics.json
  outputs/detection/dataset_statistics.html
  outputs/detection/class_distribution.png
  outputs/detection/bounding_box_geometry.png
"""

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import time
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "detection"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_YAML = PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "data.yaml"

def main():
    print("=" * 76)
    print("NutriVision AI — Generating YOLO Dataset Statistics (Step 3)")
    print("=" * 76)
    t0 = time.time()

    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    names = data_cfg.get("names", {})
    num_classes = len(names)
    print(f"Loaded {num_classes} classes from data.yaml")

    base_dir = PROJECT_ROOT / data_cfg.get("path", "data/processed/yolo_detection")
    labels_base = base_dir / "labels"
    images_base = base_dir / "images"

    splits = ["train", "val", "test"]
    split_stats = {}
    
    total_images = 0
    total_boxes = 0
    boxes_per_image = []
    
    boxes_per_class = Counter()
    images_per_class = defaultdict(set)
    
    box_widths = []
    box_heights = []
    box_areas = []
    box_aspect_ratios = []

    # Object size bins:
    # Normalized area: small < 0.0025 (32x32 in 640x640), medium: [0.0025, 0.0225], large: > 0.0225 (96x96 in 640x640)
    size_bins = {"small": 0, "medium": 0, "large": 0}

    for split in splits:
        split_lbl_dir = labels_base / split
        split_img_dir = images_base / split
        
        lbl_files = sorted(list(split_lbl_dir.glob("*.txt")))
        img_files = sorted(list(split_img_dir.glob("*.jpg")))
        
        s_images = len(img_files)
        s_boxes = 0
        s_boxes_per_img = []
        
        for lf in lbl_files:
            img_id = lf.stem
            lines = lf.read_text(encoding="utf-8").strip().splitlines()
            cnt = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    
                    area = bw * bh
                    ar = bw / max(bh, 1e-6)
                    
                    boxes_per_class[cid] += 1
                    images_per_class[cid].add(f"{split}/{img_id}")
                    
                    box_widths.append(bw)
                    box_heights.append(bh)
                    box_areas.append(area)
                    box_aspect_ratios.append(ar)
                    
                    if area < 0.0025:
                        size_bins["small"] += 1
                    elif area < 0.0225:
                        size_bins["medium"] += 1
                    else:
                        size_bins["large"] += 1
                    
                    cnt += 1
            s_boxes += cnt
            s_boxes_per_img.append(cnt)
            boxes_per_image.append(cnt)

        total_images += s_images
        total_boxes += s_boxes
        
        split_stats[split] = {
            "images": s_images,
            "labels": len(lbl_files),
            "boxes": s_boxes,
            "avg_boxes_per_image": round(s_boxes / max(s_images, 1), 2),
            "max_boxes_in_single_image": max(s_boxes_per_img) if s_boxes_per_img else 0,
            "empty_images_count": sum(1 for c in s_boxes_per_img if c == 0)
        }
        print(f"Split [{split}]: {s_images:,} images, {s_boxes:,} boxes (avg {split_stats[split]['avg_boxes_per_image']} boxes/img)")

    bw_arr = np.array(box_widths)
    bh_arr = np.array(box_heights)
    ba_arr = np.array(box_areas)
    ar_arr = np.array(box_aspect_ratios)

    geometry_stats = {
        "width": {
            "mean": round(float(np.mean(bw_arr)), 4),
            "std": round(float(np.std(bw_arr)), 4),
            "median": round(float(np.median(bw_arr)), 4),
            "p25": round(float(np.percentile(bw_arr, 25)), 4),
            "p75": round(float(np.percentile(bw_arr, 75)), 4),
            "min": round(float(np.min(bw_arr)), 4),
            "max": round(float(np.max(bw_arr)), 4)
        },
        "height": {
            "mean": round(float(np.mean(bh_arr)), 4),
            "std": round(float(np.std(bh_arr)), 4),
            "median": round(float(np.median(bh_arr)), 4),
            "p25": round(float(np.percentile(bh_arr, 25)), 4),
            "p75": round(float(np.percentile(bh_arr, 75)), 4),
            "min": round(float(np.min(bh_arr)), 4),
            "max": round(float(np.max(bh_arr)), 4)
        },
        "aspect_ratio": {
            "mean": round(float(np.mean(ar_arr)), 4),
            "median": round(float(np.median(ar_arr)), 4),
            "p25": round(float(np.percentile(ar_arr, 25)), 4),
            "p75": round(float(np.percentile(ar_arr, 75)), 4)
        },
        "object_size_distribution": {
            "small_objects_pct": round(size_bins["small"] / total_boxes * 100, 2),
            "small_objects_count": size_bins["small"],
            "medium_objects_pct": round(size_bins["medium"] / total_boxes * 100, 2),
            "medium_objects_count": size_bins["medium"],
            "large_objects_pct": round(size_bins["large"] / total_boxes * 100, 2),
            "large_objects_count": size_bins["large"]
        }
    }

    # Per-class summary
    class_stats = []
    for cid in range(num_classes):
        cname = names.get(cid, f"class_{cid}")
        c_boxes = boxes_per_class.get(cid, 0)
        c_imgs = len(images_per_class.get(cid, set()))
        class_stats.append({
            "class_id": cid,
            "class_name": cname,
            "total_boxes": c_boxes,
            "total_images": c_imgs,
            "box_frequency_pct": round(c_boxes / max(total_boxes, 1) * 100, 3)
        })

    # Top 20 and Bottom 20 classes
    sorted_by_boxes = sorted(class_stats, key=lambda x: x["total_boxes"], reverse=True)
    top20 = sorted_by_boxes[:20]
    bottom20 = sorted_by_boxes[-20:]

    dataset_statistics = {
        "timestamp": "2026-10-05T23:50:00Z",
        "dataset_name": "NutriVision Unified YOLO Food Detection Dataset",
        "dataset_yaml": str(DATA_YAML.relative_to(PROJECT_ROOT)),
        "num_classes": num_classes,
        "summary": {
            "total_images": total_images,
            "total_bounding_boxes": total_boxes,
            "average_boxes_per_image": round(total_boxes / total_images, 2),
            "min_boxes_per_image": min(boxes_per_image),
            "max_boxes_per_image": max(boxes_per_image)
        },
        "splits": split_stats,
        "bounding_box_geometry": geometry_stats,
        "class_imbalance": {
            "max_boxes_in_class": sorted_by_boxes[0]["total_boxes"],
            "min_boxes_in_class": sorted_by_boxes[-1]["total_boxes"],
            "imbalance_ratio": round(sorted_by_boxes[0]["total_boxes"] / max(sorted_by_boxes[-1]["total_boxes"], 1), 2),
            "top_10_classes": [{"name": x["class_name"], "boxes": x["total_boxes"]} for x in sorted_by_boxes[:10]],
            "bottom_10_classes": [{"name": x["class_name"], "boxes": x["total_boxes"]} for x in sorted_by_boxes[-10:]]
        },
        "all_classes": class_stats
    }

    # Save JSON
    json_path = OUTPUTS_DIR / "dataset_statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_statistics, f, indent=2)
    print(f"Saved {json_path}")

    # ---------------------------------------------------------
    # Generate Visualizations (Matplotlib)
    # ---------------------------------------------------------
    print("Generating statistical plots...")
    
    # Plot 1: Top 25 vs Bottom 25 Class Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#0b0f19")
    
    # Top 20
    top_names = [x["class_name"].replace("_", " ") for x in sorted_by_boxes[:20]][::-1]
    top_counts = [x["total_boxes"] for x in sorted_by_boxes[:20]][::-1]
    ax1.set_facecolor("#151d30")
    ax1.barh(top_names, top_counts, color="#38bdf8", edgecolor="#0284c7")
    ax1.set_title("Top 20 Most Frequent Food Classes", color="#f1f5f9", fontsize=13, fontweight="bold")
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(color="#232f48", linestyle="--", alpha=0.6)
    for spine in ax1.spines.values():
        spine.set_color("#232f48")
        
    # Bottom 20
    bot_names = [x["class_name"].replace("_", " ") for x in sorted_by_boxes[-20:]][::-1]
    bot_counts = [x["total_boxes"] for x in sorted_by_boxes[-20:]][::-1]
    ax2.set_facecolor("#151d30")
    ax2.barh(bot_names, bot_counts, color="#f59e0b", edgecolor="#d97706")
    ax2.set_title("Bottom 20 Tail Food Classes", color="#f1f5f9", fontsize=13, fontweight="bold")
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(color="#232f48", linestyle="--", alpha=0.6)
    for spine in ax2.spines.values():
        spine.set_color("#232f48")
        
    plt.tight_layout()
    plot1_path = OUTPUTS_DIR / "class_distribution.png"
    plt.savefig(plot1_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {plot1_path}")

    # Plot 2: Bounding Box Geometry & Sizes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0b0f19")

    # Bbox Width Histogram
    ax1.set_facecolor("#151d30")
    ax1.hist(bw_arr, bins=40, color="#10b981", edgecolor="#059669", alpha=0.85)
    ax1.set_title("Normalized Bounding Box Width", color="#f1f5f9", fontweight="bold")
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(color="#232f48", linestyle="--", alpha=0.6)

    # Bbox Height Histogram
    ax2.set_facecolor("#151d30")
    ax2.hist(bh_arr, bins=40, color="#6366f1", edgecolor="#4f46e5", alpha=0.85)
    ax2.set_title("Normalized Bounding Box Height", color="#f1f5f9", fontweight="bold")
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(color="#232f48", linestyle="--", alpha=0.6)

    # Boxes per Image Distribution
    ax3.set_facecolor("#151d30")
    ax3.hist(boxes_per_image, bins=range(0, 20), color="#ec4899", edgecolor="#db2777", alpha=0.85, rwidth=0.8)
    ax3.set_title("Food Items per Image Distribution", color="#f1f5f9", fontweight="bold")
    ax3.tick_params(colors="#94a3b8")
    ax3.grid(color="#232f48", linestyle="--", alpha=0.6)

    # Object Size Category Donut Chart
    ax4.set_facecolor("#151d30")
    labels = [f"Small ({size_bins['small']:,})", f"Medium ({size_bins['medium']:,})", f"Large ({size_bins['large']:,})"]
    sizes = [size_bins["small"], size_bins["medium"], size_bins["large"]]
    colors = ["#f59e0b", "#38bdf8", "#10b981"]
    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors,
                                       textprops={"color": "#f1f5f9", "fontsize": 10}, wedgeprops={"edgecolor": "#0b0f19", "width": 0.4})
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_fontweight("bold")
    ax4.set_title("Object Scale Breakdown (COCO Metric)", color="#f1f5f9", fontweight="bold")

    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_color("#232f48")

    plt.tight_layout()
    plot2_path = OUTPUTS_DIR / "bounding_box_geometry.png"
    plt.savefig(plot2_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {plot2_path}")

    # ---------------------------------------------------------
    # Generate HTML Dashboard
    # ---------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NutriVision AI — Detection Dataset Statistics</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    body {{ font-family: 'Inter', sans-serif; background: #0b0f19; color: #f1f5f9; padding: 40px 24px; line-height: 1.6; }}
    .container {{ max-width: 1300px; margin: 0 auto; }}
    .card {{ background: #151d30; border: 1px solid #232f48; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
    .badge {{ display: inline-block; padding: 4px 12px; font-size: 0.75rem; font-weight: 700; border-radius: 9999px; text-transform: uppercase; }}
    .badge-primary {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }}
    .stat-val {{ font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #fff; }}
    .stat-lbl {{ font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #232f48; }}
    th {{ background: #111827; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; }}
  </style>
</head>
<body>
  <div class="container">
    <div style="margin-bottom: 32px;">
      <span class="badge badge-primary">STEP 3 &bull; YOLO DATASET METRICS</span>
      <h1 style="font-size: 2.2rem; font-weight: 800; margin-top: 8px;">NutriVision AI — Detection Dataset Statistical Profile</h1>
      <p style="color: #94a3b8;">Comprehensive empirical distribution of images, bounding boxes, object scales, and class frequencies across 17,118 images and 198 classes.</p>
    </div>

    <div class="grid">
      <div class="card">
        <div class="stat-val">{total_images:,}</div>
        <div class="stat-lbl">Total Images</div>
      </div>
      <div class="card">
        <div class="stat-val">{total_boxes:,}</div>
        <div class="stat-lbl">Total Food Bounding Boxes</div>
      </div>
      <div class="card">
        <div class="stat-val">{num_classes}</div>
        <div class="stat-lbl">Unified Food Classes</div>
      </div>
      <div class="card">
        <div class="stat-val">{dataset_statistics['summary']['average_boxes_per_image']}</div>
        <div class="stat-lbl">Average Boxes / Image</div>
      </div>
    </div>

    <div class="card">
      <h3>Partition Breakdown</h3>
      <table>
        <thead>
          <tr>
            <th>Partition Split</th>
            <th>Image Count</th>
            <th>Percentage</th>
            <th>Bounding Boxes</th>
            <th>Avg Boxes / Image</th>
            <th>Max Boxes / Image</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Training (Train)</strong></td>
            <td>{split_stats['train']['images']:,}</td>
            <td>{round(split_stats['train']['images']/total_images*100, 1)}%</td>
            <td>{split_stats['train']['boxes']:,}</td>
            <td>{split_stats['train']['avg_boxes_per_image']}</td>
            <td>{split_stats['train']['max_boxes_in_single_image']}</td>
          </tr>
          <tr>
            <td><strong>Validation (Val)</strong></td>
            <td>{split_stats['val']['images']:,}</td>
            <td>{round(split_stats['val']['images']/total_images*100, 1)}%</td>
            <td>{split_stats['val']['boxes']:,}</td>
            <td>{split_stats['val']['avg_boxes_per_image']}</td>
            <td>{split_stats['val']['max_boxes_in_single_image']}</td>
          </tr>
          <tr>
            <td><strong>Test Set (Held-Out)</strong></td>
            <td>{split_stats['test']['images']:,}</td>
            <td>{round(split_stats['test']['images']/total_images*100, 1)}%</td>
            <td>{split_stats['test']['boxes']:,}</td>
            <td>{split_stats['test']['avg_boxes_per_image']}</td>
            <td>{split_stats['test']['max_boxes_in_single_image']}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <h3>Object Geometry and Size Categorization</h3>
      <div class="grid">
        <div>
          <p style="color: #94a3b8; font-size: 0.9rem;"><strong>Mean Box Width:</strong> {geometry_stats['width']['mean']}</p>
          <p style="color: #94a3b8; font-size: 0.9rem;"><strong>Median Box Width:</strong> {geometry_stats['width']['median']}</p>
        </div>
        <div>
          <p style="color: #94a3b8; font-size: 0.9rem;"><strong>Mean Box Height:</strong> {geometry_stats['height']['mean']}</p>
          <p style="color: #94a3b8; font-size: 0.9rem;"><strong>Median Box Height:</strong> {geometry_stats['height']['median']}</p>
        </div>
        <div>
          <p style="color: #94a3b8; font-size: 0.9rem;"><strong>Small Scale Objects:</strong> {geometry_stats['object_size_distribution']['small_objects_pct']}% ({geometry_stats['object_size_distribution']['small_objects_count']:,})</p>
          <p style="color: #94a3b8; font-size: 0.9rem;"><strong>Large Scale Objects:</strong> {geometry_stats['object_size_distribution']['large_objects_pct']}% ({geometry_stats['object_size_distribution']['large_objects_count']:,})</p>
        </div>
      </div>
    </div>
  </div>
</body>
</html>"""
    html_path = OUTPUTS_DIR / "dataset_statistics.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved {html_path}")
    print(f"Dataset statistics generation completed in {time.time()-t0:.2f}s!")

if __name__ == "__main__":
    main()
