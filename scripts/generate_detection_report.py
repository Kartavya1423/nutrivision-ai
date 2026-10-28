#!/usr/bin/env python3
"""
NutriVision AI — Detection Dataset & Annotations Report Generator
Phase 2: YOLO11 Food Detection Pipeline

Compiles outputs/detection/detection_dataset_report.json and
outputs/detection/detection_dataset_report.html with statistical summaries,
class frequencies, bounding-box densities, and visual samples with derived YOLO boxes.
"""

import base64
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
from PIL import Image, ImageDraw

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def encode_image_base64(img_pil: Image.Image, max_dim: int = 400) -> str:
    """Resizes and base64 encodes image for embedded HTML visualization."""
    from io import BytesIO
    w, h = img_pil.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img_pil = img_pil.resize((int(w * ratio), int(h * ratio)), Image.Resampling.BILINEAR)
    buf = BytesIO()
    img_pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def draw_yolo_labels(img_pil: Image.Image, label_path: Path, class_names: Dict[int, str]) -> Image.Image:
    """Draws YOLO normalized boxes on image for verification."""
    annotated = img_pil.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6"]

    if label_path.is_file():
        with open(label_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc = float(parts[1]) * w
                    yc = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h

                    x1 = int(round(xc - bw / 2.0))
                    y1 = int(round(yc - bh / 2.0))
                    x2 = int(round(xc + bw / 2.0))
                    y2 = int(round(yc + bh / 2.0))

                    color = colors[idx % len(colors)]
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    cname = class_names.get(cls_id, f"food_{cls_id}")
                    draw.text((x1 + 3, max(0, y1 - 12)), cname, fill=color)

    return annotated


def generate_reports():
    print("=" * 76)
    print("      NutriVision AI — Detection Dataset Report Generator                ")
    print("=" * 76)

    data_dir = PROJECT_ROOT / "data"
    processed_dir = data_dir / "processed"
    yolo_dir = processed_dir / "yolo_detection"
    outputs_dir = PROJECT_ROOT / "outputs" / "detection"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load bbox conversion report
    bbox_rep_file = outputs_dir / "bbox_conversion_report.json"
    bbox_stats = {}
    if bbox_rep_file.is_file():
        with open(bbox_rep_file, "r", encoding="utf-8") as f:
            bbox_stats = json.load(f)

    # 2. Load leakage report
    leak_file = outputs_dir / "data_leakage_report.json"
    leak_stats = {}
    if leak_file.is_file():
        with open(leak_file, "r", encoding="utf-8") as f:
            leak_stats = json.load(f)

    # 3. Load dataset.yaml to get class names
    yaml_file = yolo_dir / "dataset.yaml"
    class_names: Dict[int, str] = {}
    if yaml_file.is_file():
        import yaml
        with open(yaml_file, "r", encoding="utf-8") as f:
            yd = yaml.safe_load(f)
            class_names = {int(k): str(v) for k, v in yd.get("names", {}).items()}

    # 4. Box distribution calculations
    labels_train = yolo_dir / "labels" / "train"
    box_counts_per_img = []
    if labels_train.is_dir():
        for lf in labels_train.glob("*.txt"):
            with open(lf, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                box_counts_per_img.append(len(lines))

    avg_boxes = round(sum(box_counts_per_img) / len(box_counts_per_img), 2) if box_counts_per_img else 0
    min_boxes = min(box_counts_per_img) if box_counts_per_img else 0
    max_boxes = max(box_counts_per_img) if box_counts_per_img else 0

    report_json = {
        "title": "NutriVision AI — Detection Dataset Architecture & Quality Report",
        "datasets_used": ["FoodSeg103", "UECFoodPixComplete"],
        "total_classes": len(class_names),
        "total_images": sum(v.get("images", 0) for v in bbox_stats.get("boxes_per_split", {}).values()),
        "total_boxes": bbox_stats.get("total_bounding_boxes_generated", 0),
        "splits": bbox_stats.get("boxes_per_split", {}),
        "box_density": {
            "average_boxes_per_image": avg_boxes,
            "min_boxes_per_image": min_boxes,
            "max_boxes_per_image": max_boxes,
        },
        "annotations_quality": {
            "invalid_boxes": bbox_stats.get("total_invalid_boxes", 0),
            "clipped_boxes": bbox_stats.get("total_clipped_boxes", 0),
            "empty_masks": bbox_stats.get("total_empty_masks", 0),
            "annotation_type": "derived",
            "bbox_source": "derived_from_mask",
            "instance_separation": "connected_components",
        },
        "leakage_audit": leak_stats,
        "class_frequencies_top_20": list(bbox_stats.get("boxes_per_class", {}).items())[:20]
    }

    out_json_path = outputs_dir / "detection_dataset_report.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)
    print(f"  -> Saved JSON Report: {out_json_path.relative_to(PROJECT_ROOT)}")

    # 5. Visual Samples with Derived BBoxes
    visual_cards = ""
    sample_images = list((yolo_dir / "images" / "train").glob("*.*"))[:4]
    for simg in sample_images:
        slbl = yolo_dir / "labels" / "train" / f"{simg.stem}.txt"
        with Image.open(simg) as im:
            orig_b64 = encode_image_base64(im.convert("RGB"))
            annotated_im = draw_yolo_labels(im.convert("RGB"), slbl, class_names)
            ann_b64 = encode_image_base64(annotated_im)

        visual_cards += f"""
        <div style="background:#1e293b; border-radius:10px; padding:15px; border:1px solid #334155;">
          <h4 style="color:#10b981; margin-bottom:8px;">{simg.name}</h4>
          <div style="display:flex; gap:10px;">
            <div style="flex:1;">
              <p style="font-size:0.8rem; color:#94a3b8; margin-bottom:4px;">Original Image</p>
              <img src="data:image/jpeg;base64,{orig_b64}" style="width:100%; border-radius:6px;" alt="Original"/>
            </div>
            <div style="flex:1;">
              <p style="font-size:0.8rem; color:#94a3b8; margin-bottom:4px;">Derived YOLO BBoxes</p>
              <img src="data:image/jpeg;base64,{ann_b64}" style="width:100%; border-radius:6px;" alt="BBoxes"/>
            </div>
          </div>
        </div>
        """

    # 6. HTML Dashboard
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NutriVision AI — Detection Dataset Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 2.5rem; }}
    .card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
    .stat {{ background: #0f172a; border-radius: 8px; padding: 1rem; text-align: center; border: 1px solid #334155; }}
    .stat-val {{ font-size: 2rem; font-weight: 700; color: #10b981; }}
    .stat-lbl {{ color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; margin-top: 4px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }}
    .badge-pass {{ background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }}
  </style>
</head>
<body>
  <div class="card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div>
        <h1 style="margin-bottom:6px;">NutriVision AI — Detection Dataset Architecture</h1>
        <p style="color:#94a3b8;">Derived from FoodSeg103 and UECFoodPixComplete segmentation masks via connected-component analysis.</p>
      </div>
      <span class="badge badge-pass">DERIVED_FROM_SEGMENTATION</span>
    </div>
    <div class="grid" style="margin-top:1.5rem;">
      <div class="stat"><div class="stat-val">{report_json['total_images']}</div><div class="stat-lbl">Total Images</div></div>
      <div class="stat"><div class="stat-val">{report_json['total_boxes']}</div><div class="stat-lbl">Total Bounding Boxes</div></div>
      <div class="stat"><div class="stat-val">{report_json['total_classes']}</div><div class="stat-lbl">Detector Classes</div></div>
      <div class="stat"><div class="stat-val">{avg_boxes}</div><div class="stat-lbl">Avg Boxes / Image</div></div>
    </div>
  </div>

  <div class="card">
    <h2 style="margin-bottom:1rem;">Visual Verification: Derived YOLO Bounding Boxes</h2>
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(450px, 1fr)); gap:1.5rem;">
      {visual_cards}
    </div>
  </div>
</body>
</html>
"""
    out_html_path = outputs_dir / "detection_dataset_report.html"
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  -> Saved HTML Report: {out_html_path.relative_to(PROJECT_ROOT)}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    generate_reports()
