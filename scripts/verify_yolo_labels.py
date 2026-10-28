#!/usr/bin/env python3
"""
NutriVision AI — YOLO Label Correctness Verification
Phase 2.2: Inspect representative sample of training images, render YOLO bounding
boxes, and produce visual verification artifacts under outputs/detection/label_verification/.
"""

import json
import os
from pathlib import Path
import sys
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "detection" / "label_verification"


def load_class_names() -> dict:
    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        yd = yaml.safe_load(f)
    return {int(k): str(v) for k, v in yd.get("names", {}).items()}


def verify_labels():
    print("=" * 76)
    print("    NutriVision AI — Visual YOLO Label Verification Engine              ")
    print("=" * 76)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names()

    train_img_dir = PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "images" / "train"
    train_lbl_dir = PROJECT_ROOT / "data" / "processed" / "yolo_detection" / "labels" / "train"

    # Select representative samples: multiple items, single items, different datasets
    candidates = [
        "foodseg103_00000001",
        "foodseg103_00000010",
        "foodseg103_00000025",
        "foodseg103_00000050",
        "foodseg103_00000100",
        "foodseg103_00000200",
        "uecfoodpix_00001",
        "uecfoodpix_00015",
        "uecfoodpix_00050",
        "uecfoodpix_00100",
        "fruits360_apple_106",
        "fruits360_cherry_1",
    ]

    # Find existing files matching stems or fall back to any files with boxes
    all_imgs = sorted(list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png")))
    img_by_stem = {p.stem: p for p in all_imgs}

    selected_stems = []
    for cand in candidates:
        if cand in img_by_stem:
            selected_stems.append(cand)

    # Fill up to 12 if some stems didn't match exactly
    if len(selected_stems) < 12:
        for p in all_imgs:
            lbl_file = train_lbl_dir / f"{p.stem}.txt"
            if lbl_file.is_file() and lbl_file.stat().st_size > 0:
                if p.stem not in selected_stems:
                    selected_stems.append(p.stem)
                    if len(selected_stems) >= 12:
                        break

    palette = [
        (16, 185, 129),   # Emerald
        (59, 130, 246),   # Blue
        (245, 158, 11),   # Amber
        (239, 68, 68),    # Red
        (139, 92, 246),   # Purple
        (236, 72, 153),   # Pink
        (20, 184, 166),   # Teal
        (249, 115, 22),   # Orange
    ]

    verification_records = []

    for stem in selected_stems:
        img_path = img_by_stem[stem]
        lbl_path = train_lbl_dir / f"{stem}.txt"

        with Image.open(img_path) as im:
            orig = im.convert("RGB")
            w, h = orig.size

        # Create canvas with top banner for image dimensions and metadata
        banner_h = 36
        canvas = Image.new("RGB", (w, h + banner_h), (15, 23, 42))
        draw = ImageDraw.Draw(canvas)

        # Draw banner
        banner_text = f"Image: {img_path.name} | Dimensions: {w}x{h} px"
        draw.text((12, 10), banner_text, fill=(248, 250, 252))

        # Paste original image below banner
        canvas.paste(orig, (0, banner_h))
        draw_im = ImageDraw.Draw(canvas)

        boxes_info = []
        if lbl_path.is_file():
            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc = float(parts[1]) * w
                    yc = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h

                    x1 = max(0, int(round(xc - bw / 2.0)))
                    y1 = max(0, int(round(yc - bh / 2.0))) + banner_h
                    x2 = min(w - 1, int(round(xc + bw / 2.0)))
                    y2 = min(h - 1, int(round(yc + bh / 2.0))) + banner_h

                    color = palette[idx % len(palette)]
                    draw_im.rectangle([x1, y1, x2, y2], outline=color, width=3)

                    cname = class_names.get(cls_id, f"class_{cls_id}")
                    label_str = f"#{cls_id} {cname}"

                    # Tag background
                    tag_h = 16
                    tag_w = len(label_str) * 7 + 8
                    draw_im.rectangle([x1, max(banner_h, y1 - tag_h), x1 + tag_w, y1], fill=color)
                    draw_im.text((x1 + 4, max(banner_h, y1 - tag_h) + 2), label_str, fill=(255, 255, 255))

                    boxes_info.append({
                        "class_id": cls_id,
                        "class_name": cname,
                        "bbox_xyxy": [x1, y1 - banner_h, x2, y2 - banner_h],
                        "bbox_norm": [round(float(parts[1]), 4), round(float(parts[2]), 4), round(float(parts[3]), 4), round(float(parts[4]), 4)]
                    })

        out_img_path = OUTPUT_DIR / f"verified_{stem}.jpg"
        canvas.save(out_img_path, quality=92)
        print(f"Verified {img_path.name} ({w}x{h}, {len(boxes_info)} boxes) -> {out_img_path.name}")

        verification_records.append({
            "image": str(img_path.name),
            "dimensions": {"width": w, "height": h},
            "box_count": len(boxes_info),
            "boxes": boxes_info,
            "verification_image": str(out_img_path.relative_to(PROJECT_ROOT))
        })

    summary_path = OUTPUT_DIR / "verification_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_verified_images": len(verification_records),
            "records": verification_records
        }, f, indent=2)

    print(f"\n[SUCCESS] Label verification complete! Saved {len(verification_records)} artifacts to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    verify_labels()
