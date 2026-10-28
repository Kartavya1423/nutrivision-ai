"""
NutriVision AI — Dataset Inspector
Module 1: Dataset Inspection and Preparation

Recursively analyzes dataset directories to detect:
- Image files and integrity (detects corrupt or truncated images)
- Annotation formats (YOLO, COCO, Pascal VOC, Classification folder structures)
- Bounding-box and segmentation-mask availability
- Freshness labels and quality indicators
- Class counts, distribution, and imbalance
- Duplicate files (by name and SHA-256 hash)
- Train / Validation / Test splits
- Actionable recommendations for next development steps
"""

import os
import sys
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import Counter, defaultdict
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
ANNOTATION_EXTENSIONS = {".txt", ".json", ".xml", ".csv", ".yaml", ".yml"}
FRESHNESS_KEYWORDS = {"fresh", "rotten", "spoiled", "stale", "mold", "moldy", "ripe", "unripe", "good", "bad", "decay"}

class DatasetInspector:
    """Comprehensive dataset scanner and validator for computer vision pipelines."""

    def __init__(self, raw_dir: str):
        self.raw_dir = Path(raw_dir).resolve()
        self.results: Dict[str, Any] = {
            "dataset_path": str(self.raw_dir),
            "scan_status": "Not started",
            "total_images": 0,
            "corrupt_images": [],
            "duplicate_filenames": [],
            "duplicate_hash_groups": [],
            "annotation_formats_detected": [],
            "bounding_box_available": False,
            "segmentation_mask_available": False,
            "freshness_labels_available": False,
            "freshness_classes": [],
            "total_classes": 0,
            "class_names": [],
            "class_distribution": {},
            "splits_detected": {},
            "image_resolutions": {
                "min_width": None,
                "max_width": None,
                "min_height": None,
                "max_height": None,
                "channels": []
            },
            "capabilities": {
                "food_detection": False,
                "food_classification": False,
                "food_segmentation": False,
                "freshness_detection": False
            },
            "missing_data_warnings": [],
            "recommended_next_action": ""
        }

    def run_inspection(self) -> Dict[str, Any]:
        """Execute the complete recursive dataset inspection workflow."""
        if not self.raw_dir.exists():
            self.results["scan_status"] = f"Directory '{self.raw_dir}' does not exist."
            self.results["missing_data_warnings"].append(f"Directory {self.raw_dir} was not found.")
            self.results["recommended_next_action"] = f"Create directory {self.raw_dir} and place raw food dataset files into it."
            return self.results

        # 1. Recursive Scan
        all_image_paths = []
        all_annotation_paths = []
        filename_tracker = defaultdict(list)

        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                ext = Path(file).suffix.lower()
                full_path = Path(root) / file
                if ext in IMAGE_EXTENSIONS:
                    all_image_paths.append(full_path)
                    filename_tracker[file.lower()].append(str(full_path))
                elif ext in ANNOTATION_EXTENSIONS:
                    all_annotation_paths.append(full_path)

        self.results["total_images"] = len(all_image_paths)

        if len(all_image_paths) == 0:
            self.results["scan_status"] = "Empty: No image files found."
            self.results["missing_data_warnings"].append("No images (.jpg, .png, etc.) found in data/raw/.")
            self.results["recommended_next_action"] = (
                "Acquire food images or benchmark datasets (e.g. Food-101, Nutrition5k, Fresh & Rotten Fruits) "
                "and extract them into data/raw/ before proceeding to Module 2."
            )
            return self.results

        # 2. Check for Duplicate Filenames
        for fname, paths in filename_tracker.items():
            if len(paths) > 1:
                self.results["duplicate_filenames"].append({
                    "filename": fname,
                    "count": len(paths),
                    "occurrences": paths[:5]  # limit to first 5
                })

        # 3. Verify Image Integrity & Gather Dimensional Statistics
        corrupt_files = []
        hash_to_paths = defaultdict(list)
        widths, heights = [], []
        channels_set = set()

        for img_path in all_image_paths:
            # Hash check for exact byte duplicates (sample first 2000 images if huge)
            if len(all_image_paths) <= 3000:
                try:
                    with open(img_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read(65536)).hexdigest()
                        hash_to_paths[file_hash].append(str(img_path))
                except Exception:
                    pass

            # Integrity check
            try:
                with Image.open(img_path) as img:
                    img.verify()
                
                # Re-open for size & mode because verify() closes image
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    channels_set.add(img.mode)
            except Exception as e:
                corrupt_files.append({"path": str(img_path), "error": str(e)})

        self.results["corrupt_images"] = corrupt_files
        if widths:
            self.results["image_resolutions"] = {
                "min_width": min(widths),
                "max_width": max(widths),
                "min_height": min(heights),
                "max_height": max(heights),
                "mean_width": round(sum(widths) / len(widths), 1),
                "mean_height": round(sum(heights) / len(heights), 1),
                "modes": list(channels_set)
            }

        # Check duplicate hash groups
        dup_hashes = [paths for paths in hash_to_paths.values() if len(paths) > 1]
        self.results["duplicate_hash_groups"] = [
            {"count": len(paths), "paths": paths[:5]} for paths in dup_hashes
        ]

        # 4. Identify Dataset Splits
        splits = defaultdict(int)
        for img_path in all_image_paths:
            parts = [p.lower() for p in img_path.parts]
            if "train" in parts:
                splits["train"] += 1
            elif "val" in parts or "valid" in parts or "validation" in parts:
                splits["val"] += 1
            elif "test" in parts:
                splits["test"] += 1
            else:
                splits["unsplit"] += 1
        self.results["splits_detected"] = dict(splits)

        # 5. Identify Annotation Formats & Inspect Labels
        detected_formats = set()
        classes_found = Counter()
        bbox_found = False
        seg_found = False

        # 5a. Check COCO (.json files)
        json_files = [p for p in all_annotation_paths if p.suffix.lower() == ".json"]
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "annotations" in data and "images" in data:
                    detected_formats.add("COCO")
                    if "categories" in data:
                        cat_map = {c.get("id"): c.get("name") for c in data.get("categories", [])}
                        for ann in data.get("annotations", []):
                            cat_id = ann.get("category_id")
                            cat_name = cat_map.get(cat_id, str(cat_id))
                            classes_found[cat_name] += 1
                            if "bbox" in ann and len(ann["bbox"]) == 4:
                                bbox_found = True
                            if "segmentation" in ann and ann["segmentation"]:
                                seg_found = True
            except Exception:
                pass

        # 5b. Check Pascal VOC (.xml files)
        xml_files = [p for p in all_annotation_paths if p.suffix.lower() == ".xml"]
        for xf in xml_files[:500]: # inspect sample
            try:
                tree = ET.parse(xf)
                root = tree.getroot()
                if root.tag.lower() == "annotation":
                    detected_formats.add("Pascal VOC")
                    for obj in root.findall("object"):
                        name = obj.find("name")
                        if name is not None and name.text:
                            classes_found[name.text] += 1
                        bndbox = obj.find("bndbox")
                        if bndbox is not None:
                            bbox_found = True
            except Exception:
                pass

        # 5c. Check YOLO format (.txt alongside images or in labels/ directory)
        txt_files = [p for p in all_annotation_paths if p.suffix.lower() == ".txt"]
        yolo_candidate_count = 0
        for tf in txt_files:
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    lines = [line.strip().split() for line in f if line.strip()]
                if not lines:
                    continue
                valid_yolo_lines = 0
                for tokens in lines:
                    if len(tokens) == 5:
                        # class x y w h
                        try:
                            _ = int(tokens[0])
                            floats = [float(x) for x in tokens[1:]]
                            if all(0.0 <= x <= 1.0 for x in floats):
                                valid_yolo_lines += 1
                                bbox_found = True
                                classes_found[f"class_{tokens[0]}"] += 1
                        except ValueError:
                            pass
                    elif len(tokens) > 5 and len(tokens) % 2 == 1:
                        # class x1 y1 x2 y2 ... (polygon segmentation)
                        try:
                            _ = int(tokens[0])
                            floats = [float(x) for x in tokens[1:]]
                            if all(0.0 <= x <= 1.0 for x in floats):
                                valid_yolo_lines += 1
                                bbox_found = True
                                seg_found = True
                                classes_found[f"class_{tokens[0]}"] += 1
                        except ValueError:
                            pass
                if valid_yolo_lines > 0:
                    yolo_candidate_count += 1
            except Exception:
                pass

        if yolo_candidate_count > 0:
            detected_formats.add("YOLO")

        # 5d. Check Folder-based Classification Structure
        # If images are stored in subdirectories: e.g. data/raw/apple/*.jpg or data/raw/train/apple/*.jpg
        folder_classes = Counter()
        for img_path in all_image_paths:
            rel_parts = img_path.relative_to(self.raw_dir).parts
            if len(rel_parts) >= 2:
                # Filter out standard split or common structural folder names
                class_candidate = None
                filtered_parts = [p for p in rel_parts[:-1] if p.lower() not in {"train", "val", "valid", "validation", "test", "images", "labels", "data"}]
                if filtered_parts:
                    class_candidate = filtered_parts[-1]
                if class_candidate:
                    folder_classes[class_candidate] += 1

        if len(folder_classes) >= 2 and not detected_formats:
            detected_formats.add("Classification Folders")
            classes_found = folder_classes
        elif len(folder_classes) >= 2 and detected_formats:
            detected_formats.add("Classification Folders")
            # If classes_found only has generic indices like "class_0", replace/augment with folder names
            if all(k.startswith("class_") for k in classes_found.keys()):
                classes_found = folder_classes

        # 6. Freshness Label Analysis
        freshness_classes = []
        for cname in classes_found.keys():
            lower_name = cname.lower()
            if any(kw in lower_name for kw in FRESHNESS_KEYWORDS):
                freshness_classes.append(cname)

        has_freshness = len(freshness_classes) > 0

        # 7. Compile Results & Capabilities
        self.results["annotation_formats_detected"] = sorted(list(detected_formats))
        self.results["bounding_box_available"] = bbox_found
        self.results["segmentation_mask_available"] = seg_found
        self.results["freshness_labels_available"] = has_freshness
        self.results["freshness_classes"] = sorted(freshness_classes)
        self.results["total_classes"] = len(classes_found)
        self.results["class_names"] = sorted(list(classes_found.keys()))
        self.results["class_distribution"] = dict(classes_found.most_common(100))

        # Determine capabilities
        self.results["capabilities"]["food_detection"] = bbox_found
        self.results["capabilities"]["food_segmentation"] = seg_found
        self.results["capabilities"]["food_classification"] = (len(classes_found) >= 2)
        self.results["capabilities"]["freshness_detection"] = has_freshness

        # Assess missing data & generate recommendations
        self._evaluate_and_recommend()

        self.results["scan_status"] = "Completed successfully."
        return self.results

    def _evaluate_and_recommend(self):
        """Analyze gaps in dataset and generate actionable guidance."""
        warnings = []
        recs = []

        if len(self.results["corrupt_images"]) > 0:
            warnings.append(f"Detected {len(self.results['corrupt_images'])} corrupted or unreadable images.")
            recs.append("Remove or repair corrupted image files before training.")

        if len(self.results["duplicate_filenames"]) > 0:
            warnings.append(f"Detected {len(self.results['duplicate_filenames'])} duplicate filenames across directories.")
            recs.append("Standardize file naming to prevent split leakage or overwrite collisions.")

        if not self.results["bounding_box_available"]:
            warnings.append("No bounding-box annotations (YOLO/COCO/VOC) detected.")
            if self.results["capabilities"]["food_classification"]:
                warnings.append("Dataset has classification labels but lacks bounding boxes. It CANNOT be used directly for object detection.")
                recs.append("For Module 2 (Detection): Acquire detection annotations (YOLO/COCO) or prepare pre-cropped food items.")
            else:
                recs.append("Provide bounding box annotations in YOLO or COCO format to train the YOLO11 food detector.")

        if not self.results["segmentation_mask_available"]:
            warnings.append("No segmentation masks or polygon annotations detected.")
            recs.append("For Module 4 (Segmentation): SAM 2 will be used zero-shot with YOLO bounding-box prompts to extract masks.")

        if not self.results["freshness_labels_available"]:
            warnings.append("No freshness indicators (e.g., fresh/rotten) detected in class or folder names.")
            recs.append("For Module 6 (Freshness): Prepare a dataset with quality labels (e.g. Fresh and Rotten Fruits/Vegetables dataset).")

        splits = self.results["splits_detected"]
        if splits.get("train", 0) == 0:
            warnings.append("No distinct train/val/test splits detected in directory structure.")
            recs.append("Create train/validation/test splits (recommended 70/15/15) in data/processed/ during preparation.")

        if not recs:
            recs.append("Dataset is well-structured and ready for preprocessing and model module ingestion.")

        self.results["missing_data_warnings"] = warnings
        self.results["recommended_next_action"] = " | ".join(recs)

    def generate_json_report(self, output_path: str) -> str:
        """Write the structured JSON inspection report."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        return str(out)

    def generate_text_report(self, output_path: str) -> str:
        """Write a comprehensive human-readable text report."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        res = self.results
        lines = [
            "=" * 70,
            "               NutriVision AI -- Dataset Inspection Report              ",
            "=" * 70,
            f"Dataset Directory : {res['dataset_path']}",
            f"Inspection Status : {res['scan_status']}",
            f"Total Images      : {res['total_images']}",
            f"Total Classes     : {res['total_classes']}",
            f"Detected Formats  : {', '.join(res['annotation_formats_detected']) if res['annotation_formats_detected'] else 'None detected'}",
            "-" * 70,
            "MODULE CAPABILITY READINESS:",
            f"  [1] Food Detection (Bounding Boxes)   : {'YES (Supported)' if res['bounding_box_available'] else 'NO (Annotations missing)'}",
            f"  [2] Food Classification               : {'YES (Supported)' if res['capabilities']['food_classification'] else 'NO (Requires >=2 classes)'}",
            f"  [3] Food Segmentation (Masks/Polygons): {'YES (Supported)' if res['segmentation_mask_available'] else 'NO (Zero-shot SAM 2 required)'}",
            f"  [4] Freshness Detection               : {'YES (Supported)' if res['freshness_labels_available'] else 'NO (Freshness labels missing)'}",
            "-" * 70,
            "DATASET SPLITS:",
        ]

        if res["splits_detected"]:
            for split_name, count in res["splits_detected"].items():
                pct = (count / res['total_images'] * 100) if res['total_images'] > 0 else 0
                lines.append(f"  - {split_name:<15}: {count:>6} images ({pct:.1f}%)")
        else:
            lines.append("  No split structure detected.")

        lines.extend([
            "-" * 70,
            "IMAGE INTEGRITY & RESOLUTION:",
            f"  Corrupt Images       : {len(res['corrupt_images'])}",
            f"  Duplicate Filenames  : {len(res['duplicate_filenames'])}",
            f"  Duplicate Content Grp: {len(res['duplicate_hash_groups'])}",
        ])

        img_res = res.get("image_resolutions", {})
        if img_res.get("mean_width"):
            lines.extend([
                f"  Resolution (W x H)   : Min [{img_res['min_width']}x{img_res['min_height']}] | "
                f"Max [{img_res['max_width']}x{img_res['max_height']}] | "
                f"Mean [{img_res['mean_width']}x{img_res['mean_height']}]",
                f"  Color Modes          : {', '.join(img_res.get('modes', []))}"
            ])

        lines.extend([
            "-" * 70,
            "CLASSES DETECTED & TOP DISTRIBUTION:",
        ])
        if res["class_distribution"]:
            for cname, count in list(res["class_distribution"].items())[:20]:
                lines.append(f"  - {cname:<25}: {count:>6} samples")
            if len(res["class_distribution"]) > 20:
                lines.append(f"  ... ({len(res['class_distribution']) - 20} more classes omitted for brevity)")
        else:
            lines.append("  No classes detected.")

        if res["freshness_classes"]:
            lines.extend([
                "-" * 70,
                "FRESHNESS CATEGORIES FOUND:",
                f"  {', '.join(res['freshness_classes'])}"
            ])

        lines.extend([
            "-" * 70,
            "WARNINGS & GAPS:",
        ])
        if res["missing_data_warnings"]:
            for w in res["missing_data_warnings"]:
                lines.append(f"  [!] {w}")
        else:
            lines.append("  None. Dataset passed all validation checks.")

        lines.extend([
            "-" * 70,
            "RECOMMENDED NEXT ACTION:",
            f"  {res['recommended_next_action']}",
            "=" * 70
        ])

        report_content = "\n".join(lines)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report_content)
        return report_content
