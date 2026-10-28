"""
NutriVision AI — Dataset Statistics Generator
Phase 1: Dataset Inspection, Validation, Normalization, and Preparation

Computes structural dataset metrics:
- Image and class frequency distributions.
- Class imbalance metrics (max/min ratio, gini coefficient / entropy).
- Resolution distributions (min, max, mean, aspect ratios).
- Format and split breakdowns.
- Fruits-360 specific class distribution tables (Class | Train Images | Test Images | Total).
"""

from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

from src.data.dataset_scanner import DatasetScanResult


@dataclass
class DatasetStatisticsSummary:
    """Comprehensive statistical metrics for a dataset."""
    name: str
    total_images: int
    total_classes: int
    format_distribution: Dict[str, int]
    split_distribution: Dict[str, int]
    imbalance_ratio: Optional[float] = None
    min_class_count: Optional[int] = None
    max_class_count: Optional[int] = None
    mean_class_count: Optional[float] = None
    top_5_classes: List[Tuple[str, int]] = field(default_factory=list)
    bottom_5_classes: List[Tuple[str, int]] = field(default_factory=list)
    resolution_summary: Dict[str, Any] = field(default_factory=dict)
    fruits360_table: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_images": self.total_images,
            "total_classes": self.total_classes,
            "format_distribution": self.format_distribution,
            "split_distribution": self.split_distribution,
            "imbalance_ratio": round(self.imbalance_ratio, 2) if self.imbalance_ratio else None,
            "min_class_count": self.min_class_count,
            "max_class_count": self.max_class_count,
            "mean_class_count": round(self.mean_class_count, 2) if self.mean_class_count else None,
            "top_5_classes": self.top_5_classes,
            "bottom_5_classes": self.bottom_5_classes,
            "resolution_summary": self.resolution_summary,
            "fruits360_table_rows": len(self.fruits360_table)
        }


class DatasetStatistics:
    """Computes distributions, class imbalance, and resolution profiles."""

    @staticmethod
    def compute_summary(scan: DatasetScanResult) -> DatasetStatisticsSummary:
        """Derive complete statistical summary from a DatasetScanResult."""
        summary = DatasetStatisticsSummary(
            name=scan.name,
            total_images=scan.image_count,
            total_classes=scan.classes_count,
            format_distribution=scan.image_extensions,
            split_distribution=scan.splits_detected
        )

        # 1. Class Distribution & Imbalance
        dist = scan.class_distribution
        if dist:
            counts = list(dist.values())
            min_c = min(counts) if counts else 0
            max_c = max(counts) if counts else 0
            mean_c = sum(counts) / len(counts) if counts else 0
            imbalance = (max_c / min_c) if (min_c > 0) else None

            sorted_classes = sorted(dist.items(), key=lambda x: x[1], reverse=True)

            summary.min_class_count = min_c
            summary.max_class_count = max_c
            summary.mean_class_count = mean_c
            summary.imbalance_ratio = imbalance
            summary.top_5_classes = sorted_classes[:5]
            summary.bottom_5_classes = sorted_classes[-5:]

        # 2. Resolution Summary
        summary.resolution_summary = scan.image_dimensions

        # 3. Fruits-360 specific table if applicable
        if "Fruits360" in scan.name and scan.path.exists():
            summary.fruits360_table = DatasetStatistics._build_fruits360_table(scan.path)

        return summary

    @staticmethod
    def _build_fruits360_table(root: Path) -> List[Dict[str, Any]]:
        """
        Builds the Class | Train Images | Test Images | Total table
        for Fruits-360 from actual downloaded folders.
        """
        table: List[Dict[str, Any]] = []
        train_dir = None
        test_dir = None

        for cand in [root / "Training", root / "train", root / "fruits-360" / "Training"]:
            if cand.is_dir():
                train_dir = cand
                break
        for cand in [root / "Test", root / "test", root / "fruits-360" / "Test"]:
            if cand.is_dir():
                test_dir = cand
                break

        train_counts: Dict[str, int] = Counter()
        test_counts: Dict[str, int] = Counter()

        if train_dir:
            for cdir in train_dir.iterdir():
                if cdir.is_dir():
                    train_counts[cdir.name] = len([f for f in cdir.iterdir() if f.is_file() and not f.name.startswith(".")])

        if test_dir:
            for cdir in test_dir.iterdir():
                if cdir.is_dir():
                    test_counts[cdir.name] = len([f for f in cdir.iterdir() if f.is_file() and not f.name.startswith(".")])

        all_classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
        for cname in all_classes:
            tr = train_counts.get(cname, 0)
            te = test_counts.get(cname, 0)
            table.append({
                "class_name": cname,
                "train_count": tr,
                "test_count": te,
                "total_count": tr + te
            })

        return table

    @staticmethod
    def save_fruits360_statistics_csv(table: List[Dict[str, Any]], out_path: Path) -> Path:
        """Saves Fruits-360 per-class counts to data/processed/fruits360_statistics.csv."""
        import csv
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_name", "train_count", "test_count", "total_count"])
            for row in table:
                writer.writerow([
                    row["class_name"],
                    row.get("train_count", row.get("train_images", 0)),
                    row.get("test_count", row.get("test_images", 0)),
                    row.get("total_count", row.get("total_images", 0))
                ])
        return out_path
