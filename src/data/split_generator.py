"""
NutriVision AI — Train / Validation / Test Split Generator
Phase 1: Dataset Inspection, Validation, Normalization, and Preparation

Implements reproducible, leakage-free dataset splitting:
- Configurable split ratios (default: 70% Train, 15% Val, 15% Test).
- Deterministic random seed (configurable from centralized Settings).
- Strictly preserves dataset-provided official benchmark splits (e.g. FoodSeg103,
  UECFoodPixComplete, Fruits-360): official Test sets remain completely untouched.
- Saves split records in data/splits/ for downstream reproducibility.
"""

from dataclasses import dataclass, field
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.config.settings import get_settings


@dataclass
class DatasetSplitMetadata:
    """Metadata detailing exact splits assigned to images."""
    dataset_name: str
    total_samples: int
    train_count: int
    val_count: int
    test_count: int
    split_ratios: Dict[str, float]
    random_seed: int
    official_splits_preserved: bool
    split_assignments: Dict[str, str] = field(default_factory=dict)  # relative_path -> split

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "split_ratios": self.split_ratios,
            "random_seed": self.random_seed,
            "official_splits_preserved": self.official_splits_preserved,
            "sample_assignments": dict(list(self.split_assignments.items())[:100])
        }


class SplitGenerator:
    """
    Generates deterministic, leak-proof dataset partitions while
    respecting benchmark-defined official splits.
    """

    def __init__(
        self,
        splits_dir: Optional[Path] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: Optional[int] = None
    ):
        settings = get_settings()
        self.splits_dir = Path(splits_dir).resolve() if splits_dir else settings.datasets.splits_root
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed if seed is not None else settings.seed

        total = round(train_ratio + val_ratio + test_ratio, 5)
        if total != 1.0:
            raise ValueError(f"Split ratios must sum to 1.0 (got {total})")

    def generate_split(
        self,
        dataset_name: str,
        image_paths: List[Path],
        official_splits: Optional[List[str]] = None,
        class_labels: Optional[Dict[Path, str]] = None
    ) -> DatasetSplitMetadata:
        """
        Creates or derives train/val/test splits for a given set of image paths:
        - If official splits (e.g. 'train'/'test' or 'Training'/'Test') exist in paths,
          the official Test set is locked and preserved. An internal validation set is
          carved strictly from the official Training set according to val_ratio.
        - If un-split, performs deterministic stratified or random splitting.
        """
        if not image_paths:
            return DatasetSplitMetadata(
                dataset_name=dataset_name,
                total_samples=0,
                train_count=0,
                val_count=0,
                test_count=0,
                split_ratios={"train": self.train_ratio, "val": self.val_ratio, "test": self.test_ratio},
                random_seed=self.seed,
                official_splits_preserved=False
            )

        rng = random.Random(self.seed)

        # Check if paths already belong to official train/test sets
        official_test_paths = []
        official_train_paths = []
        unassigned_paths = []

        for p in sorted(image_paths):
            posix_p = p.as_posix().lower()
            if "/test/" in posix_p or "\\test\\" in posix_p or "test" in p.parts:
                official_test_paths.append(p)
            elif "/train/" in posix_p or "\\train\\" in posix_p or "training" in p.parts or "train" in p.parts:
                official_train_paths.append(p)
            else:
                unassigned_paths.append(p)

        has_official_test = len(official_test_paths) > 0
        has_official_train = len(official_train_paths) > 0

        split_assignments: Dict[str, str] = {}

        if has_official_train or has_official_test:
            # BENCHMARK PRESERVATION MODE:
            # 1. Official Test is kept 100% intact as 'test'
            for p in official_test_paths:
                split_assignments[str(p)] = "test"

            # 2. Carve Validation strictly out of official Train
            all_train_candidates = official_train_paths + unassigned_paths
            # Shuffle deterministically
            rng.shuffle(all_train_candidates)

            # Fraction of train candidates that should go to validation
            # E.g. if default is 70 train, 15 val, val fraction of (train+val) is 15/(70+15) = 17.6%
            val_frac = self.val_ratio / (self.train_ratio + self.val_ratio) if (self.train_ratio + self.val_ratio) > 0 else 0.15
            num_val = int(round(len(all_train_candidates) * val_frac))
            # Ensure at least 1 sample in val if we have sufficient samples
            if num_val == 0 and len(all_train_candidates) >= 5:
                num_val = 1

            val_paths = all_train_candidates[:num_val]
            train_paths = all_train_candidates[num_val:]

            for p in val_paths:
                split_assignments[str(p)] = "val"
            for p in train_paths:
                split_assignments[str(p)] = "train"

            official_preserved = True

        else:
            # UN-SPLIT DATASET: Generate fresh deterministic split
            all_paths = list(image_paths)
            rng.shuffle(all_paths)

            n_total = len(all_paths)
            n_test = int(round(n_total * self.test_ratio))
            n_val = int(round(n_total * self.val_ratio))
            n_train = n_total - n_test - n_val

            train_paths = all_paths[:n_train]
            val_paths = all_paths[n_train:n_train + n_val]
            test_paths = all_paths[n_train + n_val:]

            for p in train_paths:
                split_assignments[str(p)] = "train"
            for p in val_paths:
                split_assignments[str(p)] = "val"
            for p in test_paths:
                split_assignments[str(p)] = "test"

            official_preserved = False

        # Counts
        t_count = sum(1 for s in split_assignments.values() if s == "train")
        v_count = sum(1 for s in split_assignments.values() if s == "val")
        te_count = sum(1 for s in split_assignments.values() if s == "test")

        metadata = DatasetSplitMetadata(
            dataset_name=dataset_name,
            total_samples=len(image_paths),
            train_count=t_count,
            val_count=v_count,
            test_count=te_count,
            split_ratios={
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio
            },
            random_seed=self.seed,
            official_splits_preserved=official_preserved,
            split_assignments=split_assignments
        )

        # Persist split metadata to data/splits/
        self.save_split_metadata(metadata)
        return metadata

    def save_split_metadata(self, metadata: DatasetSplitMetadata) -> Path:
        """Saves partition assignments to data/splits/<dataset>_split.json."""
        out_file = self.splits_dir / f"{metadata.dataset_name}_split.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_name": metadata.dataset_name,
                "total_samples": metadata.total_samples,
                "train_count": metadata.train_count,
                "val_count": metadata.val_count,
                "test_count": metadata.test_count,
                "split_ratios": metadata.split_ratios,
                "random_seed": metadata.random_seed,
                "official_splits_preserved": metadata.official_splits_preserved,
                "assignments": metadata.split_assignments
            }, f, indent=2)
        return out_file
