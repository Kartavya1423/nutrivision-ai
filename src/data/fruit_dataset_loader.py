"""
NutriVision AI — Updated Fruit Dataset Loader, Validator, and PyTorch Dataset Pipeline
Handles:
- Relative project root resolution and Windows paths containing spaces
- Automatic discovery of the 6 fruit classes: Apple_Good, Banana_Good, Guava_Good, Lime_Good, Orange_Good, Pomegranate_Good
- Strict image validation (corrupted, unreadable, zero-byte, unsupported formats)
- Deterministic class-to-index mapping
- Stratified, leak-free train/val/test splitting with reproducible random seed
- PyTorch Dataset and DataLoader construction with training augmentations vs. deterministic validation transforms
"""

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from src.config.settings import Settings, get_settings

logger = logging.getLogger("nutrivision.fruit_dataset")

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standard ImageNet normalization used by EfficientNet backbones
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

FRUIT_CLASSES: List[str] = [
    "Apple_Good",
    "Banana_Good",
    "Guava_Good",
    "Lime_Good",
    "Orange_Good",
    "Pomegranate_Good",
]
EXPECTED_FRUIT_CLASSES = FRUIT_CLASSES


def parse_fruit_class(class_name: str) -> Tuple[str, str]:
    """
    Parses a dataset class name like 'Apple_Good' into ('Apple', 'Good').
    Extensible for future classes such as 'Apple_Bad' or 'Banana_Rotten'.
    """
    if "_" in class_name:
        parts = class_name.split("_", 1)
        return parts[0], parts[1]
    return class_name, "Unknown"


@dataclass
class FruitClassSummary:
    """Summary metrics for an individual discovered fruit class."""
    class_name: str
    class_index: int
    fruit_name: str
    dataset_label: str
    valid_count: int
    corrupt_count: int = 0
    unsupported_count: int = 0
    file_extensions: Dict[str, int] = field(default_factory=dict)
    sample_resolutions: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "class_index": self.class_index,
            "fruit_name": self.fruit_name,
            "dataset_label": self.dataset_label,
            "valid_count": self.valid_count,
            "corrupt_count": self.corrupt_count,
            "unsupported_count": self.unsupported_count,
            "file_extensions": self.file_extensions,
            "sample_resolutions": [f"{w}x{h}" for w, h in self.sample_resolutions[:3]],
        }


@dataclass
class DatasetValidationReport:
    """Consolidated validation audit across the entire updated fruit dataset."""
    dataset_dir: Path
    is_valid: bool
    total_valid_images: int
    total_corrupted_images: int
    total_unsupported_files: int
    classes: List[str]
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
    class_summaries: Dict[str, FruitClassSummary] = field(default_factory=dict)
    corrupted_files: List[Dict[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "is_valid": self.is_valid,
            "total_valid_images": self.total_valid_images,
            "total_corrupted_images": self.total_corrupted_images,
            "total_unsupported_files": self.total_unsupported_files,
            "classes": self.classes,
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
            "class_summaries": {k: v.to_dict() for k, v in self.class_summaries.items()},
            "corrupted_files": self.corrupted_files,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class FruitDatasetValidator:
    """
    Validates and inspects the updated fruit dataset:
    - Path resolution relative to project root
    - Windows paths with spaces
    - Robust PIL header verification
    - Class imbalance detection
    - Deterministic class indexing
    """

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        dataset_dir: Optional[Union[str, Path]] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        target_path = dataset_path or dataset_dir
        if target_path is not None:
            p = Path(target_path)
            self.dataset_dir = p if p.is_absolute() else (self.settings.root_dir / p).resolve()
        else:
            self.dataset_dir = self.settings.datasets.updated_fruit_dataset_dir.resolve()

    def validate(self, quick_verify: bool = False, max_verify_per_class: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """Convenience method returning (is_valid, report_dict)."""
        report = self.validate_dataset(quick_verify=quick_verify, max_verify_per_class=max_verify_per_class)
        summary = {
            "is_valid": report.is_valid,
            "discovered_classes": report.classes,
            "class_count": len(report.classes),
            "total_valid_images": report.total_valid_images,
            "corrupted_count": report.total_corrupted_images,
            "unsupported_count": report.total_unsupported_files,
            "per_class_counts": {k: v.valid_count for k, v in report.class_summaries.items()},
            "errors": report.errors,
            "warnings": report.warnings,
        }
        return report.is_valid, summary

    def validate_dataset(self, quick_verify: bool = False, max_verify_per_class: Optional[int] = None) -> DatasetValidationReport:
        """
        Executes complete verification on the dataset directory.
        Never silently marks an invalid or empty dataset as valid.
        """
        report = DatasetValidationReport(
            dataset_dir=self.dataset_dir,
            is_valid=False,
            total_valid_images=0,
            total_corrupted_images=0,
            total_unsupported_files=0,
            classes=[],
            class_to_idx={},
            idx_to_class={},
        )

        if not self.dataset_dir.exists():
            report.errors.append(f"Dataset directory does not exist: '{self.dataset_dir}'.")
            return report

        if not self.dataset_dir.is_dir():
            report.errors.append(f"Dataset path is not a directory: '{self.dataset_dir}'.")
            return report

        # Discover class directories (ignoring hidden files / folders)
        discovered_dirs = sorted([
            d for d in self.dataset_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ], key=lambda x: x.name)

        if not discovered_dirs:
            report.errors.append(f"No class subdirectories found in '{self.dataset_dir}'. Dataset is empty.")
            return report

        classes = [d.name for d in discovered_dirs]
        report.classes = classes
        # Deterministic class mapping
        report.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        report.idx_to_class = {idx: name for idx, name in enumerate(classes)}

        total_valid = 0
        total_corrupt = 0
        total_unsupported = 0
        counts_per_class: Dict[str, int] = {}

        for d in discovered_dirs:
            cname = d.name
            cidx = report.class_to_idx[cname]
            fruit_name, ds_label = parse_fruit_class(cname)

            class_sum = FruitClassSummary(
                class_name=cname,
                class_index=cidx,
                fruit_name=fruit_name,
                dataset_label=ds_label,
                valid_count=0,
            )

            all_entries = [p for p in d.iterdir() if p.is_file() and not p.name.startswith(".")]
            if not all_entries:
                report.warnings.append(f"Class folder '{cname}' contains 0 visible files.")

            valid_in_class = 0
            verify_target = all_entries if max_verify_per_class is None else all_entries[:max_verify_per_class]

            for file_path in all_entries:
                ext = file_path.suffix.lower()
                class_sum.file_extensions[ext] = class_sum.file_extensions.get(ext, 0) + 1

                if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                    class_sum.unsupported_count += 1
                    total_unsupported += 1
                    continue

                if file_path.stat().st_size == 0:
                    class_sum.corrupt_count += 1
                    total_corrupt += 1
                    report.corrupted_files.append({"path": str(file_path), "reason": "Zero-byte file"})
                    continue

                # Full or sampled PIL integrity verification
                should_verify = (not quick_verify) or (file_path in verify_target)
                if should_verify:
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                        if len(class_sum.sample_resolutions) < 5:
                            with Image.open(file_path) as img:
                                class_sum.sample_resolutions.append(img.size)
                        valid_in_class += 1
                    except Exception as exc:
                        class_sum.corrupt_count += 1
                        total_corrupt += 1
                        report.corrupted_files.append({"path": str(file_path), "reason": str(exc)})
                else:
                    valid_in_class += 1

            class_sum.valid_count = valid_in_class
            report.class_summaries[cname] = class_sum
            counts_per_class[cname] = valid_in_class
            total_valid += valid_in_class

        report.total_valid_images = total_valid
        report.total_corrupted_images = total_corrupt
        report.total_unsupported_files = total_unsupported

        # Validation acceptance check
        if total_valid == 0:
            report.errors.append("No readable images found in any class. Dataset validation FAILED.")
            report.is_valid = False
            return report

        # Check for empty classes
        empty_classes = [c for c, count in counts_per_class.items() if count == 0]
        if empty_classes:
            report.errors.append(f"The following classes contain 0 valid images: {empty_classes}.")
            report.is_valid = False
            return report

        # Class imbalance warning
        min_count = min(counts_per_class.values())
        max_count = max(counts_per_class.values())
        if max_count >= 3 * min_count:
            report.warnings.append(
                f"Significant class imbalance detected: max class has {max_count} images while min class has {min_count} images (ratio: {max_count / min_count:.1f}x). Class-weighted loss or sampling is strongly recommended."
            )

        report.is_valid = True
        return report


class FruitImageDataset(Dataset):
    """
    PyTorch Dataset for fruit image classification:
    - Loads RGB PIL images
    - Applies configurable augmentations for training
    - Applies deterministic resize & ImageNet normalization for val/test
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
        idx_to_class: Optional[Dict[int, str]] = None,
    ):
        self.samples = samples
        self.transform = transform
        self.return_metadata = return_metadata
        self.idx_to_class = idx_to_class or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Dict[str, Any]]:
        img_path, label = self.samples[idx]
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.error(f"Failed to load image at {img_path}: {exc}")
            # Fallback to neutral gray canvas on corrupted image read
            pil_img = Image.new("RGB", (224, 224), (128, 128, 128))

        tensor = self.transform(pil_img) if self.transform else T.ToTensor()(pil_img)

        if self.return_metadata:
            class_name = self.idx_to_class.get(label, f"class_{label}")
            fruit_name, ds_label = parse_fruit_class(class_name)
            return {
                "image": tensor,
                "label": label,
                "path": str(img_path),
                "class_name": class_name,
                "fruit_name": fruit_name,
                "dataset_label": ds_label,
            }
        return tensor, label


def get_fruit_transforms(
    image_size: int = 224,
    is_training: bool = False,
    mean: List[float] = NORM_MEAN,
    std: List[float] = NORM_STD,
) -> T.Compose:
    """
    Returns appropriate PyTorch transforms for EfficientNet:
    - Training: Random horizontal flip, small rotations, mild color jitter, resized crops.
    - Evaluation: Deterministic resize and ImageNet normalization.
    """
    if is_training:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def create_stratified_fruit_splits(
    dataset_dir: Union[str, Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    output_split_path: Optional[Path] = None,
    max_samples_per_class: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Creates stratified, reproducible train/val/test splits across the 6 fruit classes.
    Ensures zero split leakage and saves a persistent manifest in data/splits/.
    """
    dataset_dir = Path(dataset_dir).resolve()
    validator = FruitDatasetValidator(dataset_path=dataset_dir)
    report = validator.validate_dataset(quick_verify=True)

    if not report.is_valid:
        raise ValueError(f"Cannot generate splits: dataset validation failed with errors: {report.errors}")

    random.seed(random_seed)
    np.random.seed(random_seed)

    split_manifest = {
        "dataset_name": "updated_fruit_dataset",
        "dataset_dir": str(dataset_dir),
        "total_samples": 0,
        "classes": report.classes,
        "class_to_idx": report.class_to_idx,
        "idx_to_class": report.idx_to_class,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "random_seed": random_seed,
        "counts": {"train": 0, "val": 0, "test": 0},
        "per_class_counts": {},
        "assignments": {},  # relative path -> split
    }

    # Deduplicate image content within each class using SHA-256 hash
    hash_seen: Set[str] = set()

    for cname in report.classes:
        cdir = dataset_dir / cname
        c_files = sorted([
            f for f in cdir.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ])

        # Content deduplication
        unique_files: List[Path] = []
        for f in c_files:
            try:
                with open(f, "rb") as fp:
                    content_hash = hashlib.sha256(fp.read(8192)).hexdigest()
                if content_hash not in hash_seen:
                    hash_seen.add(content_hash)
                    unique_files.append(f)
            except Exception:
                unique_files.append(f)

        # Shuffle deterministically
        random.shuffle(unique_files)

        if max_samples_per_class is not None and max_samples_per_class > 0:
            unique_files = unique_files[:max_samples_per_class]

        n = len(unique_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_files = unique_files[:n_train]
        val_files = unique_files[n_train:n_train + n_val]
        test_files = unique_files[n_train + n_val:]

        split_manifest["per_class_counts"][cname] = {
            "total": n,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

        for f in train_files:
            rel = str(f.relative_to(dataset_dir)).replace("\\", "/")
            split_manifest["assignments"][rel] = "train"
        for f in val_files:
            rel = str(f.relative_to(dataset_dir)).replace("\\", "/")
            split_manifest["assignments"][rel] = "val"
        for f in test_files:
            rel = str(f.relative_to(dataset_dir)).replace("\\", "/")
            split_manifest["assignments"][rel] = "test"

    split_manifest["counts"]["train"] = sum(v["train"] for v in split_manifest["per_class_counts"].values())
    split_manifest["counts"]["val"] = sum(v["val"] for v in split_manifest["per_class_counts"].values())
    split_manifest["counts"]["test"] = sum(v["test"] for v in split_manifest["per_class_counts"].values())
    split_manifest["total_samples"] = sum(split_manifest["counts"].values())

    if output_split_path is not None:
        output_split_path = Path(output_split_path)
        output_split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_split_path, "w", encoding="utf-8") as fp:
            json.dump(split_manifest, fp, indent=2)
        logger.info(f"Saved fruit split manifest to: {output_split_path}")

    return split_manifest


def build_fruit_dataloaders(
    dataset_dir: Union[str, Path],
    split_manifest: Optional[Dict[str, Any]] = None,
    split_path: Optional[Union[str, Path]] = None,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    random_seed: int = 42,
    max_samples_per_class: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Constructs PyTorch DataLoaders for (train, val, test).
    Uses num_workers=0 by default on Windows to prevent multi-processing worker spawn deadlocks.
    """
    dataset_dir = Path(dataset_dir).resolve()

    if split_manifest is None:
        if split_path is not None and Path(split_path).is_file():
            with open(split_path, "r", encoding="utf-8") as fp:
                split_manifest = json.load(fp)
        else:
            split_manifest = create_stratified_fruit_splits(
                dataset_dir=dataset_dir,
                random_seed=random_seed,
                output_split_path=split_path,
                max_samples_per_class=max_samples_per_class,
            )

    class_to_idx = split_manifest["class_to_idx"]
    idx_to_class = {int(k): v for k, v in split_manifest["idx_to_class"].items()}

    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    test_samples: List[Tuple[Path, int]] = []

    for rel_str, split_type in split_manifest["assignments"].items():
        full_path = dataset_dir / rel_str
        cname = rel_str.split("/")[0]
        if cname not in class_to_idx:
            continue
        cidx = class_to_idx[cname]

        if split_type == "train":
            train_samples.append((full_path, cidx))
        elif split_type == "val":
            val_samples.append((full_path, cidx))
        elif split_type == "test":
            test_samples.append((full_path, cidx))

    train_ds = FruitImageDataset(
        samples=train_samples,
        transform=get_fruit_transforms(image_size=image_size, is_training=True),
        idx_to_class=idx_to_class,
    )
    val_ds = FruitImageDataset(
        samples=val_samples,
        transform=get_fruit_transforms(image_size=image_size, is_training=False),
        idx_to_class=idx_to_class,
    )
    test_ds = FruitImageDataset(
        samples=test_samples,
        transform=get_fruit_transforms(image_size=image_size, is_training=False),
        idx_to_class=idx_to_class,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, split_manifest
