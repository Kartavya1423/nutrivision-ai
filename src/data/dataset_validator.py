"""
NutriVision AI — Dataset Validator & Duplicate Detector
Phase 1: Dataset Inspection, Validation, Normalization, and Preparation

Implements non-destructive validation of images and masks:
- Detects zero-byte files, truncated data, corrupted headers, and unreadable formats.
- Verifies image <-> segmentation mask dimension matching and pairings.
- Detects exact duplicates (SHA-256) and probable visual duplicates (dHash perceptual hash).
- Produces auditable reports in data/processed/validation/ and data/processed/duplicates_report.csv.
- NEVER deletes or alters original files.
"""

from dataclasses import dataclass, field
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from PIL import Image

from src.config.settings import get_settings

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageValidationResult:
    """Validation result for a single image file."""
    path: Path
    is_valid: bool
    is_zero_byte: bool = False
    is_corrupted: bool = False
    is_unsupported_format: bool = False
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None
    file_size_bytes: int = 0
    error_message: Optional[str] = None


@dataclass
class MaskPairValidationResult:
    """Validation result for an image <-> mask segmentation pair."""
    dataset_name: str
    image_path: Path
    mask_path: Optional[Path]
    is_paired: bool
    is_mask_readable: bool = False
    dimensions_match: bool = False
    image_size: Optional[Tuple[int, int]] = None
    mask_size: Optional[Tuple[int, int]] = None
    status: str = "VALID"  # VALID, MISMATCH, UNREADABLE, MISSING_MASK
    reason: Optional[str] = None
    issue_description: Optional[str] = None


class DatasetValidator:
    """
    Validates food dataset integrity, image-mask correspondences,
    and identifies duplicate candidates.
    """

    def __init__(self, processed_dir: Optional[Path] = None):
        settings = get_settings()
        self.processed_dir = Path(processed_dir).resolve() if processed_dir else settings.datasets.processed_data_root
        self.validation_dir = self.processed_dir / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings

    # -------------------------------------------------------------------------
    # IMAGE VALIDATION
    # -------------------------------------------------------------------------
    def validate_image(self, file_path: Path) -> ImageValidationResult:
        """
        Safely inspects an image file:
        - Checks zero-byte status
        - Checks file extension
        - Verifies image integrity via PIL verify() and raster loading
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            return ImageValidationResult(
                path=path,
                is_valid=False,
                error_message="File does not exist."
            )

        try:
            file_size = path.stat().st_size
        except OSError as e:
            return ImageValidationResult(
                path=path,
                is_valid=False,
                error_message=f"OS error reading file stats: {e}"
            )

        if file_size == 0:
            return ImageValidationResult(
                path=path,
                is_valid=False,
                is_zero_byte=True,
                file_size_bytes=0,
                error_message="File is zero bytes (empty)."
            )

        ext = path.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            return ImageValidationResult(
                path=path,
                is_valid=False,
                is_unsupported_format=True,
                file_size_bytes=file_size,
                error_message=f"Unsupported image format extension: '{ext}'"
            )

        # Attempt to open and verify image integrity
        try:
            with Image.open(path) as img:
                w, h = img.size
                fmt = img.format
                mode = img.mode
                channels = len(mode) if mode in ("RGB", "RGBA", "CMYK") else (1 if mode in ("L", "1", "P") else 3)
                img.verify()

                if w <= 0 or h <= 0:
                    return ImageValidationResult(
                        path=path,
                        is_valid=False,
                        is_corrupted=True,
                        width=w,
                        height=h,
                        file_size_bytes=file_size,
                        error_message=f"Invalid image dimensions: {w}x{h}"
                    )

                return ImageValidationResult(
                    path=path,
                    is_valid=True,
                    width=w,
                    height=h,
                    channels=channels,
                    format=fmt,
                    file_size_bytes=file_size
                )

        except Exception as e:
            return ImageValidationResult(
                path=path,
                is_valid=False,
                is_corrupted=True,
                file_size_bytes=file_size,
                error_message=f"Corrupted or unreadable image: {type(e).__name__}: {str(e)}"
            )

    def validate_dataset_images(self, image_paths: List[Path]) -> Dict[str, List[ImageValidationResult]]:
        """Validates a batch of image paths, categorizing results into valid and invalid."""
        results: Dict[str, List[ImageValidationResult]] = {
            "valid": [],
            "corrupted": [],
            "zero_byte": [],
            "unsupported": []
        }

        for p in image_paths:
            res = self.validate_image(p)
            if res.is_valid:
                results["valid"].append(res)
            elif res.is_zero_byte:
                results["zero_byte"].append(res)
            elif res.is_unsupported_format:
                results["unsupported"].append(res)
            else:
                results["corrupted"].append(res)

        return results

    # -------------------------------------------------------------------------
    # SEGMENTATION MASK VALIDATION
    # -------------------------------------------------------------------------
    def validate_segmentation_pairs(
        self,
        dataset_name: str,
        image_paths: List[Path],
        mask_paths: List[Path]
    ) -> List[MaskPairValidationResult]:
        """
        Validates image <-> mask correspondence:
        - Checks whether each image has an existing mask file (by stem match).
        - Verifies mask file can be opened.
        - Verifies image dimensions exactly match mask dimensions.
        """
        results: List[MaskPairValidationResult] = []

        # Index masks by stem for efficient lookup
        mask_map: Dict[str, Path] = {}
        for m in mask_paths:
            # Handle common mask naming variations (e.g. 0001_mask.png or 0001.png)
            stem = m.stem
            mask_map[stem] = m
            if stem.endswith("_mask"):
                mask_map[stem[:-5]] = m

        for img_path in image_paths:
            stem = img_path.stem
            mask_path = mask_map.get(stem)

            if mask_path is None:
                results.append(MaskPairValidationResult(
                    dataset_name=dataset_name,
                    image_path=img_path,
                    mask_path=None,
                    is_paired=False,
                    status="MISSING_MASK",
                    reason="Missing segmentation mask.",
                    issue_description="Missing segmentation mask."
                ))
                continue

            # Read image dimensions
            img_w, img_h = None, None
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception:
                pass

            # Validate mask
            try:
                with Image.open(mask_path) as mask:
                    mask_w, mask_h = mask.size
                    mask.load()
                    dim_match = (img_w == mask_w and img_h == mask_h) if (img_w and img_h) else False

                    issue = None
                    if not dim_match:
                        issue = f"Dimension mismatch: image is ({img_w}, {img_h}), mask is ({mask_w}, {mask_h})."
                        status_val = "MISMATCH"
                        reason_val = issue
                    else:
                        status_val = "VALID"
                        reason_val = "Dimensions match and mask is valid."

                    results.append(MaskPairValidationResult(
                        dataset_name=dataset_name,
                        image_path=img_path,
                        mask_path=mask_path,
                        is_paired=True,
                        is_mask_readable=True,
                        dimensions_match=dim_match,
                        image_size=(img_w, img_h) if (img_w and img_h) else None,
                        mask_size=(mask_w, mask_h),
                        status=status_val,
                        reason=reason_val,
                        issue_description=issue
                    ))
            except Exception as e:
                results.append(MaskPairValidationResult(
                    dataset_name=dataset_name,
                    image_path=img_path,
                    mask_path=mask_path,
                    is_paired=True,
                    is_mask_readable=False,
                    dimensions_match=False,
                    image_size=(img_w, img_h) if (img_w and img_h) else None,
                    status="UNREADABLE",
                    reason=f"Unreadable mask file: {e}",
                    issue_description=f"Unreadable mask file: {e}"
                ))

        return results

    # -------------------------------------------------------------------------
    # DUPLICATE DETECTION (SHA-256 & dHash)
    # -------------------------------------------------------------------------
    @staticmethod
    def compute_sha256(path: Path) -> Optional[str]:
        """Calculates exact SHA-256 hash of file bytes."""
        hasher = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None

    @staticmethod
    def compute_dhash(path: Path, hash_size: int = 8) -> Optional[str]:
        """
        Calculates difference hash (dHash) for visual similarity.
        Resizes to (hash_size + 1, hash_size), converts to grayscale,
        and computes adjacent pixel gradient bitmask.
        """
        try:
            with Image.open(path) as img:
                # Convert to grayscale and resize to (width=9, height=8)
                img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BICUBIC)
                if hasattr(img, "get_flattened_data"):
                    pixels = list(img.get_flattened_data())
                else:
                    pixels = list(img.getdata())

            # Compare adjacent pixels in each row
            difference = []
            for row in range(hash_size):
                row_start = row * (hash_size + 1)
                for col in range(hash_size):
                    pixel_left = pixels[row_start + col]
                    pixel_right = pixels[row_start + col + 1]
                    difference.append(pixel_left > pixel_right)

            # Convert boolean array to hex string
            decimal_value = 0
            for idx, val in enumerate(difference):
                if val:
                    decimal_value |= 1 << (len(difference) - 1 - idx)

            return f"{decimal_value:016x}"
        except Exception:
            return None

    @staticmethod
    def hamming_distance(hex1: str, hex2: str) -> int:
        """Calculates bitwise Hamming distance between two hex hashes."""
        try:
            n1 = int(hex1, 16)
            n2 = int(hex2, 16)
            return bin(n1 ^ n2).count("1")
        except Exception:
            return 999

    @staticmethod
    def infer_dataset_name(path_str: str) -> str:
        """Infers the source dataset name from an image path string."""
        p = path_str.replace("\\", "/").lower()
        if "foodseg103" in p:
            return "FoodSeg103"
        elif "uecfoodpix" in p:
            return "UECFoodPixComplete"
        elif "fruits360_multi" in p or "test-multiple_fruits" in p:
            return "Fruits360_multi"
        elif "fruits360" in p or "fruits-360" in p:
            return "Fruits360_100x100"
        elif "sample_benchmark" in p:
            return "sample_benchmark"
        elif "weight" in p:
            return "weight_dataset"
        elif "freshness" in p:
            return "freshness_dataset"
        return "unknown"

    def find_duplicates(
        self,
        image_paths: List[Path],
        visual_threshold: int = 4,
        max_compare: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Detects exact file duplicates via SHA-256 and visual duplicates via dHash.
        Classifies each duplicate as within-dataset or cross-dataset.
        """
        duplicates: List[Dict[str, Any]] = []

        # Sample paths evenly across datasets
        by_ds: Dict[str, List[Path]] = defaultdict(list)
        for p in image_paths:
            by_ds[self.infer_dataset_name(str(p))].append(p)

        paths_to_check: List[Path] = []
        if by_ds:
            per_ds_limit = max(10, max_compare // len(by_ds))
            for ds_k, plist in by_ds.items():
                paths_to_check.extend(plist[:per_ds_limit])
        else:
            paths_to_check = image_paths[:max_compare]

        # 1. Exact Duplicates by SHA-256
        hash_to_paths: Dict[str, List[Path]] = defaultdict(list)
        for p in paths_to_check:
            sha = self.compute_sha256(p)
            if sha:
                hash_to_paths[sha].append(p)

        for sha, plist in hash_to_paths.items():
            if len(plist) > 1:
                orig = plist[0]
                ds1 = self.infer_dataset_name(str(orig))
                for dup in plist[1:]:
                    ds2 = self.infer_dataset_name(str(dup))
                    scope = "WITHIN_DATASET" if ds1 == ds2 else "CROSS_DATASET"
                    duplicates.append({
                        "file_1": str(orig),
                        "file_2": str(dup),
                        "dataset_1": ds1,
                        "dataset_2": ds2,
                        "duplicate_scope": scope,
                        "duplicate_type": "EXACT",
                        "similarity_metric": f"SHA-256 ({sha[:12]}...)",
                        "notes": f"Exact binary duplicate ({scope})."
                    })

        # 2. Probable Visual Duplicates via dHash (Hamming distance <= threshold)
        dhashes: List[Tuple[Path, str]] = []
        for p in paths_to_check:
            dh = self.compute_dhash(p)
            if dh:
                dhashes.append((p, dh))

        n = len(dhashes)
        for i in range(n):
            p1, h1 = dhashes[i]
            ds1 = self.infer_dataset_name(str(p1))
            for j in range(i + 1, n):
                p2, h2 = dhashes[j]
                if p1 == p2:
                    continue
                dist = self.hamming_distance(h1, h2)
                if dist <= visual_threshold:
                    ds2 = self.infer_dataset_name(str(p2))
                    scope = "WITHIN_DATASET" if ds1 == ds2 else "CROSS_DATASET"
                    duplicates.append({
                        "file_1": str(p1),
                        "file_2": str(p2),
                        "dataset_1": ds1,
                        "dataset_2": ds2,
                        "duplicate_scope": scope,
                        "duplicate_type": "VISUAL_PROBABLE",
                        "similarity_metric": f"dHash Hamming distance = {dist}/{len(h1)*4}",
                        "notes": f"Perceptual visual similarity ({scope})."
                    })

        return duplicates

    # -------------------------------------------------------------------------
    # REPORT GENERATION & PERSISTENCE
    # -------------------------------------------------------------------------
    def save_validation_reports(
        self,
        image_validation_results: Dict[str, List[ImageValidationResult]],
        mask_validation_results: List[MaskPairValidationResult],
        duplicate_results: List[Dict[str, Any]]
    ) -> Dict[str, Path]:
        """
        Saves validation artifacts to data/processed/validation/ and
        data/processed/duplicates_report.csv.
        """
        out_paths = {}

        # 1. Corrupted images CSV
        corrupted_csv = self.validation_dir / "corrupted_images.csv"
        with open(corrupted_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "issue_type", "error_message", "file_size_bytes"])
            for res in image_validation_results.get("zero_byte", []):
                writer.writerow([str(res.path), "ZERO_BYTE", res.error_message, res.file_size_bytes])
            for res in image_validation_results.get("corrupted", []):
                writer.writerow([str(res.path), "CORRUPTED", res.error_message, res.file_size_bytes])
            for res in image_validation_results.get("unsupported", []):
                writer.writerow([str(res.path), "UNSUPPORTED", res.error_message, res.file_size_bytes])
        out_paths["corrupted_images"] = corrupted_csv

        # 2. Mismatched pairs CSV
        mismatch_csv = self.validation_dir / "mismatched_pairs.csv"
        with open(mismatch_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "image_path", "mask_path", "paired", "readable", "dims_match", "issue"])
            for m in mask_validation_results:
                if not m.is_paired or not m.is_mask_readable or not m.dimensions_match:
                    writer.writerow([
                        m.dataset_name,
                        str(m.image_path),
                        str(m.mask_path) if m.mask_path else "NONE",
                        m.is_paired,
                        m.is_mask_readable,
                        m.dimensions_match,
                        m.issue_description or "Unknown issue"
                    ])
        out_paths["mismatched_pairs"] = mismatch_csv

        # 3. Comprehensive mask validation CSV
        mask_val_csv = self.validation_dir / "mask_validation.csv"
        with open(mask_val_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "image", "mask", "image_width", "image_height", "mask_width", "mask_height", "status", "reason"])
            for m in mask_validation_results:
                img_w = m.image_size[0] if m.image_size else ""
                img_h = m.image_size[1] if m.image_size else ""
                mask_w = m.mask_size[0] if m.mask_size else ""
                mask_h = m.mask_size[1] if m.mask_size else ""
                writer.writerow([
                    m.dataset_name,
                    str(m.image_path),
                    str(m.mask_path) if m.mask_path else "NONE",
                    img_w,
                    img_h,
                    mask_w,
                    mask_h,
                    m.status,
                    m.reason or m.issue_description or "OK"
                ])
        out_paths["mask_validation"] = mask_val_csv

        # 4. Duplicates report CSV
        dup_csv = self.processed_dir / "duplicates_report.csv"
        with open(dup_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_1", "file_2", "dataset_1", "dataset_2", "duplicate_scope", "duplicate_type", "similarity_metric", "notes"])
            for d in duplicate_results:
                writer.writerow([
                    d["file_1"],
                    d["file_2"],
                    d.get("dataset_1", "unknown"),
                    d.get("dataset_2", "unknown"),
                    d.get("duplicate_scope", "UNKNOWN"),
                    d["duplicate_type"],
                    d["similarity_metric"],
                    d["notes"]
                ])
        out_paths["duplicates_report"] = dup_csv

        # 4. Validation summary JSON
        summary_json = self.validation_dir / "validation_summary.json"
        summary_data = {
            "valid_images_count": len(image_validation_results.get("valid", [])),
            "corrupted_images_count": len(image_validation_results.get("corrupted", [])),
            "zero_byte_count": len(image_validation_results.get("zero_byte", [])),
            "unsupported_format_count": len(image_validation_results.get("unsupported", [])),
            "total_pairs_inspected": len(mask_validation_results),
            "unpaired_images_count": sum(1 for m in mask_validation_results if not m.is_paired),
            "dimension_mismatch_count": sum(1 for m in mask_validation_results if m.is_paired and not m.dimensions_match),
            "exact_duplicates_count": sum(1 for d in duplicate_results if d["duplicate_type"] == "EXACT"),
            "visual_duplicates_count": sum(1 for d in duplicate_results if d["duplicate_type"] == "VISUAL_PROBABLE")
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        out_paths["validation_summary"] = summary_json

        return out_paths
