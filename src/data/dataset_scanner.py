"""
NutriVision AI — Dataset Scanner
Phase 1: Dataset Inspection, Validation, Normalization, and Preparation

Implements automatic, non-destructive discovery and inspection of all target datasets:
- FoodSeg103 (Segmentation)
- UECFoodPixComplete (Food Segmentation Benchmark)
- Fruits-360 100x100 (Fine-grained produce classification)
- Fruits-360 multi (Multi-object image testing)
- weight_dataset (Physical weight estimation)
- freshness_dataset (Freshness quality classification)
- sample_benchmark (Synthetic local benchmark for tests)

Gracefully handles missing datasets and adapts to real directory structures.
"""

from dataclasses import dataclass, field
from collections import defaultdict, Counter
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from PIL import Image

from src.config.settings import get_settings

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNOTATION_EXTENSIONS: Set[str] = {".txt", ".json", ".xml", ".csv", ".yaml", ".yml"}

DATASET_PURPOSES: Dict[str, List[str]] = {
    "FoodSeg103": [
        "food segmentation",
        "ingredient/food recognition support"
    ],
    "UECFoodPixComplete": [
        "real-world food segmentation",
        "food-region isolation",
        "segmentation evaluation"
    ],
    "Fruits360_100x100": [
        "fruit/vegetable classification",
        "supplemental classification training"
    ],
    "Fruits360_multi": [
        "multi-object food image testing",
        "detection pipeline experimentation"
    ],
    "weight_dataset": [
        "physical weight estimation training/evaluation"
    ],
    "freshness_dataset": [
        "freshness classification"
    ],
    "sample_benchmark": [
        "pipeline smoke testing",
        "local end-to-end verification"
    ],
    "updated_fruit_dataset": [
        "fine-grained fruit recognition",
        "updated 6-class fruit classification"
    ]
}


@dataclass
class DatasetScanResult:
    """Detailed structural scan metadata for a single dataset."""
    name: str
    path: Path
    status: str  # "AVAILABLE", "NOT AVAILABLE", "EMPTY"
    is_available: bool = False
    image_count: int = 0
    image_extensions: Dict[str, int] = field(default_factory=dict)
    classes_count: int = 0
    class_names: List[str] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    annotation_files_count: int = 0
    mask_count: int = 0
    bbox_count: int = 0
    mask_available: bool = False
    bbox_available: bool = False
    splits_detected: Dict[str, int] = field(default_factory=dict)
    official_splits: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    empty_folders: List[str] = field(default_factory=list)
    duplicate_filenames: List[str] = field(default_factory=list)
    image_dimensions: Dict[str, Any] = field(default_factory=lambda: {
        "min_width": None,
        "max_width": None,
        "min_height": None,
        "max_height": None,
        "unique_resolutions": 0
    })
    corrupted_images: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    intended_modules: List[str] = field(default_factory=list)
    validation_status: str = "PENDING"  # PASS, WARN, FAIL, NOT_AVAILABLE
    notes: str = ""

    # Internal list of found paths (filtered to not serialize overly huge lists)
    image_paths: List[Path] = field(default_factory=list, repr=False)
    mask_paths: List[Path] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary for JSON reporting."""
        try:
            proj_root = Path(__file__).resolve().parent.parent.parent
            rel_p = self.path.relative_to(proj_root).as_posix()
        except Exception:
            rel_p = f"data/raw/{self.name}"

        return {
            "name": self.name,
            "path": rel_p,
            "status": self.status,
            "is_available": self.is_available,
            "image_count": self.image_count,
            "image_extensions": self.image_extensions,
            "classes_count": self.classes_count,
            "class_names": self.class_names,
            "class_distribution": self.class_distribution,
            "annotation_files_count": self.annotation_files_count,
            "mask_count": self.mask_count,
            "bbox_count": self.bbox_count,
            "mask_available": self.mask_available,
            "bbox_available": self.bbox_available,
            "splits_detected": self.splits_detected,
            "official_splits": self.official_splits,
            "missing_files_count": len(self.missing_files),
            "empty_folders_count": len(self.empty_folders),
            "duplicate_filenames_count": len(self.duplicate_filenames),
            "image_dimensions": self.image_dimensions,
            "corrupted_images_count": len(self.corrupted_images),
            "warnings": self.warnings,
            "intended_modules": self.intended_modules,
            "validation_status": self.validation_status,
            "notes": self.notes
        }


class DatasetScanner:
    """
    Scans data/raw directories to discover, inspect, and index food datasets.
    Adapts to dataset-specific folder structures without destructive alterations.
    """

    def __init__(self, raw_dir: Optional[Path] = None):
        settings = get_settings()
        self.raw_dir = Path(raw_dir).resolve() if raw_dir else settings.datasets.raw_data_root
        self.settings = settings

    def scan_all(self) -> Dict[str, DatasetScanResult]:
        """
        Scan all primary target datasets plus any discovered subdirectories in data/raw.
        Returns a dictionary mapping dataset name to DatasetScanResult.
        """
        results: Dict[str, DatasetScanResult] = {}

        # Pre-defined target datasets and their directory aliases
        target_datasets = [
            ("FoodSeg103", [
                self.settings.datasets.foodseg103_dir,
                self.raw_dir / "FoodSeg103"
            ]),
            ("UECFoodPixComplete", [
                self.settings.datasets.uecfoodpix_dir,
                self.raw_dir / "UECFoodPixComplete"
            ]),
            ("Fruits360_100x100", [
                self.settings.datasets.fruits360_dir,
                self.raw_dir / "Fruits360_100x100",
                self.raw_dir / "Fruits360"
            ]),
            ("Fruits360_multi", [
                self.settings.datasets.fruits360_multi_dir,
                self.raw_dir / "Fruits360_multi",
                self.raw_dir / "Fruits360" / "test-multiple_fruits"
            ]),
            ("weight_dataset", [
                self.settings.datasets.weight_dataset_dir,
                self.raw_dir / "weight_dataset"
            ]),
            ("freshness_dataset", [
                self.settings.datasets.freshness_dataset_dir,
                self.raw_dir / "freshness_dataset"
            ]),
            ("sample_benchmark", [
                self.raw_dir / "sample_benchmark"
            ]),
            ("updated_fruit_dataset", [
                getattr(self.settings.datasets, "updated_fruit_dataset_dir", None),
                self.raw_dir / "updated dataset"
            ]),
        ]

        scanned_paths: Set[Path] = set()

        for name, candidate_paths in target_datasets:
            chosen_path = None
            for cand in candidate_paths:
                if cand and cand.exists() and cand not in scanned_paths:
                    chosen_path = cand
                    break
            if chosen_path is None:
                # Fallback to the first candidate path (will be marked NOT AVAILABLE)
                chosen_path = candidate_paths[0] if candidate_paths else self.raw_dir / name

            res = self.scan_dataset(name, chosen_path)
            results[name] = res
            if chosen_path.exists():
                scanned_paths.add(chosen_path)

        # Also check for any extra unmapped subdirectories directly in data/raw
        if self.raw_dir.exists():
            for entry in self.raw_dir.iterdir():
                if entry.is_dir() and entry not in scanned_paths:
                    name = entry.name
                    if name not in results:
                        results[name] = self.scan_dataset(name, entry)
                        scanned_paths.add(entry)

        return results

    def scan_dataset(self, name: str, dataset_path: Path) -> DatasetScanResult:
        """Inspect a specific dataset by name and path, using dataset-specific heuristics."""
        dataset_path = Path(dataset_path).resolve()
        res = DatasetScanResult(
            name=name,
            path=dataset_path,
            status="NOT AVAILABLE",
            intended_modules=DATASET_PURPOSES.get(name, ["general food recognition"])
        )

        if not dataset_path.exists():
            res.status = "NOT AVAILABLE"
            res.validation_status = "NOT_AVAILABLE"
            res.warnings.append(f"Directory '{dataset_path}' does not exist.")
            return res

        # Check if empty (ignoring .gitkeep and hidden files)
        visible_entries = [p for p in dataset_path.iterdir() if not p.name.startswith(".")]
        if not visible_entries:
            res.status = "NOT AVAILABLE"
            res.validation_status = "NOT_AVAILABLE"
            res.warnings.append(f"Directory '{dataset_path}' is empty (no image or annotation files).")
            return res

        # Dispatch to dataset-specific inspector if available
        if name == "FoodSeg103":
            return self._scan_foodseg103(res)
        elif name == "UECFoodPixComplete":
            return self._scan_uecfoodpix(res)
        elif name == "Fruits360_100x100" or (name == "Fruits360" and not (dataset_path / "test-multiple_fruits").is_dir() and not (dataset_path / "test-multiple_fruits").exists()):
            return self._scan_fruits360_classification(res)
        elif name == "Fruits360_multi":
            return self._scan_fruits360_multi(res)
        elif name == "weight_dataset":
            return self._scan_weight_dataset(res)
        elif name == "freshness_dataset":
            return self._scan_freshness_dataset(res)
        elif name == "sample_benchmark":
            return self._scan_sample_benchmark(res)
        elif name in ("updated_fruit_dataset", "updated dataset"):
            return self._scan_updated_fruit(res)
        else:
            return self._scan_generic(res)

    # -------------------------------------------------------------------------
    # FOODSEG103 INSPECTION
    # -------------------------------------------------------------------------
    def _scan_foodseg103(self, res: DatasetScanResult) -> DatasetScanResult:
        """
        Inspects FoodSeg103 structure:
        Expects Images/img_dir/train, Images/img_dir/test and Images/ann_dir/...
        or direct img_dir / ann_dir.
        """
        root = res.path
        img_dirs: List[Path] = []
        ann_dirs: List[Path] = []

        # Look for image directory candidates
        for cand in [root / "Images" / "img_dir", root / "img_dir", root / "images", root / "Images"]:
            if cand.is_dir():
                img_dirs.append(cand)

        for cand in [root / "Images" / "ann_dir", root / "ann_dir", root / "masks", root / "annotations"]:
            if cand.is_dir():
                ann_dirs.append(cand)

        all_images = []
        all_masks = []
        split_counts = defaultdict(int)

        # Walk to find all images and masks
        for r, dirs, files in os.walk(root):
            if not dirs and not files:
                res.empty_folders.append(str(Path(r).relative_to(root)))
            rel = Path(r).relative_to(root).as_posix().lower()
            for f in files:
                if f.startswith("."):
                    continue
                ext = Path(f).suffix.lower()
                f_path = Path(r) / f

                # Determine if image or mask based on path and extension
                if "ann_dir" in rel or "mask" in rel:
                    if ext in SUPPORTED_IMAGE_EXTENSIONS or ext == ".png":
                        all_masks.append(f_path)
                elif ext in SUPPORTED_IMAGE_EXTENSIONS:
                    all_images.append(f_path)
                    if "train" in rel:
                        split_counts["train"] += 1
                    elif "test" in rel:
                        split_counts["test"] += 1
                    elif "val" in rel:
                        split_counts["val"] += 1

        res.image_paths = all_images
        res.mask_paths = all_masks
        res.image_count = len(all_images)
        res.mask_count = len(all_masks)
        res.mask_available = len(all_masks) > 0
        res.bbox_available = False

        if res.image_count == 0:
            res.status = "NOT AVAILABLE"
            res.validation_status = "NOT_AVAILABLE"
            res.warnings.append("No image files detected in FoodSeg103 directory.")
            return res

        res.status = "AVAILABLE"
        res.is_available = True
        res.splits_detected = dict(split_counts)
        res.official_splits = [s for s in ["train", "test"] if s in split_counts]

        # Check for category_id.txt or class listing
        cat_file = None
        for cand in [root / "category_id.txt", root / "category.txt", root / "classes.txt"]:
            if cand.is_file():
                cat_file = cand
                break

        if cat_file:
            try:
                with open(cat_file, "r", encoding="utf-8") as cf:
                    classes = [line.strip() for line in cf if line.strip()]
                    res.class_names = classes
                    res.classes_count = len(classes)
            except Exception as e:
                res.warnings.append(f"Failed to read category file {cat_file}: {e}")
        else:
            res.classes_count = 103  # Official FoodSeg103 class taxonomy size
            res.notes = "FoodSeg103 canonical 103 food classes (category_id.txt not found on disk, using canonical taxonomy)."

        # Extension breakdown
        ext_counter = Counter(p.suffix.lower() for p in all_images)
        res.image_extensions = dict(ext_counter)

        # Check image-mask correspondence
        mask_stems = {m.stem for m in all_masks}
        images_missing_masks = [img for img in all_images if img.stem not in mask_stems]
        if images_missing_masks:
            res.missing_files.extend([str(img) for img in images_missing_masks[:50]])
            res.warnings.append(
                f"{len(images_missing_masks)} / {len(all_images)} images in FoodSeg103 lack matching segmentation masks."
            )
            res.validation_status = "WARN"
        else:
            res.validation_status = "PASS"

        self._record_sample_resolutions(res, all_images)
        return res

    # -------------------------------------------------------------------------
    # UECFOODPIX COMPLETE INSPECTION
    # -------------------------------------------------------------------------
    def _scan_uecfoodpix(self, res: DatasetScanResult) -> DatasetScanResult:
        """
        Inspects UECFoodPixComplete:
        Expects data/train/img, data/train/mask, data/test/img, data/test/mask
        and category.txt.
        """
        root = res.path
        all_images = []
        all_masks = []
        split_counts = defaultdict(int)

        for r, dirs, files in os.walk(root):
            if not dirs and not files:
                res.empty_folders.append(str(Path(r).relative_to(root)))
            rel = Path(r).relative_to(root).as_posix().lower()
            for f in files:
                if f.startswith("."):
                    continue
                ext = Path(f).suffix.lower()
                f_path = Path(r) / f

                if "mask" in rel:
                    if ext in SUPPORTED_IMAGE_EXTENSIONS:
                        all_masks.append(f_path)
                elif ext in SUPPORTED_IMAGE_EXTENSIONS:
                    all_images.append(f_path)
                    if "train" in rel:
                        split_counts["train"] += 1
                    elif "test" in rel:
                        split_counts["test"] += 1
                    elif "val" in rel:
                        split_counts["val"] += 1

        res.image_paths = all_images
        res.mask_paths = all_masks
        res.image_count = len(all_images)
        res.mask_count = len(all_masks)
        res.mask_available = len(all_masks) > 0
        res.bbox_available = False

        if res.image_count == 0:
            res.status = "NOT AVAILABLE"
            res.validation_status = "NOT_AVAILABLE"
            res.warnings.append("No image files detected in UECFoodPixComplete directory.")
            return res

        res.status = "AVAILABLE"
        res.is_available = True
        res.splits_detected = dict(split_counts)
        res.official_splits = [s for s in ["train", "test"] if s in split_counts]

        # Look for category.txt
        cat_file = None
        for cand in [root / "data" / "category.txt", root / "category.txt", root / "category_id.txt"]:
            if cand.is_file():
                cat_file = cand
                break

        if cat_file:
            try:
                with open(cat_file, "r", encoding="utf-8") as cf:
                    classes = []
                    for line in cf:
                        line = line.strip()
                        if line:
                            # Skip header if present
                            if line.lower().startswith("id\t") or line.lower().startswith("id ") or line.lower() == "id":
                                continue
                            # Usually format is: "1 rice" or "rice"
                            parts = line.split(maxsplit=1)
                            if len(parts) == 2 and parts[0].isdigit():
                                classes.append(parts[1])
                            elif not parts[0].isdigit() and parts[0].lower() not in ("id", "index"):
                                classes.append(line)
                    res.class_names = classes
                    res.classes_count = len(classes)
            except Exception as e:
                res.warnings.append(f"Failed to read category file {cat_file}: {e}")
        else:
            res.classes_count = 102
            res.notes = "UECFoodPixComplete canonical 102 food categories (category.txt not found on disk, using canonical taxonomy)."

        ext_counter = Counter(p.suffix.lower() for p in all_images)
        res.image_extensions = dict(ext_counter)

        # Image-mask correspondence
        mask_stems = {m.stem for m in all_masks}
        images_missing_masks = [img for img in all_images if img.stem not in mask_stems]
        if images_missing_masks:
            res.missing_files.extend([str(img) for img in images_missing_masks[:50]])
            res.warnings.append(
                f"{len(images_missing_masks)} / {len(all_images)} images in UECFoodPix lack matching segmentation masks."
            )
            res.validation_status = "WARN"
        else:
            res.validation_status = "PASS"

        self._record_sample_resolutions(res, all_images)
        return res

    # -------------------------------------------------------------------------
    # FRUITS-360 100x100 INSPECTION
    # -------------------------------------------------------------------------
    def _scan_fruits360_classification(self, res: DatasetScanResult) -> DatasetScanResult:
        """
        Inspects Fruits-360 100x100:
        Directory structure with Training/<Class_Name> and Test/<Class_Name>.
        Measures class counts, train/test breakdown, and class imbalance.
        """
        root = res.path
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

        # If Training and Test aren't top-level, check if any subfolders exist
        all_images = []
        classes_train = defaultdict(int)
        classes_test = defaultdict(int)

        if train_dir:
            res.official_splits.append("Training")
            for c_dir in train_dir.iterdir():
                if c_dir.is_dir():
                    c_name = c_dir.name
                    for f in c_dir.iterdir():
                        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                            all_images.append(f)
                            classes_train[c_name] += 1

        if test_dir:
            res.official_splits.append("Test")
            for c_dir in test_dir.iterdir():
                if c_dir.is_dir():
                    c_name = c_dir.name
                    for f in c_dir.iterdir():
                        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                            all_images.append(f)
                            classes_test[c_name] += 1

        # Fallback if no Training/Test subfolders
        if not train_dir and not test_dir:
            for r, dirs, files in os.walk(root):
                rel = Path(r).relative_to(root)
                parts = rel.parts
                for f in files:
                    if Path(f).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                        f_path = Path(r) / f
                        all_images.append(f_path)
                        if len(parts) >= 1:
                            classes_train[parts[0]] += 1

        res.image_paths = all_images
        res.image_count = len(all_images)
        res.mask_available = False
        res.bbox_available = False

        if res.image_count == 0:
            res.status = "NOT AVAILABLE"
            res.validation_status = "NOT_AVAILABLE"
            res.warnings.append("No image files detected in Fruits-360 100x100 directory.")
            return res

        res.status = "AVAILABLE"
        res.is_available = True

        all_class_names = sorted(set(list(classes_train.keys()) + list(classes_test.keys())))
        res.class_names = all_class_names
        res.classes_count = len(all_class_names)

        total_distribution = {}
        for c in all_class_names:
            total_distribution[c] = classes_train[c] + classes_test[c]
        res.class_distribution = total_distribution

        res.splits_detected = {
            "Training": sum(classes_train.values()),
            "Test": sum(classes_test.values())
        }

        ext_counter = Counter(p.suffix.lower() for p in all_images)
        res.image_extensions = dict(ext_counter)
        res.validation_status = "PASS"

        self._record_sample_resolutions(res, all_images)
        return res

    # -------------------------------------------------------------------------
    # FRUITS-360 MULTI INSPECTION
    # -------------------------------------------------------------------------
    def _scan_fruits360_multi(self, res: DatasetScanResult) -> DatasetScanResult:
        """
        Inspects Fruits-360 multi:
        Explicitly checks whether bounding boxes / labels exist.
        IMPORTANT: If unavailable, explicitly reports 'Bounding-box annotations are unavailable.'
        """
        root = res.path
        all_images = []
        all_annotations = []

        for r, dirs, files in os.walk(root):
            for f in files:
                if f.startswith("."):
                    continue
                ext = Path(f).suffix.lower()
                f_path = Path(r) / f
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    all_images.append(f_path)
                elif ext in ANNOTATION_EXTENSIONS:
                    all_annotations.append(f_path)

        res.image_paths = all_images
        res.image_count = len(all_images)
        res.annotation_files_count = len(all_annotations)
        res.mask_available = False

        if res.image_count == 0:
            res.status = "NOT AVAILABLE"
            res.validation_status = "NOT_AVAILABLE"
            res.warnings.append("No image files detected in Fruits-360 multi directory.")
            return res

        res.status = "AVAILABLE"
        res.is_available = True
        res.image_extensions = dict(Counter(p.suffix.lower() for p in all_images))

        # Check for bbox annotations
        if len(all_annotations) == 0:
            res.bbox_available = False
            res.bbox_count = 0
            res.notes = "Bounding-box annotations are unavailable."
            res.warnings.append("Bounding-box annotations are unavailable.")
            res.validation_status = "WARN"
        else:
            res.bbox_available = True
            res.bbox_count = len(all_annotations)
            res.notes = f"Detected {len(all_annotations)} annotation files."
            res.validation_status = "PASS"

        self._record_sample_resolutions(res, all_images)
        return res

    # -------------------------------------------------------------------------
    # WEIGHT & FRESHNESS DATASETS
    # -------------------------------------------------------------------------
    def _scan_weight_dataset(self, res: DatasetScanResult) -> DatasetScanResult:
        """Inspects physical weight estimation dataset."""
        root = res.path
        all_images = []
        for r, dirs, files in os.walk(root):
            for f in files:
                if f.startswith("."):
                    continue
                if Path(f).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    all_images.append(Path(r) / f)

        res.image_paths = all_images
        res.image_count = len(all_images)

        if res.image_count == 0:
            res.status = "NOT AVAILABLE"
            res.validation_status = "NOT_AVAILABLE"
            res.notes = "Physical weight estimation dataset not yet provisioned in data/raw/weight_dataset/."
            return res

        res.status = "AVAILABLE"
        res.is_available = True
        res.validation_status = "PASS"
        self._record_sample_resolutions(res, all_images)
        return res

    def _scan_freshness_dataset(self, res: DatasetScanResult) -> DatasetScanResult:
        """Inspects food freshness classification dataset."""
        root = res.path
        all_images = []
        classes = defaultdict(int)

        for r, dirs, files in os.walk(root):
            rel = Path(r).relative_to(root)
            parts = rel.parts
            for f in files:
                if f.startswith("."):
                    continue
                if Path(f).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    f_path = Path(r) / f
                    all_images.append(f_path)
                    if len(parts) >= 1:
                        classes[parts[-1]] += 1

        res.image_paths = all_images
        res.image_count = len(all_images)

        if res.image_count == 0:
            res.status = "NOT AVAILABLE"
            res.validation_status = "NOT_AVAILABLE"
            res.notes = "Freshness classification dataset not yet provisioned in data/raw/freshness_dataset/."
            return res

        res.status = "AVAILABLE"
        res.is_available = True
        res.class_names = sorted(classes.keys())
        res.classes_count = len(res.class_names)
        res.class_distribution = dict(classes)
        res.validation_status = "PASS"
        self._record_sample_resolutions(res, all_images)
        return res

    # -------------------------------------------------------------------------
    # SAMPLE BENCHMARK & GENERIC
    # -------------------------------------------------------------------------
    def _scan_sample_benchmark(self, res: DatasetScanResult) -> DatasetScanResult:
        """Inspects local synthetic benchmark dataset created in Phase 0."""
        root = res.path
        all_images = []
        all_annotations = []
        classes = defaultdict(int)
        splits = defaultdict(int)

        for r, dirs, files in os.walk(root):
            rel = Path(r).relative_to(root)
            parts = rel.parts
            for f in files:
                if f.startswith("."):
                    continue
                ext = Path(f).suffix.lower()
                f_path = Path(r) / f
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    all_images.append(f_path)
                    if len(parts) >= 1:
                        splits[parts[0]] += 1
                    if len(parts) >= 2:
                        classes[parts[1]] += 1
                elif ext == ".txt":
                    all_annotations.append(f_path)

        res.image_paths = all_images
        res.image_count = len(all_images)
        res.annotation_files_count = len(all_annotations)
        res.bbox_count = len(all_annotations)
        res.bbox_available = len(all_annotations) > 0
        res.mask_available = False

        if res.image_count == 0:
            res.status = "NOT AVAILABLE"
            res.validation_status = "NOT_AVAILABLE"
            return res

        res.status = "AVAILABLE"
        res.is_available = True
        res.class_names = sorted(classes.keys())
        res.classes_count = len(res.class_names)
        res.class_distribution = dict(classes)
        res.splits_detected = dict(splits)
        res.official_splits = [s for s in ["train", "val", "test"] if s in splits]
        res.image_extensions = dict(Counter(p.suffix.lower() for p in all_images))
        res.validation_status = "PASS"
        self._record_sample_resolutions(res, all_images)
        return res

    def _scan_updated_fruit(self, res: DatasetScanResult) -> DatasetScanResult:
        """Inspects updated 6-class fruit dataset (Apple_Good, Banana_Good, etc.)."""
        from src.data.fruit_dataset_loader import FruitDatasetValidator
        validator = FruitDatasetValidator(dataset_path=res.path, settings=self.settings)
        val_rep = validator.validate_dataset(quick_verify=True, max_verify_per_class=50)

        if not val_rep.is_valid:
            res.status = "NOT AVAILABLE"
            res.validation_status = "NOT_AVAILABLE"
            res.warnings.extend(val_rep.errors)
            return res

        all_imgs = []
        for cname in val_rep.classes:
            cdir = res.path / cname
            all_imgs.extend([
                f for f in cdir.iterdir()
                if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ])

        res.image_paths = all_imgs
        res.image_count = val_rep.total_valid_images
        res.corrupted_images = [cf["path"] for cf in val_rep.corrupted_files]
        res.class_names = val_rep.classes
        res.classes_count = len(val_rep.classes)
        res.class_distribution = {k: v.valid_count for k, v in val_rep.class_summaries.items()}
        res.status = "AVAILABLE"
        res.is_available = True
        res.validation_status = "PASS"
        res.warnings.extend(val_rep.warnings)
        self._record_sample_resolutions(res, all_imgs)
        return res

    def _scan_generic(self, res: DatasetScanResult) -> DatasetScanResult:
        """Generic recursive folder scanner for arbitrary food datasets."""
        root = res.path
        all_images = []
        classes = defaultdict(int)
        splits = defaultdict(int)

        for r, dirs, files in os.walk(root):
            rel = Path(r).relative_to(root)
            parts = rel.parts
            for f in files:
                if f.startswith("."):
                    continue
                ext = Path(f).suffix.lower()
                f_path = Path(r) / f
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    all_images.append(f_path)
                    if any(s in rel.as_posix().lower() for s in ["train", "val", "test"]):
                        for s in ["train", "val", "test"]:
                            if s in rel.as_posix().lower():
                                splits[s] += 1
                                break
                    if len(parts) >= 1:
                        classes[parts[-1]] += 1

        res.image_paths = all_images
        res.image_count = len(all_images)

        if res.image_count == 0:
            res.status = "NOT AVAILABLE"
            res.validation_status = "NOT_AVAILABLE"
            return res

        res.status = "AVAILABLE"
        res.is_available = True
        res.class_names = sorted(classes.keys())
        res.classes_count = len(res.class_names)
        res.class_distribution = dict(classes)
        res.splits_detected = dict(splits)
        res.image_extensions = dict(Counter(p.suffix.lower() for p in all_images))
        res.validation_status = "PASS"
        self._record_sample_resolutions(res, all_images)
        return res

    def _record_sample_resolutions(self, res: DatasetScanResult, image_paths: List[Path], max_samples: int = 100):
        """Samples images safely to calculate resolution bounds without decoding full datasets."""
        if not image_paths:
            return

        widths = []
        heights = []
        resolutions = set()

        sample = image_paths[:max_samples]
        for p in sample:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    resolutions.add((w, h))
            except Exception as e:
                res.corrupted_images.append(str(p))

        if widths and heights:
            res.image_dimensions = {
                "min_width": min(widths),
                "max_width": max(widths),
                "min_height": min(heights),
                "max_height": max(heights),
                "unique_resolutions": len(resolutions),
                "sample_resolution": f"{widths[0]}x{heights[0]}"
            }
