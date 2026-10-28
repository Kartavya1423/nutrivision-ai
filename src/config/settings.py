"""
NutriVision AI — Centralized Configuration System
Supports environment variables (.env), YAML configuration, and runtime overrides.
Ensures zero hardcoded secrets or API keys.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_env_file(env_path: Path) -> Dict[str, str]:
    """Parse a .env file into a dictionary without requiring third-party libraries."""
    env_vars: Dict[str, str] = {}
    if not env_path.is_file():
        return env_vars

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    env_vars[key] = val
    except Exception:
        pass
    return env_vars


def _load_yaml_file(yaml_path: Path) -> Dict[str, Any]:
    """Load config.yaml if PyYAML is installed, else return empty dictionary."""
    if not yaml_path.is_file():
        return {}
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@dataclass
class DatasetPathsSettings:
    """Paths to raw, processed, and benchmark datasets supported by NutriVision AI."""
    root: Path
    data_root: Path = field(default=None)
    raw_data_root: Path = field(default=None)
    processed_data_root: Path = field(default=None)
    splits_root: Path = field(default=None)
    reports_root: Path = field(default=None)
    foodseg103_dir: Path = field(default=None)
    uecfoodpix_dir: Path = field(default=None)
    fruits360_dir: Path = field(default=None)
    fruits360_multi_dir: Path = field(default=None)
    weight_dataset_dir: Path = field(default=None)
    freshness_dataset_dir: Path = field(default=None)
    updated_fruit_dataset_dir: Path = field(default=None)

    def __post_init__(self):
        if self.data_root is None:
            self.data_root = self.root / "data"
        if self.raw_data_root is None:
            self.raw_data_root = self.data_root / "raw"
        if self.processed_data_root is None:
            self.processed_data_root = self.data_root / "processed"
        if self.splits_root is None:
            self.splits_root = self.data_root / "splits"
        if self.reports_root is None:
            self.reports_root = self.data_root / "reports"
        if self.foodseg103_dir is None:
            self.foodseg103_dir = self.raw_data_root / "FoodSeg103"
        if self.uecfoodpix_dir is None:
            self.uecfoodpix_dir = self.raw_data_root / "UECFoodPixComplete"
        if self.fruits360_dir is None:
            cand = self.raw_data_root / "Fruits360_100x100"
            if not cand.exists() and (self.raw_data_root / "Fruits360").exists():
                cand = self.raw_data_root / "Fruits360"
            self.fruits360_dir = cand
        if self.fruits360_multi_dir is None:
            cand_multi = self.raw_data_root / "Fruits360_multi"
            if not cand_multi.exists() and (self.raw_data_root / "Fruits360" / "test-multiple_fruits").exists():
                cand_multi = self.raw_data_root / "Fruits360" / "test-multiple_fruits"
            self.fruits360_multi_dir = cand_multi
        if self.weight_dataset_dir is None:
            self.weight_dataset_dir = self.raw_data_root / "weight_dataset"
        if self.freshness_dataset_dir is None:
            self.freshness_dataset_dir = self.raw_data_root / "freshness_dataset"
        if self.updated_fruit_dataset_dir is None:
            self.updated_fruit_dataset_dir = self.raw_data_root / "updated dataset"


@dataclass
class ModelPathsSettings:
    """Paths to model checkpoints and weight files."""
    root: Path
    detection_model_path: Path = field(default=None)
    segmentation_model_path: Path = field(default=None)
    classification_model_path: Path = field(default=None)
    fruit_classifier_path: Path = field(default=None)
    depth_model_path: Path = field(default=None)
    freshness_model_path: Path = field(default=None)
    weights_root: Path = field(default=None)

    def __post_init__(self):
        import os
        env_det = os.environ.get("NUTRIVISION_DETECTION_MODEL")
        if env_det and Path(env_det).is_file():
            self.detection_model_path = Path(env_det)

        if self.detection_model_path is None or not Path(self.detection_model_path).is_file():
            for cand in [
                self.root / "models" / "detection" / "yolo11" / "phase3_finetune_run" / "weights" / "best.pt",
                self.root / "models" / "detection" / "yolo11" / "best.pt",
                self.root / "yolo11n.pt",
                self.root / "models" / "detection" / "yolo11n.pt",
                self.root / "models" / "detection" / "yolo11" / "phase2_fixed_run" / "weights" / "best.pt",
                self.root / "models" / "detection" / "yolo11" / "repair_run" / "best.pt",
                self.root / "models" / "detection" / "yolo11" / "production" / "best.pt",
                self.root / "models" / "detection" / "yolo11" / "last.pt",
            ]:
                if cand.is_file():
                    self.detection_model_path = cand
                    break
            if self.detection_model_path is None:
                self.detection_model_path = self.root / "models" / "detection" / "yolo11" / "best.pt"
        if self.segmentation_model_path is None:
            for cand_name in ["sam2_hiera_base_plus.pt", "sam2.1_t.pt", "sam2_t.pt", "sam2_b.pt"]:
                cand_path = self.root / "models" / "segmentation" / cand_name
                if cand_path.is_file():
                    self.segmentation_model_path = cand_path
                    break
            if self.segmentation_model_path is None:
                self.segmentation_model_path = self.root / "models" / "segmentation" / "sam2_hiera_base_plus.pt"
        if self.classification_model_path is None:
            env_cls = os.environ.get("NUTRIVISION_CLASSIFIER_MODEL")
            if env_cls and Path(env_cls).is_file():
                self.classification_model_path = Path(env_cls)
            else:
                candidate_list = [
                    self.root / "models" / "classification" / "efficientnet_b4_food.pth",
                    self.root / "models" / "classification" / "target_foods_v1" / "best.pth",
                    self.root / "models" / "classification" / "mobilenet_v2_food50.pth",
                    self.root / "models" / "classification" / "efficientnet_b4" / "best.pth",
                ]
                for cand in candidate_list:
                    if cand.is_file():
                        self.classification_model_path = cand
                        break
                if self.classification_model_path is None:
                    self.classification_model_path = self.root / "models" / "classification" / "efficientnet_b4_food.pth"
        if self.fruit_classifier_path is None:
            env_fc = os.environ.get("NUTRIVISION_FRUIT_CLASSIFIER_MODEL")
            if env_fc and Path(env_fc).is_file():
                self.fruit_classifier_path = Path(env_fc)
            else:
                for cand in [
                    self.root / "models" / "classification" / "fruit_classifier_corrected_split" / "best_model.pth",
                    self.root / "models" / "fruit_classifier" / "best_model.pth",
                    self.root / "models" / "classification" / "fruit_classifier" / "best_model.pth",
                    self.root / "models" / "fruit_classifier" / "best.pth",
                    self.root / "models" / "fruit_classifier" / "fruit_classifier.pth",
                ]:
                    if cand.is_file():
                        self.fruit_classifier_path = cand
                        break
                if self.fruit_classifier_path is None:
                    self.fruit_classifier_path = self.root / "models" / "classification" / "fruit_classifier_corrected_split" / "best_model.pth"
        if self.depth_model_path is None:
            self.depth_model_path = self.root / "models" / "depth" / "depth_anything_v2_vits.pth"
        if self.freshness_model_path is None:
            self.freshness_model_path = self.root / "models" / "freshness" / "freshness_classifier.pth"
        if self.weights_root is None:
            self.weights_root = self.root / "models" / "weights"


@dataclass
class DetectionSettings:
    """Settings for YOLO11 multi-item food detection."""
    model_type: str = "yolo11"
    model_variant: str = "yolo11n.pt"
    confidence_threshold: float = 0.15
    iou_threshold: float = 0.45
    image_size: int = 640
    max_detections: int = 30


@dataclass
class SegmentationSettings:
    """Settings for SAM2 instance segmentation."""
    model_type: str = "sam2"
    checkpoint: str = "sam2.1_t.pt"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    box_nms_thresh: float = 0.7
    mask_threshold: float = 0.5
    min_mask_area: int = 50
    max_candidate_masks: int = 3
    selection_strategy: str = "iou_containment_confidence"
    box_margin: float = 0.05
    apply_postprocessing: bool = True


@dataclass
class DepthSettings:
    """Settings for Depth Anything V2 monocular depth estimation."""
    model_type: str = "depth_anything_v2"
    encoder: str = "vits"  # vits, vitb, vitl
    input_size: int = 518


@dataclass
class WeightSettings:
    """Settings for volumetric calibration and density-based weight calculation."""
    reference_object: str = "plate"  # plate, coin, card
    reference_dimension_cm: float = 25.0  # standard dinner plate diameter
    density_database_path: Path = field(default=None)
    default_density_g_cm3: float = 0.85

    def __post_init__(self):
        if self.density_database_path is None:
            self.density_database_path = Path("data/food_densities.json")


@dataclass
class NutritionSettings:
    """Settings for USDA FoodData Central API queries and caching."""
    usda_api_url: str = "https://api.nal.usda.gov/fdc/v1"
    usda_api_key: str = "DEMO_KEY"
    cache_ttl_hours: int = 168  # 7 days


@dataclass
class ExplainabilitySettings:
    """Settings for Grad-CAM attention visualizations."""
    target_layer: str = "conv_head"
    colormap: str = "JET"
    alpha: float = 0.5


@dataclass
class CameraSettings:
    """Settings for live camera input and video feeds."""
    device_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30


@dataclass
class OutputDirectories:
    """Target directories for saving pipeline stage artifacts."""
    root: Path
    detections: Path = field(default=None)
    segmentation: Path = field(default=None)
    crops: Path = field(default=None)
    depth: Path = field(default=None)
    results: Path = field(default=None)
    visualizations: Path = field(default=None)
    logs: Path = field(default=None)

    def __post_init__(self):
        outputs = self.root / "outputs"
        if self.detections is None:
            self.detections = outputs / "detections"
        if self.segmentation is None:
            self.segmentation = outputs / "segmentation"
        self.segmentation_masks = self.segmentation / "masks"
        self.segmentation_overlays = self.segmentation / "overlays"
        self.segmentation_crops = self.segmentation / "crops"
        self.segmentation_rgba = self.segmentation / "rgba"
        self.segmentation_predictions = self.segmentation / "predictions"
        self.segmentation_reports = self.segmentation / "reports"
        if self.crops is None:
            self.crops = outputs / "crops"
        if self.depth is None:
            self.depth = outputs / "depth"
        if self.results is None:
            self.results = outputs / "results"
        if self.visualizations is None:
            self.visualizations = outputs / "visualizations"
        if self.logs is None:
            self.logs = self.root / "logs"


class Settings:
    """Central configuration coordinator for NutriVision AI."""

    def __init__(self, root_dir: Optional[Path] = None):
        if root_dir is None:
            # Anchor to repository root (two levels up from this file)
            self.root_dir = Path(__file__).resolve().parent.parent.parent
        else:
            self.root_dir = Path(root_dir).resolve()

        # Load environment sources (.env first, then OS environ overrides)
        env_from_file = _load_env_file(self.root_dir / ".env")
        # Also try config.yaml
        yaml_config = _load_yaml_file(self.root_dir / "config.yaml")

        def get_val(key: str, default: Any, yaml_path: Optional[List[str]] = None) -> Any:
            """Priority: OS environment -> .env file -> config.yaml -> default."""
            if key in os.environ:
                return os.environ[key]
            if key in env_from_file:
                return env_from_file[key]
            if yaml_path and yaml_config:
                curr = yaml_config
                found = True
                for p in yaml_path:
                    if isinstance(curr, dict) and p in curr:
                        curr = curr[p]
                    else:
                        found = False
                        break
                if found and curr is not None:
                    return curr
            return default

        # System & Hardware
        self.project_name: str = "NutriVision AI"
        self.version: str = "0.1.0"
        self.device: str = str(get_val("NUTRIVISION_DEVICE", "auto", ["system", "device"])).lower()
        self.num_workers: int = int(get_val("NUTRIVISION_NUM_WORKERS", 4, ["system", "num_workers"]))
        self.log_level: str = str(get_val("NUTRIVISION_LOG_LEVEL", "INFO", ["system", "log_level"]))
        self.seed: int = int(get_val("NUTRIVISION_SEED", 42, ["project", "seed"]))

        # Model Paths
        default_cls = (
            self.root_dir / "models" / "classification" / "efficientnet_b4_food.pth"
            if (self.root_dir / "models" / "classification" / "efficientnet_b4_food.pth").is_file()
            else self.root_dir / "models" / "classification" / "target_foods_v1" / "best.pth"
        )
        det_path = Path(get_val("DETECTION_MODEL_PATH", self.root_dir / "models" / "detection" / "yolo11n.pt"))
        seg_path = Path(get_val("SEGMENTATION_MODEL_PATH", self.root_dir / "models" / "segmentation" / "sam2_hiera_base_plus.pt"))
        cls_path = Path(get_val("CLASSIFICATION_MODEL_PATH", default_cls))
        fc_default = (
            self.root_dir / "models" / "fruit_classifier" / "best_model.pth"
            if (self.root_dir / "models" / "fruit_classifier" / "best_model.pth").is_file()
            else self.root_dir / "models" / "classification" / "fruit_classifier" / "best_model.pth"
        )
        fc_path = Path(get_val("FRUIT_CLASSIFIER_PATH", fc_default, ["classification", "fruit_classifier_checkpoint"]))
        dep_path = Path(get_val("DEPTH_MODEL_PATH", self.root_dir / "models" / "depth" / "depth_anything_v2_vits.pth"))
        frs_path = Path(get_val("FRESHNESS_MODEL_PATH", self.root_dir / "models" / "freshness" / "freshness_classifier.pth"))
        # Check if requested seg_path exists, otherwise auto-detect in models/segmentation/
        resolved_seg = seg_path if seg_path.is_absolute() else self.root_dir / seg_path
        if not resolved_seg.is_file():
            for cand in ["sam2.1_t.pt", "sam2_t.pt", "sam2_b.pt", "sam2_hiera_base_plus.pt"]:
                p = self.root_dir / "models" / "segmentation" / cand
                if p.is_file():
                    resolved_seg = p
                    break

        self.model_paths = ModelPathsSettings(
            root=self.root_dir,
            detection_model_path=det_path if det_path.is_absolute() else self.root_dir / det_path,
            segmentation_model_path=resolved_seg,
            classification_model_path=cls_path if cls_path.is_absolute() else self.root_dir / cls_path,
            fruit_classifier_path=fc_path if fc_path.is_absolute() else self.root_dir / fc_path,
            depth_model_path=dep_path if dep_path.is_absolute() else self.root_dir / dep_path,
            freshness_model_path=frs_path if frs_path.is_absolute() else self.root_dir / frs_path,
            weights_root=self.root_dir / "models" / "weights"
        )

        # Dataset Paths Settings
        d_root = Path(get_val("DATA_ROOT", self.root_dir / "data", ["paths", "data_root"]))
        d_raw = Path(get_val("RAW_DATA_ROOT", d_root / "raw", ["paths", "raw_data"]))
        d_proc = Path(get_val("PROCESSED_DATA_ROOT", d_root / "processed", ["paths", "processed_data"]))
        d_splits = Path(get_val("SPLITS_ROOT", d_root / "splits", ["paths", "splits"]))
        d_reports = Path(get_val("REPORTS_ROOT", d_root / "reports", ["paths", "reports"]))

        fs_dir = Path(get_val("FOODSEG103_DIR", d_raw / "FoodSeg103", ["paths", "foodseg103_dir"]))
        uec_dir = Path(get_val("UECFOODPIX_DIR", d_raw / "UECFoodPixComplete", ["paths", "uecfoodpix_dir"]))
        f360_dir = Path(get_val("FRUITS360_DIR", d_raw / "Fruits360_100x100", ["paths", "fruits360_dir"]))
        f360m_dir = Path(get_val("FRUITS360_MULTI_DIR", d_raw / "Fruits360_multi", ["paths", "fruits360_multi_dir"]))
        wt_dir = Path(get_val("WEIGHT_DATASET_DIR", d_raw / "weight_dataset", ["paths", "weight_dataset_dir"]))
        fr_dir = Path(get_val("FRESHNESS_DATASET_DIR", d_raw / "freshness_dataset", ["paths", "freshness_dataset_dir"]))
        updated_fruit_dir = Path(get_val("UPDATED_FRUIT_DATASET_DIR", d_raw / "updated dataset", ["paths", "updated_fruit_dataset_dir"]))

        self.datasets = DatasetPathsSettings(
            root=self.root_dir,
            data_root=d_root if d_root.is_absolute() else self.root_dir / d_root,
            raw_data_root=d_raw if d_raw.is_absolute() else self.root_dir / d_raw,
            processed_data_root=d_proc if d_proc.is_absolute() else self.root_dir / d_proc,
            splits_root=d_splits if d_splits.is_absolute() else self.root_dir / d_splits,
            reports_root=d_reports if d_reports.is_absolute() else self.root_dir / d_reports,
            foodseg103_dir=fs_dir if fs_dir.is_absolute() else self.root_dir / fs_dir,
            uecfoodpix_dir=uec_dir if uec_dir.is_absolute() else self.root_dir / uec_dir,
            fruits360_dir=f360_dir if f360_dir.is_absolute() else self.root_dir / f360_dir,
            fruits360_multi_dir=f360m_dir if f360m_dir.is_absolute() else self.root_dir / f360m_dir,
            weight_dataset_dir=wt_dir if wt_dir.is_absolute() else self.root_dir / wt_dir,
            freshness_dataset_dir=fr_dir if fr_dir.is_absolute() else self.root_dir / fr_dir,
            updated_fruit_dataset_dir=updated_fruit_dir if updated_fruit_dir.is_absolute() else self.root_dir / updated_fruit_dir
        )

        # Detection Settings
        self.detection = DetectionSettings(
            model_type=str(get_val("DETECTION_MODEL_TYPE", "yolo11", ["detection", "model_type"])),
            model_variant=str(get_val("DETECTION_MODEL_VARIANT", "yolo11n.pt", ["detection", "model_variant"])),
            confidence_threshold=float(get_val("DETECTION_CONF_THRESHOLD", 0.35, ["detection", "conf_threshold"])),
            iou_threshold=float(get_val("DETECTION_IOU_THRESHOLD", 0.45, ["detection", "iou_threshold"])),
            image_size=int(get_val("DETECTION_IMAGE_SIZE", 640, ["detection", "image_size"])),
            max_detections=int(get_val("DETECTION_MAX_DETECTIONS", 30))
        )

        # Segmentation Settings
        self.segmentation = SegmentationSettings(
            model_type=str(get_val("SEGMENTATION_MODEL_TYPE", "sam2", ["segmentation", "model_type"])),
            checkpoint=str(get_val("SEGMENTATION_CHECKPOINT", "sam2_hiera_base_plus.pt", ["segmentation", "checkpoint"])),
            points_per_side=int(get_val("SEGMENTATION_POINTS_PER_SIDE", 32, ["segmentation", "points_per_side"])),
            pred_iou_thresh=float(get_val("SEGMENTATION_PRED_IOU_THRESH", 0.88, ["segmentation", "pred_iou_thresh"])),
            stability_score_thresh=float(get_val("SEGMENTATION_STABILITY_SCORE_THRESH", 0.95, ["segmentation", "stability_score_thresh"]))
        )

        # Depth Settings
        self.depth = DepthSettings(
            model_type=str(get_val("DEPTH_MODEL_TYPE", "depth_anything_v2", ["depth", "model_type"])),
            encoder=str(get_val("DEPTH_ENCODER", "vits", ["depth", "encoder"])),
            input_size=int(get_val("DEPTH_INPUT_SIZE", 518, ["depth", "input_size"]))
        )

        # Weight Estimation Settings
        density_path = Path(get_val("DENSITY_DATABASE_PATH", self.root_dir / "data" / "food_densities.json", ["weight_estimation", "density_database"]))
        self.weight = WeightSettings(
            reference_object=str(get_val("WEIGHT_REFERENCE_OBJECT", "plate", ["weight_estimation", "reference_object"])),
            reference_dimension_cm=float(get_val("WEIGHT_REFERENCE_DIMENSION_CM", 25.0, ["weight_estimation", "reference_dimension_cm"])),
            density_database_path=density_path if density_path.is_absolute() else self.root_dir / density_path,
            default_density_g_cm3=float(get_val("DEFAULT_FOOD_DENSITY", 0.85, ["weight_estimation", "default_density_g_cm3"]))
        )

        # Nutrition API Settings
        self.nutrition = NutritionSettings(
            usda_api_url=str(get_val("USDA_API_URL", "https://api.nal.usda.gov/fdc/v1", ["nutrition", "usda_api_url"])),
            usda_api_key=str(get_val("USDA_API_KEY", "DEMO_KEY", ["nutrition", "usda_api_key"])),
            cache_ttl_hours=int(get_val("NUTRITION_CACHE_TTL_HOURS", 168, ["nutrition", "cache_ttl_hours"]))
        )

        # Explainability Settings
        self.explainability = ExplainabilitySettings(
            target_layer=str(get_val("EXPLAINABILITY_TARGET_LAYER", "conv_head", ["explainability", "target_layer"])),
            colormap=str(get_val("EXPLAINABILITY_COLORMAP", "JET", ["explainability", "colormap"])),
            alpha=float(get_val("EXPLAINABILITY_ALPHA", 0.5, ["explainability", "alpha"]))
        )

        # Camera Settings
        self.camera = CameraSettings(
            device_index=int(get_val("CAMERA_DEVICE_INDEX", 0)),
            frame_width=int(get_val("CAMERA_FRAME_WIDTH", 1280)),
            frame_height=int(get_val("CAMERA_FRAME_HEIGHT", 720)),
            fps=int(get_val("CAMERA_FPS", 30))
        )

        # Output Directories
        self.outputs = OutputDirectories(root=self.root_dir)

        # Ensure directory creation on initialization
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Ensure all required output and log directories exist on disk."""
        dirs_to_create = [
            self.outputs.detections,
            self.outputs.segmentation,
            self.outputs.crops,
            self.outputs.depth,
            self.outputs.results,
            self.outputs.visualizations,
            self.outputs.logs,
            self.datasets.processed_data_root,
            self.datasets.splits_root,
            self.datasets.reports_root,
            self.root_dir / "outputs",
        ]
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        """Resolve 'auto' device to 'cuda' (if PyTorch and CUDA available) or 'cpu'."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def dataset_paths(self) -> DatasetPathsSettings:
        """Alias for self.datasets for compatibility."""
        return self.datasets


# Singleton instance
settings = Settings()


def get_settings() -> Settings:
    """Return singleton settings instance."""
    return settings
