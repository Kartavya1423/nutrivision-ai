"""
NutriVision AI — Detection Configuration Models
Defines dataclasses and parameters for YOLO11 training, inference,
and segmentation mask conversion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MaskConversionConfig:
    """Configuration for converting segmentation masks to YOLO bounding boxes."""
    min_area_pixels: int = 16          # Minimum pixel area to consider a valid object
    min_dimension_pixels: int = 4      # Minimum width or height in pixels
    boundary_padding_ratio: float = 0.0 # Extra padding around bounding box
    use_connected_components: bool = True
    connectivity: int = 8             # 4 or 8 connectivity
    save_crops: bool = False
    crop_output_dir: Optional[Path] = None


@dataclass
class YOLO11TrainingConfig:
    """Hyperparameters and runtime configuration for YOLO11 training."""
    model: str = "yolo11n.pt"         # Nano baseline variant
    data: str = "data/processed/yolo_detection/dataset.yaml"
    epochs: int = 5                   # Baseline verification epochs
    batch_size: int = 8               # Conservative batch size for CPU/memory
    image_size: int = 640
    patience: int = 10
    device: str = "cpu"               # Detected dynamically (cpu or cuda)
    workers: int = 2
    seed: int = 42
    deterministic: bool = True
    project: str = "models/detection/yolo11"
    run_name: str = "train_run"
    exist_ok: bool = True
    save: bool = True
    save_period: int = -1
    val: bool = True
    plots: bool = True
    verbose: bool = True


@dataclass
class DetectionInferenceConfig:
    """Inference parameters for food detection."""
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    max_detections: int = 100
    save_annotated: bool = True
    save_crops: bool = True
    crops_dir: str = "outputs/detection/crops"
    predictions_dir: str = "outputs/detection/predictions"
