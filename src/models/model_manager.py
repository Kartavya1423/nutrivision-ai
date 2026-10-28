"""
NutriVision AI — Centralized Model Manager (Phase 8)
Responsible for:
- One-time loading and lifecycle management of all 5 deep neural networks
- Hardware device detection (CUDA vs CPU) without false claims
- Model readiness, checkpoint verification, and structured status reporting
- Reusing loaded model instances across API and UI requests
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import torch

from src.classification.efficientnet_classifier import EfficientNetB4Classifier
from src.classification.fruit_classifier import FruitClassifier
from src.config.settings import Settings, get_settings
from src.depth.depth_anything import DepthAnythingV2Estimator
from src.detection.detector import ModelStatus
from src.detection.yolo_detector import YOLO11FoodDetector
from src.freshness.freshness_detector import EfficientNetFreshnessDetector
from src.segmentation.sam2_segmenter import SAM2FoodSegmenter

logger = logging.getLogger("nutrivision.model_manager")


class ModelManager:
    """
    Centralized Model Manager.
    Loads and caches neural network backends once to eliminate redundant disk I/O.
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self, settings: Optional[Settings] = None, preload: bool = False):
        self.settings = settings or get_settings()
        self.root_dir = self.settings.root_dir
        self.device = self._resolve_device()

        # Checkpoint candidates
        import os
        env_det = os.environ.get("NUTRIVISION_DETECTION_MODEL")
        self._yolo_checkpoint = self._find_checkpoint([
            env_det if env_det else None,
            self.settings.model_paths.detection_model_path,
            self.root_dir / "models" / "detection" / "yolo11" / "best.pt",
            self.root_dir / "yolo11n.pt",
            self.root_dir / "models" / "detection" / "yolo11n.pt",
            self.root_dir / "models" / "detection" / "yolo11" / "phase2_fixed_run" / "weights" / "best.pt",
            self.root_dir / "models" / "detection" / "yolo11" / "repair_run" / "best.pt",
            self.root_dir / "models" / "detection" / "yolo11" / "production" / "best.pt",
            self.root_dir / "models" / "detection" / "yolo11" / "last.pt",
        ])

        self._sam2_checkpoint = self._find_checkpoint([
            self.root_dir / "models" / "segmentation" / "sam2.1_t.pt",
            self.root_dir / "models" / "segmentation" / "sam2_hiera_base_plus.pt",
            self.settings.model_paths.segmentation_model_path,
        ])

        import os
        env_cls = os.environ.get("NUTRIVISION_CLASSIFIER_MODEL")
        cls_candidates = [
            env_cls if env_cls else None,
            self.root_dir / "models" / "classification" / "efficientnet_b4_food.pth",
            self.settings.model_paths.classification_model_path,
            self.root_dir / "models" / "classification" / "target_foods_v1" / "best.pth",
            self.root_dir / "models" / "classification" / "efficientnet_b4" / "phase4_final_run" / "best.pth",
            self.root_dir / "models" / "classification" / "efficientnet_b4" / "best.pth",
            self.root_dir / "models" / "classification" / "mobilenet_v2_food50.pth",
        ]
        self._classifier_checkpoint = self._find_checkpoint(cls_candidates)

        self._freshness_checkpoint = self._find_checkpoint([
            self.root_dir / "models" / "freshness" / "best.pth",
            self.root_dir / "models" / "freshness" / "freshness_classifier.pth",
            self.settings.model_paths.freshness_model_path,
        ])

        self._depth_checkpoint = self._find_checkpoint([
            self.root_dir / "models" / "depth" / "depth_anything_v2_vits.pth",
            self.settings.model_paths.depth_model_path,
        ])

        env_fruit = os.environ.get("NUTRIVISION_FRUIT_CLASSIFIER_MODEL")
        fruit_candidates = [
            env_fruit if env_fruit else None,
            getattr(self.settings.model_paths, "fruit_classifier_path", None),
            self.root_dir / "models" / "classification" / "fruit_classifier_corrected_split" / "best_model.pth",
            self.root_dir / "models" / "fruit_classifier" / "best_model.pth",
            self.root_dir / "models" / "classification" / "fruit_classifier" / "best_model.pth",
            self.root_dir / "models" / "fruit_classifier" / "last_model.pth",
            self.root_dir / "models" / "fruit_classifier" / "final_model.pth",
        ]
        self._fruit_classifier_checkpoint = self._find_checkpoint(fruit_candidates)

        # Lazy model instances
        self._detector: Optional[YOLO11FoodDetector] = None
        self._segmenter: Optional[SAM2FoodSegmenter] = None
        self._classifier: Optional[EfficientNetB4Classifier] = None
        self._fruit_classifier: Optional[FruitClassifier] = None
        self._freshness_detector: Optional[EfficientNetFreshnessDetector] = None
        self._depth_estimator: Optional[DepthAnythingV2Estimator] = None

        if preload:
            self.preload_all()

    def _resolve_device(self) -> str:
        """Accurately determine device: CUDA or CPU without false claims."""
        configured = getattr(self.settings, "device", "auto")
        if configured == "cuda" and torch.cuda.is_available():
            dev = f"cuda:{torch.cuda.current_device()}"
        elif configured == "auto":
            dev = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        else:
            dev = "cpu"
        return dev

    @staticmethod
    def _find_checkpoint(candidates: list) -> Optional[Path]:
        for c in candidates:
            if c is not None:
                p = Path(c)
                if p.is_file():
                    return p
        return None

    @property
    def detector(self) -> YOLO11FoodDetector:
        if self._detector is None:
            logger.info("Initializing YOLO11 detector...")
            conf = getattr(self.settings.detection, "confidence_threshold", 0.25)
            self._detector = YOLO11FoodDetector(
                model_path=self._yolo_checkpoint,
                conf_threshold=conf,
                device=self.device,
                settings=self.settings,
            )
        return self._detector

    @property
    def segmenter(self) -> SAM2FoodSegmenter:
        if self._segmenter is None:
            logger.info("Initializing SAM 2 segmenter...")
            self._segmenter = SAM2FoodSegmenter(
                checkpoint_path=self._sam2_checkpoint,
                device=self.device,
                settings=self.settings,
            )
        return self._segmenter

    @property
    def classifier(self) -> EfficientNetB4Classifier:
        if self._classifier is None:
            logger.info("Initializing EfficientNet-B4 classifier...")
            self._classifier = EfficientNetB4Classifier(
                model_path=self._classifier_checkpoint,
                device=self.device,
                settings=self.settings,
            )
        return self._classifier

    @property
    def fruit_classifier(self) -> FruitClassifier:
        if self._fruit_classifier is None:
            logger.info("Initializing Fruit Classifier...")
            self._fruit_classifier = FruitClassifier(
                model_path=self._fruit_classifier_checkpoint,
                device=self.device,
                settings=self.settings,
            )
        return self._fruit_classifier

    @property
    def freshness_detector(self) -> EfficientNetFreshnessDetector:
        if self._freshness_detector is None:
            logger.info("Initializing Freshness classifier...")
            self._freshness_detector = EfficientNetFreshnessDetector(
                model_path=self._freshness_checkpoint,
                device=self.device,
                settings=self.settings,
            )
        return self._freshness_detector

    @property
    def depth_estimator(self) -> DepthAnythingV2Estimator:
        if self._depth_estimator is None:
            logger.info("Initializing Depth Anything V2 estimator...")
            self._depth_estimator = DepthAnythingV2Estimator(
                model_path=self._depth_checkpoint,
                device=self.device,
                settings=self.settings,
            )
        return self._depth_estimator

    def preload_all(self):
        """Eagerly load all models into memory/GPU."""
        _ = self.detector
        _ = self.segmenter
        _ = self.classifier
        _ = self.freshness_detector
        _ = self.depth_estimator

    def get_status(self) -> Dict[str, Any]:
        """Detailed status report for all models."""
        yolo_ready = self._yolo_checkpoint is not None and self._yolo_checkpoint.is_file()
        sam2_ready = self._sam2_checkpoint is not None and self._sam2_checkpoint.is_file()
        eff_ready = self._classifier_checkpoint is not None and self._classifier_checkpoint.is_file()
        fruit_ready = self._fruit_classifier_checkpoint is not None and self._fruit_classifier_checkpoint.is_file()
        fresh_ready = self._freshness_checkpoint is not None and self._freshness_checkpoint.is_file()
        depth_ready = self._depth_checkpoint is not None and self._depth_checkpoint.is_file()

        models_info = {
            "yolo11": {
                "name": "YOLO11 Food Detector",
                "checkpoint": str(self._yolo_checkpoint) if self._yolo_checkpoint else None,
                "loaded": self._detector is not None and self._detector.is_ready(),
                "ready": yolo_ready,
                "status": "READY" if yolo_ready else "CHECKPOINT_NOT_FOUND",
                "device": self.device,
                "error": None if yolo_ready else "YOLO11 best.pt checkpoint not found",
            },
            "sam2": {
                "name": "SAM 2 Instance Segmenter",
                "checkpoint": str(self._sam2_checkpoint) if self._sam2_checkpoint else None,
                "loaded": self._segmenter is not None and self._segmenter.is_ready(),
                "ready": sam2_ready,
                "status": "READY" if sam2_ready else "CHECKPOINT_NOT_FOUND",
                "device": self.device,
                "error": None if sam2_ready else "SAM2 checkpoint not found",
            },
            "efficientnet_b4": {
                "name": "EfficientNet-B4 Food Classifier",
                "checkpoint": str(self._classifier_checkpoint) if self._classifier_checkpoint else None,
                "model_version": (
                    "target_foods_v1"
                    if self._classifier_checkpoint and "target_foods_v1" in str(self._classifier_checkpoint)
                    else "apple_baseline_198"
                ),
                "loaded": self._classifier is not None and self._classifier.is_ready(),
                "ready": eff_ready,
                "status": "READY" if eff_ready else "CHECKPOINT_NOT_FOUND",
                "device": self.device,
                "error": None if eff_ready else "EfficientNet-B4 best.pth checkpoint not found",
            },
            "fruit_classifier": {
                "name": "EfficientNet-B4 6-Class Fruit Classifier",
                "checkpoint": str(self._fruit_classifier_checkpoint) if self._fruit_classifier_checkpoint else None,
                "model_version": "updated_fruit_dataset_v1",
                "classes": ["Apple_Good", "Banana_Good", "Guava_Good", "Lime_Good", "Orange_Good", "Pomegranate_Good"],
                "loaded": self._fruit_classifier is not None and self._fruit_classifier.is_ready(),
                "ready": fruit_ready,
                "status": "READY" if fruit_ready else "CHECKPOINT_NOT_FOUND",
                "device": self.device,
                "error": (
                    None
                    if fruit_ready
                    else "Fruit classifier checkpoint not found. Train with 'python scripts/train_fruit_classifier.py'."
                ),
            },
            "freshness": {
                "name": "EfficientNet Freshness Classifier",
                "checkpoint": str(self._freshness_checkpoint) if self._freshness_checkpoint else None,
                "loaded": self._freshness_detector is not None and self._freshness_detector.is_ready(),
                "ready": fresh_ready,
                "status": "READY" if fresh_ready else "CHECKPOINT_NOT_FOUND",
                "device": self.device,
                "error": None if fresh_ready else "Freshness best.pth checkpoint not found",
            },
            "depth_anything_v2": {
                "name": "Depth Anything V2 Monocular Depth",
                "checkpoint": str(self._depth_checkpoint) if self._depth_checkpoint else None,
                "loaded": self._depth_estimator is not None and self._depth_estimator.is_ready(),
                "ready": depth_ready,
                "status": "READY" if depth_ready else "CHECKPOINT_NOT_FOUND",
                "device": self.device,
                "error": None if depth_ready else "Depth Anything V2 checkpoint not found",
            },
        }

        all_ready = all(m["ready"] for m in models_info.values())

        return {
            "status": "READY" if all_ready else "PARTIALLY_READY",
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_models": len(models_info),
            "ready_models_count": sum(1 for m in models_info.values() if m["ready"]),
            "models": models_info,
        }


def get_model_manager(settings: Optional[Settings] = None) -> ModelManager:
    """Singleton getter for centralized model manager."""
    if ModelManager._instance is None:
        ModelManager._instance = ModelManager(settings=settings)
    return ModelManager._instance
