"""
NutriVision AI — Food Freshness Detection Module (Phase 6)
Assesses quality states: fresh vs rotten/spoiled.

Features:
- Fine-tuned EfficientNet-B4 backbone on real produce benchmark data (apples, bananas, oranges)
- Deterministic, non-fabricated classification probabilities
- Configurable confidence thresholding with explicit LOW_CONFIDENCE handling
- Isolates inference to segmented food crops (excluding background/plate artifacts)
- Exposes convolutional feature layer (features[-1]) for Phase 7 Grad-CAM explainability
- Clear error handling for missing models, invalid crops, and uncalibrated assessments
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from src.config.settings import Settings, get_settings
from src.detection.detector import ModelStatus


class FreshnessState(str, Enum):
    """Quality states for food preservation assessment."""
    FRESH = "fresh"
    AGING = "aging"
    SPOILED = "spoiled"
    ROTTEN = "rotten"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"


@dataclass
class FreshnessEvaluation:
    """Freshness prediction output for an individual food crop."""
    state: FreshnessState = FreshnessState.UNKNOWN
    confidence: Optional[float] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    food_class: str = "food"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "confidence": round(float(self.confidence), 4) if self.confidence is not None else None,
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
            "food_class": self.food_class,
            "notes": self.notes,
        }


@dataclass
class FreshnessResult:
    """Envelope containing freshness assessment."""
    status: ModelStatus
    evaluation: Optional[FreshnessEvaluation] = None
    inference_time_ms: float = 0.0
    model_name: str = "BaseFreshnessDetector"
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status == ModelStatus.READY and self.evaluation is not None

    @property
    def predicted_state(self) -> Optional[FreshnessState]:
        return self.evaluation.state if self.evaluation else FreshnessState.UNKNOWN

    @property
    def confidence(self) -> float:
        return float(self.evaluation.confidence) if (self.evaluation and self.evaluation.confidence is not None) else 0.0

    @property
    def probabilities(self) -> Dict[str, float]:
        return self.evaluation.probabilities if self.evaluation else {}

    @property
    def assessments(self) -> List[Any]:
        """Provides backwards compatibility for legacy pipeline callers."""
        if self.evaluation is not None:
            class _LegacyAssessment:
                def __init__(self, ev):
                    self.freshness_score = ev.confidence or 0.0
                    self.state = ev.state
                    self.notes = ev.notes
            return [_LegacyAssessment(self.evaluation)]
        return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "model_name": self.model_name,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "evaluation": self.evaluation.to_dict() if self.evaluation else {
                "state": "unknown",
                "confidence": None,
                "probabilities": {},
                "food_class": "unknown",
                "notes": "",
            },
            "error_message": self.error_message,
        }


class BaseFreshnessDetector(ABC):
    """Abstract interface for food freshness classification."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else None
        self.model: Optional[nn.Module] = None
        self._status: ModelStatus = ModelStatus.MODEL_NOT_TRAINED
        self._error_msg: Optional[str] = None

    @abstractmethod
    def load_model(self) -> bool:
        pass

    @abstractmethod
    def assess_freshness(
        self,
        food_crop: Union[str, Path, np.ndarray, Image.Image, torch.Tensor],
        class_name: str = "food",
        item_index: int = 0,
    ) -> FreshnessResult:
        pass

    def is_ready(self) -> bool:
        return self._status == ModelStatus.READY

    @property
    def status(self) -> ModelStatus:
        return self._status

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass


class EfficientNetFreshnessModel(nn.Module):
    """
    EfficientNet-B4 fine-tuned for freshness detection.
    Exposes final convolutional layer for Grad-CAM explainability.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4, pretrained: bool = False):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)

        # Classifier head
        in_features = self.backbone.classifier[1].in_features  # 1792
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns final convolutional feature layer for Phase 7 explainability."""
        return self.backbone.features[-1]


class EfficientNetFreshnessDetector(BaseFreshnessDetector):
    """
    Production EfficientNet-B4 food freshness condition detector.
    Loads real trained weights, evaluates crops, and provides confidence & probabilities.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        settings: Optional[Settings] = None,
        confidence_threshold: float = 0.60,
    ):
        cfg = settings or get_settings()

        # Resolve model path
        resolved_path = None
        if model_path is not None:
            resolved_path = Path(model_path)
            if not resolved_path.is_absolute():
                resolved_path = cfg.root_dir / resolved_path
        else:
            # Check candidate paths in order of preference
            candidates = [
                cfg.root_dir / "models" / "freshness" / "best.pth",
                cfg.root_dir / "models" / "freshness" / "freshness_classifier.pth",
                cfg.model_paths.freshness_model_path,
            ]
            for cand in candidates:
                if cand.is_file():
                    resolved_path = cand
                    break
            if resolved_path is None:
                resolved_path = cfg.model_paths.freshness_model_path

        super().__init__(model_path=resolved_path)

        # Hardware device
        self.device = torch.device(device or cfg.resolve_device())
        self.confidence_threshold = float(confidence_threshold)
        self.image_size: int = 224
        self.class_mapping: Dict[int, str] = {0: "fresh", 1: "rotten"}
        self.food_classes: List[str] = ["apple", "banana", "orange"]

        # Preprocessing transforms
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.load_model()

    def load_model(self) -> bool:
        """Loads trained EfficientNet-B4 weights into model."""
        if self.model_path is None or not self.model_path.is_file():
            self._status = ModelStatus.MODEL_NOT_TRAINED
            self._error_msg = (
                f"Freshness weights not found at: '{self.model_path}'. "
                "Freshness classifier has not yet been trained. "
                "Returning MODEL_NOT_TRAINED."
            )
            self.model = None
            return False

        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)

            num_classes = ckpt.get("num_classes", 2)
            self.image_size = ckpt.get("image_size", 224)
            mapping = ckpt.get("class_mapping", {0: "fresh", 1: "rotten"})
            self.class_mapping = {int(k): str(v) for k, v in mapping.items()}

            # Rebuild transform with checkpoint image size
            norm = ckpt.get("normalization", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
            self.transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=norm["mean"], std=norm["std"]),
            ])

            self.model = EfficientNetFreshnessModel(num_classes=num_classes, pretrained=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            self._status = ModelStatus.READY
            self._error_msg = None
            return True

        except Exception as e:
            self._status = ModelStatus.ERROR
            self._error_msg = f"Failed to load freshness model: {str(e)}"
            self.model = None
            return False

    def assess_freshness(
        self,
        food_crop: Union[str, Path, np.ndarray, Image.Image, torch.Tensor],
        class_name: str = "food",
        item_index: int = 0,
    ) -> FreshnessResult:
        """
        Assesses freshness of a single food crop.

        Returns:
            FreshnessResult containing predicted state, confidence, and class probabilities.
        """
        t0 = time.perf_counter()

        if not self.is_ready():
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return FreshnessResult(
                status=self._status,
                evaluation=FreshnessEvaluation(
                    state=FreshnessState.UNKNOWN,
                    confidence=None,
                    probabilities={},
                    food_class=class_name,
                    notes=self._error_msg or "MODEL_NOT_TRAINED: Freshness classifier not ready.",
                ),
                inference_time_ms=elapsed_ms,
                model_name="EfficientNetFreshnessDetector",
                error_message=self._error_msg or "MODEL_NOT_TRAINED",
            )

        # 1. Preprocess input to PIL Image
        pil_img: Optional[Image.Image] = None
        try:
            if isinstance(food_crop, (str, Path)):
                p = Path(food_crop)
                if not p.is_file():
                    return FreshnessResult(
                        status=ModelStatus.ERROR,
                        evaluation=FreshnessEvaluation(state=FreshnessState.UNKNOWN, food_class=class_name),
                        error_message=f"INVALID_IMAGE: File does not exist at '{p}'.",
                    )
                with open(p, "rb") as f:
                    pil_img = Image.open(f).convert("RGB")

            elif isinstance(food_crop, np.ndarray):
                if food_crop.size == 0 or food_crop.ndim < 2:
                    return FreshnessResult(
                        status=ModelStatus.ERROR,
                        evaluation=FreshnessEvaluation(state=FreshnessState.UNKNOWN, food_class=class_name),
                        error_message="FOOD_CROP_UNAVAILABLE: Empty or invalid numpy array provided.",
                    )
                if food_crop.ndim == 2:
                    pil_img = Image.fromarray(food_crop).convert("RGB")
                elif food_crop.shape[2] == 4:
                    # RGBA crop from SAM2
                    rgba_img = Image.fromarray(food_crop)
                    bg = Image.new("RGB", rgba_img.size, (255, 255, 255))
                    bg.paste(rgba_img, mask=rgba_img.split()[3])
                    pil_img = bg
                elif food_crop.shape[2] == 3:
                    # Assume BGR if coming from OpenCV, convert to RGB
                    rgb_arr = cv2.cvtColor(food_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_arr)
                else:
                    pil_img = Image.fromarray(food_crop).convert("RGB")

            elif isinstance(food_crop, Image.Image):
                if food_crop.size[0] == 0 or food_crop.size[1] == 0:
                    return FreshnessResult(
                        status=ModelStatus.ERROR,
                        evaluation=FreshnessEvaluation(state=FreshnessState.UNKNOWN, food_class=class_name),
                        error_message="FOOD_CROP_UNAVAILABLE: Zero-dimension image provided.",
                    )
                if food_crop.mode == "RGBA":
                    bg = Image.new("RGB", food_crop.size, (255, 255, 255))
                    bg.paste(food_crop, mask=food_crop.split()[3])
                    pil_img = bg
                else:
                    pil_img = food_crop.convert("RGB")

            elif isinstance(food_crop, torch.Tensor):
                # Convert 3D or 4D tensor back to PIL for consistent transform
                if food_crop.ndim == 4:
                    food_crop = food_crop.squeeze(0)
                arr = food_crop.detach().cpu().permute(1, 2, 0).numpy()
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(arr)

            else:
                return FreshnessResult(
                    status=ModelStatus.ERROR,
                    evaluation=FreshnessEvaluation(state=FreshnessState.UNKNOWN, food_class=class_name),
                    error_message=f"INVALID_IMAGE: Unsupported food crop type '{type(food_crop)}'.",
                )

        except Exception as e:
            return FreshnessResult(
                status=ModelStatus.ERROR,
                evaluation=FreshnessEvaluation(state=FreshnessState.UNKNOWN, food_class=class_name),
                error_message=f"FRESHNESS_INFERENCE_FAILED: Preprocessing error: {str(e)}",
            )

        # 2. Run inference
        try:
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1).squeeze(0)

            prob_fresh = float(probs[0].item())
            prob_rotten = float(probs[1].item())
            pred_idx = int(torch.argmax(probs).item())
            max_prob = max(prob_fresh, prob_rotten)

            probabilities = {
                "fresh": round(prob_fresh, 4),
                "rotten": round(prob_rotten, 4),
            }

            # 3. Confidence thresholding
            if max_prob < self.confidence_threshold:
                state = FreshnessState.UNKNOWN
                notes = (
                    f"LOW_CONFIDENCE: Maximum prediction probability ({max_prob:.2f}) is below "
                    f"threshold ({self.confidence_threshold:.2f}). Visual assessment uncertain."
                )
            else:
                state = FreshnessState.FRESH if pred_idx == 0 else FreshnessState.ROTTEN
                notes = (
                    f"Predicted {state.value} ({max_prob*100:.1f}% confidence). "
                    "Visual assessment estimate only; not a laboratory food-safety certification."
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            return FreshnessResult(
                status=ModelStatus.READY,
                evaluation=FreshnessEvaluation(
                    state=state,
                    confidence=round(max_prob, 4),
                    probabilities=probabilities,
                    food_class=class_name,
                    notes=notes,
                ),
                inference_time_ms=elapsed_ms,
                model_name="EfficientNetFreshnessDetector",
                error_message=None,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return FreshnessResult(
                status=ModelStatus.ERROR,
                evaluation=FreshnessEvaluation(state=FreshnessState.UNKNOWN, food_class=class_name),
                inference_time_ms=elapsed_ms,
                model_name="EfficientNetFreshnessDetector",
                error_message=f"FRESHNESS_INFERENCE_FAILED: Model execution error: {str(e)}",
            )

    def get_gradcam_target_layer(self) -> Optional[nn.Module]:
        """Exposes final convolutional layer for Phase 7 explainability."""
        if self.model is not None and hasattr(self.model, "get_gradcam_target_layer"):
            return self.model.get_gradcam_target_layer()
        return None

    def get_model_info(self) -> Dict[str, Any]:
        """Returns structured metadata about model state, device, and capabilities."""
        return {
            "model_type": "efficientnet_b4_freshness",
            "weights_path": str(self.model_path) if self.model_path else None,
            "status": self._status.value,
            "device": str(self.device),
            "image_size": self.image_size,
            "confidence_threshold": self.confidence_threshold,
            "states_supported": [s.value for s in FreshnessState if s not in (FreshnessState.UNKNOWN, FreshnessState.LOW_CONFIDENCE)],
            "produce_categories_supported": self.food_classes,
            "is_ready": self.is_ready(),
            "error_message": self._error_msg,
        }
