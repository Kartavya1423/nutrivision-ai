"""
NutriVision AI — EfficientNet-B4 Fruit Classifier (Updated Fruit Dataset)
Recognizes 6 fruit categories from the updated dataset:
- Apple_Good
- Banana_Good
- Guava_Good
- Lime_Good
- Orange_Good
- Pomegranate_Good

Features:
- Deterministic 6-class mapping
- Honest freshness handling: reports 'Good' as the dataset category label without claiming generic rotten detection
- Safe checkpoint loading with informative error messages when weights are not yet trained
- Connects predictions to USDA FoodData Central nutrition service
- Supports Grad-CAM visual attention heatmaps
"""
import torch

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.classification.classifier import (
    BaseFoodClassifier,
    ClassificationOutput,
    ClassificationResult,
    ClassificationStatus,
)
from src.config.settings import Settings, get_settings
from src.data.fruit_dataset_loader import NORM_MEAN, NORM_STD, parse_fruit_class
from src.detection.detector import ModelStatus


# Canonical 6 fruit classes in deterministic alphabetical order
FRUIT_CLASSES: List[str] = [
    "Apple_Good",
    "Banana_Good",
    "Guava_Good",
    "Lime_Good",
    "Orange_Good",
    "Pomegranate_Good",
]


class FruitClassifier(BaseFoodClassifier):
    """
    Dedicated 6-Class Fruit Classifier using EfficientNet-B4.
    Trained on data/raw/updated dataset/.
    """

    CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(FRUIT_CLASSES)}
    IDX_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(FRUIT_CLASSES)}

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        settings: Optional[Settings] = None,
        confidence_threshold: float = 0.35,
        image_size: int = 224,
    ):
        self.settings = settings or get_settings()
        init_path = model_path or getattr(self.settings.model_paths, "fruit_classifier_path", None)
        super().__init__(model_path=init_path)

        self.device = device or self.settings.resolve_device()
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.class_names: List[str] = list(FRUIT_CLASSES)
        self.class_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class: Dict[int, str] = {i: name for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.norm_mean = list(NORM_MEAN)
        self.norm_std = list(NORM_STD)
        self.training_config: Dict[str, Any] = {}
        self.val_metrics: Dict[str, Any] = {}
        self.test_metrics: Dict[str, Any] = {}

        self.load_model()

    @property
    def classes(self) -> List[str]:
        """Class names list."""
        return self.class_names

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        settings: Optional[Settings] = None,
    ) -> "FruitClassifier":
        dev_str = str(device) if device is not None else None
        return cls(model_path=Path(checkpoint_path), device=dev_str, settings=settings)

    def eval(self) -> "FruitClassifier":
        if self.model is not None:
            self.model.eval()
        return self

    def _resolve_checkpoint_path(self) -> Optional[Path]:
        """Check primary path, or fallback candidate locations."""
        if self.model_path is not None:
            p = Path(self.model_path)
            if p.is_file():
                return p

        # Check candidate locations
        root = self.settings.root_dir
        for cand in [
            root / "models" / "classification" / "fruit_classifier_corrected_split" / "best_model.pth",
            root / "models" / "fruit_classifier" / "best_model.pth",
            root / "models" / "classification" / "fruit_classifier" / "best_model.pth",
            root / "models" / "fruit_classifier" / "best.pth",
            root / "models" / "fruit_classifier" / "fruit_classifier.pth",
        ]:
            if cand.is_file():
                return cand
        return None

    def load_model(self) -> bool:
        """
        Load trained EfficientNet-B4 fruit classification checkpoint.
        Returns False and sets MODEL_NOT_TRAINED if weights are absent.
        """
        resolved_path = self._resolve_checkpoint_path()
        if resolved_path is None:
            self._status = ModelStatus.MODEL_NOT_TRAINED
            self._error_msg = (
                f"Fruit classifier weights not found at: '{self.model_path}'. "
                "The 6-class fruit classifier has not yet been trained on the updated dataset. "
                "Please run: 'python scripts/train_fruit_classifier.py' to train the model."
            )
            self.model = None
            return False

        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models

            self.model_path = resolved_path
            ckpt = torch.load(resolved_path, map_location=self.device, weights_only=False)

            if isinstance(ckpt, dict) and "class_names" in ckpt:
                self.class_names = ckpt["class_names"]
                self.class_to_idx = ckpt.get("class_to_idx", {name: i for i, name in enumerate(self.class_names)})
                self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}
                self.num_classes = len(self.class_names)
                if "image_size" in ckpt:
                    self.image_size = int(ckpt["image_size"])
                if "normalization" in ckpt:
                    self.norm_mean = ckpt["normalization"].get("mean", self.norm_mean)
                    self.norm_std = ckpt["normalization"].get("std", self.norm_std)
                self.training_config = ckpt.get("training_config", {})
                self.val_metrics = ckpt.get("val_metrics", {})
                self.test_metrics = ckpt.get("test_metrics", {})
                state_dict = ckpt.get("model_state_dict", ckpt)
            else:
                state_dict = ckpt

            # Build EfficientNet-B4 architecture with 6-class head
            base_model = models.efficientnet_b4(weights=None)
            in_features = base_model.classifier[1].in_features
            base_model.classifier[1] = nn.Linear(in_features, self.num_classes)
            base_model.load_state_dict(state_dict)

            self.model = base_model.to(self.device)
            self.model.eval()
            self._status = ModelStatus.READY
            self._error_msg = None
            return True

        except Exception as exc:
            self._status = ModelStatus.ERROR
            self._error_msg = f"Failed to load fruit classifier weights: {str(exc)}"
            self.model = None
            return False

    def is_ready(self) -> bool:
        return self._status == ModelStatus.READY and self.model is not None

    def get_target_layer(self):
        """Exposes the final convolutional layer of EfficientNet-B4 for Grad-CAM."""
        if self.model is None:
            return None
        return self.model.features[-1]

    def _preprocess_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> "torch.Tensor":
        """
        Preprocesses food image or crop for EfficientNet-B4 inference:
        1. Converts RGBA by compositing against a clean white background.
        2. Bilinear resize to (image_size, image_size).
        3. Scales to float32 Tensor in [0, 1].
        4. Applies standard ImageNet normalization.
        """
        import torch

        if isinstance(image, (str, Path)):
            pil_img = Image.open(image)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            if image.ndim == 2:
                pil_img = Image.fromarray(image).convert("RGB")
            elif image.ndim == 3 and image.shape[2] == 4:
                pil_img = Image.fromarray(image, mode="RGBA")
            else:
                pil_img = Image.fromarray(image, mode="RGB")
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            raise ValueError(f"Unsupported image input type: {type(image)}")

        # Handle RGBA transparency
        if pil_img.mode == "RGBA":
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])
            pil_img = bg
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        pil_resized = pil_img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        np_arr = np.array(pil_resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np_arr).permute(2, 0, 1)

        mean = torch.tensor(self.norm_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.norm_std, dtype=torch.float32).view(3, 1, 1)
        normalized = (tensor - mean) / std

        return normalized.unsqueeze(0).to(self.device)

    def classify(
        self,
        food_crop: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 5,
        confidence_threshold: Optional[float] = None,
    ) -> ClassificationResult:
        """
        Classifies fruit crop into one of the 6 fruit categories.
        Returns Top-1 classification, Top-K candidates, and confidence status.
        """
        t0 = time.perf_counter()

        if not self.is_ready():
            return ClassificationResult(
                status=self._status,
                result=ClassificationOutput(class_id=None, class_name=None, confidence=None),
                inference_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                model_name="FruitClassifier",
                confidence_status=ClassificationStatus.CLASSIFICATION_MODEL_NOT_READY.value,
                error_message=self._error_msg or "MODEL_NOT_TRAINED: Fruit classifier weights not available.",
            )

        try:
            import torch

            t_pre_0 = time.perf_counter()
            input_tensor = self._preprocess_image(food_crop)
            t_pre_ms = (time.perf_counter() - t_pre_0) * 1000.0

            t_fwd_0 = time.perf_counter()
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
            t_fwd_ms = (time.perf_counter() - t_fwd_0) * 1000.0

            t_post_0 = time.perf_counter()
            k = min(top_k, len(self.class_names))
            top_probs, top_indices = torch.topk(probs, k=k)

            top_candidates = []
            for rank, (p, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
                c_name = self.idx_to_class.get(idx, f"class_{idx}")
                fruit_name, ds_label = parse_fruit_class(c_name)
                top_candidates.append({
                    "rank": rank + 1,
                    "class_id": int(idx),
                    "class_name": c_name,
                    "fruit_name": fruit_name,
                    "dataset_label": ds_label,
                    "confidence": round(float(p), 4),
                })

            top1_id = int(top_indices[0].item())
            top1_class_name = self.idx_to_class.get(top1_id, f"class_{top1_id}")
            top1_fruit_name, top1_ds_label = parse_fruit_class(top1_class_name)
            top1_conf = float(top_probs[0].item())

            thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
            if top1_conf >= thresh:
                conf_status = ClassificationStatus.CLASSIFICATION_SUCCESS.value
            else:
                conf_status = ClassificationStatus.CLASSIFICATION_LOW_CONFIDENCE.value

            t_post_ms = (time.perf_counter() - t_post_0) * 1000.0
            total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

            return ClassificationResult(
                status=ModelStatus.READY,
                result=ClassificationOutput(
                    class_id=top1_id,
                    class_name=top1_class_name,
                    confidence=round(top1_conf, 4),
                    top_k_candidates=top_candidates,
                ),
                inference_time_ms=round(total_elapsed_ms, 2),
                model_name="FruitClassifier",
                confidence_status=conf_status,
                breakdown_ms={
                    "preprocess_ms": t_pre_ms,
                    "forward_ms": t_fwd_ms,
                    "postprocess_ms": t_post_ms,
                    "total_ms": total_elapsed_ms,
                },
            )

        except Exception as exc:
            total_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return ClassificationResult(
                status=ModelStatus.ERROR,
                result=ClassificationOutput(class_id=None, class_name=None, confidence=None),
                inference_time_ms=round(total_elapsed_ms, 2),
                model_name="FruitClassifier",
                confidence_status=ClassificationStatus.CLASSIFICATION_FAILED.value,
                error_message=f"Inference error: {str(exc)}",
            )

    def predict_structured(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        weight_grams: Optional[float] = 150.0,
        enable_nutrition: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes fruit classification and returns the structured response matching
        the project contract:
        {
          "fruit_name": "Apple",
          "dataset_class": "Apple_Good",
          "freshness_status": "Good",
          "confidence": 0.94,
          "classification_model": "EfficientNet-B4",
          "nutrition": { ... }
        }
        """
        cls_res = self.classify(image)
        if not cls_res.is_success or cls_res.result is None:
            return {
                "fruit_name": None,
                "dataset_class": None,
                "freshness_status": "Unknown",
                "freshness_note": "No valid prediction available.",
                "confidence": 0.0,
                "confidence_status": cls_res.confidence_status,
                "classification_model": "EfficientNet-B4",
                "nutrition": {
                    "calories": None,
                    "protein": None,
                    "carbohydrates": None,
                    "fat": None,
                },
                "status": cls_res.status.value,
                "error_message": cls_res.error_message,
            }

        class_name = cls_res.result.class_name
        fruit_name, ds_label = parse_fruit_class(class_name)
        conf = cls_res.result.confidence

        nutrition_summary = {
            "calories": None,
            "protein": None,
            "carbohydrates": None,
            "fat": None,
            "status": "UNAVAILABLE",
        }

        if enable_nutrition:
            try:
                from src.nutrition.nutrition_service import NutritionService
                svc = NutritionService()
                nut = svc.get_nutrition_for_item(fruit_name.lower(), weight_grams=weight_grams)
                if nut and nut.status == "AVAILABLE":
                    nutrition_summary = {
                        "calories": nut.calories_kcal,
                        "protein": nut.protein_g,
                        "carbohydrates": nut.carbohydrates_g,
                        "fat": nut.fat_g,
                        "fiber": nut.fiber_g,
                        "status": "AVAILABLE",
                        "serving_size_g": weight_grams,
                        "source": nut.source,
                        "fdc_id": nut.usda_fdc_id,
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch nutrition for {fruit_name}: {e}")

        return {
            "fruit_name": fruit_name,
            "dataset_class": class_name,
            "freshness_status": ds_label,
            "freshness_note": (
                f"Label '{ds_label}' is derived directly from the dataset category '{class_name}'. "
                "The updated dataset contains exclusively 'Good' condition produce samples and was not trained to detect spoiled/rotten fruit."
            ),
            "confidence": conf,
            "confidence_status": cls_res.confidence_status,
            "classification_model": "EfficientNet-B4",
            "nutrition": nutrition_summary,
            "top_k_candidates": cls_res.result.top_k_candidates,
            "inference_time_ms": cls_res.inference_time_ms,
        }

    def predict_crop(self, crop: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Convenience method for crop prediction inside the pipeline."""
        res = self.predict_structured(crop, enable_nutrition=False)
        # Add boolean helper fields
        conf = res.get("confidence", 0.0)
        res["is_low_confidence"] = conf < self.confidence_threshold
        res["probabilities"] = {
            cand.get("class_name", ""): cand.get("confidence", 0.0)
            for cand in res.get("top_k_candidates", [])
        }
        return res

    def predict_image_path(
        self,
        image_path: Union[str, Path],
        weight_grams: Optional[float] = 150.0,
        enable_nutrition: bool = True,
    ) -> Dict[str, Any]:
        """Convenience method for path prediction."""
        res = self.predict_structured(image_path, weight_grams=weight_grams, enable_nutrition=enable_nutrition)
        conf = res.get("confidence", 0.0)
        res["is_low_confidence"] = conf < self.confidence_threshold
        return res

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "fruit_classifier_efficientnet_b4",
            "weights_path": str(self.model_path) if self.model_path else None,
            "status": self._status.value,
            "device": self.device,
            "is_ready": self.is_ready(),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "image_size": self.image_size,
            "confidence_threshold": self.confidence_threshold,
            "dataset": "updated fruit dataset",
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "error_message": self._error_msg,
        }
