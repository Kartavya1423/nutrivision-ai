"""
NutriVision AI — Visual Explainability (Grad-CAM) Service (Phase 7)
Provides high-level interpretability interfaces for:
1. Fine-grained food classification (EfficientNet-B4)
2. Freshness and spoilage detection (Phase 6 EfficientNetFreshnessModel)

Saves normalized heatmaps, alpha overlays, and 3-panel composite figures with complete metadata.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.classification.efficientnet_classifier import EfficientNetB4Classifier
from src.config.settings import Settings, get_settings
from src.detection.detector import ModelStatus
from src.explainability.gradcam import GradCAM
from src.freshness.freshness_detector import EfficientNetFreshnessDetector


class NutriVisionExplainer:
    """
    Unified Grad-CAM explainability coordinator for NutriVision AI.
    Integrates both the Food Classifier and Freshness Classifier.
    """

    def __init__(
        self,
        food_classifier: Optional[EfficientNetB4Classifier] = None,
        freshness_detector: Optional[EfficientNetFreshnessDetector] = None,
        settings: Optional[Settings] = None,
        colormap: str = "JET",
        alpha: float = 0.5,
    ):
        self.settings = settings or get_settings()
        self.colormap = colormap
        self.alpha = alpha

        # Load or initialize models
        self.food_classifier = food_classifier or EfficientNetB4Classifier(settings=self.settings)
        self.freshness_detector = freshness_detector or EfficientNetFreshnessDetector(settings=self.settings)

        self._food_gradcam: Optional[GradCAM] = None
        self._freshness_gradcam: Optional[GradCAM] = None

        self._init_gradcam_engines()

    def _init_gradcam_engines(self):
        """Initialize underlying GradCAM engines for available models."""
        if self.food_classifier.is_ready() and self.food_classifier.model is not None:
            target_layer = self.food_classifier.get_target_layer()
            if target_layer is not None:
                self._food_gradcam = GradCAM(
                    model=self.food_classifier.model,
                    target_layer=target_layer,
                    device=self.food_classifier.device,
                    image_size=self.food_classifier.image_size,
                    norm_mean=self.food_classifier.norm_mean,
                    norm_std=self.food_classifier.norm_std,
                )

        if self.freshness_detector.is_ready() and self.freshness_detector.model is not None:
            fr_target_layer = self.freshness_detector.model.get_gradcam_target_layer()
            if fr_target_layer is not None:
                self._freshness_gradcam = GradCAM(
                    model=self.freshness_detector.model,
                    target_layer=fr_target_layer,
                    device=self.freshness_detector.device,
                    image_size=getattr(self.freshness_detector, "image_size", 224),
                    norm_mean=getattr(self.freshness_detector, "norm_mean", [0.485, 0.456, 0.406]),
                    norm_std=getattr(self.freshness_detector, "norm_std", [0.229, 0.224, 0.225]),
                )

    def explain_food_classification(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        target_class: Optional[Union[int, str]] = None,
        colormap: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM explanation for food classification model.
        """
        if self._food_gradcam is None:
            return {
                "status": "GRADCAM_UNAVAILABLE",
                "error_code": "MODEL_NOT_READY",
                "message": "Food classification model is not loaded or target layer is unavailable.",
                "has_heatmap": False,
            }

        target_id = None
        if target_class is not None:
            if isinstance(target_class, int):
                target_id = target_class
            elif isinstance(target_class, str):
                target_id = self.food_classifier.class_to_idx.get(target_class.lower())
                if target_id is None:
                    return {
                        "status": "GRADCAM_UNAVAILABLE",
                        "error_code": "INVALID_TARGET_CLASS",
                        "message": f"Class name '{target_class}' not found in canonical food class mapping.",
                        "has_heatmap": False,
                    }

        res = self._food_gradcam.generate(
            image=image,
            target_class_id=target_id,
            class_names=self.food_classifier.class_names,
            colormap=colormap or self.colormap,
            alpha=alpha if alpha is not None else self.alpha,
        )
        res["model_name"] = "EfficientNet-B4 Food Classifier"
        res["checkpoint_path"] = str(self.food_classifier.model_path)
        return res

    def explain_freshness(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        target_class: Optional[Union[int, str]] = None,
        colormap: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM explanation for freshness detection model.
        """
        if self._freshness_gradcam is None:
            return {
                "status": "GRADCAM_UNAVAILABLE",
                "error_code": "MODEL_NOT_READY",
                "message": "Freshness model is not loaded or target layer is unavailable.",
                "has_heatmap": False,
            }

        target_id = None
        if target_class is not None:
            if isinstance(target_class, int):
                target_id = target_class
            elif isinstance(target_class, str):
                target_str = target_class.lower().strip()
                if target_str in ["fresh", "0"]:
                    target_id = 0
                elif target_str in ["rotten", "spoiled", "1"]:
                    target_id = 1
                else:
                    return {
                        "status": "GRADCAM_UNAVAILABLE",
                        "error_code": "INVALID_TARGET_CLASS",
                        "message": f"Freshness class '{target_class}' must be 'fresh' or 'rotten'.",
                        "has_heatmap": False,
                    }

        class_names = ["fresh", "rotten"]
        res = self._freshness_gradcam.generate(
            image=image,
            target_class_id=target_id,
            class_names=class_names,
            colormap=colormap or self.colormap,
            alpha=alpha if alpha is not None else self.alpha,
        )
        res["model_name"] = "EfficientNet-B4 Freshness Classifier"
        res["checkpoint_path"] = str(self.freshness_detector.model_path)
        return res

    def save_artifacts(
        self,
        explanation_result: Dict[str, Any],
        output_dir: Union[str, Path],
        file_prefix: str,
    ) -> Dict[str, str]:
        """
        Save heatmap image, overlay image, composite image, and metadata JSON.
        Returns dictionary of saved file paths.
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        saved = {}
        if not explanation_result.get("has_heatmap"):
            return saved

        # 1. Overlay
        if "overlay" in explanation_result:
            ov_path = out_p / f"{file_prefix}_overlay.png"
            cv2.imwrite(str(ov_path), cv2.cvtColor(explanation_result["overlay"], cv2.COLOR_RGB2BGR))
            saved["overlay_path"] = str(ov_path)

        # 2. Colored Heatmap
        if "heatmap_colored" in explanation_result:
            hm_path = out_p / f"{file_prefix}_heatmap.png"
            cv2.imwrite(str(hm_path), cv2.cvtColor(explanation_result["heatmap_colored"], cv2.COLOR_RGB2BGR))
            saved["heatmap_path"] = str(hm_path)

        # 3. 3-Panel Composite
        if "composite" in explanation_result:
            comp_path = out_p / f"{file_prefix}_gradcam_3panel.png"
            cv2.imwrite(str(comp_path), cv2.cvtColor(explanation_result["composite"], cv2.COLOR_RGB2BGR))
            saved["composite_path"] = str(comp_path)

        # 4. Metadata JSON
        meta = {
            "model_name": explanation_result.get("model_name"),
            "checkpoint_path": explanation_result.get("checkpoint_path"),
            "status": explanation_result.get("status"),
            "target_class_id": explanation_result.get("target_class_id"),
            "target_class_name": explanation_result.get("target_class_name"),
            "predicted_class_id": explanation_result.get("predicted_class_id"),
            "predicted_class_name": explanation_result.get("predicted_class_name"),
            "confidence": explanation_result.get("confidence"),
            "target_confidence": explanation_result.get("target_confidence"),
            "metadata": explanation_result.get("metadata", {}),
            "artifacts": saved,
        }
        meta_path = out_p / f"{file_prefix}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        saved["metadata_path"] = str(meta_path)

        return saved

    def explain(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        food_output_dir: Optional[Union[str, Path]] = None,
        freshness_output_dir: Optional[Union[str, Path]] = None,
        prefix: str = "item",
    ) -> Dict[str, Any]:
        """Coordinator method to generate and save both food and freshness Grad-CAM heatmaps."""
        class _CamResult:
            def __init__(self, artifacts: Dict[str, str], target_layer: str):
                self.composite_path = artifacts.get("composite_path")
                self.target_layer = target_layer

        results: Dict[str, Any] = {}

        # 1. Food classification Grad-CAM
        food_res = self.explain_food_classification(image=image)
        if food_res.get("has_heatmap") and food_output_dir:
            f_artifacts = self.save_artifacts(food_res, food_output_dir, f"{prefix}_food")
            layer = food_res.get("metadata", {}).get("target_layer", "features[-1]")
            results["food_gradcam"] = _CamResult(f_artifacts, str(layer))

        # 2. Freshness classification Grad-CAM
        fresh_res = self.explain_freshness(image=image)
        if fresh_res.get("has_heatmap") and freshness_output_dir:
            fr_artifacts = self.save_artifacts(fresh_res, freshness_output_dir, f"{prefix}_freshness")
            layer = fresh_res.get("metadata", {}).get("target_layer", "features[-1]")
            results["freshness_gradcam"] = _CamResult(fr_artifacts, str(layer))

        return results


# Backward-compatible wrapper for existing pipeline callers
class GradCAMExplainer(NutriVisionExplainer):
    """Alias for NutriVisionExplainer preserving legacy constructor and methods."""

    def generate_heatmap(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        target_class_id: Optional[int] = None,
        model: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return self.explain_food_classification(image=image, target_class=target_class_id)
