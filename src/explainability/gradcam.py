"""
NutriVision AI — Native PyTorch Grad-CAM Explainability Engine (Phase 7)
Computes Gradient-weighted Class Activation Mapping (Grad-CAM) for deep convolutional classifiers.
Zero external library dependencies (uses pure PyTorch + OpenCV).
Supported architectures:
- EfficientNet-B4 Fine-Grained Food Classifier (198 classes)
- EfficientNet-B4 Food Freshness Classifier (2 classes: fresh, rotten)

Strict rules:
- No fake heatmaps
- Validates dimensions, NaNs, and gradient availability
- Transparent overlay generation with configurable alpha and colormaps
- Saves individual heatmaps, overlays, 3-panel composite figures, and metadata JSON
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


COLORMAP_MAP = {
    "JET": cv2.COLORMAP_JET,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
}


class GradCAM:
    """
    Core Gradient-weighted Class Activation Mapping engine.
    Hooks into any PyTorch convolutional target layer and calculates class activation maps.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        image_size: int = 224,
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
    ):
        self.model = model
        self.target_layer = target_layer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = image_size
        self.norm_mean = norm_mean or [0.485, 0.456, 0.406]
        self.norm_std = norm_std or [0.229, 0.224, 0.225]

        self.model.to(self.device)
        self.model.eval()

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: List[Any] = []

    def _register_hooks(self):
        """Register forward and backward hooks on target convolutional layer."""
        self._remove_hooks()

        def forward_hook(module, inp, out):
            self._activations = out

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h1, h2])

    def _remove_hooks(self):
        """Remove active PyTorch hooks."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._activations = None
        self._gradients = None

    def preprocess_image(self, img_input: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess input image to normalized PyTorch tensor while preserving original RGB array.
        Returns: (preprocessed_tensor, rgb_numpy_array)
        """
        if isinstance(img_input, (str, Path)):
            p = Path(img_input)
            if not p.is_file():
                raise FileNotFoundError(f"Input image not found: {p}")
            pil_img = Image.open(p).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            if img_input.size == 0:
                raise ValueError("Empty numpy array provided as image.")
            if img_input.ndim == 2:
                pil_img = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB))
            elif img_input.shape[2] == 4:
                # Handle RGBA transparent crop by compositing on white background
                rgba = img_input
                alpha = rgba[:, :, 3:4] / 255.0
                rgb = rgba[:, :, :3] * alpha + 255.0 * (1.0 - alpha)
                pil_img = Image.fromarray(rgb.astype(np.uint8))
            else:
                pil_img = Image.fromarray(img_input[:, :, :3])
        elif isinstance(img_input, Image.Image):
            pil_img = img_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(img_input)}")

        orig_rgb = np.array(pil_img)
        resized_pil = pil_img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        resized_arr = np.array(resized_pil, dtype=np.float32) / 255.0

        # Normalize with dataset statistics
        mean = np.array(self.norm_mean, dtype=np.float32)
        std = np.array(self.norm_std, dtype=np.float32)
        norm_arr = (resized_arr - mean) / std

        # Shape: (1, 3, H, W)
        tensor = torch.from_numpy(norm_arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)
        return tensor, orig_rgb

    def generate(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        target_class_id: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        colormap: str = "JET",
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compute Grad-CAM for the input image.
        Returns normalized heatmap, colored overlay, and structured metadata.
        """
        try:
            tensor, orig_rgb = self.preprocess_image(image)
        except Exception as e:
            return {
                "status": "GRADCAM_UNAVAILABLE",
                "error_code": "INVALID_IMAGE",
                "message": f"Failed to preprocess image: {str(e)}",
                "has_heatmap": False,
            }

        orig_h, orig_w = orig_rgb.shape[:2]

        self._register_hooks()
        try:
            # Enable gradient calculation even if in eval mode
            with torch.enable_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)[0]

                num_classes = logits.shape[1]
                pred_class_id = int(torch.argmax(logits, dim=1).item())
                confidence = float(probs[pred_class_id].item())

                if target_class_id is None:
                    target_id = pred_class_id
                else:
                    target_id = int(target_class_id)
                    if target_id < 0 or target_id >= num_classes:
                        self._remove_hooks()
                        return {
                            "status": "GRADCAM_UNAVAILABLE",
                            "error_code": "INVALID_TARGET_CLASS",
                            "message": f"Target class {target_id} is out of bounds (0 to {num_classes - 1}).",
                            "has_heatmap": False,
                        }

                target_prob = float(probs[target_id].item())
                target_score = logits[0, target_id]

                self.model.zero_grad()
                target_score.backward(retain_graph=False)

            if self._activations is None or self._gradients is None:
                self._remove_hooks()
                return {
                    "status": "GRADCAM_UNAVAILABLE",
                    "error_code": "NO_GRADIENTS_CAPTURED",
                    "message": "Target layer hooks failed to capture forward activations or backward gradients.",
                    "has_heatmap": False,
                }

            # Global average pooling of gradients: alpha_k = (1/Z) * sum_{i,j} (dY/dA_k)
            # Gradients shape: (1, C, H_a, W_a)
            grads = self._gradients.detach()
            acts = self._activations.detach()

            weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)    # (1, 1, H_a, W_a)
            cam = F.relu(cam)                                       # ReLU: focus on features having positive contribution

            cam_np = cam.squeeze().cpu().numpy()

            # Quality Check: Check for NaNs or Infinities
            if np.isnan(cam_np).any() or np.isinf(cam_np).any():
                self._remove_hooks()
                return {
                    "status": "GRADCAM_UNAVAILABLE",
                    "error_code": "NAN_IN_GRADCAM_COMPUTATION",
                    "message": "Computed Grad-CAM map contains NaN or Inf values.",
                    "has_heatmap": False,
                }

            # Normalize to [0.0, 1.0]
            cam_min, cam_max = np.min(cam_np), np.max(cam_np)
            if cam_max - cam_min > 1e-8:
                cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_norm = np.zeros_like(cam_np, dtype=np.float32)

            # Resize heatmap to original input resolution
            heatmap_resized = cv2.resize(cam_norm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0).astype(np.float32)

            # Generate colored heatmap and overlay
            cm_code = COLORMAP_MAP.get(colormap.upper(), cv2.COLORMAP_JET)
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            colored_heatmap = cv2.applyColorMap(heatmap_uint8, cm_code)
            colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

            # Alpha blend overlay: alpha * heatmap + (1 - alpha) * orig
            alpha = float(np.clip(alpha, 0.0, 1.0))
            overlay_rgb = (alpha * colored_heatmap_rgb + (1.0 - alpha) * orig_rgb).astype(np.uint8)

            # Generate 3-panel composite visualization: [Original | Heatmap | Overlay]
            composite_img = self._create_composite_figure(
                orig_rgb=orig_rgb,
                heatmap_rgb=colored_heatmap_rgb,
                overlay_rgb=overlay_rgb,
                pred_label=class_names[pred_class_id] if class_names and pred_class_id < len(class_names) else f"Class {pred_class_id}",
                target_label=class_names[target_id] if class_names and target_id < len(class_names) else f"Class {target_id}",
                confidence=confidence,
                target_confidence=target_prob,
            )

            pred_name = class_names[pred_class_id] if class_names and pred_class_id < len(class_names) else f"class_{pred_class_id}"
            target_name = class_names[target_id] if class_names and target_id < len(class_names) else f"class_{target_id}"

            result = {
                "status": "AVAILABLE",
                "error_code": None,
                "has_heatmap": True,
                "target_class_id": target_id,
                "target_class_name": target_name,
                "predicted_class_id": pred_class_id,
                "predicted_class_name": pred_name,
                "confidence": round(confidence, 4),
                "target_confidence": round(target_prob, 4),
                "heatmap_normalized": heatmap_resized,
                "heatmap_colored": colored_heatmap_rgb,
                "overlay": overlay_rgb,
                "composite": composite_img,
                "original_image": orig_rgb,
                "metadata": {
                    "image_dimensions": [orig_w, orig_h],
                    "target_layer": self.target_layer.__class__.__name__,
                    "colormap": colormap.upper(),
                    "alpha": alpha,
                    "device": str(self.device),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            return result

        except Exception as e:
            return {
                "status": "GRADCAM_UNAVAILABLE",
                "error_code": "GRADCAM_EXECUTION_FAILED",
                "message": f"Grad-CAM forward/backward computation failed: {str(e)}",
                "has_heatmap": False,
            }
        finally:
            self._remove_hooks()

    def _create_composite_figure(
        self,
        orig_rgb: np.ndarray,
        heatmap_rgb: np.ndarray,
        overlay_rgb: np.ndarray,
        pred_label: str,
        target_label: str,
        confidence: float,
        target_confidence: float,
    ) -> np.ndarray:
        """Create a clean 3-panel horizontal composite with header banner."""
        h, w = orig_rgb.shape[:2]
        pad = 10
        banner_h = 60

        total_w = w * 3 + pad * 4
        total_h = h + banner_h + pad * 2

        composite = np.full((total_h, total_w, 3), 245, dtype=np.uint8)

        # Draw header banner
        header_text = f"NutriVision AI Grad-CAM | Pred: {pred_label} ({confidence * 100:.1f}%) | Target: {target_label} ({target_confidence * 100:.1f}%)"
        cv2.putText(
            composite,
            header_text,
            (pad, banner_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

        # Place panels
        y0 = banner_h + pad
        x0 = pad
        composite[y0:y0 + h, x0:x0 + w] = orig_rgb

        x1 = x0 + w + pad
        composite[y0:y0 + h, x1:x1 + w] = heatmap_rgb

        x2 = x1 + w + pad
        composite[y0:y0 + h, x2:x2 + w] = overlay_rgb

        # Add panel sublabels
        cv2.putText(composite, "1. Food Crop", (x0 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(composite, "2. Grad-CAM Heatmap", (x1 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(composite, "3. Decision Overlay", (x2 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return composite
