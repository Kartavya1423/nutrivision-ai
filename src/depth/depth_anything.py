"""
NutriVision AI — Depth Anything V2 Monocular Depth Model
Produces relative depth maps from single RGB images using official Depth Anything V2 architecture.
Explicitly distinguishes relative depth from metric depth in cm.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Union
import cv2
import numpy as np
from PIL import Image
import torch

from src.config.settings import Settings, get_settings
from src.depth.depth_estimator import BaseDepthEstimator, DepthResult, DepthType
from src.detection.detector import ModelStatus
from src.utils.image_ops import load_image

# Ensure the local Depth Anything V2 package is importable
_CURRENT_DIR = Path(__file__).resolve().parent
_PKG_DIR = _CURRENT_DIR / "depth_anything_v2"
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    _DEPTH_ANYTHING_AVAILABLE = True
except ImportError:
    _DEPTH_ANYTHING_AVAILABLE = False


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


class DepthAnythingV2Estimator(BaseDepthEstimator):
    """
    Depth Anything V2 monocular depth prediction wrapper.
    Relative depth output requires external camera calibration or reference object
    (e.g., plate diameter) to compute metric depth in cm.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        encoder: Optional[str] = None,
        device: Optional[str] = None,
        input_size: Optional[int] = None,
        settings: Optional[Settings] = None,
    ):
        cfg = settings or get_settings()
        path = model_path if model_path is not None else cfg.model_paths.depth_model_path
        super().__init__(model_path=path)
        self.encoder = encoder or cfg.depth.encoder
        self.device = device or cfg.resolve_device()
        self.input_size = input_size or cfg.depth.input_size
        self.model = None
        self.load_model()

    def load_model(self) -> bool:
        """Load Depth Anything V2 architecture and checkpoint weights."""
        if self.model_path is None or not Path(self.model_path).is_file():
            self._status = ModelStatus.MODEL_NOT_LOADED
            self._error_msg = (
                f"Depth Anything V2 checkpoint not found at: '{self.model_path}'. "
                "Ensure weights are configured under models/depth/. "
                "Do not fabricate depth maps."
            )
            self.model = None
            return False

        if not _DEPTH_ANYTHING_AVAILABLE:
            self._status = ModelStatus.ERROR
            self._error_msg = "Depth Anything V2 architecture could not be imported from src/depth/depth_anything_v2."
            self.model = None
            return False

        if self.encoder not in MODEL_CONFIGS:
            self._status = ModelStatus.ERROR
            self._error_msg = f"Unknown Depth Anything V2 encoder: '{self.encoder}'. Choose from {list(MODEL_CONFIGS.keys())}."
            self.model = None
            return False

        try:
            config = MODEL_CONFIGS[self.encoder]
            model = DepthAnythingV2(**config)
            state_dict = torch.load(str(self.model_path), map_location=self.device)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            self.model = model
            self._status = ModelStatus.READY
            self._error_msg = None
            return True
        except Exception as e:
            self._status = ModelStatus.ERROR
            self._error_msg = f"Failed to load Depth Anything V2 checkpoint: {str(e)}"
            self.model = None
            return False

    def estimate_depth(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        calibration: Optional[Dict[str, Any]] = None,
    ) -> DepthResult:
        """
        Run real monocular depth inference on an input RGB image.
        Returns a DepthResult containing:
        - raw 2D depth map matching input dimensions (H, W)
        - relative or calibrated metric status
        - robust statistics (min, max, mean, median, percentiles)
        - execution time and device info
        """
        t0 = time.perf_counter()

        if not self.is_ready():
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DepthResult(
                status=self._status,
                depth_map=None,
                depth_type=DepthType.RELATIVE,
                is_metric_calibrated=False,
                calibration_status="DEPTH_MODEL_NOT_AVAILABLE",
                inference_time_ms=elapsed_ms,
                model_name="DepthAnythingV2Estimator",
                device=self.device,
                error_message=self._error_msg or "Depth Anything V2 model is not loaded.",
            )

        try:
            np_rgb, _ = load_image(image)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DepthResult(
                status=ModelStatus.ERROR,
                depth_map=None,
                depth_type=DepthType.RELATIVE,
                is_metric_calibrated=False,
                calibration_status="INVALID_IMAGE",
                inference_time_ms=elapsed_ms,
                model_name="DepthAnythingV2Estimator",
                device=self.device,
                error_message=f"Failed to load input image: {str(e)}",
            )

        orig_h, orig_w = np_rgb.shape[:2]

        try:
            # Convert RGB to BGR for OpenCV-based transforms in Depth Anything V2
            bgr_image = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                raw_depth = self.model.infer_image(bgr_image, input_size=self.input_size)

            # Ensure alignment to input dimensions
            if raw_depth.shape != (orig_h, orig_w):
                raw_depth = cv2.resize(raw_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Calculate robust statistics
            d_min = float(np.min(raw_depth))
            d_max = float(np.max(raw_depth))
            d_mean = float(np.mean(raw_depth))
            d_median = float(np.median(raw_depth))
            percentiles = {
                "p10": float(np.percentile(raw_depth, 10)),
                "p25": float(np.percentile(raw_depth, 25)),
                "p50": d_median,
                "p75": float(np.percentile(raw_depth, 75)),
                "p90": float(np.percentile(raw_depth, 90)),
                "p95": float(np.percentile(raw_depth, 95)),
            }

            # Check if explicit calibration parameters are provided
            is_calibrated = False
            calib_status = "DEPTH_SCALE_NOT_CALIBRATED"
            ref_obj = None

            if calibration and isinstance(calibration, dict):
                if "metric_scale" in calibration or "cm_per_pixel" in calibration:
                    is_calibrated = True
                    calib_status = "CALIBRATED"
                    ref_obj = calibration.get("reference_object")

            return DepthResult(
                status=ModelStatus.READY,
                depth_map=raw_depth.astype(np.float32),
                depth_type=DepthType.METRIC if is_calibrated else DepthType.RELATIVE,
                is_metric_calibrated=is_calibrated,
                reference_object_used=ref_obj,
                calibration_status=calib_status,
                min_depth=d_min,
                max_depth=d_max,
                mean_depth=d_mean,
                median_depth=d_median,
                percentiles=percentiles,
                inference_time_ms=elapsed_ms,
                model_name=f"DepthAnythingV2Estimator({self.encoder})",
                device=self.device,
                error_message=None,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DepthResult(
                status=ModelStatus.ERROR,
                depth_map=None,
                depth_type=DepthType.RELATIVE,
                is_metric_calibrated=False,
                calibration_status="DEPTH_INFERENCE_FAILED",
                inference_time_ms=elapsed_ms,
                model_name="DepthAnythingV2Estimator",
                device=self.device,
                error_message=f"Depth inference failed: {str(e)}",
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Return diagnostic and configuration metadata for Depth Anything V2."""
        return {
            "model_type": "depth_anything_v2",
            "encoder": self.encoder,
            "checkpoint": str(self.model_path) if self.model_path else None,
            "status": self._status.value,
            "device": self.device,
            "input_size": self.input_size,
            "depth_type": DepthType.RELATIVE.value,
            "is_metric": False,
            "calibration_note": (
                "Monocular depth maps are relative. Metric depth scale in cm requires camera "
                "intrinsics calibration or reference object (e.g. plate diameter)."
            ),
            "is_ready": self.is_ready(),
            "error_message": self._error_msg,
        }
