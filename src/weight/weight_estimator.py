"""
NutriVision AI — Food Weight Estimation Module
Formula: weight (g) = volume (cm³) × density (g/cm³)
Strictly distinguishes estimated weight from actual measured weight.
Reports honest status when volume is uncalibrated or density is unavailable.
Multi-food total is ONLY computed when all items are valid and calibrated.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from src.volume.volume_estimator import VolumeEstimateOutput, MultiVolumeResult, ItemVolumeResult
from src.weight.density_database import DensityDatabase


@dataclass
class ItemWeightEstimate:
    """Individual item weight result separating estimated from actual measured weight."""
    class_name: str
    estimated_weight_grams: Optional[float]
    actual_measured_weight_grams: Optional[float] = None  # Ground-truth scale measurement if provided
    volume_cm3: Optional[float] = None
    density_g_cm3: Optional[float] = None
    density_source: Optional[str] = None
    calibration_status: str = "VOLUME_SCALE_NOT_CALIBRATED"
    is_estimated: bool = True
    confidence: float = 0.0
    status: str = "READY"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "food": self.class_name,
            "class_name": self.class_name,
            "volume_cm3": round(float(self.volume_cm3), 2) if self.volume_cm3 is not None else None,
            "density_g_cm3": round(float(self.density_g_cm3), 4) if self.density_g_cm3 is not None else None,
            "density_source": self.density_source,
            "estimated_weight_g": round(float(self.estimated_weight_grams), 2) if self.estimated_weight_grams is not None else None,
            "estimated_weight_grams": round(float(self.estimated_weight_grams), 2) if self.estimated_weight_grams is not None else None,
            "actual_measured_weight_grams": round(float(self.actual_measured_weight_grams), 2) if self.actual_measured_weight_grams is not None else None,
            "unit": "grams",
            "calibration": self.calibration_status,
            "calibration_status": self.calibration_status,
            "is_estimated": self.is_estimated,
            "confidence": round(float(self.confidence), 3),
            "status": self.status,
            "error_message": self.error_message,
        }

    @property
    def is_success(self) -> bool:
        return self.status == "READY" and self.estimated_weight_grams is not None

    @property
    def weight_grams(self) -> float:
        """Backward-compatibility accessor for pipeline integration."""
        return self.estimated_weight_grams if self.estimated_weight_grams is not None else 100.0


@dataclass
class WeightEstimationResult:
    """Consolidated weight estimation for one or more meal items."""
    items: List[ItemWeightEstimate]
    total_estimated_weight_grams: Optional[float] = None
    total_actual_measured_weight_grams: Optional[float] = None
    status: str = "READY"
    error_message: Optional[str] = None

    @property
    def weights(self) -> List[ItemWeightEstimate]:
        return self.items

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "total_estimated_weight_g": round(float(self.total_estimated_weight_grams), 2) if self.total_estimated_weight_grams is not None else None,
            "total_estimated_weight_grams": round(float(self.total_estimated_weight_grams), 2) if self.total_estimated_weight_grams is not None else None,
            "total_actual_measured_weight_grams": round(float(self.total_actual_measured_weight_grams), 2) if self.total_actual_measured_weight_grams is not None else None,
            "items": [item.to_dict() for item in self.items],
            "uncalibrated_items_count": sum(1 for item in self.items if item.status != "READY"),
            "error_message": self.error_message,
        }


# Backward compatibility alias
WeightResult = WeightEstimationResult


class WeightEstimator:
    """
    Computes food mass from physical volume and density:
        Weight (g) = Volume (cm³) × Density (g/cm³)
    Never fabricates weights when volume or density is missing.
    """

    def __init__(
        self,
        density_db: Optional[DensityDatabase] = None,
        settings: Optional[Any] = None,
    ):
        self.density_db = density_db or DensityDatabase()

    def estimate_weight(
        self,
        class_name: Union[str, Any],
        volume_output: Optional[Union[Dict[str, Any], VolumeEstimateOutput, MultiVolumeResult]] = None,
        actual_measured_weight_grams: Optional[float] = None,
    ) -> Union[ItemWeightEstimate, WeightEstimationResult]:
        """
        Estimate weight for a single food item or a multi-volume result.
        """
        # Handle multi-item input (MultiVolumeResult or object with .volumes)
        active_volumes = None
        if hasattr(class_name, "volumes"):
            active_volumes = getattr(class_name, "volumes")
        elif volume_output is not None and hasattr(volume_output, "volumes"):
            active_volumes = getattr(volume_output, "volumes")

        if active_volumes is not None:
            if len(active_volumes) == 0:
                return WeightEstimationResult(
                    items=[],
                    total_estimated_weight_grams=None,
                    status="NO_FOOD_DETECTED",
                    error_message="No food items provided for weight estimation.",
                )

            items: List[ItemWeightEstimate] = []
            for vol_item in active_volumes:
                c_name = getattr(vol_item, "class_name", "food")
                vol_val = getattr(vol_item, "volume_cm3", getattr(vol_item, "volume", None))
                vol_st = getattr(vol_item, "status", "UNKNOWN")
                calib_st = getattr(vol_item, "calibration_status", "VOLUME_SCALE_NOT_CALIBRATED")
                conf = getattr(vol_item, "confidence", 0.0)

                single_vol_out = VolumeEstimateOutput(
                    volume=vol_val,
                    unit="cm3",
                    status=vol_st,
                    calibration_status=calib_st,
                    confidence=conf,
                )
                item_est = self.estimate_weight(
                    class_name=c_name,
                    volume_output=single_vol_out,
                )
                items.append(item_est)

            # Enforce rule: total_estimated_weight_g is ONLY calculated when ALL items are valid
            all_valid = len(items) > 0 and all(
                it.status == "READY" and it.estimated_weight_grams is not None for it in items
            )
            if all_valid:
                total_g = sum(it.estimated_weight_grams for it in items)
                status_res = "READY"
                err_msg = None
            else:
                total_g = None
                invalid_reasons = [f"{it.class_name}: {it.status}" for it in items if it.status != "READY"]
                status_res = "WEIGHT_ESTIMATION_UNAVAILABLE"
                err_msg = f"Cannot calculate total weight; uncalibrated or missing density items: {', '.join(invalid_reasons)}"

            return WeightEstimationResult(
                items=items,
                total_estimated_weight_grams=total_g,
                status=status_res,
                error_message=err_msg,
            )

        # Single item calculation
        class_name = str(class_name)

        if volume_output is None:
            return ItemWeightEstimate(
                class_name=class_name,
                estimated_weight_grams=None,
                actual_measured_weight_grams=actual_measured_weight_grams,
                volume_cm3=None,
                density_g_cm3=None,
                calibration_status="VOLUME_SCALE_NOT_CALIBRATED",
                status="VOLUME_SCALE_NOT_CALIBRATED",
                error_message="Volume output not provided.",
            )

        if isinstance(volume_output, dict):
            vol_val = volume_output.get("volume")
            vol_status = volume_output.get("status", "UNKNOWN")
            calib_status = volume_output.get("calibration_status", "VOLUME_SCALE_NOT_CALIBRATED")
            conf = volume_output.get("confidence", 0.0)
        else:
            vol_val = volume_output.volume
            vol_status = volume_output.status
            calib_status = getattr(volume_output, "calibration_status", "VOLUME_SCALE_NOT_CALIBRATED")
            conf = getattr(volume_output, "confidence", 0.0)

        # Check if volume is uncalibrated or unavailable
        if vol_val is None or vol_status != "READY":
            report_status = vol_status
            return ItemWeightEstimate(
                class_name=class_name,
                estimated_weight_grams=None,
                actual_measured_weight_grams=actual_measured_weight_grams,
                volume_cm3=vol_val,
                density_g_cm3=None,
                calibration_status=calib_status,
                is_estimated=True,
                confidence=conf,
                status=report_status,
                error_message=f"Physical volume unavailable: {vol_status}",
            )

        # Lookup density in database
        density_lookup = self.density_db.get_density(class_name)
        if density_lookup["status"] != "AVAILABLE" or density_lookup["density"] is None:
            return ItemWeightEstimate(
                class_name=class_name,
                estimated_weight_grams=None,
                actual_measured_weight_grams=actual_measured_weight_grams,
                volume_cm3=vol_val,
                density_g_cm3=None,
                density_source=None,
                calibration_status=calib_status,
                is_estimated=True,
                confidence=conf,
                status="DENSITY_NOT_AVAILABLE",
                error_message=density_lookup.get("message", f"No empirical density recorded for '{class_name}'."),
            )

        density = float(density_lookup["density"])
        density_src = density_lookup.get("source", "USDA FoodData Central")

        # Weight (g) = Volume (cm³) × Density (g/cm³)
        calculated_grams = vol_val * density

        return ItemWeightEstimate(
            class_name=class_name,
            estimated_weight_grams=calculated_grams,
            actual_measured_weight_grams=actual_measured_weight_grams,
            volume_cm3=vol_val,
            density_g_cm3=density,
            density_source=density_src,
            calibration_status=calib_status,
            is_estimated=True,
            confidence=conf,
            status="READY",
            error_message=None,
        )
