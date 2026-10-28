"""
NutriVision AI — OpenCV Contour-Based Weight Estimation Module
Based on: "Vision-Based Approach for Food Calories And Weight Estimation from Images"
Author: Sonu Kumar (Centurion University of Technology and Management)

Implements:
1. Grayscale conversion & Gaussian Blurring
2. Canny Edge Detection & Contour Extraction
3. Food Area Ratio (Ar = (w * h) / (Wi * Hi))
4. Empirical Weight Formula: We = Wb * (0.6 + 1.2 * Ar)
5. Base weights table (Wb) for food categories
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("nutrivision.contour_weight")

# Table of empirical reference base portion weights (Wb in grams)
# Source: Research Paper Table & USDA Standard Portion Reference
DEFAULT_FOOD_BASE_WEIGHTS: Dict[str, float] = {
    # Fruits
    "apple": 182.0,
    "banana": 118.0,
    "orange": 131.0,
    "mango": 200.0,
    "raw_mango": 200.0,
    "green_mango": 200.0,
    "tomato": 105.0,
    "raw_tomato": 105.0,
    "green_tomato": 105.0,
    "strawberry": 120.0,
    "grape": 100.0,
    "watermelon": 286.0,
    "pineapple": 165.0,
    "lemon": 58.0,
    "peach": 150.0,
    "pear": 178.0,
    "cherry": 80.0,
    "blueberry": 75.0,
    "kiwi": 69.0,
    "plum": 66.0,

    # Vegetables
    "potato": 173.0,
    "carrot": 72.0,
    "broccoli": 91.0,
    "cucumber": 133.0,
    "onion": 110.0,
    "garlic": 15.0,
    "corn": 102.0,
    "mushroom": 70.0,
    "cabbage": 110.0,
    "cauliflower": 107.0,
    "spinach": 85.0,
    "lettuce": 55.0,
    "chili": 25.0,
    "eggplant": 160.0,
    "zucchini": 180.0,
    "asparagus": 90.0,
    "bell_pepper": 119.0,
    "pepper": 119.0,

    # Staples & Prepared Meals
    "rice": 186.0,
    "cooked_rice": 186.0,
    "fried_rice": 210.0,
    "pizza": 135.0,
    "hamburger": 215.0,
    "burger": 215.0,
    "bread": 60.0,
    "french_fries": 117.0,
    "curry": 220.0,
    "beef_curry": 230.0,
    "soup": 240.0,
    "salad": 150.0,
    "sushi": 165.0,
    "dumpling": 120.0,
    "pancake": 110.0,
    "noodle": 180.0,
    "pasta_noodle": 190.0,
    "pasta": 190.0,
    "sandwich": 155.0,
    "samosa": 85.0,

    # Proteins & Dairy
    "egg": 50.0,
    "chicken": 172.0,
    "chicken_duck": 180.0,
    "chicken_breast": 172.0,
    "beef": 185.0,
    "steak": 210.0,
    "fish": 170.0,
    "salmon": 175.0,
    "cheese": 40.0,
    "cheese_butter": 45.0,

    # Desserts & Snacks
    "cake": 125.0,
    "chocolate": 45.0,
    "ice_cream": 105.0,
    "biscuit": 50.0,
    "candy": 30.0,
}

DEFAULT_FALLBACK_BASE_WEIGHT = 150.0


@dataclass
class ContourWeightResult:
    """Output dataclass from contour weight estimation."""
    food_name: str
    estimated_weight_g: float
    base_weight_g: float
    area_ratio: float
    contour_bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    status: str
    method: str = "OpenCV_Contour_Area_Ratio"
    error_message: Optional[str] = None


class ContourWeightEstimator:
    """
    Implements 2D OpenCV contour area detection and weight estimation.
    Formula:
        Ar = (w * h) / (Wi * Hi)
        We = Wb * (0.6 + 1.2 * Ar)
    """

    def __init__(self, base_weights: Optional[Dict[str, float]] = None):
        self.base_weights = base_weights or DEFAULT_FOOD_BASE_WEIGHTS

    def get_base_weight(self, food_name: str) -> float:
        """Retrieves base weight (Wb) for the food class with safe fallback."""
        norm_name = str(food_name).lower().strip().replace(" ", "_").replace("-", "_")
        if norm_name in self.base_weights:
            return self.base_weights[norm_name]
        # Partial match
        for k, v in self.base_weights.items():
            if k in norm_name or norm_name in k:
                return v
        return DEFAULT_FALLBACK_BASE_WEIGHT

    def estimate_weight(
        self,
        image: Union[np.ndarray, Image.Image],
        food_name: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> ContourWeightResult:
        """
        Calculates estimated weight from image contour using the research paper formula.
        
        Args:
            image: Input RGB numpy array or PIL Image
            food_name: Recognized food category
            bbox: Optional bounding box [x1, y1, x2, y2]
            mask: Optional binary segmentation mask
        """
        # Convert to numpy array
        if isinstance(image, Image.Image):
            np_img = np.array(image.convert("RGB"))
        else:
            np_img = image

        h_img, w_img = np_img.shape[:2]
        total_image_area = float(w_img * h_img)

        wb = self.get_base_weight(food_name)

        # Case 1: Mask is available from SAM 2
        if mask is not None and isinstance(mask, np.ndarray) and np.sum(mask) > 0:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                mask_u8 = (mask.astype(np.uint8) * 255)
                we, ar, item_count = self._compute_portion_weight(food_name, wb, w, h, total_image_area, mask=mask_u8)
                return ContourWeightResult(
                    food_name=food_name,
                    estimated_weight_g=we,
                    base_weight_g=wb,
                    area_ratio=ar,
                    contour_bbox=(x1, y1, w, h),
                    status="SUCCESS",
                )

        # Case 2: Bounding box is provided by detector
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            we, ar, item_count = self._compute_portion_weight(food_name, wb, w, h, total_image_area)
            return ContourWeightResult(
                food_name=food_name,
                estimated_weight_g=we,
                base_weight_g=wb,
                area_ratio=ar,
                contour_bbox=(x1, y1, w, h),
                status="SUCCESS",
            )

        # Case 3: Autonomous robust whole-food contour extraction from 2D image (Section III-D)
        try:
            solid_mask, (x, y, w, h), cov = self._extract_robust_food_mask(np_img)
            if solid_mask is not None and w * h > (0.01 * total_image_area):
                ar = float(w * h) / total_image_area
                we, ar, item_count = self._compute_portion_weight(food_name, wb, w, h, total_image_area, mask=solid_mask)
                return ContourWeightResult(
                    food_name=food_name,
                    estimated_weight_g=we,
                    base_weight_g=wb,
                    area_ratio=ar,
                    contour_bbox=(x, y, w, h),
                    status="SUCCESS",
                )

            # Fallback Canny for fine-line edges if solid mask had no components
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                all_pts = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(all_pts)
                if w * h > (0.01 * total_image_area):
                    ar = float(w * h) / total_image_area
                    we, ar, item_count = self._compute_portion_weight(food_name, wb, w, h, total_image_area)
                    return ContourWeightResult(
                        food_name=food_name,
                        estimated_weight_g=we,
                        base_weight_g=wb,
                        area_ratio=ar,
                        contour_bbox=(x, y, w, h),
                        status="SUCCESS",
                    )

            # No significant contour found (e.g., solid color, blank image, featureless background)
            return ContourWeightResult(
                food_name=food_name,
                estimated_weight_g=0.0,
                base_weight_g=wb,
                area_ratio=0.0,
                contour_bbox=(0, 0, 0, 0),
                status="NO_CONTOUR_DETECTED",
                error_message="No food contour detected in image",
            )

        except Exception as e:
            return ContourWeightResult(
                food_name=food_name,
                estimated_weight_g=0.0,
                base_weight_g=wb,
                area_ratio=0.0,
                contour_bbox=(0, 0, 0, 0),
                status="CONTOUR_FAILED",
                error_message=f"Contour warning: {str(e)}",
            )

    def _extract_robust_food_mask(
        self,
        np_img: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int], float]:
        """
        Extracts robust, solid whole-food mask avoiding partial/severed contours.
        Uses Otsu thresholding, color saturation saliency, and morphological closing.
        Returns: (binary_mask, (x, y, w, h), coverage_ratio)
        """
        h_img, w_img = np_img.shape[:2]
        total_area = float(w_img * h_img)

        # 1. Grayscale & blur
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        if float(np.std(gray)) < 6.0:
            return None, (0, 0, 0, 0), 0.0

        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # 2. Otsu thresholding with background polarity detection
        _, otsu_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, otsu_norm = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cw, ch = max(2, int(0.08 * w_img)), max(2, int(0.08 * h_img))
        corners = np.concatenate([
            gray[:ch, :cw].flatten(),
            gray[:ch, -cw:].flatten(),
            gray[-ch:, :cw].flatten(),
            gray[-ch:, -cw:].flatten(),
        ])
        bg_is_light = np.median(corners) > 120
        base_otsu = otsu_inv if bg_is_light else otsu_norm

        # 3. Saturation channel saliency
        hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        sat_mask = (sat > 25).astype(np.uint8) * 255

        # 4. Fusion of contrast and chromaticity
        fg_candidate = cv2.bitwise_or(base_otsu, sat_mask)

        # 5. Clean small noise, then bridge gaps to create solid object
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_cleaned = cv2.morphologyEx(fg_candidate, cv2.MORPH_OPEN, kernel_open)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(fg_cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # 6. Extract solid food contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) >= 0.02 * total_area]

        if not significant_contours:
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) >= 0.01 * total_area:
                    significant_contours = [largest]

        if not significant_contours:
            return None, (0, 0, 0, 0), 0.0

        # Combine all significant contours so multi-part dishes or single items are not cut in half
        all_points = np.vstack(significant_contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Add a 3% padding margin so the edge of the food is never clipped
        margin_x = int(0.03 * w)
        margin_y = int(0.03 * h)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)
        w_final = max(1, x2 - x1)
        h_final = max(1, y2 - y1)

        solid_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(solid_mask, significant_contours, -1, 255, -1)

        coverage = float(w_final * h_final) / total_area
        return solid_mask, (x1, y1, w_final, h_final), coverage

    def _compute_portion_weight(
        self,
        food_name: str,
        wb: float,
        w: int,
        h: int,
        total_image_area: float,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, int]:
        """
        Computes portion weight using the research paper formula:
            W_e = W_b * (0.6 + 1.2 * A_r)
        with bounded effective area ratio (A_r) to ensure realistic physical portion weights
        aligned with USDA standard reference portions.
        Returns: (estimated_weight_g, area_ratio, item_count)
        """
        w = max(1, w)
        h = max(1, h)
        ar = float(w * h) / max(1.0, total_image_area)

        # For close-ups or tight crops, clamp effective area ratio to standard perspective bounds [0.08, 0.55]
        # This keeps portion weights within physical realistic bounds [0.70 * W_b, 1.26 * W_b]
        ar_effective = min(0.55, max(0.08, ar))
        scale = 0.6 + 1.2 * ar_effective
        we = wb * scale

        return round(float(we), 2), round(float(ar), 4), 1
