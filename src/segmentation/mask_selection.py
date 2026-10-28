"""
NutriVision AI — SAM 2 Candidate Mask Selection Module
Phase 3: SAM 2 Food Instance Segmentation

Implements a deterministic multi-criteria candidate selection strategy:
1. Overlap (IoU) with the YOLO bounding box prompt
2. Containment: Fraction of mask pixels remaining inside the detected food region (penalizing leaks into background/table)
3. SAM 2 predicted quality / confidence score
4. Area validity: Filtering out tiny noise (< min_mask_area) or runaway masks covering the entire image.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection-over-Union between two axis-aligned bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0
    return float(intersection_area / union_area)


def compute_mask_tight_bbox(mask: np.ndarray) -> List[float]:
    """Derives tight bounding box [x1, y1, x2, y2] enclosing non-zero mask pixels."""
    if not np.any(mask):
        return [0.0, 0.0, 0.0, 0.0]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    if len(y_indices) == 0 or len(x_indices) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(x_indices[0]), float(y_indices[0]), float(x_indices[-1]), float(y_indices[-1])]


def evaluate_candidate_mask(
    mask: np.ndarray,
    prompt_bbox: List[float],
    image_shape: Tuple[int, int],
    quality_score: float = 1.0,
    min_area: int = 50,
) -> Dict[str, Any]:
    """
    Evaluates spatial suitability of a candidate mask against a YOLO bounding box prompt.

    Args:
        mask: 2D boolean numpy array (H, W).
        prompt_bbox: Bounding box [x1, y1, x2, y2] from YOLO.
        image_shape: (H, W) dimensions of the parent image.
        quality_score: SAM 2 predicted quality/IoU score.
        min_area: Minimum acceptable pixel area.

    Returns:
        Dictionary of spatial metrics and the overall composite selection score.
    """
    h, w = image_shape[:2]
    total_pixels = float(h * w)
    mask_bool = mask > 0
    area = int(np.sum(mask_bool))

    if area < min_area:
        return {
            "is_valid": False,
            "failure_reason": "MASK_TOO_SMALL",
            "area_pixels": area,
            "composite_score": 0.0,
        }

    # Runaway mask check (e.g. prompt was a small berry, but mask captured entire plate/table)
    bx1, by1, bx2, by2 = prompt_bbox
    box_w = max(1.0, bx2 - bx1)
    box_h = max(1.0, by2 - by1)
    box_area = box_w * box_h

    if area > (total_pixels * 0.95) and box_area < (total_pixels * 0.70):
        return {
            "is_valid": False,
            "failure_reason": "RUNAWAY_MASK",
            "area_pixels": area,
            "composite_score": 0.0,
        }

    # Mask tight bounding box
    mask_bbox = compute_mask_tight_bbox(mask_bool)
    box_iou = compute_box_iou(mask_bbox, prompt_bbox)

    # Box containment: what percentage of food mask pixels lie INSIDE the YOLO box?
    ix1, iy1 = max(0, int(np.floor(bx1))), max(0, int(np.floor(by1)))
    ix2, iy2 = min(w, int(np.ceil(bx2))), min(h, int(np.ceil(by2)))

    box_submask = np.zeros_like(mask_bool, dtype=bool)
    box_submask[iy1:iy2, ix1:ix2] = True

    pixels_inside = int(np.sum(np.logical_and(mask_bool, box_submask)))
    inside_ratio = float(pixels_inside / area) if area > 0 else 0.0
    leak_ratio = 1.0 - inside_ratio

    # Box coverage: fraction of box area filled by mask
    box_coverage = float(pixels_inside / box_area) if box_area > 0 else 0.0

    # Composite scoring formula:
    # 35% SAM predicted quality score
    # 35% Bounding box IoU alignment
    # 30% Containment ratio (food should stay within food detector's region)
    # Severe penalty for large leaks outside the bounding box
    leak_penalty = max(0.0, leak_ratio - 0.15) * 0.60
    composite_score = (
        (0.35 * max(0.0, min(1.0, quality_score))) +
        (0.35 * box_iou) +
        (0.30 * inside_ratio) -
        leak_penalty
    )
    composite_score = float(max(0.0, min(1.0, composite_score)))

    return {
        "is_valid": True,
        "area_pixels": area,
        "mask_bbox": mask_bbox,
        "box_iou": round(box_iou, 4),
        "inside_ratio": round(inside_ratio, 4),
        "leak_ratio": round(leak_ratio, 4),
        "box_coverage": round(box_coverage, 4),
        "quality_score": round(quality_score, 4),
        "composite_score": round(composite_score, 4),
    }


def select_best_mask(
    candidate_masks: List[np.ndarray],
    candidate_scores: Optional[List[float]],
    prompt_bbox: List[float],
    image_shape: Tuple[int, int],
    min_mask_area: int = 50,
    strategy: str = "iou_containment_confidence",
) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
    """
    Deterministically selects the optimal food mask from multiple candidates.

    Args:
        candidate_masks: List of candidate 2D boolean masks.
        candidate_scores: Optional SAM 2 quality scores for each candidate.
        prompt_bbox: YOLO bounding box [x1, y1, x2, y2].
        image_shape: (H, W) of the image.
        min_mask_area: Minimum acceptable pixel area.
        strategy: Selection heuristic name.

    Returns:
        Tuple[best_mask, best_score, evaluation_metadata]
    """
    if not candidate_masks:
        return None, 0.0, {"status": "NO_CANDIDATE_MASKS"}

    scores = candidate_scores or [1.0] * len(candidate_masks)
    evaluated = []

    for idx, (m, sc) in enumerate(zip(candidate_masks, scores)):
        eval_metrics = evaluate_candidate_mask(
            mask=m,
            prompt_bbox=prompt_bbox,
            image_shape=image_shape,
            quality_score=sc,
            min_area=min_mask_area,
        )
        eval_metrics["candidate_index"] = idx
        evaluated.append((idx, eval_metrics))

    # Filter valid candidates
    valid_candidates = [item for item in evaluated if item[1].get("is_valid", False)]

    if not valid_candidates:
        # All candidates failed validation
        worst_reason = evaluated[0][1].get("failure_reason", "INVALID_MASK")
        return None, 0.0, {
            "status": worst_reason,
            "evaluated_candidates": [e[1] for e in evaluated],
        }

    # Sort descending by composite score
    valid_candidates.sort(key=lambda x: x[1]["composite_score"], reverse=True)
    best_idx, best_metrics = valid_candidates[0]
    best_mask = candidate_masks[best_idx]
    best_score = best_metrics["composite_score"]

    best_metrics["selection_strategy"] = strategy
    best_metrics["total_candidates"] = len(candidate_masks)
    best_metrics["valid_candidates_count"] = len(valid_candidates)

    return best_mask, best_score, best_metrics
