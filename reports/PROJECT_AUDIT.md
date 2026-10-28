# NutriVision AI — Comprehensive Repository & Pipeline Audit Report

**Project Title:** NutriVision AI: An Intelligent Food Recognition, Weight Estimation, Freshness Detection, and Nutritional Analysis System Using Computer Vision  
**Document Version:** 1.0.0  
**Audit Date:** 2026-10-15  
**Auditor:** Senior ML, Computer Vision, & Systems Integration Engineer  
**Host Architecture:** Windows 11 x86_64 | Intel Core i7-13620H (10 physical cores, 16 threads, 15.75 GB RAM) | CPU Device Execution  

---

## 1. Executive Summary

A comprehensive architectural and empirical audit was conducted on the NutriVision AI codebase. Rather than relying on historical claims or self-reported completions, every model checkpoint was inspected on disk, loaded into memory, and evaluated against authentic test images. All automated test suites were executed to reveal genuine failures and pipeline bottlenecks.

### High-Level Audit Findings:
1. **Core Vision Models Exist and Load Successfully:**
   * **YOLO11 Detector:** Verified weights at `models/detection/yolo11/repair_run/best.pt` (5.62 MB, 198 classes strictly aligned with `data.yaml`).
   * **SAM 2 Segmenter:** Verified weights at `models/segmentation/sam2.1_t.pt` (78.1 MB, Tiny Hiera-T architecture).
   * **EfficientNet-B4 Classifier:** Verified weights at `models/classification/efficientnet_b4_food.pth` (72.3 MB, 198 fine-grained food categories).
   * **Depth Anything V2:** Verified weights at `models/depth/depth_anything_v2_vits.pth` (99.2 MB, DINOv2 Small backbone).
   * **Freshness Detector:** Verified weights at `models/freshness/freshness_classifier.pth` (72.4 MB, 3-class produce freshness head).

2. **Automated Test Suite Status:**
   * **37 Test Suites Collected**: 197 total test cases.
   * **Pass Rate:** **100% (197 / 197 tests passing)** after resolving 6 critical regressions.

3. **Critical Root Causes Diagnosed and Repaired During Audit:**
   * **USDA API Fuzzy Leakage in `fooddata_central.py`:** Unlisted or mystery query terms (e.g. `unlisted_fabricated_item_12345` or `unknown_mystery_item`) previously matched unrelated items in USDA's database (matching the word `"item"` to `"Latino bakery item"`) and erroneously marked items as `AVAILABLE`. This was fixed by enforcing non-stopword query token matching and score positivity.
   * **Missing Method in `NutritionService`:** `NutriVisionPipeline` called `self.nutrition_service.calculate_nutrition_for_weight(...)`, but `NutritionService` only provided `get_nutrition_for_item(...)`. This caused an unhandled `AttributeError` on every single meal item in previous Phase 8 pipeline runs, resulting in 0 successfully calculated nutrition items. This has been resolved by adding `calculate_nutrition_for_weight()`.
   * **FoodCropResult Dataclass Compatibility:** `SegmentedFoodObject` in Step 4 had replaced `FoodCropResult` as an alias without backward-compatible constructor keyword arguments, breaking Phase 3 schema tests. A dedicated subclass wrapper was implemented to preserve full two-way compatibility.
   * **Model Manager Checkpoint Fallback Priority:** `ModelManager` previously searched for `phase2_final_run` (the legacy scrambled run) before `repair_run/best.pt`, and hardcoded `conf = 0.001`. This was repaired to prioritize `self.settings.model_paths.detection_model_path` (`repair_run/best.pt`) and use standard configurable thresholds (`conf = 0.25`).

---

## 2. Checkpoint & Asset Inventory

| Model Role | Checkpoint Path | File Size | Architecture / Classes | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO11 Detection (Repaired)** | `models/detection/yolo11/repair_run/best.pt` | 5,619,738 B | YOLO11n (2.68M params, 198 classes) | **VERIFIED (Taxonomy aligned, CPU under-trained)** |
| **YOLO11 Detection (Legacy)** | `models/detection/yolo11/production/best.pt` | 5,619,354 B | YOLO11n (198 classes) | **VERIFIED (Legacy taxonomy scrambled)** |
| **SAM 2 Segmentation** | `models/segmentation/sam2.1_t.pt` | 78,105,722 B | SAM 2.1 Tiny (Hiera-T) | **VERIFIED (Production-ready on CPU/GPU)** |
| **SAM 2 Base+ (Alternative)** | `models/segmentation/sam2_hiera_base_plus.pt` | 163,842,109 B | SAM 2 Hiera Base+ | **VERIFIED (Storage archive)** |
| **Food Classification** | `models/classification/efficientnet_b4_food.pth` | 72,377,763 B | EfficientNet-B4 (198 classes) | **VERIFIED (Top-1: 25.86%, Top-5: 51.28%)** |
| **Monocular Depth** | `models/depth/depth_anything_v2_vits.pth` | 99,218,434 B | Depth Anything V2 Small (DINOv2) | **VERIFIED (Relative depth operational)** |
| **Produce Freshness** | `models/freshness/freshness_classifier.pth` | 72,410,219 B | EfficientNet-B4 (3 produce classes) | **VERIFIED (Produce freshness operational)** |

---

## 3. Dataset Availability & Integrity Audit

1. **FoodSeg103 (`data/raw/FoodSeg103`):**
   * High-quality dining images with authentic pixel-level segmentation masks and bounding annotations.
   * Ground truth masks in `ann_dir/test` verify SAM 2 segmentation empirical fidelity (Mean IoU: 86.03%, Mean Dice: 91.76%).
2. **UECFoodPixComplete (`data/raw/UECFoodPixComplete`):**
   * Multi-food plated meals. Ground truth masks and segmentation maps present and usable.
3. **Produce Freshness Dataset (`data/raw/freshness_dataset`):**
   * Structured into `freshapples`, `freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`, `rottenoranges`.
   * Evaluated accuracy: 96.67% across 180 samples. Limited to produce; does not generalize to cooked meals or dairy spoilage.
4. **Weight Estimation Dataset (`data/raw/weight_dataset`):**
   * Contains only `.gitkeep`. No physical scale ground-truth measurements exist locally.
   * In strict accordance with the anti-cheating mandate, weight estimation is reported as **PROVISIONAL / UNCALIBRATED** rather than fabricating ground-truth gram errors.

---

## 4. API & Frontend Audit

1. **FastAPI Application (`api/main.py`):**
   * Full REST API with OpenAPI documentation at `/docs`.
   * Standard endpoints:
     - `GET /` — Service discovery and metadata.
     - `GET /health` — Hardware and service uptime.
     - `GET /models/status` — Model manager readiness report.
     - `POST /analyze` — End-to-end 10-stage image analysis (file upload or Base64).
     - `POST /analyze/options` — Configurable analysis toggling Grad-CAM, nutrition, thresholds.
     - `GET /results/{result_id}` — Result caching and retrieval.
     - `GET /results/{result_id}/image` — Annotated visualization retrieval.
   * Automated tests in `tests/test_api.py` and `tests/test_api_endpoints.py`: **11/11 tests pass (100%)**.

2. **Streamlit Frontend (`frontend/streamlit_app.py`):**
   * Supports file upload (JPG, PNG, WEBP).
   * Live camera integration workflow documented.
   * Displays annotated bounding boxes, segmentation masks, isolated food crops, freshness scores, calibrated/uncalibrated weight notices, USDA nutrition cards, and Grad-CAM visual heatmaps.
   * Automated tests in `tests/test_frontend.py`: **4/4 tests pass (100%)**.

---

## 5. Identified Blockers & Recommended Actions

1. **YOLO11 Detection Under-Training on CPU:**
   * The 3-epoch CPU repair run resolved taxonomy scrambling, but achieved only $\text{mAP@50} = 0.145\%$ on the 3,135-image test split due to CPU compute limits.
   * **Action:** Retrain for 60 epochs on Google Colab GPU using `notebooks/NutriVision_YOLO11_Training_Colab.ipynb` to reach $\ge 52\%$ mAP@50.
2. **Real-World Metric Scale Calibration for Weight Estimation:**
   * Monocular depth maps from Depth Anything V2 are inherently scale-ambiguous.
   * **Action:** Use standard reference objects (e.g. 25 cm dinner plate, credit card) or calibrated stereo cameras for true metric scale recovery.
