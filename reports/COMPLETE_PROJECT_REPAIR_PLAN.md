# NutriVision AI — Comprehensive Project Repair & Completion Plan (Phase 1)

> **Document Version:** 1.0.0 (Phase 1 Audit & Repair Blueprint)  
> **Date:** 2026-10-15T20:30:00+05:30  
> **Author:** Antigravity Senior Computer Vision & Deep Learning Engineer  
> **Target Environment:** Windows 11 x86_64, Intel Core i7-13620H, Python 3.13.15, PyTorch 2.14.0+cpu  

---

## 1. Executive Status & Diagnostic Overview

This comprehensive repair plan establishes an evidence-based roadmap to bring every module of NutriVision AI from provisional/partial states to robust, production-grade performance on real-world food photographs.

### Current Operating Summary:
- **Baseline Test Suite:** **197 / 197 tests passing (100%)**.
- **Services Operational:** FastAPI backend (`http://localhost:8000`) and Streamlit frontend (`http://localhost:8501`).
- **Immediate Bug Repaired (Phase 1):** Resolved the *"Depth map visualization unavailable"* issue by decoupling Monocular Depth Estimation from detection success so that depth maps are always generated for the uploaded scene. Added a configurable detection confidence slider in the UI.

---

## 2. Dataset Discovery, Verification & Class Audit

Datasets physically discovered and verified on the local host:

| Dataset Name | Physical Location on Machine | Image Count | Structure & Annotations | Supported Tasks in NutriVision AI | Class Count & Mapping Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FoodSeg103** | `data/raw/FoodSeg103` | 14,236 images | Multi-food images, pixel-level semantic masks, `category_id.txt` | YOLO11 Detection, SAM2 Segmentation, EfficientNet Classification | 103 food classes (includes `banana` at class ID 27, `apple`, `bread`, etc.) |
| **UECFoodPixComplete**| `data/raw/UECFoodPixComplete` | 20,000 images | Plate images, multi-food masks, `category.txt` (102 classes) | YOLO11 Detection, SAM2 Segmentation, Volume Calibration | 100/102 Japanese & Western dish classes (`rice`, `ramen`, `curry`, etc.) |
| **Fruits-360 Multi** | `data/raw/Fruits360_multi` | 919 images | Multi-fruit composite scenes | Produce Multi-Object Detection & Freshness Evaluation | Fruit produce (`banana`, `apple`, `orange`, `lemon`, etc.) |
| **Fruits-360 Single** | `C:\Users\mdahs\Downloads\fruits-360-100x100-main\fruits-360-100x100-main` | ~90,000 images (`Training`, `Test`) | Isolated 100x100 produce images | Produce Freshness & Fine-Grained Produce Classification | 131 fruit and vegetable categories |
| **Freshness Dataset** | `data/raw/freshness_dataset` | 1,200 images | Binary fresh vs. rotten produce crops (apples, bananas, oranges) | Freshness Classification (EfficientNet) & Grad-CAM | 2 classes: `fresh` (0), `rotten` (1) |
| **Processed YOLO Detection** | `data/processed/yolo_detection` | 17,118 images (34,253 files) | YOLO normalized txt bounding boxes (`train`, `val`, `test`), `data.yaml` | YOLO11 Object Detection Training & Evaluation | **198 unified food classes** |
| **Processed Classification** | `data/processed/classification` | 7,997 crop images across 594 subdirectories | Food-only crops organized by class name | EfficientNet-B4 Food Classification | 198 unified target classes |

### Data Leakage Audit:
- Detection split manifest (`data/processed/yolo_detection/data.yaml`): Train: 11,540 images, Val: 2,789 images, Test: 2,789 images.
- Verification confirms zero overlap between train and validation image hash sets. All crops in `data/processed/classification` are derived strictly within their corresponding dataset split boundaries.

---

## 3. Banana Detection Failure: Evidence-Based Root Cause Analysis

### Symptoms:
When a real photograph of a banana is uploaded to Streamlit or submitted to `POST /analyze`, the system outputs:
> *"No food items were localized in this image."*

### Empirical Evidence & Tracing:
1. **Checkpoint Audit:**
   - Active YOLO checkpoint: [`models/detection/yolo11/repair_run/best.pt`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/models/detection/yolo11/repair_run/best.pt) (5.36 MB).
   - Class ID 27 in this checkpoint is mapped to `banana`.
   - Training parameters in [`scripts/train_repaired_yolo11.py`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/scripts/train_repaired_yolo11.py):
     ```python
     "epochs": 3,
     "fraction": 0.03,  # Only 345 images sampled out of 11,540!
     "device": "cpu"
     ```
   - **Trained Quality:** Model was trained on CPU for only 3 epochs on a 3% random fraction. Its overall evaluation mAP@50 is **0.145% (0.00145)**.
2. **Inference Tracing on Banana Images:**
   - Evaluated on validation banana image `data/processed/yolo_detection/images/val/foodseg103_00001059.jpg`:
     - Standard threshold (`conf=0.25`): **0 detections**.
     - Extremely sensitive threshold (`conf=0.001`): Detects only phantom boxes classified as `corn` (conf: 0.0136).
   - Evaluated on banana crop `foodseg103_00004428_c00.jpg`:
     - Pretrained COCO weights (`yolo11n.pt`): Detects `banana` with confidence **0.2717**.
     - Undertrained custom 198-class weights: **0 detections**.
3. **Conclusion:**
   The failure is **NOT** a code bug, missing class mapping, or image preprocessing error. The root cause is severe model undertraining (only 345 random images seen for 3 epochs across 198 classes). Visual features for bananas were never learned by the custom weights.

---

## 4. Prioritized Project Repair Matrix

| Priority | Module | Component | Current State | Root Cause / Evidence | Proposed Fix | Required Data / Models | Acceptance Criteria |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P0** | **UI / Pipeline** | Depth Map on 0 Detections | **FIXED** in Phase 1 | Early return bypassed depth estimation | Run Depth Anything V2 before early return | `depth_anything_v2_vits.pth` | Depth map displayed on all valid image uploads |
| **P0** | **UI** | Detection Threshold Control | **FIXED** in Phase 1 | UI lacked confidence threshold slider | Added slider (0.01 - 0.90) in Streamlit sidebar | Frontend & API integration | User can tune sensitivity directly from UI |
| **P1** | **Detection** | YOLO11 Food Localization | **PARTIAL** (mAP 0.145%) | 3-epoch CPU undertraining on 3% data fraction | Train YOLO11 on full/curated FoodSeg103+UEC dataset with transfer learning | `data/processed/yolo_detection` (17,118 images) | mAP@50 ≥ 35%; Banana and staple foods detected at conf ≥ 0.25 |
| **P1** | **Classification** | EfficientNet-B4 Food Classifier | **PARTIAL** (Top-1 ~18%) | 5-epoch CPU training on small crop sample | Fine-tune classifier on balanced class crops with cosine annealing | `data/processed/classification` (7,997 crops) | Top-1 accuracy ≥ 65%, Top-5 ≥ 85% on validation set |
| **P2** | **Segmentation** | SAM 2.1 Tiny Segmenter | **VERIFIED** | Meta SAM 2.1 Tiny foundation model working cleanly | Retain existing model; add prompt refinement | `sam2.1_t.pt` | IoU ≥ 0.80 on localized food bounding boxes |
| **P2** | **Freshness** | EfficientNet Freshness Detector | **VERIFIED** | Model loads and classifies fresh vs. rotten with Grad-CAM | Validate produce generalization on Fruits-360 test set | `freshness_dataset` & Fruits-360 | Binary accuracy ≥ 85% on produce test crops |
| **P2** | **Depth** | Depth Anything V2 | **VERIFIED** | Model loads and produces clean relative depth maps | Retain foundation weights; add camera calibration helper | `depth_anything_v2_vits.pth` | Smooth depth gradients, zero NaN outputs |
| **P3** | **Volume & Weight**| Scale & Density Estimation | **PROVISIONAL** | Monocular depth is relative; requires metric calibration | Implement reference-object auto-detection (e.g. coin / plate diameter) | `food_densities.json`, coin detector | Documented calibration error margin ≤ 15% |
| **P3** | **Nutrition** | USDA FoodData Central | **VERIFIED** | Lookups and scaling active; cache functioning | Expand synonym mapping table for compound dishes | USDA API & offline cache | 100% of detected classes match valid FDC food entries |
| **P3** | **Explainability** | Dual Grad-CAM | **VERIFIED** | Dual 3-panel composites generated for food & freshness | Retain existing architecture | EfficientNet conv layers | Heatmaps focus on discriminative food regions |

---

## 5. Model Retraining Proposal & Protocol (No Premature Training)

Per non-negotiable guidelines, no multi-hour training job has been started without approval. Below is the exact proposed training plan:

### A. YOLO11 Food Detection Retraining Proposal
- **Dataset:** `data/processed/yolo_detection/data.yaml` (11,540 train images, 2,789 val images).
- **Target Classes:** 198 food categories (including `banana`, `apple`, `bread`, `rice`, `chicken`, etc.).
- **Base Checkpoint:** `yolo11n.pt` (transfer learning from COCO pretrained backbone).
- **Proposed Training Parameters:**
  - Image size: 640x640 (standard YOLO resolution).
  - Epochs: 30 epochs (or 15 epochs on GPU/Google Colab).
  - Batch size: 16 (or 8 on CPU).
  - Optimizer: AdamW, initial LR $0.001$.
- **Expected Outcome:** Increases mAP@50 from 0.145% to >35.0%, enabling reliable localization of bananas and common plate items at standard confidence thresholds (`conf ≥ 0.25`).

### B. EfficientNet-B4 Classification Fine-Tuning Proposal
- **Dataset:** `data/processed/classification` (7,997 verified crops across 198 food categories).
- **Base Checkpoint:** ImageNet-pretrained EfficientNet-B4 backbone.
- **Proposed Training Parameters:**
  - Image size: 380x380.
  - Epochs: 15 epochs with mixed precision.
  - Optimizer: AdamW, LR $3 \times 10^{-4}$ with CosineAnnealingLR.

---

## 6. Module Maturity Classification

```
┌─────────────────────────────────────────────────────────────┐
│                      NUTRIVISION AI                         │
├──────────────────────────────┬──────────────────────────────┤
│ ✅ VERIFIED & WORKING        │ ⚠️ PARTIAL / PENDING TRAIN   │
│ • FastAPI Backend Endpoints  │ • YOLO11 Detection Model     │
│ • Streamlit Frontend App     │   (Requires planned retrain) │
│ • SAM 2.1 Tiny Segmenter     │ • EfficientNet Food Classifier│
│ • Depth Anything V2 Model    │   (Requires planned fine-tune│
│ • Freshness Classification   │ • Physical Weight Accuracy   │
│ • Dual Grad-CAM Heatmaps     │   (Requires scale reference) │
│ • USDA FoodData Lookups      │                              │
│ • PyTest Test Suite (197/197)│                              │
│ • Depth Map on 0 Detections  │                              │
│ • UI Confidence Slider       │                              │
└──────────────────────────────┴──────────────────────────────┘
```
