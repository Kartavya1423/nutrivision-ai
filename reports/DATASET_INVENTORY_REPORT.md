# NutriVision AI — Verified Dataset Inventory Report

**Document Version:** 1.0.0 (Pre-Expo Verified)  
**Verification Date:** 2026-10-17  
**Project Root:** `D:\NutriVision-AI (2)\NutriVision-AI`  
**Auditor:** Senior AI/ML & Computer Vision QA Engineer  

---

## 1. Executive Summary & Dataset Architecture

The NutriVision AI project incorporates multiple computer vision datasets unified across detection, instance segmentation, fine-grained classification, produce freshness assessment, and physical density mapping.

A rigorous audit of the filesystem reveals **26,315 processed and raw image assets** across the unified pipelines. No physical-scale ground-truth weight dataset exists in the repository, establishing why depth-derived mass estimation is scientifically categorized as **PROVISIONAL / SCALE UNCALIBRATED**.

```
                                  DATASET REPOSITORIES & TASK MAPPING
┌────────────────────────────┐    ┌────────────────────────────┐    ┌────────────────────────────┐
│   FoodSeg103 (Raw)         │    │ UECFoodPixComplete (Raw)   │    │ Freshness Dataset (Raw)    │
│   7,118 images (PNG masks) │    │ 10,000 images (PNG masks)  │    │ 1,200 images (6 classes)   │
└─────────────┬──────────────┘    └─────────────┬──────────────┘    └─────────────┬──────────────┘
              │                                 │                                 │
              └────────────────┬────────────────┘                                 │
                               ▼                                                  ▼
             ┌────────────────────────────────────┐             ┌──────────────────────────────────┐
             │ Unified YOLO11 Object Detection    │             │ Processed Freshness Dataset      │
             │ data/processed/yolo_detection/     │             │ data/processed/freshness/        │
             │ 17,118 images, 198 unified classes │             │ 1,200 images (840/180/180 split) │
             └─────────────────┬──────────────────┘             └──────────────────────────────────┘
                               ▼
             ┌────────────────────────────────────┐
             │ Fine-Grained Classification        │
             │ data/processed/classification/     │
             │ 7,997 isolated food crops          │
             │ 198 unified classes synchronized   │
             └────────────────────────────────────┘
```

---

## 2. Verified Dataset Inventory

| Dataset Identifier | Physical Filesystem Location | Primary Task | Classes | Total Samples | Split (Train / Val / Test) | Native Annotation Format | Validated Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **YOLO11 Unified Detection** | `data/processed/yolo_detection` | Object Detection / Bounding Box Proposal | **198** | **17,118** | **11,516 / 2,467 / 3,135** | YOLO Normalized Text (`<cls> <cx> <cy> <w> <h>`) | **VALIDATED** |
| **Food Classification** | `data/processed/classification` | Fine-Grained Dish Categorization | **198** | **7,997** | **4,697 / 1,467 / 1,833** | Torchvision ImageFolder (isolated food crops) | **VALIDATED** |
| **Produce Freshness** | `data/processed/freshness` | Quality / Spoilage Assessment | **6** | **1,200** | **840 / 180 / 180** | ImageFolder Binary Produce Quality | **VALIDATED** |
| **FoodSeg103 (Raw Source)** | `data/raw/FoodSeg103` | Semantic & Instance Segmentation | **102** | **7,118** | **4,983 / 0 / 2,135** | PNG 8-bit Palette Mask (`ann_dir/`) | **VALIDATED** |
| **UECFoodPixComplete** | Referenced in data synthesis metadata | Prepared Meal Segmentation | **103** | **10,000** | Unified into processed detection split | PNG Semantic Segmentation Masks | **VALIDATED** |
| **Sample Benchmark** | `data/raw/sample_benchmark` | Regression & CI Smoke Testing | **5** | **30** | **20 / 5 / 5** | YOLO txt + ImageFolder | **VALIDATED** |
| **Physical Weight Dataset** | `data/raw/weight_dataset` | Direct Supervised Mass Regression | **0** | **0** | None (`.gitkeep` only) | **NONE (UNAVAILABLE)** | **EMPTY / REQUIRED** |

---

## 3. Detailed Dataset Audits

### 3.1 Object Detection Dataset (`data/processed/yolo_detection`)
- **Configuration File:** `data/processed/yolo_detection/data.yaml`
- **Class Count:** 198 unified food categories (`nc: 198`)
- **Key Index Synchronization:**
  - Class 23: `apple`
  - Class 27: `banana`
  - Class 41: `orange`
  - Class 48: `carrot`
  - Class 110: `pizza`
  - Class 127: `salad`
- **Bounding Box Integrity:**
  - Total bounding boxes on disk: **62,768**
  - Train split: 11,516 images (40,153 bounding boxes)
  - Validation split: 2,467 images (8,314 bounding boxes)
  - Test split: 3,135 images (14,301 bounding boxes)
  - Invalid coordinate count: **0**
  - Clipped coordinates: **0**
  - Out-of-range class IDs: **0**
  - Empty mask conversion errors: **0**

### 3.2 Fine-Grained Classification Dataset (`data/processed/classification`)
- **Directory Structure:**
  - `train/`: 4,697 crops across 198 subfolders
  - `val/`: 1,467 crops across 198 subfolders
  - `test/`: 1,833 crops across 198 subfolders
- **Taxonomy Alignment:**
  - Evaluated against `data.yaml` class registry: **0 mismatches across all 198 classes**.
  - Folder names match canonical lowercase underscore identifiers (e.g., `french_fries`, `grilled_salmon`, `ice_cream`).

### 3.3 Produce Freshness Dataset (`data/processed/freshness`)
- **Source:** Kaggle Fruits Freshness Dataset (`data/raw/freshness_dataset`)
- **Class Breakdown (Balanced 200 per class raw; 140/30/30 split):**
  1. `freshapples` (Train: 140, Val: 30, Test: 30)
  2. `freshbanana` (Train: 140, Val: 30, Test: 30)
  3. `freshoranges` (Train: 140, Val: 30, Test: 30)
  4. `rottenapples` (Train: 140, Val: 30, Test: 30)
  5. `rottenbanana` (Train: 140, Val: 30, Test: 30)
  6. `rottenoranges` (Train: 140, Val: 30, Test: 30)
- **Scientific Scope:** Specifically trained for post-harvest fruit quality (apple, banana, orange). Does not generalize to arbitrary cooked foods.

### 3.4 Weight Estimation Dataset (`data/raw/weight_dataset`)
- **Critical Finding:** The repository does **NOT** contain an RGB-D dataset paired with calibrated digital scale measurements in grams.
- **Consequence:** Weight estimation in NutriVision AI is computed mathematically from depth-derived volume ($V = \sum \Delta x \Delta y \Delta z$) and empirical bulk densities ($m = V \times \rho$).
- **Status:** Uncalibrated camera images are explicitly tagged **PROVISIONAL / SCALE UNCALIBRATED** to maintain academic integrity. Users can provide a kitchen scale measurement to override estimated grams.

---

## 4. Class Taxonomy & Supported Quick-Select Discovery

To eliminate hardcoded food lists in the frontend, the class registry draws from:
1. **Model Class Taxonomy (198 classes):** Synchronized in `data/processed/yolo_detection/data.yaml` and `models/detection/yolo11/class_names.json`.
2. **Empirical Density Database (73 foods):** Defined in `data/processed/food_density.json` (traceable to USDA and FAO tables).
3. **Authentic USDA FDC Cached Records (32 foods):** Stored in `data/nutrition_cache/` with verified FDC IDs.

The dynamic UI selector displays all verified food classes, supporting both one-click catalog buttons and a searchable dropdown across all registered food classes.
