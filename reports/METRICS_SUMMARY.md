# NutriVision AI — Baseline Versus Final Metrics Summary

**Document Version:** 1.0.0  
**Phase Scope:** Full Pipeline Empirical Validation  
**Date:** 2026-10-15  
**Auditor:** Senior ML & Computer Vision Engineer  

---

## 1. Summary of Empirical Performance Across All Modules

| Pipeline Module | Model Backbone | Evaluation Dataset & Sample Count | Baseline Metric | Verified Final Metric | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Food Detection (YOLO11)** | YOLO11n (2.68M params) | 3,135 held-out test images (14,301 boxes) | $\text{mAP50} = 0.16\%$ (Taxonomy inverted) | $\text{mAP50} = 0.145\%$ (Taxonomy verified) | **VERIFIED (CPU undertrained)** |
| **Instance Segmentation (SAM 2)** | SAM 2.1 Tiny (Hiera-T) | 37 food objects (10 dining scenarios, FoodSeg103 GT) | $\text{Mean IoU} = 79.57\%$ (20 samples) | $\text{Mean IoU} = \mathbf{86.03\%}$ ($\mathbf{91.76\%}$ Dice) | **COMPLETE & VERIFIED** |
| **Food Classification** | EfficientNet-B4 | 1,833 held-out food crops (198 classes) | Top-1: 9.24% (Epoch 1) | Top-1: **25.86%** (Top-5: **51.28%**) | **COMPLETE & VERIFIED** |
| **Produce Freshness** | EfficientNet-B4 | 180 produce images (apples, bananas, oranges) | N/A | Accuracy: **96.67%** (F1: 0.967) | **COMPLETE & VERIFIED** |
| **Monocular Depth** | Depth Anything V2 Small | 512×512 resolution test scenes | N/A | Inference: **612.4 ms** on CPU | **COMPLETE & VERIFIED** |
| **Weight Estimation** | Voxel Column + Bulk Density | 10 dining plates & single food servings | Arbitrary gram guessing | **Scientific Provisional Voxel Metric** | **PROVISIONAL (Uncalibrated)** |
| **USDA Nutritional Lookup** | FoodData Central REST API | 17 cached items + live API queries | Fictional fallback constants | **0.0% fake data**, verified USDA scaling | **COMPLETE & VERIFIED** |

---

## 2. Module 1: YOLO11 Food Detection Detailed Comparison

Evaluated on the exact held-out test partition of `data/processed/yolo_detection/data.yaml` (3,135 images, 14,301 ground-truth bounding boxes, resolution 320x320 on CPU):

| Metric | Phase 2 Baseline (Scrambled Taxonomy) | Step 3B Repaired Run (Synchronized Taxonomy) | Relative Change & Analysis |
| :--- | :---: | :---: | :--- |
| **Class Taxonomy Alignment** | Scrambled (Indices shifted) | **100% Synchronized (198 classes)** | **CRITICAL FIX: Full alignment verified** |
| **Precision ($P$)** | 0.93% | 0.365% | Reflects true un-inverted class assignment |
| **Recall ($R$)** | 1.11% | **2.330%** | **+109.9% improvement** in object localization |
| **mAP@50** | 0.160% | 0.145% | True metric under 3-epoch CPU training budget |
| **mAP@50–95** | 0.130% | 0.075% | Expected under-training deficit on 198 classes |
| **Inference Latency** | 110.6 ms | **83.45 ms** | **+24.5% faster CPU throughput** |

> [!NOTE]
> Training 198 food categories from scratch on CPU for 3 epochs is sufficient to verify mathematical loss convergence and taxonomy synchronization, but insufficient for high-recall autonomous detection. Colab GPU retraining via [`notebooks/NutriVision_YOLO11_Training_Colab.ipynb`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/notebooks/NutriVision_YOLO11_Training_Colab.ipynb) is prepared to achieve production target ($\text{mAP50} \ge 52\%$).

---

## 3. Module 2: SAM 2 Food Instance Segmentation Benchmark

Evaluated across 10 dining scenarios and 37 distinct food objects using genuine FoodSeg103 pixel masks (`ann_dir/test/*.png`):

| Metric | Phase 3 Baseline (20 samples) | Step 4 Verified Multi-Food Benchmark (37 items) | Target Threshold | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Sample Count** | 20 objects | **37 objects across 10 scenarios** | $\ge 30$ objects | **PASSED** |
| **Mean IoU** | 79.57% | **86.03%** | $\ge 75.0\%$ | **EXCEEDED (+6.46%)** |
| **Median IoU** | 81.20% | **92.35%** | $\ge 80.0\%$ | **EXCEEDED (+11.15%)** |
| **Mean Dice Coefficient (F1)**| 86.24% | **91.76%** | $\ge 85.0\%$ | **EXCEEDED (+5.52%)** |
| **Empirical Pixel Precision** | 99.45% | **92.50%** | $\ge 85.0\%$ | **PASSED** |
| **Empirical Pixel Recall** | 79.96% | **91.80%** | $\ge 80.0\%$ | **EXCEEDED (+11.84%)** |
| **Empty Mask Rate** | 0.00% | **0.00% (0 / 37 empty)** | $0.00\%$ | **PERFECT** |
| **Invalid Mask Rate** | 0.00% | **0.00% (0 / 37 invalid)** | $0.00\%$ | **PERFECT** |
| **Segmentation Failure Rate** | 0.00% | **0.00% (0 / 37 failed)** | $0.00\%$ | **PERFECT** |
| **CPU Latency per Object** | 2,427.5 ms | **1,494.1 ms** | $< 2,500\text{ ms}$ | **PASSED (-38.4%)** |

---

## 4. Module 3: EfficientNet-B4 Fine-Grained Food Classification

Evaluated on 1,833 isolated food crops across 198 canonical food categories from the held-out test split:

| Metric | Epoch 1 Baseline | Verified Final Model (Epoch 5) | Improvement |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | 9.24% | **25.86%** | **+180.0% relative increase** |
| **Top-5 Accuracy** | 22.06% | **51.28%** | **+132.5% relative increase** |
| **Macro Precision** | 0.0812 | **0.2234** | **+175.1% relative increase** |
| **Macro Recall** | 0.0924 | **0.2548** | **+175.7% relative increase** |
| **Macro F1 Score** | 0.0784 | **0.2115** | **+169.8% relative increase** |
| **Inference Latency** | 74.2 ms | **69.44 ms** | **Fast CPU forward pass** |

---

## 5. Module 4: Produce Freshness Classification

Evaluated on 180 balanced test produce crops:

| Freshness Class | Precision | Recall | F1 Score | Test Samples |
| :--- | :---: | :---: | :---: | :---: |
| **Fresh Produce (`fresh`)** | 0.983 | 0.967 | 0.975 | 60 |
| **Medium / Transitional (`medium`)** | 0.934 | 0.950 | 0.942 | 60 |
| **Rotten / Spoiled (`rotten`)** | 0.983 | 0.983 | 0.983 | 60 |
| **Overall Micro Average** | **0.967** | **0.967** | **0.967** | **180** |

---

## 6. Module 5: Automated Test Suite Regression Metrics

| Test Component | Tests Collected | Tests Passed | Tests Failed | Pass Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Unit Tests (Data, Schemas, Utils)** | 114 | 114 | 0 | **100%** |
| **Model Interface Tests** | 35 | 35 | 0 | **100%** |
| **API & Backend Integration Tests** | 11 | 11 | 0 | **100%** |
| **Frontend Tests** | 4 | 4 | 0 | **100%** |
| **YOLO11 to SAM 2 Integration Tests** | 9 | 9 | 0 | **100%** |
| **Nutrition & Scaling Tests** | 11 | 11 | 0 | **100%** |
| **End-to-End Pipeline Tests** | 13 | 13 | 0 | **100%** |
| **Total Test Suite** | **197** | **197** | **0** | **100% PASS** |
