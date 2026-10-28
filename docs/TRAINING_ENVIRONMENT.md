# NutriVision AI — Training Environment & GPU Readiness Specification

**Document Version:** 1.0.0  
**Phase:** Step 2 — Production Dataset & Training Environment Preparation  
**System Target:** Production-Grade Food Scanning & Nutritional Analysis System  
**Last Updated:** 2026-10-05  

---

## 1. Executive Summary

This document establishes the hardware, software, and training configuration profile for **NutriVision AI**. It provides rigorous capacity planning for multi-task model training (YOLO11 food detection, EfficientNet-B4 multi-class classification, SAM2 prompt-driven segmentation, Depth Anything V2 monocular depth estimation, and 3D volume/weight regression).

The current host is an **Intel Core i7-13620H** workstation running Windows 11 with **16 GB RAM** and **CPU-only execution**. While the local CPU environment is verified, stable, and fully functional for dataset preprocessing, annotation normalization, API backend services, and single-image smoke tests, **full multi-epoch model training requires dedicated GPU acceleration**.

---

## 2. Current Host Environment

### 2.1 Hardware Profile
| Subsystem | Specification | Current Status / Utilization |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel(R) Core(TM) i7-13620H (10 Cores: 6 P-cores + 4 E-cores, 16 Threads, up to 4.9 GHz) | Verified, active. High single-thread and multi-thread performance. |
| **System RAM** | 15.75 GB Total | ~7.96 GB free (~50.5% available for caching/data loading). |
| **Local Storage** | 474.9 GB NVMe SSD (C:) | 262.4 GB free (55.3% available). Sufficient for raw & processed sets. |
| **Local GPU** | Intel(R) UHD Graphics (Integrated) | 128 MB dedicated VRAM. No CUDA cores. Not suitable for deep learning training. |
| **NVIDIA CUDA** | Unavailable (`torch.cuda.is_available() == False`) | CPU execution fallback active. |

### 2.2 Software Stack & Versions
| Package / Component | Installed Version | Production Recommendation | Status |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 11 Home Single Language (Build 26100) | Windows 11 / Ubuntu 22.04 LTS | Supported |
| **Python** | 3.13.0 (64-bit) | Python 3.11.x (LTS) | Functional locally; PyTorch ecosystem standard is 3.11 |
| **PyTorch** | `2.14.0.dev20250212+cpu` | `torch==2.5.1` (CUDA 12.1 / 12.4) | Pre-release dev build; CPU-only |
| **Torchvision** | `0.29.0` | `torchvision==0.20.1` | Compatibility anomaly identified (see Section 3) |
| **Ultralytics YOLO** | `8.3.16` (YOLO11 compatible) | `ultralytics>=8.3.20` | Operational |
| **OpenCV** | `4.11.0.86` (headless) | `opencv-python==4.10.0` | Operational |
| **FastAPI / Uvicorn** | `0.115.11` / `0.34.0` | Production standard | Verified |
| **Streamlit** | `1.42.2` | Production standard | Verified |

---

## 3. Known Compatibility Issues & Diagnostic Resolution

### 3.1 Torchvision NMS Operator Anomaly (`torchvision::nms`)
* **Symptom:** During certain torchvision operations or bounding-box non-maximum suppression invocations on Python 3.13, PyTorch emits:  
  `RuntimeError: operator torchvision::nms does not exist`
* **Root Cause Diagnosis:**  
  The current environment utilizes `Python 3.13.0` with nightly/pre-release build `torch==2.14.0.dev20250212+cpu` and `torchvision==0.29.0`. In this pre-release combination on Windows x86_64, the C++ compiled shared library `_C.pyd` in torchvision is either missing the CPU registration for `torchvision::nms` or the torch dispatcher registry failed to link dynamically across pre-release binary boundaries.
* **Resolution Strategy (Strict No-Arbitrary Downgrade Rule):**  
  1. For existing CPU tasks, Ultralytics YOLO uses its own internal pure-PyTorch / NumPy NMS implementations (`ops.non_max_suppression`), completely bypassing `torchvision::nms`. Detection validation remains 100% operational.
  2. For GPU Training Environment, provision a dedicated Python 3.11 virtual environment with official matched stable release wheels:
     ```bash
     pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
     ```

---

## 4. Model-by-Model Hardware & VRAM Requirements

| Model Architecture | Task | Input Resolution | Target Batch Size | Min. VRAM (Inference) | Recommended VRAM (Training) | Estimated Training Time (50 Epochs) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11m** | Multi-class Food Detection | $640 \times 640$ | 16 | 2.5 GB | **8.0 GB - 12.0 GB** | ~2.5 hrs on RTX 3080/4070 (12 GB) |
| **YOLO11s** (Lightweight) | Edge Detection | $640 \times 640$ | 32 | 1.8 GB | **6.0 GB** | ~1.2 hrs on T4 (16 GB) |
| **EfficientNet-B4** | 198-Class Meal Classifier | $380 \times 380$ | 32 (AMP FP16) | 2.0 GB | **8.0 GB - 10.0 GB** | ~3.0 hrs on T4 / RTX 4070 |
| **SAM2 (tiny/small)** | Instance Food Masking | $1024 \times 1024$ | 1 (prompted) | 4.0 GB | 16.0 GB (fine-tuning) | N/A (Frozen prompt inference) |
| **Depth Anything V2-S** | Metric / Monocular Depth | $518 \times 518$ | 1 (feed-forward) | 2.5 GB | 12.0 GB | N/A (Zero-shot depth inference) |
| **3D Volume & Weight MLP** | Regress grams from V & $\rho$ | Tabular + Feature | 64 | 0.5 GB | **2.0 GB (or CPU)** | ~15 mins on CPU / GPU |

---

## 5. Production Training Configurations

### 5.1 YOLO11 Training Configuration (`yolo_detection`)
```yaml
# Recommended training parameters for YOLO11 on Cloud / GPU Workstation
task: detect
mode: train
model: yolo11m.pt
data: data/processed/yolo_detection/data.yaml
epochs: 60
patience: 15
batch: 16
imgsz: 640
device: 0
workers: 4
optimizer: AdamW
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
box: 7.5
cls: 0.5
dfl: 1.5
amp: True            # Automatic Mixed Precision (FP16) saves ~35% VRAM
cos_lr: True
save: True
project: outputs/detection/runs
name: yolo11m_production_v1
```

### 5.2 EfficientNet-B4 Training Configuration (`classification`)
```python
# Recommended parameters for PyTorch / Timm EfficientNet-B4
training_config = {
    "model_name": "tf_efficientnet_b4.ns_jft_in1k",
    "num_classes": 198,
    "pretrained": True,
    "input_size": (3, 380, 380),
    "batch_size": 32,
    "gradient_accumulation_steps": 2, # Effective batch size = 64
    "epochs": 40,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "min_lr": 1e-6,
    "mixed_precision": "fp16",         # torch.cuda.amp.autocast()
    "label_smoothing": 0.1,
    "class_weights": "balanced",       # Address long-tail food classes
    "early_stopping_patience": 8
}
```

---

## 6. Recommended Cloud / Off-Host Training Environments

When the user is ready to begin training (Phases 3 and 4), any of the following verified options provides frictionless execution:

1. **Google Colab Pro (Recommended for Quick Execution):**
   - GPU: NVIDIA T4 (16 GB) or A100 (40 GB VRAM).
   - Cost: Free tier (T4) or Pro ($10/mo).
   - Compatibility: PyTorch 2.5.1 + CUDA 12.1 native, pre-installed.
2. **RunPod / Lambda Labs (Recommended for Cost Efficiency):**
   - GPU: RTX 4090 (24 GB) @ ~$0.34/hr, or RTX A4000 (16 GB) @ ~$0.20/hr.
   - Storage: 50 GB persistent network volume.
   - Ready-made Docker container: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
3. **Dedicated On-Premise GPU Workstation:**
   - Any desktop equipped with NVIDIA RTX 3060 (12 GB), RTX 4070 (12 GB), or RTX 4080/4090.

---

## 7. Current Actionable Guidance

* **Local Machine Mandate:** Keep all preprocessing, validation, class mapping, pipeline integration, API development, and UI testing on this local Windows machine.
* **Zero Disruption:** Do not force-reinstall CUDA or PyTorch locally if no NVIDIA GPU exists. Local CPU mode is verified and optimal for Step 2.
