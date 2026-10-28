# NutriVision AI — Phase 1: YOLO11 GPU Retraining & Verification Guide

**Document Version:** 1.0.0  
**Target:** Google Colab (Free T4 or Pro V100/A100) / Linux GPU Cloud  
**Dataset:** 17,118 images (11,516 train, 2,467 val, 3,135 test) across 198 unified classes  
**Target Architecture:** Ultralytics YOLO11s / YOLO11n  
**Expected Duration:** ~45–60 minutes on NVIDIA A100 (~2.5 hours on T4)

---

## 1. Prerequisites & Required Assets

To retrain YOLO11 on GPU, ensure you have:
1. **Google Account & Google Colab:** [https://colab.research.google.com](https://colab.research.google.com)
2. **Project Archive:** The synchronized project zip archive:
   - Primary: `C:\Users\mdahs\OneDrive\Attachments\NutriVision-AI.zip` (80.61 MB)
   - Or Local: [`NutriVision-AI-Latest.zip`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/NutriVision-AI-Latest.zip) (31.56 MB)
   - *Note on Dataset:* Ensure `data/processed/yolo_detection` is included or uploaded to Google Drive.
3. **Colab Notebook:** Located at [`notebooks/NutriVision_YOLO11_Training_Colab.ipynb`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/notebooks/NutriVision_YOLO11_Training_Colab.ipynb).

---

## 2. Step-by-Step Google Colab Execution Workflow

### Step 1: Open Notebook & Allocate GPU Runtime
1. Navigate to Google Colab and select **File → Upload notebook**.
2. Upload [`notebooks/NutriVision_YOLO11_Training_Colab.ipynb`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/notebooks/NutriVision_YOLO11_Training_Colab.ipynb).
3. In the top-right menu, click **Change runtime type**:
   - **Hardware accelerator:** Select **GPU** (T4, V100, or A100).
4. Run the first cell to verify GPU allocation:
   ```python
   !nvidia-smi
   import torch
   print(f"CUDA Available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0)}")
   ```

### Step 2: Mount Google Drive & Extract Project
```python
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile

# If zip file is in Google Drive root:
zip_path = '/content/drive/MyDrive/NutriVision-AI.zip'
if os.path.exists(zip_path):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall('/content/')
    print("Extraction complete.")
else:
    print("Zip not found in MyDrive. Upload directly via Colab Files sidebar.")
```

### Step 3: Install Ultralytics and Validate Taxonomy
```python
!pip install --upgrade ultralytics opencv-python pyyaml pandas

import yaml, glob
from pathlib import Path

PROJECT_ROOT = Path('/content/NutriVision-AI')
data_yaml = PROJECT_ROOT / 'data/processed/yolo_detection/data.yaml'

with open(data_yaml, 'r') as f:
    cfg = yaml.safe_load(f)

# Assertions to ensure zero silent taxonomy failures
assert len(cfg['names']) == 198, f"Expected 198 classes, found {len(cfg['names'])}"
assert cfg['names'][27] == 'banana', f"Class 27 must be banana, found: {cfg['names'][27]}"
assert cfg['names'][23] == 'apple', f"Class 23 must be apple, found: {cfg['names'][23]}"

print("[PASS] Taxonomy verified: 198 unified food classes aligned.")
```

### Step 4: Launch Full GPU Training
```python
from ultralytics import YOLO

# Load pretrained base checkpoint
model = YOLO('yolo11s.pt')  # or 'yolo11n.pt' for lightweight inference

train_args = {
    'data': str(data_yaml),
    'epochs': 60,
    'patience': 12,
    'batch': 16,
    'imgsz': 640,
    'device': 0,
    'workers': 4,
    'seed': 42,
    'deterministic': True,
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'cos_lr': True,
    'box': 7.5,
    'cls': 1.2,
    'dfl': 1.5,
    'label_smoothing': 0.05,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'mosaic': 1.0,
    'mixup': 0.15,
    'project': '/content/NutriVision-AI/models/detection/yolo11/production',
    'name': 'gpu_production_run',
    'save': True,
    'exist_ok': True,
    'verbose': True
}

results = model.train(**train_args)
```

### Step 5: Post-Training Validation on Held-Out Test Set
```python
# Evaluate the best checkpoint on the untouched held-out test split
best_weights = '/content/NutriVision-AI/models/detection/yolo11/production/gpu_production_run/weights/best.pt'
trained_model = YOLO(best_weights)

# 1. Validation split
val_metrics = trained_model.val(data=str(data_yaml), split='val', imgsz=640, device=0)
print(f"Validation mAP@50: {val_metrics.box.map50:.4f}, mAP@50-95: {val_metrics.box.map:.4f}")

# 2. Test split (3,135 untouched held-out images)
test_metrics = trained_model.val(data=str(data_yaml), split='test', imgsz=640, device=0)
print(f"Test mAP@50: {test_metrics.box.map50:.4f}, mAP@50-95: {test_metrics.box.map:.4f}")
print(f"Test Precision: {test_metrics.box.p:.4f}, Recall: {test_metrics.box.r:.4f}")
```

### Step 6: Export Checkpoint to Google Drive
```python
import shutil

dest_drive = '/content/drive/MyDrive/NutriVision_yolo11_best.pt'
shutil.copy2(best_weights, dest_drive)
print(f"[SUCCESS] Checkpoint saved to Google Drive: {dest_drive}")
```

---

## 3. Local Integration Workflow (Once Checkpoint is Downloaded)

Once training completes on Colab, follow these steps to integrate into your local repository:

1. **Download Checkpoint:** Download `NutriVision_yolo11_best.pt` from Google Drive to your PC.
2. **Backup Previous Production Checkpoint:**
   ```powershell
   Copy-Item "models\detection\yolo11\production\best.pt" "models\detection\yolo11\production\best_backup_20261013.pt"
   ```
3. **Install New Checkpoint:**
   ```powershell
   Copy-Item "C:\Users\mdahs\Downloads\NutriVision_yolo11_best.pt" "models\detection\yolo11\production\best.pt" -Force
   Copy-Item "C:\Users\mdahs\Downloads\NutriVision_yolo11_best.pt" "models\detection\yolo11\repair_run\best.pt" -Force
   ```
4. **Run Verification on Real Food Images:**
   ```powershell
   python scripts/test_step6a_api_endpoints.py
   python -m pytest tests/ -q
   ```
5. **Synchronize Project Archives:**
   ```powershell
   python scripts/package_project_zip.py
   ```

---

## 4. Expected Performance Benchmark Comparison

| Metric | Undertrained Baseline (`repair_run/best.pt`) | Target GPU Model (`yolo11s` / 60 ep) | Acceptance Threshold |
| :--- | :---: | :---: | :---: |
| **mAP@50 (Held-Out Test)** | 0.091% (0.00091) | **35.0% – 48.0%** | $\ge 30.0\%$ |
| **Recall (Held-Out Test)** | 2.32% (0.02319) | **40.0% – 55.0%** | $\ge 35.0\%$ |
| **Precision** | 0.22% (0.00221) | **45.0% – 60.0%** | $\ge 40.0\%$ |
| **Banana Test Confidence** | < 0.01 (0 detections) | **0.65 – 0.90** | $\ge 0.50$ |
| **Apple Test Confidence** | < 0.01 (0 detections) | **0.70 – 0.92** | $\ge 0.50$ |
| **Status** | Non-Functional | Fully Production-Ready | Verified on Test Set |
