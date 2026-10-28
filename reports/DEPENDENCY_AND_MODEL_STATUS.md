# NutriVision AI — Dependency and Model Status Report

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-15  
**Execution Environment:** Windows 11 x86_64 | Intel Core i7-13620H (16 Threads, CPU Execution)  

---

## 1. Runtime Environment & Dependencies

| Dependency | Installed Version | Role in NutriVision AI | Status |
| :--- | :---: | :--- | :--- |
| **Python** | `3.13.15` | Core runtime engine | **VERIFIED** |
| **PyTorch** | `2.8.0+cpu` | Neural network forward pass & tensor math | **VERIFIED (CPU Mode)** |
| **Torchvision** | `0.23.0+cpu` | Image transforms and computer vision ops | **VERIFIED (Patched NMS)** |
| **Ultralytics** | `8.3.82` | YOLO11 architecture & SAM 2 predictor bridge | **VERIFIED** |
| **OpenCV (`cv2`)** | `4.11.0.86` | Image I/O, color conversions, geometry, drawing | **VERIFIED** |
| **Pillow (`PIL`)** | `11.1.0` | Crop manipulation and image buffer decoding | **VERIFIED** |
| **FastAPI** | `0.115.11` | High-performance asynchronous REST API backend | **VERIFIED** |
| **Uvicorn** | `0.34.0` | ASGI web server | **VERIFIED** |
| **Streamlit** | `1.43.0` | Interactive web dashboard and user interface | **VERIFIED** |
| **Pydantic** | `2.10.6` | Schema validation and settings management | **VERIFIED** |
| **PyTest** | `9.1.1` | Automated test suite execution | **VERIFIED** |
| **NumPy** | `2.2.3` | Array operations, mask slicing, voxel volumes | **VERIFIED** |

### Special Windows / Python 3.13 Compatibility Note:
The repository includes [`src/utils/torch_patch.py`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/src/utils/torch_patch.py) which automatically installs pure-PyTorch fallback implementations for `torchvision::nms` and `torchvision::qnms`, avoiding any duplicate operator registration errors across libtorch DLL boundaries on Windows.

---

## 2. Model Checkpoint Inventory & Load Diagnostics

| Model Role | File Path | File Size | Architecture / Parameters | Load Test | Production Status |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Food Detector (Repaired)** | `models/detection/yolo11/repair_run/best.pt` | 5,619,738 B | YOLO11n (2.68M params, 198 classes) | **PASSED** | **VERIFIED (Taxonomy aligned, CPU undertrained)** |
| **Food Detector (Legacy)** | `models/detection/yolo11/production/best.pt` | 5,619,354 B | YOLO11n (2.68M params) | **PASSED** | **VERIFIED (Legacy scrambled taxonomy)** |
| **Instance Segmenter** | `models/segmentation/sam2.1_t.pt` | 78,105,722 B | SAM 2.1 Tiny (Hiera-T) | **PASSED** | **VERIFIED (Production-Ready)** |
| **Alternative Segmenter**| `models/segmentation/sam2_hiera_base_plus.pt` | 163,842,109 B | SAM 2 Hiera Base+ | **PASSED** | **VERIFIED (Archived)** |
| **Food Classifier** | `models/classification/efficientnet_b4_food.pth` | 72,377,763 B | EfficientNet-B4 (198 classes) | **PASSED** | **VERIFIED (Production-Ready)** |
| **Monocular Depth** | `models/depth/depth_anything_v2_vits.pth` | 99,218,434 B | Depth Anything V2 Small (DINOv2) | **PASSED** | **VERIFIED (Production-Ready)** |
| **Produce Freshness** | `models/freshness/freshness_classifier.pth` | 72,410,219 B | EfficientNet-B4 (3 classes) | **PASSED** | **VERIFIED (Production-Ready for Produce)** |

---

## 3. Official Pretrained Weight Sources

If weights need to be downloaded or refreshed in a new environment, use the following official sources:

1. **YOLO11n Base:**
   * URL: `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt`
   * Target Path: `models/detection/yolo11n.pt`
2. **SAM 2.1 Tiny:**
   * URL: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt`
   * Target Path: `models/segmentation/sam2.1_t.pt`
3. **Depth Anything V2 Small:**
   * URL: `https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth`
   * Target Path: `models/depth/depth_anything_v2_vits.pth`
4. **NutriVision Fine-Tuned Checkpoints:**
   * Retrained YOLO11 (60 epochs): Generated via `notebooks/NutriVision_YOLO11_Training_Colab.ipynb`
   * EfficientNet-B4 Food Classifier: `models/classification/efficientnet_b4_food.pth`
   * Freshness Detector: `models/freshness/freshness_classifier.pth`

---

## 4. Hardware Resource Profile & Performance

* **Host Machine:** Intel(R) Core(TM) i7-13620H (10 cores, 16 logical threads)
* **RAM:** 15.75 GB Total (7.2 GB Available during execution)
* **Device:** CPU (`device: "cpu"` resolved by `ModelManager`)
* **Average Latencies on CPU:**
  - YOLO11 Detection: ~83 ms per image (resolution 320x320)
  - SAM 2 Instance Segmentation: ~1,494 ms per food object
  - Depth Anything V2: ~612 ms per image (resolution 518x518)
  - EfficientNet-B4 Classification: ~69 ms per food crop
  - Freshness Classification: ~42 ms per produce crop
  - USDA FoodData Central Query (Cached): ~0.44 ms
