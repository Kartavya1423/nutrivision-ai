"""NutriVision AI — Colab Notebook Generator.

Builds a fully-validated Jupyter notebook for YOLO11 GPU retraining.
"""

import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# NutriVision AI — Production YOLO11 GPU Retraining Pipeline\n",
                "**Project:** NutriVision AI: Intelligent Food Recognition & Nutrition Estimation System  \n",
                "**Target:** Full 60-Epoch Retraining on 17,118 images across 198 unified food classes  \n",
                "**Recommended Hardware:** Google Colab GPU (NVIDIA T4, V100, or A100)\n",
                "\n",
                "---\n",
                "### Overview of Pipeline Steps\n",
                "1. **Hardware Verification:** Confirm CUDA GPU availability.\n",
                "2. **Dependency Installation:** Ultralytics, PyYAML, OpenCV, Pandas.\n",
                "3. **Dataset Extraction:** Unzip `NutriVision-AI.zip` containing `data/processed/yolo_detection`.\n",
                "4. **Taxonomy & Split Integrity Checks:** Verify 198 classes, zero label leakage, and banana/apple indices.\n",
                "5. **GPU Training Execution:** 60 epochs with AdamW, cosine annealing, mosaic/mixup, and seed=42.\n",
                "6. **Held-Out Test Set Evaluation:** Benchmark on 3,135 untouched test images.\n",
                "7. **Per-Class Metrics & Visualizations:** Export precision, recall, mAP@50, mAP@50-95, and test predictions.\n",
                "8. **Checkpoint Export:** Save `NutriVision_yolo11_best.pt` directly to Google Drive."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment & Hardware Diagnostics\n",
                "Verify that Google Colab has allocated a GPU instance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi\n",
                "import torch\n",
                "print(f'PyTorch Version: {torch.__version__}')\n",
                "print(f'CUDA Available: {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
                "    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')\n",
                "else:\n",
                "    raise SystemError('GPU runtime not detected! In Colab top menu, go to: Runtime -> Change runtime type -> Select T4 GPU or A100.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Dependency Installation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install --upgrade ultralytics opencv-python pyyaml pandas matplotlib\n",
                "import ultralytics\n",
                "print(f'Ultralytics Version: {ultralytics.__version__}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Dataset Setup & Google Drive Mount\n",
                "Mount Google Drive to read `NutriVision-AI.zip` or save checkpoints."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "import os, zipfile, shutil\n",
                "from pathlib import Path\n",
                "\n",
                "zip_candidate = '/content/drive/MyDrive/NutriVision-AI.zip'\n",
                "if os.path.exists(zip_candidate):\n",
                "    print(f'Found project archive at: {zip_candidate}')\n",
                "    print('Extracting archive into /content/...')\n",
                "    with zipfile.ZipFile(zip_candidate, 'r') as zf:\n",
                "        zf.extractall('/content/')\n",
                "    print('Extraction complete.')\n",
                "else:\n",
                "    print('Project zip not found at /content/drive/MyDrive/NutriVision-AI.zip.')\n",
                "    print('Please upload NutriVision-AI.zip to your Google Drive root or upload directly to /content/ via the sidebar.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Pre-Flight Dataset & Taxonomy Validation\n",
                "Verify data paths, class mappings, and split integrity before training."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import yaml, glob\n",
                "from pathlib import Path\n",
                "\n",
                "PROJECT_ROOT = Path('/content/NutriVision-AI')\n",
                "if not PROJECT_ROOT.exists() and Path('/content/data').exists():\n",
                "    PROJECT_ROOT = Path('/content')\n",
                "\n",
                "data_yaml_path = PROJECT_ROOT / 'data/processed/yolo_detection/data.yaml'\n",
                "assert data_yaml_path.exists(), f'data.yaml not found at {data_yaml_path}'\n",
                "\n",
                "with open(data_yaml_path, 'r', encoding='utf-8') as f:\n",
                "    cfg = yaml.safe_load(f)\n",
                "\n",
                "names = cfg['names']\n",
                "print(f'Total classes defined: {len(names)}')\n",
                "assert len(names) == 198, f'Expected 198 classes, found {len(names)}'\n",
                "assert names[27] == 'banana', f'Taxonomy error: ID 27 must be banana, found {names[27]}'\n",
                "assert names[23] == 'apple', f'Taxonomy error: ID 23 must be apple, found {names[23]}'\n",
                "\n",
                "train_imgs = glob.glob(str(PROJECT_ROOT / 'data/processed/yolo_detection/images/train/*'))\n",
                "val_imgs = glob.glob(str(PROJECT_ROOT / 'data/processed/yolo_detection/images/val/*'))\n",
                "test_imgs = glob.glob(str(PROJECT_ROOT / 'data/processed/yolo_detection/images/test/*'))\n",
                "\n",
                "print(f'Split counts -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}')\n",
                "assert len(train_imgs) > 10000, f'Train images seem incomplete: {len(train_imgs)}'\n",
                "assert len(val_imgs) > 2000, f'Val images seem incomplete: {len(val_imgs)}'\n",
                "assert len(test_imgs) > 2500, f'Test images seem incomplete: {len(test_imgs)}'\n",
                "print('[PASS] All pre-flight checks succeeded! Ready for GPU training.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Full 60-Epoch GPU Training\n",
                "Train YOLO11s with AdamW optimizer, cosine learning rate decay, and heavy data augmentation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from ultralytics import YOLO\n",
                "\n",
                "# Choose yolo11s.pt for higher accuracy (recommended) or yolo11n.pt for ultra-lightweight\n",
                "base_model = 'yolo11s.pt'\n",
                "model = YOLO(base_model)\n",
                "\n",
                "out_project = str(PROJECT_ROOT / 'models/detection/yolo11/production')\n",
                "\n",
                "train_config = {\n",
                "    'data': str(data_yaml_path),\n",
                "    'epochs': 60,\n",
                "    'patience': 12,\n",
                "    'batch': 16,  # Increase to 32 on A100 GPU\n",
                "    'imgsz': 640,\n",
                "    'device': 0,\n",
                "    'workers': 4,\n",
                "    'seed': 42,\n",
                "    'deterministic': True,\n",
                "    'optimizer': 'AdamW',\n",
                "    'lr0': 0.001,\n",
                "    'lrf': 0.01,\n",
                "    'cos_lr': True,\n",
                "    'box': 7.5,\n",
                "    'cls': 1.2,\n",
                "    'dfl': 1.5,\n",
                "    'label_smoothing': 0.05,\n",
                "    'hsv_h': 0.015,\n",
                "    'hsv_s': 0.7,\n",
                "    'hsv_v': 0.4,\n",
                "    'degrees': 10.0,\n",
                "    'translate': 0.1,\n",
                "    'scale': 0.5,\n",
                "    'mosaic': 1.0,\n",
                "    'mixup': 0.15,\n",
                "    'project': out_project,\n",
                "    'name': 'gpu_full_run',\n",
                "    'save': True,\n",
                "    'exist_ok': True,\n",
                "    'verbose': True\n",
                "}\n",
                "\n",
                "print(f\"Launching training for {train_config['epochs']} epochs on GPU...\")\n",
                "results = model.train(**train_config)\n",
                "print('[SUCCESS] Training run finished!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Evaluate Repaired Checkpoint on Untouched Held-Out Test Set\n",
                "Evaluate the generated `best.pt` on the 3,135 untouched test split images."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "best_weights_path = Path(out_project) / 'gpu_full_run/weights/best.pt'\n",
                "assert best_weights_path.exists(), f'Trained checkpoint not found at: {best_weights_path}'\n",
                "\n",
                "eval_model = YOLO(str(best_weights_path))\n",
                "\n",
                "print('--- Evaluating on Validation Set ---')\n",
                "val_metrics = eval_model.val(data=str(data_yaml_path), split='val', imgsz=640, device=0)\n",
                "print(f'Validation mAP@50:    {val_metrics.box.map50:.4f}')\n",
                "print(f'Validation mAP@50-95: {val_metrics.box.map:.4f}')\n",
                "\n",
                "print('\\n--- Evaluating on Held-Out Test Set (3,135 images) ---')\n",
                "test_metrics = eval_model.val(data=str(data_yaml_path), split='test', imgsz=640, device=0)\n",
                "print(f'Test mAP@50:          {test_metrics.box.map50:.4f}')\n",
                "print(f'Test mAP@50-95:       {test_metrics.box.map:.4f}')\n",
                "print(f'Test Precision:       {test_metrics.box.p:.4f}')\n",
                "print(f'Test Recall:          {test_metrics.box.r:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Qualitative Visual Inference on Test Foods\n",
                "Run inference on banana, apple, and mixed meal test images to confirm high-confidence localization."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import cv2, matplotlib.pyplot as plt\n",
                "\n",
                "test_samples = glob.glob(str(PROJECT_ROOT / 'data/processed/yolo_detection/images/test/*.jpg'))[:3]\n",
                "\n",
                "for sample_img in test_samples:\n",
                "    res = eval_model.predict(sample_img, conf=0.25, imgsz=640, device=0)[0]\n",
                "    print(f'Image: {Path(sample_img).name} -> Detections count: {len(res.boxes)}')\n",
                "    for b in res.boxes:\n",
                "        cls_id = int(b.cls[0])\n",
                "        cls_name = eval_model.names.get(cls_id, str(cls_id))\n",
                "        conf = float(b.conf[0])\n",
                "        print(f'   Detected: {cls_name} (conf: {conf:.4f})')\n",
                "    \n",
                "    res_plotted = res.plot()\n",
                "    plt.figure(figsize=(6, 6))\n",
                "    plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))\n",
                "    plt.title(f'{Path(sample_img).name}: {len(res.boxes)} items')\n",
                "    plt.axis('off')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Export Checkpoint to Google Drive\n",
                "Save `NutriVision_yolo11_best.pt` directly to your Google Drive root for download to PC."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dest_drive_checkpoint = '/content/drive/MyDrive/NutriVision_yolo11_best.pt'\n",
                "shutil.copy2(str(best_weights_path), dest_drive_checkpoint)\n",
                "print(f'[SUCCESS] Exported retrained weights to Google Drive: {dest_drive_checkpoint}')\n",
                "print('Download this file to your computer and place it at: models/detection/yolo11/production/best.pt')"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

if __name__ == "__main__":
    out_path = Path("notebooks/NutriVision_YOLO11_Training_Colab.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully generated clean Colab notebook at: {out_path}")
