import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# NutriVision AI — Production YOLO11 GPU Retraining Pipeline\n",
                    "**Project:** NutriVision AI: Intelligent Food Recognition & Nutrition Estimation System  \n",
                    "**Target:** Retraining YOLO11 on 17,118 images across 198 unified food classes  \n",
                    "**Recommended Runtime:** GPU (Google Colab T4, V100, or A100)\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Environment & Hardware Diagnostics\n",
                    "Verify NVIDIA CUDA acceleration and GPU VRAM."
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
                    "    raise SystemError('GPU runtime not detected! Go to Runtime -> Change runtime type -> Select T4 GPU.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Dependency Installation\n",
                    "Install Ultralytics and imaging libraries."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install --upgrade ultralytics opencv-python pyyaml\n",
                    "import ultralytics\n",
                    "print(f'Ultralytics Version: {ultralytics.__version__}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Dataset Setup & Google Drive Mount\n",
                    "Mount Google Drive or upload `NutriVision-AI.zip`."
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
                    "# Unzip dataset archive if stored in Drive:\n",
                    "# !unzip -q /content/drive/MyDrive/NutriVision-AI.zip -d /content/"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Verify Dataset Configuration\n",
                    "Ensure `data.yaml` has 198 classes and points to image splits."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import yaml, glob\n",
                    "yaml_path = 'NutriVision-AI/data/processed/yolo_detection/data.yaml'\n",
                    "with open(yaml_path, 'r') as f:\n",
                    "    cfg = yaml.safe_load(f)\n",
                    "print(f'Loaded {len(cfg[\"names\"])} classes. Class 0: {cfg[\"names\"][0]}, Class 19: {cfg[\"names\"][19]}')\n",
                    "train_imgs = glob.glob('NutriVision-AI/data/processed/yolo_detection/images/train/*')\n",
                    "val_imgs = glob.glob('NutriVision-AI/data/processed/yolo_detection/images/val/*')\n",
                    "test_imgs = glob.glob('NutriVision-AI/data/processed/yolo_detection/images/test/*')\n",
                    "print(f'Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Train YOLO11 on GPU\n",
                    "Train YOLO11s (or YOLO11n) for 60 epochs with AdamW and cosine learning rate."
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
                    "# Load pretrained YOLO11 base\n",
                    "model = YOLO('yolo11s.pt')\n",
                    "\n",
                    "# Launch full GPU training run\n",
                    "results = model.train(\n",
                    "    data=yaml_path,\n",
                    "    epochs=60,\n",
                    "    patience=12,\n",
                    "    batch=16,\n",
                    "    imgsz=640,\n",
                    "    optimizer='AdamW',\n",
                    "    lr0=0.001,\n",
                    "    lrf=0.01,\n",
                    "    cos_lr=True,\n",
                    "    box=7.5,\n",
                    "    cls=1.2,\n",
                    "    dfl=1.5,\n",
                    "    label_smoothing=0.05,\n",
                    "    hsv_h=0.015,\n",
                    "    hsv_s=0.7,\n",
                    "    hsv_v=0.4,\n",
                    "    degrees=10.0,\n",
                    "    translate=0.1,\n",
                    "    scale=0.5,\n",
                    "    mosaic=1.0,\n",
                    "    mixup=0.15,\n",
                    "    project='NutriVision-AI/models/detection/yolo11/production',\n",
                    "    name='gpu_train_run',\n",
                    "    save=True,\n",
                    "    device=0\n",
                    ")\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Evaluate Repaired Checkpoint on Validation and Test Sets\n",
                    "Measure precision, recall, and mAP@50-95."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "val_metrics = model.val(data=yaml_path, split='val', imgsz=640, device=0)\n",
                    "print(f'Validation mAP50: {val_metrics.box.map50:.4f}, mAP50-95: {val_metrics.box.map:.4f}')\n",
                    "\n",
                    "test_metrics = model.val(data=yaml_path, split='test', imgsz=640, device=0)\n",
                    "print(f'Test mAP50: {test_metrics.box.map50:.4f}, mAP50-95: {test_metrics.box.map:.4f}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Export Model Checkpoint\n",
                    "Copy `best.pt` to Google Drive or download."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import shutil\n",
                    "best_path = 'NutriVision-AI/models/detection/yolo11/production/gpu_train_run/weights/best.pt'\n",
                    "shutil.copy2(best_path, '/content/drive/MyDrive/NutriVision_yolo11_best.pt')\n",
                    "print('Saved trained model to Google Drive: NutriVision_yolo11_best.pt')"
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

    out_p = Path("notebooks/NutriVision_YOLO11_Training_Colab.ipynb")
    out_p.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Wrote {out_p}")

if __name__ == "__main__":
    create_notebook()
