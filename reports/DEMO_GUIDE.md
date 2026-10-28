# NutriVision AI — College HOD Demonstration Guide & Presentation Script

**Project Title:** NutriVision AI: An Intelligent Food Recognition, Weight Estimation, Freshness Detection, and Nutritional Analysis System Using Computer Vision  
**Presentation Audience:** Head of Department (HOD), External Examiners, Evaluation Committee  
**Version:** 2.1.0 (Production Verified — Dataset-Driven Edition)  
**Date:** 2026-10-17  

---

## 1. Quick Launch for Demo Day

### One-Click Launch (Recommended)
Double-click `start_nutrivision.bat` in the project root directory:
```powershell
.\start_nutrivision.bat
```
*The launcher checks Python 3.10+, validates dependencies, starts the FastAPI backend in a dedicated console, waits for the `/health` endpoint to return HTTP 200, and opens the Streamlit frontend at `http://127.0.0.1:8501`.*

### Manual Two-Terminal Launch
If running manually in PowerShell:
* **Terminal 1 (Backend API):**
  ```powershell
  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
  ```
  *Verify OpenAPI Swagger documentation at: `http://127.0.0.1:8000/docs`*

* **Terminal 2 (Streamlit Frontend):**
  ```powershell
  python -m streamlit run frontend/streamlit_app.py --server.port 8501
  ```
  *Access Web Dashboard at: `http://127.0.0.1:8501`*

---

## 2. 10-Step HOD Presentation Flow

### Step 1: Introduce the Problem & Academic Motivation (1 Minute)
* **What to Say:**
  > "Good morning, respected HOD and faculty. Dietary monitoring in healthcare and fitness traditionally relies on manual food journaling, which suffers from up to 50% underreporting. Our project, **NutriVision AI**, addresses this challenge through an automated multi-stage computer vision system that localizes individual food items, segments their geometric boundaries, estimates physical volume and mass, assesses produce freshness, and computes authentic USDA nutritional values."

### Step 2: System Architecture Overview (1 Minute)
* **What to Show:**
  * Point to the top engine status pills in the Streamlit header:
    - **YOLO11:** Real-time multi-food object proposal and bounding box localization.
    - **SAM 2.1 (Segment Anything 2):** Prompted instance segmentation for pixel-accurate food masks.
    - **Depth Anything V2:** Monocular depth estimation mapping 3D scene geometry.
    - **EfficientNet-B4:** 198-class fine-grained food classification on isolated crops.
    - **Binary Freshness Classifier:** Produce spoilage detection (Fresh vs. Rotten).
    - **USDA FoodData Central:** REST API and cached USDA nutrient tables for authentic macronutrients.
    - **Grad-CAM:** Neural activation heatmaps providing explainability for predictions.

### Step 3: Dataset-Driven Food Selector & Searchable Catalog (1.5 Minutes)
* **What to Show:**
  * Show the **Food Category filter** (All, Fast Food, Fruit, Vegetable, Bakery & Sweets, Meals & Mains).
  * Demonstrate the **Searchable Food Selector** (`st.selectbox`) allowing the user to select any of the 32 verified USDA foods or 198 model taxonomy classes.
  * Click on dynamic quick-select buttons such as **🍕 Pizza**, **🍔 Burger**, **🥭 Mango**, or **🥟 Samosa**.
  * Adjust the portion weight slider (e.g. from 100g to 200g).
  * Show how Calories, Protein, Carbohydrates, Fat, and Fiber scale dynamically in real time according to $N = N_{100g} \times \frac{W}{100}$.
* **What to Explain:**
  > "Instead of a hardcoded 8-food list, our UI is now completely dataset-driven via `FoodRegistry`. It automatically discovers 32 authentic USDA FoodData Central profiles with verified FDC IDs and synchronizes with our 198-class model taxonomy. Notice that the interface clearly labels this as a *Catalog-Based Reference Estimation* to maintain scientific honesty, rather than pretending an image detection was executed."

### Step 4: Live Food Plate Image Upload & Validation (1.5 Minutes)
* **What to Show:**
  * Upload a validation image from `data/processed/yolo_detection/images/val/foodseg103_00000015.jpg` or use the webcam capture tab.
  * Point out the pre-flight validation badge (Resolution, format, integrity check).
  * Point out the optional *User-Entered Measured Weight* field for when users have a kitchen scale.
  * Click **🚀 Analyze Food Plate & Nutrition**.

### Step 5: Explain Detection vs. Classification (1.5 Minutes)
* **What to Explain:**
  > "In real-world food analysis, a plate often contains multiple overlapping items. Detection (YOLO11) answers *'where is the food and what are its boundaries?'*, while Classification (EfficientNet-B4) answers *'what specific culinary dish is in this isolated crop?'*. SAM 2 extracts the exact pixel mask, removing background plates and cutlery so the classifier evaluates only food pixels."

### Step 6: Monocular Depth & Volumetric Estimation (1.5 Minutes)
* **What to Show:**
  * Show the two side-by-side visualization panels:
    1. Left: YOLO11 annotated bounding boxes.
    2. Right: Depth Anything V2 continuous depth map.
* **What to Explain:**
  > "To estimate weight from a single 2D image, we employ Depth Anything V2 to predict relative distance from the camera. We isolate the table plane as a baseline and perform voxel column integration ($\sum \text{pixel\_area} \times \text{height}$) across the SAM 2 mask. Because single RGB monocular depth lacks absolute metric scale, our system honestly tags this as **PROVISIONAL / SCALE UNCALIBRATED** and allows camera or reference-plate calibration."

### Step 7: USDA Nutritional Aggregation (1 Minute)
* **What to Show:**
  * Show the nutrition breakdown cards for each item and the complete **Meal Nutrition Summary**.
  * Highlight the macronutrient distribution bar (Protein vs. Carbs vs. Fat percentages).
  * Show the traceable USDA FDC ID (e.g., FDC ID `2708951`).

### Step 8: Freshness Classification & Produce Scope (1 Minute)
* **What to Show:**
  * Show the Freshness badge (`🟢 FRESH` or `🔴 ROTTEN`).
* **What to Explain:**
  > "Our freshness classifier is an empirical model trained on produce datasets (apples, bananas, and oranges) achieving 96.67% test accuracy. The interface transparently clarifies that freshness assessment is specialized for these supported fruits and does not make unverified claims on cooked dishes."

### Step 9: Neural Explainability via Grad-CAM (1 Minute)
* **What to Show:**
  * Expand the Grad-CAM Neural Attention Maps section.
  * Show the 3-panel display (Isolated Crop, Heatmap, Overlay).
* **What to Explain:**
  > "In medical and nutritional AI, black-box predictions are unacceptable. Grad-CAM computes gradients at the final convolutional layer to highlight the visual features—such as texture and surface coloration—that influenced the model's decision."

### Step 10: Quantitative Metrics, Limitations & Future Work (1.5 Minutes)
* **What to Conclude:**
  * Mention our rigorous test suite: **224 out of 224 automated tests pass (100% pass rate)**.
  * Address the YOLO11 model status: Forensic audit proved the baseline was undertrained on CPU (3 epochs, 3% data); a full 60-epoch Google Colab GPU retraining pipeline has been prepared with synchronized 198-class canonical taxonomy.
  * Invite questions from the HOD and faculty.

---

## 3. Anticipated Questions from HOD & Answers

1. **Q: How does the system estimate weight from a single RGB camera?**  
   * **Answer:** "It combines SAM 2 instance masks with Depth Anything V2 monocular depth maps. By isolating the surface plane (e.g., the plate rim), it computes relative height for each pixel and integrates voxel columns ($V = \sum A_{\text{pixel}} \times h$). It then maps the identified food class to an empirical bulk density from our USDA/FAO database ($\text{Weight} = \text{Volume} \times \text{Density}$). When reference plate dimensions are uncalibrated, it honestly tags the weight as Provisional."

2. **Q: Why does the detector perform better on some foods than others?**  
   * **Answer:** "The initial baseline checkpoint was trained on CPU for only 3 epochs on a small 3% fraction of the dataset to verify convergence. We have developed and validated a 60-epoch Colab GPU retraining notebook with verified 198-class alignment to achieve production mAP."

3. **Q: How do you prevent hallucinated or fake nutrition numbers?**  
   * **Answer:** "NutriVision AI strictly queries the official USDA FoodData Central REST API and authentic cached FDC records. If an item is unlisted, it returns an explicit `NUTRITION_DATA_NOT_AVAILABLE` status rather than fabricating default numbers."

4. **Q: Is the system resilient to backend disconnections?**  
   * **Answer:** "Yes. The frontend features typed exception handling (`BackendConnectionError`, `InvalidImageApiError`, `ModelNotReadyApiError`). If the backend is offline, it provides clear PowerShell commands to start the service, and also supports an optional in-process execution fallback."

---

## 4. Final Demo Day Checklist

- [ ] Laptop charger plugged in.
- [ ] Working directory set to `D:\NutriVision-AI (2)\NutriVision-AI`.
- [ ] Run `python -m pytest tests/ -q` to confirm all 224 tests pass.
- [ ] Double-click `start_nutrivision.bat` or run `.\start_nutrivision.ps1`.
- [ ] Confirm browser opens at `http://127.0.0.1:8501`.
- [ ] Confirm OpenAPI Swagger responds at `http://127.0.0.1:8000/docs`.
- [ ] Have sample images ready in `data/processed/yolo_detection/images/val/`.
