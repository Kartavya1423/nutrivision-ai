# NutriVision AI — Phase 1 Manual Verification Guide

**Guide Version:** 1.0.0  
**Target Environment:** Windows 11 PowerShell  
**Project Root:** `C:\Users\mdahs\OneDrive\Attachments\NutriVision-AI`

---

## 1. Prerequisites & Terminal Configuration

To manually verify the repaired backend-frontend connection, open **two separate PowerShell terminal windows** from the project root directory.

### Terminal 1: FastAPI Backend Service
From `C:\Users\mdahs\OneDrive\Attachments\NutriVision-AI`:
```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

> **Why `--host 127.0.0.1 --port 8000`?**
> - `127.0.0.1` binds directly to loopback interface on your Windows PC, preventing Windows Defender Firewall alerts.
> - `8000` matches the default `API_BASE_URL` defined in `frontend/streamlit_app.py` and `.env.example`.
> - If you wish to enable hot-reloading during code edits, append `--reload`:
>   `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload`

Expected console output:
```
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Terminal 2: Streamlit Interactive Frontend
From `C:\Users\mdahs\OneDrive\Attachments\NutriVision-AI`:
```powershell
python -m streamlit run frontend/streamlit_app.py --server.port 8501
```

> **Why `--server.port 8501`?**
> - Port 8501 is the standard Streamlit HTTP port.
> - Streamlit will automatically launch your default browser to `http://localhost:8501`.

---

## 2. Step-by-Step Manual Verification Scenarios

### Test Scenario 1: Verify Initial Health & Model Status
1. Open your browser to `http://localhost:8501`.
2. Inspect the **Sidebar (left panel)**:
   - Verify the top status shows: `🟢 FastAPI Backend: Online (CPU)`.
   - Inspect the **Model Status** list:
     - `✅ YOLO11 Food Detector: READY`
     - `✅ SAM 2 Instance Segmenter: READY`
     - `✅ EfficientNet-B4 Food Classifier: READY`
     - `✅ EfficientNet Freshness Classifier: READY`
     - `✅ Depth Anything V2 Monocular Depth: READY`
3. In a browser tab, open `http://127.0.0.1:8000/docs` to inspect the Swagger UI documentation.

---

### Test Scenario 2: Valid Image Analysis Flow
1. Under **1. Food Plate Image Input**, choose **File Upload**.
2. Select a food image. You can use an existing sample from the repository:
   `data\raw\UECFoodPixComplete\data\UECFoodPIXCOMPLETE\train\img\1.jpg`
3. Observe the preview image displayed in column 1 (`Uploaded Image (W×H px, JPEG)`).
4. Click **🚀 Analyze Food Plate**.
5. Observe the spinner: *"Analyzing food plate through NutriVision 10-stage pipeline..."*.
6. Review the results:
   - If food is localized: View the annotated image, depth map, food crops, freshness badges, USDA nutrition values, and Grad-CAM attention heatmaps.
   - If zero items are localized: Observe the blue badge `:blue[NO_FOOD_DETECTED (0 Items)]`, the exact processing latency in milliseconds, and the threshold adjustment advice.

---

### Test Scenario 3: Threshold Tuning for Undertrained Detector
1. Keep the sample food image loaded.
2. In the sidebar, adjust **Detection Confidence Threshold** slider from `0.25` down to `0.05` or `0.01`.
3. Click **🚀 Analyze Food Plate**.
4. Observe that lower confidence threshold allows candidate proposals from the baseline YOLO11 model to be passed downstream to SAM2 and EfficientNet.

---

### Test Scenario 4: Testing Invalid Image Handling
1. Create a dummy text file named `test_corrupt.jpg` containing plain text (e.g. `THIS IS NOT AN IMAGE`).
2. Upload `test_corrupt.jpg` using the file uploader.
3. Observe the immediate frontend error banner:
   ```
   🖼️ Invalid or Unreadable Image
   Corrupted or invalid image: Corrupted or unreadable image file...
   Requirements: Allowed formats: JPG, JPEG, PNG, WEBP. Resolution: 10×10 to 4096×4096 px.
   ```
4. Verify that the Streamlit application **does not crash** or display an unhandled Python traceback.

---

### Test Scenario 5: Testing Backend Offline Resilience
1. In Terminal 1, stop the FastAPI backend by pressing `Ctrl + C`.
2. Return to the Streamlit browser window at `http://localhost:8501`.
3. Click **🚀 Analyze Food Plate**.
4. Observe the clean diagnostic error banner:
   ```
   🔌 Backend Connection Failure
   Could not connect to FastAPI backend at 'http://127.0.0.1:8000'...
   How to Fix:
   1. Open a new PowerShell terminal in the project directory:
      python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
5. Verify that the user is not left in a frozen state and is provided with the exact command to recover.
6. Restart Terminal 1 (`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`) and verify connection is immediately restored.

---

### Test Scenario 6: Exporting Analysis Artifacts
1. Execute an analysis run.
2. Scroll down to **5. 📥 Export Analysis Results**.
3. Test all 3 export actions:
   - **Download Complete JSON:** Saves `<result_id>.json` containing full timing, detections, nutrition, and paths.
   - **Download Nutrition CSV:** Saves `<result_id>_summary.csv` containing itemized tabular nutrition.
   - **Download Annotated Image:** Downloads the visualized PNG bounding boxes and freshness tags.

---

## 3. Quick Automated Sanity Check

If you prefer to run a single command from PowerShell to verify all endpoints and the frontend client without opening a browser:
```powershell
python -m pytest tests/test_api.py tests/test_frontend.py -v
```
Expected result: **30 passed in <10s**.
