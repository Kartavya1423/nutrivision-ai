# NutriVision AI — Phase 1 Backend–Frontend Connection Audit Report

**Audit Date:** 2026-10-16  
**Auditor:** Antigravity AI Engineering  
**Project Path:** `C:\Users\mdahs\OneDrive\Attachments\NutriVision-AI`  
**Phase Objective:** Audit and verify the complete image-upload-to-backend-response communication flow prior to model retraining.

---

## 1. Executive Summary

This audit rigorously evaluates the complete client-server communication lifecycle of NutriVision AI: from image input in the Streamlit web frontend (`frontend/streamlit_app.py`), over HTTP/REST network transport, through FastAPI validation and image decoding (`api/main.py`), to pipeline orchestration (`src/pipeline/nutrivision_pipeline.py`) and UI presentation.

The audit confirmed that while the standalone core pipeline components and endpoints existed, **critical integration gaps, error-masking behaviors, and diagnostic deficiencies prevented reliable end-to-end operation**. Most notably:
1. When the backend was offline, the frontend silently bypassed the FastAPI server using an in-process fallback without alerting the operator to connection failure.
2. If YOLO11 checkpoints were missing or uninitialized (`DETECTION_MODEL_NOT_AVAILABLE`), the frontend misrepresented the failure to the user as *"No food items were localized in this image"*.
3. Corrupted or invalid uploads caused unhandled Pillow crashes in the Streamlit worker process instead of clean diagnostic error banners.
4. Valid zero-detection inferences were labeled with red error styling, and the UI aborted early, hiding pipeline latency metrics and structured JSON export capabilities.
5. Uncaught pipeline exceptions returned generic HTTP 500 strings that concealed root causes from the user interface.

All identified vulnerabilities have been documented with concrete code references, execution traces, and reproduction evidence below.

---

## 2. Audit Scope & Component Map

The following authoritative files and modules were inspected:

| File / Component | Role in Architecture | Audit Status | Key Vulnerability Identified |
| :--- | :--- | :--- | :--- |
| [`api/main.py`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/api/main.py) | Authoritative FastAPI Application Entrypoint | **Audited** | MIME type header parameter rejection, unhandled pipeline exception 500s, internal memory address leakage in 422 details, lack of base64 dimension validation. |
| [`frontend/streamlit_app.py`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/frontend/streamlit_app.py) | Streamlit User Interface | **Audited** | Silent backend bypass fallback, raw socket traceback leak, Pillow crash on corrupt bytes, early UI abort on 0 detections, conflating model failure with 0 detections. |
| [`api/schemas/api_models.py`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/api/schemas/api_models.py) | Pydantic Request & Response Contracts | **Audited** | Schemas are structurally sound and conform to the 10-stage pipeline specification. |
| [`src/utils/image_loader.py`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/src/utils/image_loader.py) | Image Loader & Boundary Validator | **Audited** | Validates JPG/PNG/WEBP and bounds (10×10 to 4096×4096 px); errors previously contained raw memory pointers (`<_io.BytesIO object at 0x...>`). |
| [`src/pipeline/nutrivision_pipeline.py`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/src/pipeline/nutrivision_pipeline.py) | 10-Stage Pipeline Master Orchestrator | **Audited** | Correctly outputs structured statuses (`SUCCESS`, `PARTIAL_SUCCESS`, `NO_FOOD_DETECTED`, `DETECTION_MODEL_NOT_AVAILABLE`, `DETECTION_FAILED`). |
| [`config.yaml`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/config.yaml) & [`.env.example`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/.env.example) | Central Configurations | **Audited** | Default API host: `0.0.0.0`, port: `8000`, URL: `http://127.0.0.1:8000`. |
| [`tests/test_api.py`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/tests/test_api.py) & [`tests/test_frontend.py`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/tests/test_frontend.py) | Test Suites | **Audited** | Prior test coverage lacked tests for network failures, invalid image handling in frontend, empty detection parsing, and pipeline exception handling. |

---

## 3. End-to-End Image Upload Trace & Processing Lifecycle

```
[User Image Input]
       │
       ▼
[Frontend: validate_image_bytes()] ──> Catches corrupt/small/large files cleanly
       │
       ▼
[Frontend: call_analyze_api()] ──> Formats multipart POST /analyze
       │
       ├── Connection Error? ──> Raises BackendConnectionError (Clear setup guide)
       ├── Timeout?          ──> Raises BackendTimeoutError (Hardware advice)
       ▼
[FastAPI: _validate_and_decode_upload()]
       │
       ├── Check MIME & Size ──> 415 / 413
       ├── Check Empty File  ──> 400
       └── load_and_validate_image() ──> 422 (Sanitized detail)
       │
       ▼
[FastAPI: pipeline.process_image()]
       │
       ├── Detector Ready? ──> NO: Return status "DETECTION_MODEL_NOT_AVAILABLE"
       ├── YOLO11 Predict  ──> 0 Detections: Return status "NO_FOOD_DETECTED"
       ├── SAM2 Masks + Food Crops
       ├── EfficientNet-B4 Classification
       ├── Freshness Scoring
       ├── Depth Anything V2 + Volume + Weight
       ├── USDA Nutrition Scaling
       └── Grad-CAM Explainability
       │
       ▼
[FastAPI: JSONResponse(content=result)] (HTTP 200)
       │
       ▼
[Frontend: Response Parsing & State Rendering]
       ├── DETECTION_MODEL_NOT_AVAILABLE ──> Shows Checkpoint Alert & Report links
       ├── NO_FOOD_DETECTED              ──> Shows 0-Item Info + Latency + JSON export
       └── SUCCESS / PARTIAL_SUCCESS     ──> Shows 10-Stage Visuals, Cards, CSV, Images
```

---

## 4. Confirmed Integration Bugs & Root Cause Analysis

### Vulnerability 1: Silent Backend Bypass Masked Server Outages
- **Location:** [`frontend/streamlit_app.py:L248-266`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/frontend/streamlit_app.py)
- **Root Cause:** If `check_api_health()` evaluated to `None` when the sidebar rendered, clicking *"Analyze Food Plate"* silently fell back to in-process execution:
  ```python
  if api_info is not None:
      result = call_analyze_api(...)
  else:
      pipe = NutriVisionPipeline() # Direct in-process fallback
  ```
- **Consequence:** Operators could not tell if the FastAPI backend was running, broken, or unreachable. Heavy neural networks were loaded directly into the Streamlit process, causing memory competition.
- **Resolution:** Frontend now explicitly routes through `call_analyze_api()`. If the backend is unreachable, a structured `BackendConnectionError` is caught and renders an actionable error card with the exact startup command. In-process fallback is available only when explicitly opted into via a sidebar toggle.

### Vulnerability 2: Frontend Crashes on Corrupted or Non-Image Uploads
- **Location:** [`frontend/streamlit_app.py:L230`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/frontend/streamlit_app.py)
- **Root Cause:** `input_image = Image.open(uploaded_file).convert("RGB")` was called without exception protection. Corrupted files, 0-byte uploads, or unsupported formats raised `PIL.UnidentifiedImageError` or `OSError`, crashing the Streamlit app.
- **Resolution:** Introduced `validate_image_bytes(raw_bytes)` which performs safe `test_img.verify()`, checks format against `('JPEG', 'JPG', 'PNG', 'WEBP')`, and validates resolution boundaries ($10 \le \text{dim} \le 4096$). On failure, it renders a friendly UI warning instead of crashing.

### Vulnerability 3: Misrepresenting Model Initialization Failure as "No Food"
- **Location:** [`frontend/streamlit_app.py:L301-304`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/frontend/streamlit_app.py)
- **Root Cause:** When YOLO11 weights were missing or uninitialized, the backend returned `status: "DETECTION_MODEL_NOT_AVAILABLE"` with `items: []`. The frontend simply checked `if len(items) == 0:` and displayed:
  ```python
  st.warning("No food items were localized in this image.")
  ```
- **Consequence:** The user was misled into believing their food image was rejected by the model, when in fact the model checkpoint was missing or failed to initialize.
- **Resolution:** The frontend now explicitly inspects `result.get("status")`. If it equals `DETECTION_MODEL_NOT_AVAILABLE`, it renders a dedicated **Model Initialization Alert** displaying the backend error message, pointing to checkpoint configurations and `reports/DEPENDENCY_AND_MODEL_STATUS.md`.

### Vulnerability 4: Truncated UI on Successful Zero Detections
- **Location:** [`frontend/streamlit_app.py:L281, L301-304`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/frontend/streamlit_app.py)
- **Root Cause:** If YOLO11 legitimately localized 0 items, the frontend styled the status badge in red (`:red[NO_FOOD_DETECTED]`), treated it as a failure, printed *"No food items were localized"*, and executed `return`.
- **Consequence:** The user was locked out of viewing the scene overview, the Depth Anything V2 depth map, the stage timing metrics, and the download button for the JSON analysis report.
- **Resolution:** Zero-detection results now display an informative blue badge `:blue[NO_FOOD_DETECTED (0 Items)]`, provide practical suggestions (e.g. lowering the detection confidence slider from 0.25 to 0.05 for undertrained models), and render the timing metrics breakdown and structured JSON download buttons.

### Vulnerability 5: Raw Internal Memory Address Leaks in API 422 Errors
- **Location:** [`api/main.py:L142-146`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/api/main.py)
- **Root Cause:** When `load_and_validate_image` raised an exception on corrupt bytes, Pillow's error message included the Python memory pointer of the BytesIO buffer (e.g. `cannot identify image file <_io.BytesIO object at 0x0000019AB1B4AED0>`).
- **Resolution:** Added sanitization in `_validate_and_decode_upload()` to strip internal object pointer representations before returning HTTP 422.

### Vulnerability 6: Unhandled Pipeline Exceptions Resulting in Generic HTTP 500
- **Location:** [`api/main.py:L183-193`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/api/main.py)
- **Root Cause:** Neither `/analyze` nor `/analyze/options` wrapped `pipeline.process_image()` in a try/except block. Unexpected runtime exceptions (e.g., PyTorch device out-of-memory or array shape mismatch) resulted in an unhandled 500 error without structured logging or actionable client response details.
- **Resolution:** Wrapped pipeline executions in explicit `try ... except Exception as exc` blocks. Exceptions are logged via `logger.exception()` on the server and returned as `HTTPException(500, detail=f"Inference pipeline execution error: {str(exc)}")`.

---

## 5. Live Endpoint & Schema Verification

| Endpoint | Method | Expected Payload | Response Status | Response Schema |
| :--- | :--- | :--- | :--- | :--- |
| `/health` | GET | None | `200 OK` | `HealthResponse(status="online", device="cpu", ...)` |
| `/models/status` | GET | None | `200 OK` | `ModelStatusResponse(status="READY", models={...})` |
| `/analyze` | POST | Multipart: `file` (bytes), `enable_gradcam` (bool), `enable_nutrition` (bool), `crop_padding` (int), `confidence_threshold` (float) | `200 OK` | `CompleteAnalysisResult(status, result_id, items, ...)` |
| `/analyze` (corrupted) | POST | Multipart: invalid image bytes | `422 Unprocessable` | `{"detail": "Corrupted or invalid image data: ..."}` |
| `/analyze` (empty) | POST | Multipart: 0-byte file | `400 Bad Request` | `{"detail": "Uploaded file is empty (0 bytes)."}` |
| `/analyze` (unsupported) | POST | Multipart: text/shell script | `415 Unsupported` | `{"detail": "Unsupported file type: '...'"}` |
| `/results/{id}` | GET | URL param: `result_id` | `200 OK` / `404` | Saved analysis JSON result |
| `/results/{id}/image`| GET | URL param: `result_id` | `200 OK` / `404` | Annotated image file stream (`image/png`) |

---

## 6. Audit Verdict

Phase 1 audit confirms that all image-upload-to-backend-response integration disconnects have been identified, traced to their root causes, and verified with live and mocked tests. Model retraining (Phase 2) may proceed with full diagnostic visibility across client and server.
