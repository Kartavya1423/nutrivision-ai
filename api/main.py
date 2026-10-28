"""
NutriVision AI — FastAPI Application Entrypoint (Phase 8)
Serves the Authoritative NutriVision Endpoints:
- GET  /                     Application metadata and discovery
- GET  /health               Service health check and uptime
- GET  /models/status        Centralized model readiness diagnostics
- POST /analyze              Complete 10-stage food analysis on uploaded image
- POST /analyze/options      Configurable analysis (toggle Grad-CAM, nutrition, thresholds)
- GET  /results/{result_id}  Retrieve cached/persisted analysis report
- GET  /docs                 Interactive OpenAPI Swagger documentation
"""

import json
import logging
import sys
from pathlib import Path

# Ensure repository root is present in sys.path for direct invocation: `python api/main.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import time
from typing import Any, Dict, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import numpy as np

from api.schemas.api_models import (
    CompleteAnalysisResult,
    HealthResponse,
    ModelStatusResponse,
)
from src.config.settings import get_settings
from src.models.model_manager import get_model_manager
from src.pipeline.nutrivision_pipeline import NutriVisionPipeline
from src.utils.image_loader import load_and_validate_image
from src.utils.image_ops import base64_to_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nutrivision.api")

cfg = get_settings()

app = FastAPI(
    title=cfg.project_name,
    version=cfg.version,
    description=(
        "“NutriVision AI: An Intelligent Food Recognition, Weight Estimation, "
        "Freshness Detection, and Nutritional Analysis System Using Computer Vision”"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes.nutrition import router as nutrition_router
if nutrition_router is not None:
    app.include_router(nutrition_router)

# Global Authoritative Pipeline and Model Manager
model_manager = get_model_manager(settings=cfg)
pipeline = NutriVisionPipeline(settings=cfg)

# Allowed MIME types for image uploads
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/pjpeg",
    "image/jfif",
    "image/x-png",
    "application/octet-stream",  # Raw buffer fallbacks
}
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


@app.get("/", tags=["System"])
def root():
    """Application metadata and API discovery endpoint."""
    return {
        "project": cfg.project_name,
        "version": cfg.version,
        "status": "OPERATIONAL",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "models_status": "/models/status",
            "analyze": "/analyze",
            "results": "/results/{id}",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check and hardware acceleration reporting."""
    return HealthResponse(
        status="online",
        project=cfg.project_name,
        version=cfg.version,
        device=model_manager.device,
        timestamp=time.time(),
    )


@app.get("/models/status", response_model=ModelStatusResponse, tags=["Diagnostics"])
@app.get("/api/v1/models/status", response_model=ModelStatusResponse, tags=["Diagnostics"])
def models_status():
    """Detailed model readiness and checkpoint diagnostic report."""
    status_dict = model_manager.get_status()
    return ModelStatusResponse(**status_dict)


async def _validate_and_decode_upload(file: Optional[UploadFile], image_base64: Optional[str]) -> np.ndarray:
    """Validate uploaded file size, MIME type, and decode image into numpy RGB array."""
    if file is not None:
        if file.content_type:
            raw_mime = file.content_type.lower().split(";")[0].strip()
            if raw_mime and raw_mime not in ALLOWED_MIME_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported file type: '{file.content_type}'. Allowed types: JPG, PNG, WEBP.",
                )
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )
        if len(contents) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size ({MAX_IMAGE_SIZE_BYTES / (1024*1024):.0f} MB).",
            )
        try:
            np_img, _ = load_and_validate_image(contents)
            return np_img
        except Exception as e:
            err_msg = str(e)
            if "<" in err_msg and ">" in err_msg:
                err_msg = "The image byte stream could not be identified or decoded as a valid image."
            raise HTTPException(
                status_code=422,
                detail=f"Corrupted or invalid image data: {err_msg}",
            )
    elif image_base64:
        try:
            np_img, _ = base64_to_image(image_base64)
            np_img, _ = load_and_validate_image(np_img)
            return np_img
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid base64 encoded image: {str(e)}",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image provided. Upload a file or provide image_base64.",
        )


@app.post(
    "/analyze",
    response_model=CompleteAnalysisResult,
    tags=["Analysis"],
    summary="Execute Complete NutriVision Analysis",
)
@app.post("/api/v1/analyze/image", response_model=CompleteAnalysisResult, tags=["Analysis"])
async def analyze_food_image(
    file: Optional[UploadFile] = File(None, description="Image file (JPG, PNG, WEBP)"),
    image_base64: Optional[str] = Form(None, description="Optional Base64 encoded image"),
    enable_gradcam: bool = Form(True, description="Generate Grad-CAM visual heatmaps"),
    enable_nutrition: bool = Form(True, description="Query USDA FoodData Central for nutritional scaling"),
    crop_padding: int = Form(10, description="Pixel padding around food-only crop"),
    confidence_threshold: Optional[float] = Form(None, description="Detection confidence threshold"),
    classifier_mode: Optional[str] = Form(None, description="Classifier selection: 'fruit_classifier' or 'efficientnet_b4'"),
):
    """
    Execute the authoritative 10-stage NutriVision computer vision & nutrition analysis:
    YOLO11 -> SAM2 -> Crop -> EfficientNet-B4 / FruitClassifier -> Freshness -> Depth -> Volume -> Weight -> USDA -> Grad-CAM.
    """
    np_img = await _validate_and_decode_upload(file=file, image_base64=image_base64)
    try:
        result = pipeline.process_image(
            image=np_img,
            enable_gradcam=enable_gradcam,
            enable_nutrition=enable_nutrition,
            crop_padding=crop_padding,
            confidence_threshold=confidence_threshold,
            classifier_mode=classifier_mode,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("Pipeline execution error in /analyze: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(exc)}",
        )


@app.post(
    "/analyze/options",
    response_model=CompleteAnalysisResult,
    tags=["Analysis"],
    summary="Execute NutriVision Analysis with Query Options",
)
async def analyze_with_options(
    file: UploadFile = File(..., description="Image file"),
    enable_gradcam: bool = Query(True),
    enable_nutrition: bool = Query(True),
    crop_padding: int = Query(10),
    confidence_threshold: Optional[float] = Query(None),
    classifier_mode: Optional[str] = Query(None),
):
    """Analyze image with URL query parameters for fine-grained workflow configuration."""
    np_img = await _validate_and_decode_upload(file=file, image_base64=None)
    try:
        result = pipeline.process_image(
            image=np_img,
            enable_gradcam=enable_gradcam,
            enable_nutrition=enable_nutrition,
            crop_padding=crop_padding,
            confidence_threshold=confidence_threshold,
            classifier_mode=classifier_mode,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("Pipeline execution error in /analyze/options: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(exc)}",
        )


@app.post(
    "/classify/fruit",
    tags=["Fruit Recognition"],
    summary="Direct 6-Class Fruit Classification & Nutrition",
)
@app.post(
    "/analyze/fruit",
    tags=["Fruit Recognition"],
    summary="Direct 6-Class Fruit Classification & Nutrition",
)
async def classify_fruit_endpoint(
    file: Optional[UploadFile] = File(None, description="Fruit image file (JPG, PNG, WEBP)"),
    image_base64: Optional[str] = Form(None, description="Optional Base64 encoded image"),
    serving_weight_g: float = Form(150.0, description="Portion weight in grams for USDA nutrition scaling"),
):
    """
    Direct fruit recognition endpoint for the updated 6-class fruit dataset.
    Returns structured output with fruit_name, dataset_class, freshness_status,
    confidence, classification_model, and scaled USDA nutrition metrics.
    """
    np_img = await _validate_and_decode_upload(file=file, image_base64=image_base64)
    try:
        from src.classification.fruit_classifier import FruitClassifier
        fc = model_manager.fruit_classifier
        pred = fc.predict_image_path(np_img, weight_grams=serving_weight_g, enable_nutrition=True)
        return JSONResponse(content=pred)
    except Exception as exc:
        logger.exception("Error in /classify/fruit: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fruit recognition error: {str(exc)}",
        )


@app.get(
    "/results/{result_id}",
    tags=["Analysis"],
    summary="Retrieve Analysis Result by ID",
)
def get_analysis_result(result_id: str):
    """Retrieve previously saved structured result JSON from server storage."""
    clean_id = Path(result_id).stem
    result_path = pipeline.results_dir / f"{clean_id}.json"
    if not result_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis result ID '{result_id}' not found.",
        )
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read result file: {str(e)}",
        )


@app.get(
    "/results/{result_id}/image",
    tags=["Analysis"],
    summary="Retrieve Annotated Image Visualization",
)
def get_annotated_image(result_id: str):
    """Retrieve generated annotated visualization image."""
    clean_id = Path(result_id).stem
    img_path = pipeline.viz_dir / f"{clean_id}_annotated.png"
    if not img_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotated visualization for '{result_id}' not found.",
        )
    return FileResponse(path=str(img_path), media_type="image/png")


def create_app():
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
