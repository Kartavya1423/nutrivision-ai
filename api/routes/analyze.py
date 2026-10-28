"""
NutriVision AI — Meal Image Analysis Route
Accepts image upload or base64 data and executes the full modular pipeline.
"""

try:
    from fastapi import APIRouter, File, Form, UploadFile
    from fastapi.responses import JSONResponse
    router = APIRouter(prefix="/analyze", tags=["Meal Analysis"])
except ImportError:
    router = None

from src.pipeline.pipeline import NutriVisionPipeline
from src.utils.image_ops import base64_to_image, load_image


if router is not None:
    @router.post("/upload")
    async def analyze_uploaded_file(
        file: UploadFile = File(...),
        save_artifacts: bool = Form(False),
        crop_padding: int = Form(10),
    ):
        """Analyze an uploaded food image file."""
        contents = await file.read()
        pipeline = NutriVisionPipeline()
        np_img, _ = load_image(contents)
        result = pipeline.analyze_image(
            image=np_img,
            save_artifacts=save_artifacts,
            crop_padding=crop_padding,
        )
        return JSONResponse(content=result.to_dict())

    @router.post("/base64")
    async def analyze_base64_image(
        payload: dict,
    ):
        """Analyze a base64-encoded image string."""
        b64_str = payload.get("image_base64")
        if not b64_str:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'image_base64' in request body."},
            )
        save_artifacts = payload.get("save_artifacts", False)
        crop_padding = payload.get("crop_padding", 10)

        pipeline = NutriVisionPipeline()
        np_img, _ = base64_to_image(b64_str)
        result = pipeline.analyze_image(
            image=np_img,
            save_artifacts=save_artifacts,
            crop_padding=crop_padding,
        )
        return JSONResponse(content=result.to_dict())
