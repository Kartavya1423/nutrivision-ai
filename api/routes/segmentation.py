"""
NutriVision AI — Direct Food Segmentation Route
"""

try:
    from fastapi import APIRouter, File, UploadFile
    from fastapi.responses import JSONResponse
    router = APIRouter(prefix="/segment", tags=["Segmentation"])
except ImportError:
    router = None

from src.segmentation.sam2_segmenter import SAM2FoodSegmenter
from src.utils.image_ops import load_image


if router is not None:
    @router.post("/")
    async def segment_food(file: UploadFile = File(...)):
        """Run SAM 2 food segmenter on uploaded image."""
        contents = await file.read()
        segmenter = SAM2FoodSegmenter()
        np_img, _ = load_image(contents)
        result = segmenter.predict_masks(np_img)
        return JSONResponse(content=result.to_dict())
