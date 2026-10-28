"""
NutriVision AI — Direct Food Detection Route
"""

try:
    from fastapi import APIRouter, File, UploadFile
    from fastapi.responses import JSONResponse
    router = APIRouter(prefix="/detect", tags=["Detection"])
except ImportError:
    router = None

from src.detection.yolo_detector import YOLO11FoodDetector
from src.utils.image_ops import load_image


if router is not None:
    @router.post("/")
    async def detect_food(file: UploadFile = File(...)):
        """Run YOLO11 food detector on uploaded image."""
        contents = await file.read()
        detector = YOLO11FoodDetector()
        np_img, _ = load_image(contents)
        result = detector.predict(np_img)
        return JSONResponse(content=result.to_dict())
