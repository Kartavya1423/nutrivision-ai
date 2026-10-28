"""NutriVision AI API Routes Module"""

from api.routes.analyze import router as analyze_router
from api.routes.detection import router as detection_router
from api.routes.health import router as health_router
from api.routes.nutrition import router as nutrition_router
from api.routes.segmentation import router as segmentation_router

__all__ = [
    "health_router",
    "analyze_router",
    "detection_router",
    "segmentation_router",
    "nutrition_router",
]
