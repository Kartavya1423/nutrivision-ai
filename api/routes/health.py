"""
NutriVision AI — Health and System Diagnostics Endpoints
"""

import time
from typing import Any, Dict

try:
    from fastapi import APIRouter
    router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])
except ImportError:
    router = None

from src.config.settings import get_settings
from src.pipeline.pipeline import NutriVisionPipeline


def get_health_data() -> Dict[str, Any]:
    cfg = get_settings()
    pipeline = NutriVisionPipeline(settings=cfg)
    diag = pipeline.get_pipeline_status()
    return {
        "status": "online",
        "project": cfg.project_name,
        "version": cfg.version,
        "device": cfg.resolve_device(),
        "models_ready": diag["all_ready"],
        "diagnostics": diag,
        "timestamp": time.time(),
    }


if router is not None:
    @router.get("/")
    def health_check():
        """Basic service liveness check."""
        return get_health_data()

    @router.get("/diagnostics")
    def diagnostics():
        """Detailed status of all model backends and checkpoints."""
        return get_health_data()
