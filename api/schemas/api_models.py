"""
NutriVision AI — Pydantic API Schemas (Phase 8)
Defines strongly-typed request and response contracts for all FastAPI endpoints.
Strictly conforms to Section 8.3 & 8.4 structured pipeline results.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = Field(default="online", description="Service health status")
    project: str = Field(default="NutriVision AI", description="Project title")
    version: str = Field(default="0.1.0", description="API version")
    device: str = Field(default="cpu", description="Hardware device running models")
    timestamp: float = Field(default=0.0, description="Unix timestamp of health check")
    models_ready: bool = Field(default=True, description="Backwards compatibility flag")


class ModelStatusItem(BaseModel):
    model_name: str
    status: str
    is_ready: bool
    details: Optional[Dict[str, Any]] = None


class SystemDiagnosticsResponse(BaseModel):
    timestamp: float
    system_status: str
    models: Dict[str, Any]


class AnalyzeImageRequest(BaseModel):
    image_base64: str
    save_artifacts: bool = False
    crop_padding: int = 10


class NutritionLookupRequest(BaseModel):
    food_name: str
    weight_grams: float = 100.0


class NutritionResponse(BaseModel):
    item_name: str
    weight_grams: float
    calories: float
    macronutrients: Dict[str, float]
    micronutrients: Dict[str, float]
    source: str


class ModelDetail(BaseModel):
    name: str
    checkpoint: Optional[str] = None
    loaded: bool
    ready: bool
    status: str
    device: str
    model_version: Optional[str] = None
    error: Optional[str] = None


class ModelStatusResponse(BaseModel):
    status: str = Field(..., description="Overall model readiness: READY or PARTIALLY_READY")
    device: str
    cuda_available: bool
    cuda_device_name: Optional[str] = None
    total_models: int
    ready_models_count: int
    models: Dict[str, ModelDetail]


class DetectionInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    class_name: str = Field(..., alias="class")
    confidence: float
    bbox: List[int]


class SegmentationInfo(BaseModel):
    mask_available: bool
    mask_area: int
    crop_path: Optional[str] = None


class TopKCandidate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    class_name: str = Field(..., alias="class")
    confidence: float


class ClassificationInfo(BaseModel):
    label: str
    confidence: float
    label_origin: str = "classifier"  # "classifier", "detector", "user_corrected"
    top_k: List[TopKCandidate] = []
    status: str
    model_version: Optional[str] = None
    support_tier: Optional[str] = None
    alternative_suggestions: Optional[List[Dict[str, Any]]] = None


class FreshnessInfo(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float] = {}
    status: str


class DepthInfo(BaseModel):
    status: str
    calibration: str
    depth_map_path: Optional[str] = None


class VolumeInfo(BaseModel):
    value_cm3: Optional[float] = None
    status: str


class WeightInfo(BaseModel):
    value_g: Optional[float] = None
    status: str
    source: str = "vision_estimated"
    label: str = "Vision Estimated Portion"
    monocular_depth_disclaimer: Optional[str] = (
        "Physical weight is an empirical estimate calibrated from 2D contour area "
        "and reference food density. Monocular depth provides relative surface relief only, "
        "not absolute physical mass or scale."
    )


class NutritionInfo(BaseModel):
    status: str
    serving_size_g: float = 100.0
    serving_weight_g: Optional[float] = None
    calories_kcal: Optional[float] = None
    protein_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    fat_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    potassium_mg: Optional[float] = None
    calcium_mg: Optional[float] = None
    iron_mg: Optional[float] = None
    sugars_g: Optional[float] = None
    source: str = "USDA FoodData Central"
    matched_food: Optional[str] = None
    fdc_id: Optional[Union[int, str]] = None
    preparation_variant: Optional[str] = None
    variant_name: Optional[str] = None
    is_generic_match: bool = True
    available_variants: Optional[List[Dict[str, Any]]] = None
    weight_source: str = "user_measured"
    weight_label: str = "User-Entered Measured Weight"
    scaling_formula: str = "nutrient_scaled = nutrient_per_100g * (portion_weight_g / 100.0)"
    nutrients_per_100g: Optional[Dict[str, float]] = None


class ExplainabilityInfo(BaseModel):
    status: str
    food_gradcam_path: Optional[str] = None
    freshness_gradcam_path: Optional[str] = None
    food_target_layer: Optional[str] = None
    freshness_target_layer: Optional[str] = None


class FoodItemSchema(BaseModel):
    food: str
    label_origin: str = "classifier"
    detection: DetectionInfo
    segmentation: SegmentationInfo
    classification: ClassificationInfo
    freshness: FreshnessInfo
    depth: DepthInfo
    volume: VolumeInfo
    weight: WeightInfo
    nutrition: NutritionInfo
    explainability: ExplainabilityInfo


class MealSummarySchema(BaseModel):
    total_items: int
    calculated_items_count: int
    unavailable_items_count: int
    total_weight_g: float
    total_calories_kcal: float
    total_protein_g: float
    total_carbohydrates_g: float
    total_fat_g: float
    total_fiber_g: float
    total_sodium_mg: Optional[float] = None
    source: str = "USDA FoodData Central"
    unavailable_foods: List[str] = []


class TimingSchema(BaseModel):
    detection_ms: float = 0.0
    segmentation_ms: float = 0.0
    classification_ms: float = 0.0
    freshness_ms: float = 0.0
    depth_ms: float = 0.0
    volume_ms: float = 0.0
    weight_ms: float = 0.0
    nutrition_ms: float = 0.0
    gradcam_ms: float = 0.0
    total_ms: float = 0.0


class CompleteAnalysisResult(BaseModel):
    status: str
    result_id: str
    timestamp: str
    source_image: str
    device: str
    image_dimensions: Dict[str, int]
    total_foods_detected: int
    items: List[FoodItemSchema] = []
    meal_summary: Optional[MealSummarySchema] = None
    visualizations: Dict[str, Optional[str]] = {}
    timing: TimingSchema
    errors: List[str] = []
