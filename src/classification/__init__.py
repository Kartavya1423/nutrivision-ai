"""NutriVision AI Food Classification Module"""

from src.classification.classifier import (
    BaseFoodClassifier,
    ClassificationOutput,
    ClassificationResult,
    ClassificationStatus,
)
from src.classification.efficientnet_classifier import (
    EfficientNetB4Classifier,
)
from src.classification.fruit_classifier import (
    FruitClassifier,
    FRUIT_CLASSES,
)
from src.classification.mock_classifier import MockFoodClassifier

# Alias for backward compatibility
EfficientNetFoodClassifier = EfficientNetB4Classifier

__all__ = [
    "BaseFoodClassifier",
    "ClassificationOutput",
    "ClassificationResult",
    "ClassificationStatus",
    "EfficientNetB4Classifier",
    "EfficientNetFoodClassifier",
    "FruitClassifier",
    "FRUIT_CLASSES",
    "MockFoodClassifier",
]
