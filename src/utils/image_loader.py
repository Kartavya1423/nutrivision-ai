"""
NutriVision AI — Image Loader & Validator
Supports: JPG, JPEG, PNG, WEBP.
Validates:
- Supported format / extension
- Corrupted images
- Minimum and maximum dimensions (resolution limit)
Standardizes inputs to RGB NumPy arrays and Pillow Images.
"""

from io import BytesIO
from pathlib import Path
from typing import Set, Tuple, Union
import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MAX_DIMENSION: int = 4096
DEFAULT_MIN_DIMENSION: int = 10


class ImageValidationError(ValueError):
    """Raised when an image fails format, corruption, or resolution checks."""
    pass


def validate_image_path(path: Union[str, Path]) -> Path:
    """Validate file existence and extension."""
    p = Path(path)
    if not p.is_file():
        raise ImageValidationError(f"Image file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported image extension '{p.suffix}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return p


def load_and_validate_image(
    source: Union[str, Path, bytes, np.ndarray, Image.Image],
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    min_dimension: int = DEFAULT_MIN_DIMENSION,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Load and thoroughly validate an image.
    
    Args:
        source: Filepath (str/Path), raw bytes, numpy array, or PIL Image.
        max_dimension: Maximum allowed width or height in pixels.
        min_dimension: Minimum allowed width or height in pixels.
        
    Returns:
        Tuple[np.ndarray, Image.Image]: (RGB numpy array uint8, RGB PIL Image)
        
    Raises:
        ImageValidationError: If format unsupported, image corrupt, or dimensions invalid.
    """
    pil_img: Image.Image

    if isinstance(source, (str, Path)):
        p = validate_image_path(source)
        try:
            # Check for corruption via verify
            with Image.open(p) as test_img:
                test_img.verify()
            # Open actual image data
            pil_img = Image.open(p).convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise ImageValidationError(f"Corrupted or unreadable image file '{p}': {str(e)}")

    elif isinstance(source, bytes):
        try:
            with Image.open(BytesIO(source)) as test_img:
                test_img.verify()
            pil_img = Image.open(BytesIO(source)).convert("RGB")
        except Exception as e:
            raise ImageValidationError(f"Corrupted byte stream image: {str(e)}")

    elif isinstance(source, Image.Image):
        pil_img = source.convert("RGB")

    elif isinstance(source, np.ndarray):
        arr = source.copy()
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ImageValidationError(f"Invalid array shape for RGB image: {arr.shape}")

        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)

    else:
        raise ImageValidationError(f"Unsupported image input type: {type(source)}")

    w, h = pil_img.size

    # Dimension validation
    if w < min_dimension or h < min_dimension:
        raise ImageValidationError(
            f"Image dimensions too small: ({w}x{h}). Minimum required: {min_dimension}x{min_dimension}"
        )

    if w > max_dimension or h > max_dimension:
        raise ImageValidationError(
            f"Image dimensions exceed maximum resolution: ({w}x{h}). Maximum allowed: {max_dimension}x{max_dimension}"
        )

    np_arr = np.array(pil_img, dtype=np.uint8)
    return np_arr, pil_img
