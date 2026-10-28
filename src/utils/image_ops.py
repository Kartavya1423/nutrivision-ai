"""
NutriVision AI — Image Utilities
Robust conversions between NumPy, Pillow, filepaths, and base64 strings.
Always operates in RGB color space by default.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Tuple, Union
import numpy as np
from PIL import Image


def load_image(image_input: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[np.ndarray, Image.Image]:
    """
    Standardize any image input into both a NumPy array (H, W, 3) RGB and a PIL Image.
    
    Args:
        image_input: Filepath (str/Path), numpy array, or PIL Image.
        
    Returns:
        Tuple[np.ndarray, Image.Image]: (RGB numpy array uint8, RGB PIL Image)
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            raise FileNotFoundError(f"Image file not found: {p}")
        pil_img = Image.open(p).convert("RGB")
        np_img = np.array(pil_img, dtype=np.uint8)
        return np_img, pil_img

    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        np_img = np.array(pil_img, dtype=np.uint8)
        return np_img, pil_img

    if isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        if arr.ndim == 2:
            # Grayscale -> RGB
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            # RGBA -> RGB
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        pil_img = Image.fromarray(arr)
        return arr, pil_img

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def image_to_base64(image: Union[np.ndarray, Image.Image], format: str = "PNG") -> str:
    """Encode an image array or PIL Image to a base64 data string."""
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8))
    else:
        pil_img = image

    buffer = BytesIO()
    pil_img.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def base64_to_image(b64_str: str) -> Tuple[np.ndarray, Image.Image]:
    """Decode a base64 string to (RGB numpy array, PIL Image)."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    pil_img = Image.open(BytesIO(data)).convert("RGB")
    np_img = np.array(pil_img, dtype=np.uint8)
    return np_img, pil_img
