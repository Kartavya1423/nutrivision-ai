"""
NutriVision AI — Live Camera Architecture
OpenCV-based video stream interface.
Provides frames to NutriVisionPipeline.
Includes standalone camera testing function (open, display, quit safely).
Does NOT run continuous AI inference yet.
"""

from typing import Any, Dict, Generator, Optional, Tuple
import numpy as np

from src.config.settings import get_settings


class CameraStream:
    """
    Manages OpenCV camera capture and frame acquisition.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None

    def open(self) -> bool:
        """Initialize and open the video capture device."""
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.device_index)
            if not self._cap.isOpened():
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame and convert BGR -> RGB.
        Returns: (success: bool, rgb_frame: np.ndarray | None)
        """
        if not self.is_opened():
            return False, None

        import cv2
        ret, bgr_frame = self._cap.read()
        if not ret or bgr_frame is None:
            return False, None

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return True, rgb_frame

    def release(self) -> None:
        """Safely release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def test_camera_stream(device_index: int = 0, window_name: str = "NutriVision Camera Test"):
    """
    Simple camera test function:
    - Opens camera
    - Displays raw frames with instructions
    - Quits safely on 'q' or Esc
    - Does NOT run continuous AI inference yet.
    """
    try:
        import cv2
    except ImportError:
        print("[Error] OpenCV (cv2) is not installed. Install via: pip install opencv-python")
        return False

    print(f"Testing camera device index {device_index}...")
    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        print(f"[Notice] Camera device {device_index} could not be opened (no hardware webcam detected or in use).")
        return False

    print("Camera initialized successfully!")
    print("Press 'q' or 'Esc' in the camera window to quit safely.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab frame.")
                break

            # Overlay information banner
            cv2.putText(
                frame,
                "NutriVision AI - Camera Stream Test (Press 'q' to Quit)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 128),
                2,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed safely.")
    return True


if __name__ == "__main__":
    test_camera_stream()
