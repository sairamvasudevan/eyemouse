"""MediaPipe Face Landmarker wrapper operating in sequential VIDEO mode."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .resources import get_face_landmarker_path


class LandmarkerError(RuntimeError):
    """Raised when landmarker initialization or inference fails."""
    pass


class FaceLandmarkerWrapper:
    """Wrapper managing MediaPipe FaceLandmarker lifecycle in VIDEO mode."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: vision.RunningMode = vision.RunningMode.VIDEO,
    ) -> None:
        if model_path is None:
            model_path = get_face_landmarker_path()
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise LandmarkerError(f"Model file not found: {self.model_path}")

        self.num_faces = num_faces
        self.running_mode = running_mode
        self._last_timestamp_ms = -1

        base_options = BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker: Optional[vision.FaceLandmarker] = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        """Release native MediaPipe C++ resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> FaceLandmarkerWrapper:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        timestamp_ms: int,
    ) -> Tuple[int, Optional[np.ndarray], Dict[str, float], Optional[np.ndarray]]:
        """
        Run inference on an RGB frame with a monotonically increasing timestamp.
        
        Args:
            rgb_frame: (H, W, 3) uint8 numpy array in RGB channel order.
            timestamp_ms: Monotonically increasing timestamp in milliseconds.
            
        Returns:
            (num_faces, landmarks_normalized_478x3, blendshapes_dict, transform_matrix_4x4)
        """
        if self._landmarker is None:
            raise LandmarkerError("FaceLandmarker has been closed.")

        if self.running_mode == vision.RunningMode.VIDEO:
            if timestamp_ms <= self._last_timestamp_ms:
                # Enforce strictly monotonic timestamps required by MediaPipe VIDEO mode
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self.running_mode == vision.RunningMode.VIDEO:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        num_faces = len(result.face_landmarks) if result.face_landmarks else 0
        if num_faces == 0:
            return 0, None, {}, None

        # Primary face (index 0)
        face_lms = result.face_landmarks[0]
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in face_lms], dtype=np.float32)

        # Blendshapes dictionary
        blendshapes = {}
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            for category in result.face_blendshapes[0]:
                blendshapes[category.category_name] = float(category.score)

        # Facial transformation matrix
        matrix = None
        if result.facial_transformation_matrixes and len(result.facial_transformation_matrixes) > 0:
            matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)

        return num_faces, landmarks_array, blendshapes, matrix

