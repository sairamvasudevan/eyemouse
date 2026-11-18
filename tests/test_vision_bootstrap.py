"""Bootstrap validation test for MediaPipe FaceLandmarker on Apple Silicon ARM64."""

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from pathlib import Path
import pytest


def generate_synthetic_face(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a deterministic synthetic face image for automated inference testing."""
    img = np.full((height, width, 3), (210, 205, 200), dtype=np.uint8)
    cx, cy = width // 2, height // 2

    # Face oval
    cv2.ellipse(img, (cx, cy), (90, 120), 0, 0, 360, (180, 195, 235), -1)
    # Eyes
    cv2.ellipse(img, (cx - 38, cy - 15), (20, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (cx + 38, cy - 15), (20, 10), 0, 0, 360, (255, 255, 255), -1)
    # Irises
    cv2.circle(img, (cx - 38, cy - 15), 5, (90, 60, 40), -1)
    cv2.circle(img, (cx + 38, cy - 15), 5, (90, 60, 40), -1)
    # Pupils
    cv2.circle(img, (cx - 38, cy - 15), 2, (10, 10, 10), -1)
    cv2.circle(img, (cx + 38, cy - 15), 2, (10, 10, 10), -1)
    # Eyebrows
    cv2.ellipse(img, (cx - 38, cy - 32), (22, 5), 0, 180, 360, (60, 40, 30), 3)
    cv2.ellipse(img, (cx + 38, cy - 32), (22, 5), 0, 180, 360, (60, 40, 30), 3)
    # Nose
    cv2.line(img, (cx, cy - 15), (cx, cy + 20), (140, 160, 200), 2)
    cv2.line(img, (cx, cy + 20), (cx + 8, cy + 20), (140, 160, 200), 2)
    # Mouth
    cv2.ellipse(img, (cx, cy + 55), (28, 8), 0, 0, 180, (100, 110, 180), -1)
    cv2.GaussianBlur(img, (5, 5), 1.2, dst=img)
    return img


def test_mediapipe_landmarker_inference():
    """Verify that MediaPipe FaceLandmarker initializes and detects 478 landmarks."""
    repo_root = Path(__file__).resolve().parent.parent
    model_path = repo_root / "assets" / "models" / "face_landmarker.task"
    assert model_path.is_file(), f"Model asset missing: {model_path}"

    base_options = BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_facial_transformation_matrixes=True,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        test_img = generate_synthetic_face()
        rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        result = landmarker.detect(mp_image)
        assert result.face_landmarks, "Expected at least one face to be detected"
        landmarks = result.face_landmarks[0]
        assert len(landmarks) == 478, f"Expected 478 landmarks, got {len(landmarks)}"

        # Verify left iris (468) and right iris (473)
        left_iris = landmarks[468]
        right_iris = landmarks[473]
        assert 0.0 < left_iris.x < 1.0
        assert 0.0 < left_iris.y < 1.0
        assert 0.0 < right_iris.x < 1.0
        assert 0.0 < right_iris.y < 1.0
        # Right eye should be to the right of left eye in image space
        assert right_iris.x > left_iris.x

