"""Minimal standalone vision smoke-test executable for PyInstaller bundling verification."""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


def get_asset_path(filename: str) -> str:
    """Find bundled assets in PyInstaller _MEIPASS or local development tree."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_dir = Path(sys._MEIPASS)
    else:
        base_dir = Path(__file__).resolve().parent.parent
    asset_path = base_dir / "assets" / "models" / filename
    if not asset_path.is_file():
        # Fallback to direct model path
        asset_path = base_dir / filename
    if not asset_path.is_file():
        raise FileNotFoundError(f"Asset not found: {asset_path}")
    return str(asset_path)


def main():
    print("=== Frozen Vision Engine Smoke Test ===")
    print("Python Executable:", sys.executable)
    print("Is Frozen:", getattr(sys, "frozen", False))
    print("NumPy Version:", np.__version__)
    print("MediaPipe Version:", mp.__version__)
    print("OpenCV Version:", cv2.__version__)

    model_path = get_asset_path("face_landmarker.task")
    print(f"Loading Model from: {model_path}")

    # Generate synthetic face
    width, height = 640, 480
    img = np.full((height, width, 3), (210, 205, 200), dtype=np.uint8)
    cx, cy = width // 2, height // 2
    cv2.ellipse(img, (cx, cy), (90, 120), 0, 0, 360, (180, 195, 235), -1)
    cv2.ellipse(img, (cx - 38, cy - 15), (20, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (cx + 38, cy - 15), (20, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (cx - 38, cy - 15), 5, (90, 60, 40), -1)
    cv2.circle(img, (cx + 38, cy - 15), 5, (90, 60, 40), -1)
    cv2.circle(img, (cx - 38, cy - 15), 2, (10, 10, 10), -1)
    cv2.circle(img, (cx + 38, cy - 15), 2, (10, 10, 10), -1)
    cv2.ellipse(img, (cx - 38, cy - 32), (22, 5), 0, 180, 360, (60, 40, 30), 3)
    cv2.ellipse(img, (cx + 38, cy - 32), (22, 5), 0, 180, 360, (60, 40, 30), 3)
    cv2.line(img, (cx, cy - 15), (cx, cy + 20), (140, 160, 200), 2)
    cv2.line(img, (cx, cy + 20), (cx + 8, cy + 20), (140, 160, 200), 2)
    cv2.ellipse(img, (cx, cy + 55), (28, 8), 0, 0, 180, (100, 110, 180), -1)
    cv2.GaussianBlur(img, (5, 5), 1.2, dst=img)

    base_options = BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            print("ERROR: Face not detected in smoke test")
            sys.exit(1)
        landmarks = result.face_landmarks[0]
        if len(landmarks) != 478:
            print(f"ERROR: Expected 478 landmarks, got {len(landmarks)}")
            sys.exit(1)

        print(f"SUCCESS: Face detected with {len(landmarks)} landmarks.")
        print(f"Left Iris: ({landmarks[468].x:.4f}, {landmarks[468].y:.4f})")
        print(f"Right Iris: ({landmarks[473].x:.4f}, {landmarks[473].y:.4f})")
        print("FROZEN_VISION_SMOKE_TEST_SUCCESS")


if __name__ == "__main__":
    main()

