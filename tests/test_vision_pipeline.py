"""Automated tests for the VisionPipeline core inference and failure modes."""

import cv2
import numpy as np
import pytest

from src.vision_engine.pipeline import VisionPipeline
from src.vision_engine.types import PixelFormat


def create_synthetic_face_bgra(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a deterministic synthetic face frame in 32-bit BGRA format."""
    img = np.full((height, width, 4), (200, 205, 210, 255), dtype=np.uint8)
    cx, cy = width // 2, height // 2

    # Face oval (skin tone in BGRA)
    cv2.ellipse(img, (cx, cy), (90, 120), 0, 0, 360, (180, 195, 235, 255), -1)
    # Eyes
    cv2.ellipse(img, (cx - 38, cy - 15), (20, 10), 0, 0, 360, (255, 255, 255, 255), -1)
    cv2.ellipse(img, (cx + 38, cy - 15), (20, 10), 0, 0, 360, (255, 255, 255, 255), -1)
    # Irises
    cv2.circle(img, (cx - 38, cy - 15), 5, (40, 60, 90, 255), -1)
    cv2.circle(img, (cx + 38, cy - 15), 5, (40, 60, 90, 255), -1)
    # Pupils
    cv2.circle(img, (cx - 38, cy - 15), 2, (10, 10, 10, 255), -1)
    cv2.circle(img, (cx + 38, cy - 15), 2, (10, 10, 10, 255), -1)
    # Eyebrows
    cv2.ellipse(img, (cx - 38, cy - 32), (22, 5), 0, 180, 360, (30, 40, 60, 255), 3)
    cv2.ellipse(img, (cx + 38, cy - 32), (22, 5), 0, 180, 360, (30, 40, 60, 255), 3)
    # Nose
    cv2.line(img, (cx, cy - 15), (cx, cy + 20), (140, 160, 200, 255), 2)
    cv2.line(img, (cx, cy + 20), (cx + 8, cy + 20), (140, 160, 200, 255), 2)
    # Mouth
    cv2.ellipse(img, (cx, cy + 55), (28, 8), 0, 0, 180, (100, 110, 180, 255), -1)
    cv2.GaussianBlur(img, (5, 5), 1.2, dst=img)
    return img


@pytest.fixture(scope="module")
def pipeline():
    pipe = VisionPipeline()
    yield pipe
    pipe.close()


def test_face_present_inference(pipeline):
    """Verify that a valid face produces complete landmarks, pose, and eye metrics."""
    width, height = 640, 480
    frame = create_synthetic_face_bgra(width, height)
    stride = width * 4

    output = pipeline.process_raw_frame(
        pixel_data=frame,
        width=width,
        height=height,
        stride=stride,
        pixel_format=PixelFormat.BGRA,
        sequence_num=1,
        timestamp_ns=1_000_000_000,
    )

    assert output.face_detected is True
    assert output.confidence > 0.7
    assert output.landmarks is not None
    assert len(output.landmarks) == 478
    assert output.eye_metrics is not None
    assert output.head_pose is not None
    assert output.blink_event is not None
    assert output.gaze_estimate is not None
    assert output.gaze_estimate.is_valid is True
    assert output.processing_time_ms > 0.0


def test_face_absent_returns_explicit_invalid_state(pipeline):
    """Verify face loss returns explicit unpopulated data, never fabricating coordinates."""
    width, height = 640, 480
    # Blank plain gray background without any facial features
    blank_frame = np.full((height, width, 4), (180, 180, 180, 255), dtype=np.uint8)
    stride = width * 4

    output = pipeline.process_raw_frame(
        pixel_data=blank_frame,
        width=width,
        height=height,
        stride=stride,
        pixel_format=PixelFormat.BGRA,
        sequence_num=2,
        timestamp_ns=1_033_333_333,
    )

    assert output.face_detected is False
    assert output.confidence == 0.0
    assert output.landmarks is None
    assert output.eye_metrics is None
    assert output.head_pose is None
    assert output.gaze_estimate is None


def test_invalid_pixel_format_handling(pipeline):
    """Verify unsupported pixel formats raise an explicit ValueError."""
    data = np.zeros(640 * 480 * 4, dtype=np.uint8)
    with pytest.raises(ValueError) as exc:
        pipeline.process_raw_frame(
            pixel_data=data,
            width=640,
            height=480,
            stride=2560,
            pixel_format=PixelFormat.UNKNOWN,
            sequence_num=3,
            timestamp_ns=1_066_666_666,
        )
    assert "Unsupported pixel format" in str(exc.value)

