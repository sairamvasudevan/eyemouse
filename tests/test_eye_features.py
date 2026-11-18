"""Automated tests for normalized eye feature extraction and geometric invariance."""

import math
import numpy as np
import pytest

from src.vision_engine.eye_features import (
    extract_eye_metrics,
    _normalize_iris_to_corners,
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    LEFT_EYE_UPPER,
    LEFT_EYE_LOWER,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    RIGHT_EYE_UPPER,
    RIGHT_EYE_LOWER,
    RIGHT_IRIS_CENTER,
)


def test_iris_normalization_centered():
    """Verify that a centered iris produces iris_x = 0.5 and iris_y = 0.0."""
    inner = np.array([100.0, 100.0])
    outer = np.array([140.0, 100.0])  # Width = 40 px
    iris = np.array([120.0, 100.0])   # Exactly mid-point
    upper = np.array([120.0, 95.0])
    lower = np.array([120.0, 105.0])

    norm = _normalize_iris_to_corners(iris, inner, outer, upper, lower)
    assert pytest.approx(norm.iris_x, abs=1e-4) == 0.5
    assert pytest.approx(norm.iris_y, abs=1e-4) == 0.0
    assert pytest.approx(norm.openness, abs=1e-4) == 0.25  # 10 / 40


def test_iris_normalization_roll_invariance():
    """Verify that rotating the head (roll) does not change normalized iris offsets."""
    inner = np.array([100.0, 100.0])
    outer = np.array([140.0, 100.0])
    iris = np.array([124.0, 102.0])  # slightly off-center
    upper = np.array([120.0, 95.0])
    lower = np.array([120.0, 105.0])

    base_norm = _normalize_iris_to_corners(iris, inner, outer, upper, lower)

    # Rotate all points by 25 degrees around inner corner
    theta = math.radians(25.0)
    c, s = math.cos(theta), math.sin(theta)
    rot_mat = np.array([[c, -s], [s, c]])

    def rotate_pt(pt):
        return inner + rot_mat @ (pt - inner)

    rot_outer = rotate_pt(outer)
    rot_iris = rotate_pt(iris)
    rot_upper = rotate_pt(upper)
    rot_lower = rotate_pt(lower)

    rot_norm = _normalize_iris_to_corners(rot_iris, inner, rot_outer, rot_upper, rot_lower)

    assert pytest.approx(rot_norm.iris_x, abs=1e-4) == base_norm.iris_x
    assert pytest.approx(rot_norm.iris_y, abs=1e-4) == base_norm.iris_y
    assert pytest.approx(rot_norm.openness, abs=1e-4) == base_norm.openness


def test_iris_normalization_scale_invariance():
    """Verify that user distance changes (scaling) do not change normalized iris offsets."""
    inner = np.array([100.0, 100.0])
    outer = np.array([140.0, 100.0])  # width 40
    iris = np.array([124.0, 102.0])
    upper = np.array([120.0, 95.0])
    lower = np.array([120.0, 105.0])

    base_norm = _normalize_iris_to_corners(iris, inner, outer, upper, lower)

    # Scale by 2.0x (user moves closer to camera)
    scale = 2.0
    scale_outer = inner + (outer - inner) * scale
    scale_iris = inner + (iris - inner) * scale
    scale_upper = inner + (upper - inner) * scale
    scale_lower = inner + (lower - inner) * scale

    scale_norm = _normalize_iris_to_corners(scale_iris, inner, scale_outer, scale_upper, scale_lower)

    assert pytest.approx(scale_norm.iris_x, abs=1e-4) == base_norm.iris_x
    assert pytest.approx(scale_norm.iris_y, abs=1e-4) == base_norm.iris_y
    assert pytest.approx(scale_norm.openness, abs=1e-4) == base_norm.openness


def test_extract_eye_metrics_from_full_array():
    """Verify extraction across the 478 landmark array."""
    landmarks = np.zeros((478, 3), dtype=np.float32)

    # Left eye setup
    landmarks[LEFT_EYE_INNER] = [300.0, 200.0, 0.0]
    landmarks[LEFT_EYE_OUTER] = [340.0, 200.0, 0.0]
    landmarks[LEFT_EYE_UPPER] = [320.0, 195.0, 0.0]
    landmarks[LEFT_EYE_LOWER] = [320.0, 205.0, 0.0]
    landmarks[LEFT_IRIS_CENTER] = [320.0, 200.0, 0.0]

    # Right eye setup
    landmarks[RIGHT_EYE_INNER] = [260.0, 200.0, 0.0]
    landmarks[RIGHT_EYE_OUTER] = [220.0, 200.0, 0.0]
    landmarks[RIGHT_EYE_UPPER] = [240.0, 195.0, 0.0]
    landmarks[RIGHT_EYE_LOWER] = [240.0, 205.0, 0.0]
    landmarks[RIGHT_IRIS_CENTER] = [240.0, 200.0, 0.0]

    metrics = extract_eye_metrics(landmarks)
    assert metrics is not None
    assert len(metrics.gaze_feature_vector) == 4
    assert pytest.approx(metrics.left_normalized.iris_x, abs=1e-3) == 0.5
    assert pytest.approx(metrics.right_normalized.iris_x, abs=1e-3) == 0.5


def test_conjugate_gaze_coordinate_consistency():
    """Verify that looking right or looking down shifts both eyes with identical sign and magnitude."""
    landmarks = np.zeros((478, 3), dtype=np.float32)

    # Left eye: nasal corner (362) at x=300, temporal corner (263) at x=340. Width = 40.
    landmarks[LEFT_EYE_INNER] = [300.0, 200.0, 0.0]
    landmarks[LEFT_EYE_OUTER] = [340.0, 200.0, 0.0]
    landmarks[LEFT_EYE_UPPER] = [320.0, 195.0, 0.0]
    landmarks[LEFT_EYE_LOWER] = [320.0, 205.0, 0.0]

    # Right eye: temporal corner (33) at x=220, nasal corner (133) at x=260. Width = 40.
    landmarks[RIGHT_EYE_INNER] = [260.0, 200.0, 0.0]
    landmarks[RIGHT_EYE_OUTER] = [220.0, 200.0, 0.0]
    landmarks[RIGHT_EYE_UPPER] = [240.0, 195.0, 0.0]
    landmarks[RIGHT_EYE_LOWER] = [240.0, 205.0, 0.0]

    # Subject looks right: both irises shift +10 px in image X.
    # Subject looks down: both irises shift +5 px in image Y.
    landmarks[LEFT_IRIS_CENTER] = [320.0 + 10.0, 200.0 + 5.0, 0.0]
    landmarks[RIGHT_IRIS_CENTER] = [240.0 + 10.0, 200.0 + 5.0, 0.0]

    metrics = extract_eye_metrics(landmarks)
    assert metrics is not None

    # Both eyes must increase in norm_x from 0.50 to 0.75 (+0.25)
    assert pytest.approx(metrics.left_normalized.iris_x, abs=1e-4) == 0.75
    assert pytest.approx(metrics.right_normalized.iris_x, abs=1e-4) == 0.75

    # Both eyes must increase in norm_y from 0.0 to +0.125
    assert pytest.approx(metrics.left_normalized.iris_y, abs=1e-4) == 0.125
    assert pytest.approx(metrics.right_normalized.iris_y, abs=1e-4) == 0.125

