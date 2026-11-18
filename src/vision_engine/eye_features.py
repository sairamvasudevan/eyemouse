"""Normalized eye feature and iris geometry extraction from MediaPipe landmarks.

Coordinate Conventions and Landmark Index Map:
----------------------------------------------
MediaPipe Face Landmarker assigns 478 3D landmarks (0..467: face mesh, 468..477: irises).
Landmarks are indexed from the subject's anatomical perspective:

Subject's Left Eye (appears on image right in unmirrored frame):
    - Nasal corner (inner): 362
    - Temporal corner (outer): 263
    - Upper eyelid: 386
    - Lower eyelid: 374
    - Iris center: 473
    - Iris boundary: [474, 475, 476, 477]

Subject's Right Eye (appears on image left in unmirrored frame):
    - Temporal corner (outer): 33
    - Nasal corner (inner): 133
    - Upper eyelid: 159
    - Lower eyelid: 145
    - Iris center: 468
    - Iris boundary: [469, 470, 471, 472]

Iris Socket Normalization:
--------------------------
Let inner corner = A, outer corner = B.
The axis vector V = B - A spans the physical eye opening.
Width W = ||V||.
Projecting the iris center I onto V yields:
    norm_x = ((I - A) . V) / (W^2)
    norm_y = (V_x * (I_y - A_y) - V_y * (I_x - A_x)) / (W^2)

Properties:
- Scale Invariant: Changes in user distance from webcam scale W and (I - A) equally.
- Roll Invariant: Head roll tilts V and (I - A) by the exact same angle.
- Translation Invariant: Translating the head in (x, y) does not change relative offsets.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np

from .types import EyeMetrics, NormalizedEyeCoords, Point2D


# Landmark index definitions
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
LEFT_EYE_UPPER = 386
LEFT_EYE_LOWER = 374
LEFT_IRIS_CENTER = 473

RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33
RIGHT_EYE_UPPER = 159
RIGHT_EYE_LOWER = 145
RIGHT_IRIS_CENTER = 468


def _normalize_iris_to_corners(
    iris_px: np.ndarray,
    corner_left_px: np.ndarray,
    corner_right_px: np.ndarray,
    upper_px: np.ndarray,
    lower_px: np.ndarray,
) -> NormalizedEyeCoords:
    """
    Project iris center into the normalized eye coordinate system defined by eye corners.
    
    The coordinate axis points horizontally from image-left to image-right for BOTH eyes:
    - Subject's left eye (appears on image right): inner (nasal 362) -> outer (temporal 263).
    - Subject's right eye (appears on image left): outer (temporal 33) -> inner (nasal 133).
    
    This ensures conjugate eye movements (e.g. looking right) yield identical positive
    normalized offsets for both eyes, preventing directional cancellation or inversion.
    
    Returns:
        NormalizedEyeCoords with iris_x, iris_y, openness, and EAR.
    """
    axis = corner_right_px[:2] - corner_left_px[:2]
    width_sq = float(axis @ axis)
    width = math.sqrt(width_sq)

    if width < 1e-4:
        # Degenerate eye (too small or eye closed)
        return NormalizedEyeCoords(iris_x=0.5, iris_y=0.0, openness=0.0, ear=0.0)

    rel = iris_px[:2] - corner_left_px[:2]
    # Projected x along corner axis: 0.0 at image-left corner, 1.0 at image-right corner
    norm_x = float(rel @ axis) / width_sq
    # Orthogonal y perpendicular to corner axis: positive = looking downward
    norm_y = float(axis[0] * rel[1] - axis[1] * rel[0]) / width_sq

    # Vertical eyelid distance
    height = float(np.linalg.norm(upper_px[:2] - lower_px[:2]))
    openness = height / width
    ear = height / (2.0 * width)

    return NormalizedEyeCoords(
        iris_x=norm_x,
        iris_y=norm_y,
        openness=openness,
        ear=ear,
    )


def extract_eye_metrics(landmarks_px: np.ndarray) -> Optional[EyeMetrics]:
    """
    Extract normalized eye features and iris coordinates from 478 landmark points.
    
    Args:
        landmarks_px: (478, 2) or (478, 3) pixel array.
        
    Returns:
        EyeMetrics object or None if landmarks are degenerate.
    """
    if len(landmarks_px) < 478:
        return None

    # Left eye points (subject's left, image-right)
    # Nasal corner (362) is on the image-left, temporal corner (263) is on the image-right.
    l_in = landmarks_px[LEFT_EYE_INNER][:2]
    l_out = landmarks_px[LEFT_EYE_OUTER][:2]
    l_up = landmarks_px[LEFT_EYE_UPPER][:2]
    l_lo = landmarks_px[LEFT_EYE_LOWER][:2]
    l_iris = landmarks_px[LEFT_IRIS_CENTER][:2]

    # Right eye points (subject's right, image-left)
    # Temporal corner (33) is on the image-left, nasal corner (133) is on the image-right.
    r_in = landmarks_px[RIGHT_EYE_INNER][:2]
    r_out = landmarks_px[RIGHT_EYE_OUTER][:2]
    r_up = landmarks_px[RIGHT_EYE_UPPER][:2]
    r_lo = landmarks_px[RIGHT_EYE_LOWER][:2]
    r_iris = landmarks_px[RIGHT_IRIS_CENTER][:2]

    left_norm = _normalize_iris_to_corners(l_iris, l_in, l_out, l_up, l_lo)
    right_norm = _normalize_iris_to_corners(r_iris, r_out, r_in, r_up, r_lo)

    # Combined 4D eye gaze feature: [left_norm_x, left_norm_y, right_norm_x, right_norm_y]
    feature_vector = np.array([
        left_norm.iris_x,
        left_norm.iris_y,
        right_norm.iris_x,
        right_norm.iris_y,
    ], dtype=np.float32)

    return EyeMetrics(
        left_corner_inner=Point2D(float(l_in[0]), float(l_in[1])),
        left_corner_outer=Point2D(float(l_out[0]), float(l_out[1])),
        left_iris_center=Point2D(float(l_iris[0]), float(l_iris[1])),
        left_normalized=left_norm,
        right_corner_inner=Point2D(float(r_in[0]), float(r_in[1])),
        right_corner_outer=Point2D(float(r_out[0]), float(r_out[1])),
        right_iris_center=Point2D(float(r_iris[0]), float(r_iris[1])),
        right_normalized=right_norm,
        gaze_feature_vector=feature_vector,
    )

