"""Replaceable gaze estimation interface and uncalibrated baseline estimator.

DISCLAIMER:
-----------
Raw MediaPipe landmarks and normalized iris positions DO NOT directly correspond
to calibrated screen gaze coordinates. Screen calibration requires mapping the
user's individual pupil-corneal geometry, display position, and head distance.
This module defines the architectural interface for calibrated models, while
providing an uncalibrated optical proxy estimator for testing.
"""

from __future__ import annotations

from typing import Protocol, Optional
import numpy as np

from .types import EyeMetrics, HeadPose, GazeEstimate
from .filters import PointFilter2D


class GazeEstimatorProtocol(Protocol):
    """Protocol interface for replaceable screen gaze estimation models."""

    @property
    def is_calibrated(self) -> bool:
        """Whether a valid calibration model is active."""
        ...

    def estimate(
        self,
        eye_metrics: EyeMetrics,
        head_pose: HeadPose,
        timestamp_s: float,
    ) -> GazeEstimate:
        """Estimate gaze position from normalized eye features and head pose."""
        ...

    def reset(self) -> None:
        """Reset internal filter states."""
        ...


class UncalibratedGazeEstimator:
    """
    Uncalibrated optical proxy estimator.
    
    Combines normalized iris vector offsets with head rotation angles to provide
    a continuous directional estimate. Coordinates are normalized [-1.0, 1.0].
    """

    def __init__(
        self,
        smoothing_min_cutoff: float = 0.5,
        smoothing_beta: float = 0.005,
    ) -> None:
        self._filter = PointFilter2D(min_cutoff=smoothing_min_cutoff, beta=smoothing_beta)

    @property
    def is_calibrated(self) -> bool:
        return False

    def reset(self) -> None:
        self._filter.reset()

    def estimate(
        self,
        eye_metrics: EyeMetrics,
        head_pose: HeadPose,
        timestamp_s: float,
    ) -> GazeEstimate:
        """
        Derive optical proxy gaze from eye metrics and head pose.
        
        Left and right iris positions are normalized such that 0.5 is centered.
        Relative offset = (norm_x - 0.5) * 2.0.
        """
        # Average iris offset across both eyes
        l_norm = eye_metrics.left_normalized
        r_norm = eye_metrics.right_normalized
        avg_eye_x = ((l_norm.iris_x - 0.5) + (r_norm.iris_x - 0.5))
        avg_eye_y = ((l_norm.iris_y - 0.0) + (r_norm.iris_y - 0.0))

        # Head pose influence (yaw contributes to X, pitch contributes to Y)
        # Moderate head turn (+-30 deg) maps to normalized displacement
        head_x = head_pose.yaw / 30.0
        head_y = -head_pose.pitch / 25.0

        # Uncalibrated composite directional proxy
        raw_x = float(np.clip(avg_eye_x + 0.3 * head_x, -1.0, 1.0))
        raw_y = float(np.clip(avg_eye_y + 0.3 * head_y, -1.0, 1.0))

        filtered_x, filtered_y = self._filter(raw_x, raw_y, timestamp_s)

        # Baseline confidence based on eye openness
        min_openness = min(l_norm.openness, r_norm.openness)
        confidence = float(np.clip(min_openness / 0.25, 0.0, 1.0))

        return GazeEstimate(
            raw_x=raw_x,
            raw_y=raw_y,
            filtered_x=filtered_x,
            filtered_y=filtered_y,
            confidence=confidence,
            is_valid=True,
        )

