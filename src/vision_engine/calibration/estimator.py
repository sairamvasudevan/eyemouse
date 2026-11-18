"""Calibrated screen gaze estimator implementing GazeEstimatorProtocol."""

from __future__ import annotations

from typing import Optional
import numpy as np

from ..types import EyeMetrics, HeadPose, GazeEstimate
from ..filters import PointFilter2D
from ..gaze import GazeEstimatorProtocol
from .models import BaseGazeModel
from .profile import CalibrationProfile


class CalibratedGazeEstimator(GazeEstimatorProtocol):
    """Calibrated screen gaze estimator.
    
    Transforms normalized eye metrics and head pose into calibrated normalized screen coordinates
    (u, v) in [0.0, 1.0], with OneEuroFilter temporal smoothing.
    """

    def __init__(
        self,
        model: BaseGazeModel,
        profile: Optional[CalibrationProfile] = None,
        smoothing_min_cutoff: float = 0.5,
        smoothing_beta: float = 0.008,
    ) -> None:
        self.model = model
        self.profile = profile
        self._filter = PointFilter2D(min_cutoff=smoothing_min_cutoff, beta=smoothing_beta)
        self._last_valid_x = 0.5
        self._last_valid_y = 0.5

    @property
    def is_calibrated(self) -> bool:
        return True

    def reset(self) -> None:
        """Reset temporal filter state on face loss or tracking interruption."""
        self._filter.reset()

    def estimate(
        self,
        eye_metrics: EyeMetrics,
        head_pose: HeadPose,
        timestamp_s: float,
    ) -> GazeEstimate:
        """Predict calibrated screen coordinates from eye features and head pose."""
        l_norm = eye_metrics.left_normalized
        r_norm = eye_metrics.right_normalized

        # Construct 7-dimensional feature vector:
        # [left_iris_x, left_iris_y, right_iris_x, right_iris_y, yaw, pitch, roll]
        feature_vector = np.array([
            l_norm.iris_x,
            l_norm.iris_y,
            r_norm.iris_x,
            r_norm.iris_y,
            head_pose.yaw,
            head_pose.pitch,
            head_pose.roll,
        ], dtype=np.float64)

        # Model prediction (u, v) in [0.0, 1.0]
        preds = self.model.predict(feature_vector)
        raw_u = float(np.clip(preds[0, 0], 0.0, 1.0))
        raw_v = float(np.clip(preds[0, 1], 0.0, 1.0))

        # Velocity-based adaptive smoothing (OneEuroFilter)
        filt_u, filt_v = self._filter(raw_u, raw_v, timestamp_s)
        filt_u = float(np.clip(filt_u, 0.0, 1.0))
        filt_v = float(np.clip(filt_v, 0.0, 1.0))

        self._last_valid_x = filt_u
        self._last_valid_y = filt_v

        # Confidence based on eye openness and reasonable head pose limits
        min_openness = min(l_norm.openness, r_norm.openness)
        openness_conf = float(np.clip(min_openness / 0.25, 0.0, 1.0))
        pose_penalty = 1.0
        if abs(head_pose.yaw) > 40.0 or abs(head_pose.pitch) > 30.0:
            pose_penalty = 0.6

        confidence = float(np.clip(openness_conf * pose_penalty, 0.0, 1.0))

        return GazeEstimate(
            raw_x=raw_u,
            raw_y=raw_v,
            filtered_x=filt_u,
            filtered_y=filt_v,
            confidence=confidence,
            is_valid=True,
        )

    @classmethod
    def from_profile(
        cls,
        profile: CalibrationProfile,
        smoothing_min_cutoff: float = 0.5,
        smoothing_beta: float = 0.008,
    ) -> CalibratedGazeEstimator:
        model = profile.instantiate_model()
        return cls(
            model=model,
            profile=profile,
            smoothing_min_cutoff=smoothing_min_cutoff,
            smoothing_beta=smoothing_beta,
        )

