"""Sample collection, stability filtering, and robust aggregation for gaze calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass(slots=True)
class RawGazeSample:
    """Individual frame observation during calibration fixation."""
    timestamp_s: float
    left_iris_x: float
    left_iris_y: float
    right_iris_x: float
    right_iris_y: float
    head_yaw: float
    head_pitch: float
    head_roll: float
    confidence: float
    is_blink: bool
    face_detected: bool

    def to_feature_vector(self) -> np.ndarray:
        """Extract 7-dimensional raw gaze feature vector:
        [left_iris_x, left_iris_y, right_iris_x, right_iris_y, yaw, pitch, roll]
        """
        return np.array([
            self.left_iris_x,
            self.left_iris_y,
            self.right_iris_x,
            self.right_iris_y,
            self.head_yaw,
            self.head_pitch,
            self.head_roll,
        ], dtype=np.float64)


@dataclass(slots=True)
class TargetSampleAggregate:
    """Aggregated, filtered calibration observation for a single screen target."""
    target_index: int
    target_name: str
    target_norm_x: float
    target_norm_y: float
    feature_median: np.ndarray       # (7,) robust central tendency
    feature_std: np.ndarray          # (7,) variance / stability metric
    total_samples_collected: int
    valid_samples_accepted: int
    outliers_rejected: int
    blinks_rejected: int
    is_valid: bool
    is_center: bool = False
    is_corner: bool = False
    is_edge: bool = False


class TargetSampleCollector:
    """Collects, validates, filters, and aggregates gaze samples for a single target."""

    def __init__(
        self,
        target_index: int,
        target_name: str,
        target_norm_x: float,
        target_norm_y: float,
        settling_duration_s: float = 0.5,
        collection_duration_s: float = 1.0,
        min_required_samples: int = 12,
        outlier_mad_threshold: float = 2.5,
    ) -> None:
        self.target_index = target_index
        self.target_name = target_name
        self.target_norm_x = target_norm_x
        self.target_norm_y = target_norm_y
        self.settling_duration_s = settling_duration_s
        self.collection_duration_s = collection_duration_s
        self.min_required_samples = min_required_samples
        self.outlier_mad_threshold = outlier_mad_threshold

        self.start_timestamp: Optional[float] = None
        self._raw_samples: List[RawGazeSample] = []
        self._accepted_samples: List[RawGazeSample] = []
        self._blinks_count = 0
        self._face_loss_count = 0
        self._low_confidence_count = 0

    def start(self, timestamp_s: float) -> None:
        self.start_timestamp = timestamp_s
        self._raw_samples.clear()
        self._accepted_samples.clear()
        self._blinks_count = 0
        self._face_loss_count = 0
        self._low_confidence_count = 0

    @property
    def is_settling(self) -> bool:
        """Whether current time falls within the initial saccade settling window."""
        if self.start_timestamp is None:
            return True
        return False

    def add_sample(self, sample: RawGazeSample) -> bool:
        """Add a frame observation. Returns True if accepted into steady fixation window."""
        if self.start_timestamp is None:
            self.start(sample.timestamp_s)

        elapsed = sample.timestamp_s - self.start_timestamp

        # 1. Reject during settling interval
        if elapsed < self.settling_duration_s:
            return False

        # 2. Reject if past collection window
        if elapsed > (self.settling_duration_s + self.collection_duration_s):
            return False

        self._raw_samples.append(sample)

        # 3. Quality validation
        if not sample.face_detected:
            self._face_loss_count += 1
            return False

        if sample.is_blink:
            self._blinks_count += 1
            return False

        if sample.confidence < 0.60:
            self._low_confidence_count += 1
            return False

        self._accepted_samples.append(sample)
        return True

    def is_finished(self, current_timestamp_s: float) -> bool:
        """Check whether the full collection duration has elapsed."""
        if self.start_timestamp is None:
            return False
        return (current_timestamp_s - self.start_timestamp) >= (
            self.settling_duration_s + self.collection_duration_s
        )

    def aggregate(self) -> TargetSampleAggregate:
        """Filter outliers using MAD and compute robust per-target aggregated features."""
        if len(self._accepted_samples) < self.min_required_samples:
            return TargetSampleAggregate(
                target_index=self.target_index,
                target_name=self.target_name,
                target_norm_x=self.target_norm_x,
                target_norm_y=self.target_norm_y,
                feature_median=np.zeros(7, dtype=np.float64),
                feature_std=np.zeros(7, dtype=np.float64),
                total_samples_collected=len(self._raw_samples),
                valid_samples_accepted=len(self._accepted_samples),
                outliers_rejected=0,
                blinks_rejected=self._blinks_count,
                is_valid=False,
            )

        features = np.array([s.to_feature_vector() for s in self._accepted_samples], dtype=np.float64)  # (N, 7)

        # Outlier rejection using Median Absolute Deviation (MAD)
        median_vec = np.median(features, axis=0)
        abs_deviations = np.abs(features - median_vec)
        mad = np.median(abs_deviations, axis=0)
        # Avoid division by zero on perfectly constant synthetic data
        mad = np.maximum(mad, 1e-6)

        # Normalized distance for each sample
        norm_deviations = abs_deviations / mad
        max_dev = np.max(norm_deviations, axis=1)
        inlier_mask = max_dev <= self.outlier_mad_threshold

        inliers = features[inlier_mask]
        num_rejected = int(np.sum(~inlier_mask))

        if len(inliers) < (self.min_required_samples // 2):
            # Too many outliers rejected; fall back to all accepted samples to avoid total drop
            inliers = features

        robust_median = np.median(inliers, axis=0)
        robust_std = np.std(inliers, axis=0)

        return TargetSampleAggregate(
            target_index=self.target_index,
            target_name=self.target_name,
            target_norm_x=self.target_norm_x,
            target_norm_y=self.target_norm_y,
            feature_median=robust_median,
            feature_std=robust_std,
            total_samples_collected=len(self._raw_samples),
            valid_samples_accepted=len(inliers),
            outliers_rejected=num_rejected,
            blinks_rejected=self._blinks_count,
            is_valid=True,
        )

