"""Confidence evaluation and face tracking loss detection."""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np


class TrackingQualityEvaluator:
    """Evaluates face tracking confidence and flags tracking loss."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence

    def evaluate(
        self,
        landmarks: Optional[np.ndarray],
        num_faces_detected: int,
    ) -> tuple[bool, float]:
        """
        Evaluate if a primary face is validly tracked.
        
        Returns:
            (is_valid, confidence_score)
        """
        if num_faces_detected == 0 or landmarks is None or len(landmarks) < 478:
            return False, 0.0

        if not np.all(np.isfinite(landmarks)):
            return False, 0.0

        # Sample rigid anchor landmarks (nose tip = 1, chin = 152, eye corners = 33, 263)
        anchors = landmarks[[1, 152, 33, 263]]
        # Bounds check: all anchors must be inside normalized [0..1] range
        if np.any(anchors < -0.2) or np.any(anchors > 1.2):
            return False, 0.2

        # Check eye corner separation (face too small / far away)
        eye_distance = float(np.linalg.norm(landmarks[33][:2] - landmarks[263][:2]))
        if eye_distance < 0.05:  # Less than 5% of screen width
            return False, 0.3

        confidence = 0.95
        if num_faces_detected > 1:
            # Degrade confidence if multiple people are in frame
            confidence *= 0.8

        return True, confidence

