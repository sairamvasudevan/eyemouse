"""Independent blink detection state machine using MediaPipe blendshape signals.

States and Hysteresis:
-----------------------
Blink detection uses MediaPipe's `eyeBlinkLeft` and `eyeBlinkRight` blendshape weights (0.0 = wide open, 1.0 = fully closed).

Hysteresis thresholds:
    CLOSE_THRESHOLD = 0.55  (must rise above to count as closed)
    OPEN_THRESHOLD  = 0.35  (must fall below to count as open)

State transitions:
    OPEN -> (score > OPEN_THRESHOLD) -> CLOSING
    CLOSING -> (score >= CLOSE_THRESHOLD) -> CLOSED
    CLOSED -> (duration >= min_sustained_ms) -> SUSTAINED_CLOSED
    (CLOSED | SUSTAINED_CLOSED) -> (score < CLOSE_THRESHOLD) -> REOPENING
    REOPENING -> (score <= OPEN_THRESHOLD) -> OPEN
"""

from __future__ import annotations

import time
from typing import Optional, Dict
from .types import BlinkPhase, BlinkEvent


# Hysteresis parameters
DEFAULT_CLOSE_THRESHOLD = 0.55
DEFAULT_OPEN_THRESHOLD = 0.35

# Duration boundaries in milliseconds
MIN_NATURAL_BLINK_MS = 80.0
MAX_NATURAL_BLINK_MS = 300.0
MIN_SUSTAINED_BLINK_MS = 350.0
MAX_SUSTAINED_BLINK_MS = 2000.0


class BlinkDetector:
    """State machine tracking blink dynamics with temporal smoothing and hysteresis."""

    def __init__(
        self,
        close_threshold: float = DEFAULT_CLOSE_THRESHOLD,
        open_threshold: float = DEFAULT_OPEN_THRESHOLD,
        smoothing_alpha: float = 0.6,
        min_sustained_ms: float = MIN_SUSTAINED_BLINK_MS,
        max_sustained_ms: float = MAX_SUSTAINED_BLINK_MS,
    ) -> None:
        if open_threshold >= close_threshold:
            raise ValueError("open_threshold must be strictly less than close_threshold for hysteresis")
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.smoothing_alpha = smoothing_alpha
        self.min_sustained_ms = min_sustained_ms
        self.max_sustained_ms = max_sustained_ms

        self._phase = BlinkPhase.OPEN
        self._smoothed_left = 0.0
        self._smoothed_right = 0.0
        self._close_start_time_ns: Optional[int] = None
        self._last_event: Optional[BlinkEvent] = None

    @property
    def phase(self) -> BlinkPhase:
        return self._phase

    def reset(self) -> None:
        """Reset state machine to initial open state."""
        self._phase = BlinkPhase.OPEN
        self._smoothed_left = 0.0
        self._smoothed_right = 0.0
        self._close_start_time_ns = None
        self._last_event = None

    def update(
        self,
        blendshapes: Dict[str, float],
        timestamp_ns: int,
    ) -> BlinkEvent:
        """
        Update state machine with raw blendshape scores from the current frame.
        
        Args:
            blendshapes: Dictionary mapping blendshape names (e.g. 'eyeBlinkLeft') to float [0..1].
            timestamp_ns: Monotonic frame timestamp in nanoseconds.
            
        Returns:
            BlinkEvent reflecting current phase and durations.
        """
        raw_left = blendshapes.get("eyeBlinkLeft", 0.0)
        raw_right = blendshapes.get("eyeBlinkRight", 0.0)

        # Exponential moving average filter for noise suppression
        self._smoothed_left = (self.smoothing_alpha * raw_left) + ((1.0 - self.smoothing_alpha) * self._smoothed_left)
        self._smoothed_right = (self.smoothing_alpha * raw_right) + ((1.0 - self.smoothing_alpha) * self._smoothed_right)

        # Bilateral blink score is the mean of both eyes
        combined_score = (self._smoothed_left + self._smoothed_right) / 2.0

        duration_ms = 0.0
        if self._close_start_time_ns is not None:
            duration_ms = (timestamp_ns - self._close_start_time_ns) / 1_000_000.0

        is_intentional = False

        # State machine transition logic with hysteresis
        if self._phase == BlinkPhase.OPEN:
            if combined_score >= self.close_threshold:
                self._phase = BlinkPhase.CLOSED
                self._close_start_time_ns = timestamp_ns
            elif combined_score > self.open_threshold:
                self._phase = BlinkPhase.CLOSING

        elif self._phase == BlinkPhase.CLOSING:
            if combined_score >= self.close_threshold:
                self._phase = BlinkPhase.CLOSED
                self._close_start_time_ns = timestamp_ns
            elif combined_score <= self.open_threshold:
                self._phase = BlinkPhase.OPEN

        elif self._phase == BlinkPhase.CLOSED:
            if duration_ms >= self.min_sustained_ms:
                self._phase = BlinkPhase.SUSTAINED_CLOSED
                is_intentional = (duration_ms <= self.max_sustained_ms)
            elif combined_score < self.close_threshold:
                self._phase = BlinkPhase.REOPENING

        elif self._phase == BlinkPhase.SUSTAINED_CLOSED:
            is_intentional = (self.min_sustained_ms <= duration_ms <= self.max_sustained_ms)
            if combined_score < self.close_threshold:
                self._phase = BlinkPhase.REOPENING

        elif self._phase == BlinkPhase.REOPENING:
            if combined_score <= self.open_threshold:
                self._phase = BlinkPhase.OPEN
                self._close_start_time_ns = None
            elif combined_score >= self.close_threshold:
                self._phase = BlinkPhase.CLOSED

        event = BlinkEvent(
            phase=self._phase,
            left_blink_score=float(self._smoothed_left),
            right_blink_score=float(self._smoothed_right),
            closure_duration_ms=duration_ms,
            is_intentional_sustained=is_intentional,
        )
        self._last_event = event
        return event

