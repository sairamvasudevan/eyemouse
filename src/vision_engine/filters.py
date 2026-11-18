"""One Euro filter for adaptive velocity-based low-pass smoothing.

Academic Reference:
    Géry Casiez, Nicolas Roussel, and Daniel Vogel. 2012.
    1 € filter: a simple speed-based low-pass filter for noisy input in human-computer interaction.
    In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '12).
    ACM, New York, NY, USA, 2527-2530. https://doi.org/10.1145/2207676.2208639

Adapted from:
    eye-tracker (https://github.com/bugraskl/eye-tracker)
    Copyright (c) 2026 Buğra Şıkel. MIT License.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np


# Default parameters for human eye gaze / cursor tracking:
# min_cutoff (Hz): Cutoff frequency when signal is stationary. Lower = heavier smoothing (0.3 - 1.0 Hz).
# beta: Sensitivity to velocity. Higher = faster response to saccades (0.001 - 0.01).
# d_cutoff (Hz): Cutoff frequency for the derivative (speed) filter.
DEFAULT_MIN_CUTOFF = 0.5
DEFAULT_BETA = 0.005
DEFAULT_D_CUTOFF = 1.0


class OneEuroFilter:
    """One-dimensional One Euro filter with adaptive cutoff frequency."""

    def __init__(
        self,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
        d_cutoff: float = DEFAULT_D_CUTOFF,
    ) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
            raise ValueError("min_cutoff and d_cutoff must be > 0 and beta >= 0")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: Optional[float] = None
        self._dx = 0.0
        self._t: Optional[float] = None

    @property
    def value(self) -> Optional[float]:
        """Last filtered value (None before the first sample or after reset)."""
        return self._x

    def reset(self) -> None:
        """Reset filter state."""
        self._x = None
        self._dx = 0.0
        self._t = None

    def __call__(self, x: float, t: float) -> float:
        """Filter input sample x measured at timestamp t (in seconds)."""
        x = float(x)
        if not math.isfinite(x):
            # Do not poison state with NaN or Infinity; retain last valid value
            return self._x if self._x is not None else 0.0

        if self._x is None or self._t is None:
            self._x = x
            self._dx = 0.0
            self._t = t
            return x

        dt = t - self._t
        if dt <= 1e-6:
            # Timestamp did not advance; return previous value
            return self._x

        self._t = t

        # Estimate derivative of the signal
        edx = (x - self._x) / dt
        a_d = self._smoothing_factor(dt, self.d_cutoff)
        self._dx = (1.0 - a_d) * self._dx + a_d * edx

        # Dynamically adjust the cutoff frequency based on velocity magnitude
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._smoothing_factor(dt, cutoff)
        self._x = (1.0 - a) * self._x + a * x
        return self._x

    @staticmethod
    def _smoothing_factor(dt: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)


class PointFilter2D:
    """2D One Euro filter for (x, y) coordinates (e.g. eye features or cursor)."""

    def __init__(
        self,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
        d_cutoff: float = DEFAULT_D_CUTOFF,
    ) -> None:
        self.fx = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.fy = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x: float, y: float, t: float) -> Tuple[float, float]:
        return self.fx(x, t), self.fy(y, t)

    @property
    def value(self) -> Optional[Tuple[float, float]]:
        if self.fx.value is not None and self.fy.value is not None:
            return self.fx.value, self.fy.value
        return None

