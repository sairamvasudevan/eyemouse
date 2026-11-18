"""Automated tests for OneEuroFilter numerical stability and velocity adaptation."""

import math
import numpy as np
import pytest

from src.vision_engine.filters import OneEuroFilter, PointFilter2D


def test_one_euro_filter_initialization():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01, d_cutoff=1.0)
    assert f.value is None
    val = f(10.0, 1.0)
    assert val == 10.0
    assert f.value == 10.0


def test_one_euro_filter_jitter_smoothing():
    """Verify that stationary jitter is heavily smoothed."""
    f = OneEuroFilter(min_cutoff=0.5, beta=0.001)
    t = 1.0
    dt = 1.0 / 30.0

    raw_samples = [100.0 + (1.0 if i % 2 == 0 else -1.0) for i in range(30)]
    filtered_samples = []

    for s in raw_samples:
        filtered_samples.append(f(s, t))
        t += dt

    # Filtered variance should be significantly smaller than raw jitter variance
    raw_var = np.var(raw_samples)
    filt_var = np.var(filtered_samples[-15:])  # steady-state tail
    assert filt_var < raw_var * 0.2, f"Expected smoothing reduction, got {filt_var} vs {raw_var}"


def test_one_euro_filter_saccade_adaptation():
    """Verify that sudden high-velocity movements open the filter cutoff immediately."""
    f = OneEuroFilter(min_cutoff=0.3, beta=0.05)
    t = 1.0
    # Settle at 100.0
    for _ in range(10):
        f(100.0, t)
        t += 0.033

    # Sudden saccade jump to 600.0 (500px displacement)
    jump_val = f(600.0, t)
    # Filter should adapt quickly: jump_val should reach at least 50% of the target in 1 frame
    assert jump_val > 350.0, f"Expected rapid adaptation during saccade, got {jump_val}"


def test_one_euro_filter_nan_inf_safety():
    """Verify NaN or Inf inputs do not poison filter state."""
    f = OneEuroFilter()
    f(50.0, 1.0)
    last_valid = f(52.0, 1.033)

    # Inject NaN
    val_nan = f(float("nan"), 1.066)
    assert not math.isnan(val_nan)
    assert val_nan == last_valid  # Retains last valid value

    # Inject Inf
    val_inf = f(float("inf"), 1.1)
    assert not math.isinf(val_inf)
    assert val_inf == last_valid


def test_point_filter_2d():
    pf = PointFilter2D()
    x, y = pf(10.0, 20.0, 1.0)
    assert x == 10.0 and y == 20.0
    assert pf.value == (10.0, 20.0)
