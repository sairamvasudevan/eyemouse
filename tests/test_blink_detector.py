"""Automated tests for the BlinkDetector state machine, hysteresis, and durations."""

import pytest
from src.vision_engine.blink import BlinkDetector
from src.vision_engine.types import BlinkPhase


def test_blink_detector_initial_state():
    detector = BlinkDetector()
    assert detector.phase == BlinkPhase.OPEN


def test_hysteresis_and_state_transitions():
    """Verify hysteresis: intermediate scores do not flap state."""
    detector = BlinkDetector(close_threshold=0.55, open_threshold=0.35, smoothing_alpha=1.0)
    t0 = 1_000_000_000

    # 1. Neutral open eyes
    ev = detector.update({"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1}, t0)
    assert ev.phase == BlinkPhase.OPEN

    # 2. Score rises into intermediate band (0.45) -> CLOSING
    ev = detector.update({"eyeBlinkLeft": 0.45, "eyeBlinkRight": 0.45}, t0 + 33_000_000)
    assert ev.phase == BlinkPhase.CLOSING

    # 3. Score exceeds close threshold (0.60) -> CLOSED
    ev = detector.update({"eyeBlinkLeft": 0.60, "eyeBlinkRight": 0.60}, t0 + 66_000_000)
    assert ev.phase == BlinkPhase.CLOSED

    # 4. Score drops into intermediate band (0.45) -> REOPENING (does not flip to OPEN yet)
    ev = detector.update({"eyeBlinkLeft": 0.45, "eyeBlinkRight": 0.45}, t0 + 99_000_000)
    assert ev.phase == BlinkPhase.REOPENING

    # 5. Score drops below open threshold (0.20) -> OPEN
    ev = detector.update({"eyeBlinkLeft": 0.20, "eyeBlinkRight": 0.20}, t0 + 132_000_000)
    assert ev.phase == BlinkPhase.OPEN


def test_sustained_closure_vs_quick_blink():
    """Verify intentional sustained closure (>350ms) vs quick natural blink (<300ms)."""
    detector = BlinkDetector(
        close_threshold=0.55,
        open_threshold=0.35,
        smoothing_alpha=1.0,
        min_sustained_ms=350.0,
        max_sustained_ms=1500.0,
    )
    t0 = 1_000_000_000

    # Frame 1: Eyes close
    detector.update({"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.8}, t0)
    assert detector.phase == BlinkPhase.CLOSED

    # Frame 2: 150ms elapsed -> Still standard CLOSED, NOT intentional sustained yet
    ev = detector.update({"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.8}, t0 + 150_000_000)
    assert ev.phase == BlinkPhase.CLOSED
    assert ev.is_intentional_sustained is False
    assert pytest.approx(ev.closure_duration_ms, abs=1.0) == 150.0

    # Frame 3: 400ms elapsed -> Transitions to SUSTAINED_CLOSED, marked intentional
    ev = detector.update({"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.8}, t0 + 400_000_000)
    assert ev.phase == BlinkPhase.SUSTAINED_CLOSED
    assert ev.is_intentional_sustained is True
    assert pytest.approx(ev.closure_duration_ms, abs=1.0) == 400.0

