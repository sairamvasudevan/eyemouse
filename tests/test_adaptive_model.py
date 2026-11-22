"""Tests for AdaptiveGazeModel, Bayesian prior adaptation, and RLS online calibration."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.vision_engine.calibration.adaptive_model import (
    AdaptiveGazeModel,
    get_base_prior_weights,
    get_default_lambda_diag,
    FEATURE_DIM,
)
from src.vision_engine.calibration.profile import CalibrationProfile
from src.vision_engine.calibration.estimator import CalibratedGazeEstimator
from src.vision_engine.types import (
    EyeMetrics,
    NormalizedEyeCoords,
    Point2D,
    HeadPose,
)


def test_base_prior_weights_and_lambda():
    W0 = get_base_prior_weights()
    assert W0.shape == (FEATURE_DIM, 2)
    # Bias term should predict center (0.5, 0.5)
    assert abs(W0[0, 0] - 0.5) < 1e-4
    assert abs(W0[0, 1] - 0.5) < 1e-4

    lam = get_default_lambda_diag()
    assert lam.shape == (FEATURE_DIM,)
    # Bias penalty should be low to allow instant 1-point centering
    assert lam[0] <= 0.05
    # Eye sensitivity penalty should be moderate
    assert lam[1] > 1.0


def test_adaptive_model_base_prior_prediction():
    model = AdaptiveGazeModel()
    assert model.adaptation_points_count == 0

    # Resting eye features at neutral center:
    # lx=0.5, ly=0.0, rx=0.5, ry=0.0, yaw=0, pitch=0, roll=0
    resting_feats = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    u, v = model.predict(resting_feats)[0]
    assert abs(u - 0.5) < 0.05
    assert abs(v - 0.5) < 0.05

    # Look right: iris moves in +x direction in camera frame
    look_right_feats = np.array([0.58, 0.0, 0.58, 0.0, 0.0, 0.0, 0.0])
    u_right, v_right = model.predict(look_right_feats)[0]
    assert u_right > u

    # Look down: iris moves in +y direction
    look_down_feats = np.array([0.5, 0.08, 0.5, 0.08, 0.0, 0.0, 0.0])
    u_down, v_down = model.predict(look_down_feats)[0]
    assert v_down > v


def test_single_point_center_adaptation():
    """Test instant 1-point center recalibration (anchors bias to user's personal eye resting position)."""
    model = AdaptiveGazeModel()

    # Suppose user's physical resting eye position has a slight offset:
    # lx=0.48, ly=0.02, rx=0.49, ry=0.03, yaw=2.0 deg, pitch=-1.5 deg
    user_resting_feats = np.array([0.48, 0.02, 0.49, 0.03, 2.0, -1.5, 0.5])

    # Before adaptation, base model has a small bias error:
    u_pre, v_pre = model.predict(user_resting_feats)[0]
    err_pre = np.hypot(u_pre - 0.5, v_pre - 0.5)
    assert err_pre > 0.01  # > 15-20 pixels off

    # User looks at center (0.5, 0.5) and presses Spacebar -> adapt_point
    pred_u, pred_v = model.adapt_point(user_resting_feats, 0.5, 0.5, weight=1.0)

    # Immediately after 1 point adaptation, center error should collapse to < 1 pixel (< 0.002 in normalized coords)
    u_post, v_post = model.predict(user_resting_feats)[0]
    err_post = np.hypot(u_post - 0.5, v_post - 0.5)
    assert err_post < 0.002, f"Center error post-adaptation: {err_post}"
    assert model.adaptation_points_count == 1


def test_multi_point_online_rls_updates():
    model = AdaptiveGazeModel()

    # User fixates 5 targets
    targets = [
        (np.array([0.50, 0.00, 0.50, 0.00, 0.0, 0.0, 0.0]), 0.5, 0.5),      # Center
        (np.array([0.38, -0.08, 0.38, -0.08, 4.0, 4.0, 0.0]), 0.1, 0.1),   # Top-Left
        (np.array([0.62, -0.08, 0.62, -0.08, -4.0, 4.0, 0.0]), 0.9, 0.1),   # Top-Right
        (np.array([0.38, 0.08, 0.38, 0.08, 4.0, -4.0, 0.0]), 0.1, 0.9),    # Bottom-Left
        (np.array([0.62, 0.08, 0.62, 0.08, -4.0, -4.0, 0.0]), 0.9, 0.9),   # Bottom-Right
    ]

    import time
    t0 = time.perf_counter()
    for feats, tx, ty in targets:
        model.adapt_point(feats, tx, ty)
    elapsed = (time.perf_counter() - t0) * 1e6 / len(targets)
    # Each update should be under 200 microseconds
    assert elapsed < 200.0, f"Average RLS update time: {elapsed} us"
    assert model.adaptation_points_count == 5

    # Check that predictions for these points track the targets accurately
    errors = []
    for feats, tx, ty in targets:
        pu, pv = model.predict(feats)[0]
        errors.append(np.hypot(pu - tx, pv - ty))
    mean_err = np.mean(errors)
    assert mean_err < 0.10, f"Mean error after 5 points: {mean_err}"


def test_serialization_and_deserialization():
    model = AdaptiveGazeModel()
    user_feats = np.array([0.47, 0.03, 0.48, 0.04, 1.0, -1.0, 0.0])
    model.adapt_point(user_feats, 0.5, 0.5)

    d = model.to_dict()
    assert d["model_name"] == "adaptive_online_ridge"
    assert d["adaptation_points_count"] == 1
    assert "weights" in d
    assert "p_matrix" in d

    restored = AdaptiveGazeModel.from_dict(d)
    assert restored.adaptation_points_count == 1
    u1, v1 = model.predict(user_feats)[0]
    u2, v2 = restored.predict(user_feats)[0]
    assert abs(u1 - u2) < 1e-6
    assert abs(v1 - v2) < 1e-6

    # Test file roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model.json"
        model.save_to_file(path)
        from_file = AdaptiveGazeModel.load_from_file(path)
        u3, v3 = from_file.predict(user_feats)[0]
        assert abs(u1 - u3) < 1e-6
        assert abs(v1 - v3) < 1e-6


def test_profile_instantiation_with_adaptive_model():
    model = AdaptiveGazeModel()
    feats = np.array([0.48, 0.02, 0.48, 0.02, 0.0, 0.0, 0.0])
    model.adapt_point(feats, 0.5, 0.5)

    profile = CalibrationProfile(
        version=1,
        created_at_iso="2026-11-22T00:00:00Z",
        camera_unique_id="",
        camera_name="MacBook Camera",
        camera_resolution="1920x1080",
        pixel_format="BGRA",
        screen_width_points=1470.0,
        screen_height_points=956.0,
        backing_scale_factor=2.0,
        model_name="adaptive_online_ridge",
        model_dict=model.to_dict(),
        validation_metrics={"quality_grade": "ADAPTIVE"},
        quality_grade="ADAPTIVE",
        is_valid=True,
    )

    estimator = CalibratedGazeEstimator.from_profile(profile)
    assert estimator.is_calibrated
    assert isinstance(estimator.model, AdaptiveGazeModel)

    # Estimate using eye metrics and head pose
    eye_metrics = EyeMetrics(
        left_corner_inner=Point2D(90, 100),
        left_corner_outer=Point2D(110, 100),
        left_iris_center=Point2D(100, 100),
        left_normalized=NormalizedEyeCoords(iris_x=0.48, iris_y=0.02, openness=0.8, ear=0.3),
        right_corner_inner=Point2D(140, 100),
        right_corner_outer=Point2D(160, 100),
        right_iris_center=Point2D(150, 100),
        right_normalized=NormalizedEyeCoords(iris_x=0.48, iris_y=0.02, openness=0.8, ear=0.3),
        gaze_feature_vector=np.array([0.48, 0.02, 0.48, 0.02]),
    )
    head_pose = HeadPose(yaw=0.0, pitch=0.0, roll=0.0, tx=0.0, ty=0.0, tz=0.0)

    est = estimator.estimate(eye_metrics, head_pose, timestamp_s=1.0)
    assert est is not None
    assert est.is_valid
    # In normalized screen points, (0.5, 0.5) corresponds to (0.5, 0.5)
    assert abs(est.raw_x - 0.5) < 0.01
    assert abs(est.raw_y - 0.5) < 0.01
