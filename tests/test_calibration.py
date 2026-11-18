"""Comprehensive test suite for eye-screen calibration geometry, models, and validation."""

import math
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.vision_engine.calibration.geometry import (
    ScreenGeometry,
    generate_13_point_calibration_grid,
    generate_9_point_validation_grid,
)
from src.vision_engine.calibration.collector import (
    RawGazeSample,
    TargetSampleCollector,
)
from src.vision_engine.calibration.models import (
    AffineGazeModel,
    PolynomialGazeModel,
    solve_ridge_regression,
    cross_validate_model,
)
from src.vision_engine.calibration.evaluator import CalibrationEvaluator
from src.vision_engine.calibration.profile import (
    CalibrationProfile,
    create_profile,
)
from src.vision_engine.calibration.session import CalibrationSession
from src.vision_engine.calibration.estimator import CalibratedGazeEstimator
from src.vision_engine.types import (
    EyeMetrics,
    NormalizedEyeCoords,
    Point2D,
    HeadPose,
)


@pytest.fixture
def macbook_air_screen():
    return ScreenGeometry(
        width_points=1470.0,
        height_points=956.0,
        backing_scale=2.0,
        viewing_distance_cm=55.0,
        diagonal_inches=13.6,
    )


# --- Geometry & Coordinate Transformation Tests ---

def test_screen_geometry_scaling(macbook_air_screen):
    screen = macbook_air_screen
    assert screen.width_pixels == 2940.0
    assert screen.height_pixels == 1912.0
    assert abs(screen.diagonal_points - 1753.5) < 1.0
    assert abs(screen.diagonal_pixels - 3507.0) < 1.0
    assert screen.ppi > 250.0  # ~257.8 PPI

    # Normalized to points
    pts = screen.normalized_to_points(0.5, 0.5)
    assert pts == (735.0, 478.0)

    # Normalized bounds clipping
    assert screen.normalized_to_points(-0.5, 1.5) == (0.0, 956.0)

    # Cocoa vs CoreGraphics flip
    cocoa_y = screen.points_to_cocoa(100.0, 200.0)
    assert cocoa_y == (100.0, 756.0)
    assert screen.cocoa_to_points(100.0, 756.0) == (100.0, 200.0)


def test_visual_degrees_calculation(macbook_air_screen):
    screen = macbook_air_screen
    # 0 pixels error = 0 visual degrees
    assert screen.pixel_error_to_visual_degrees(0.0) == 0.0

    # Approximately 100 physical pixels at 258 PPI is ~0.98 cm.
    # At 55 cm, angle is 2 * atan((0.98/2)/55) ~ 1.02 degrees
    deg = screen.pixel_error_to_visual_degrees(100.0)
    assert 0.95 < deg < 1.10


def test_target_grids_independence():
    calib_targets = generate_13_point_calibration_grid(margin=0.08)
    val_targets = generate_9_point_validation_grid()

    assert len(calib_targets) == 13
    assert len(val_targets) == 9

    # Verify no coordinates coincide (held-out independence)
    calib_coords = {(round(t.norm_x, 3), round(t.norm_y, 3)) for t in calib_targets}
    for vt in val_targets:
        vt_coord = (round(vt.norm_x, 3), round(vt.norm_y, 3))
        assert vt_coord not in calib_coords, f"Validation target {vt.name} overlaps with calibration grid"


# --- Sample Collector & Filtering Tests ---

def test_collector_settling_window():
    collector = TargetSampleCollector(
        target_index=0,
        target_name="center",
        target_norm_x=0.5,
        target_norm_y=0.5,
        settling_duration_s=0.5,
        collection_duration_s=1.0,
    )

    t0 = 100.0
    collector.start(t0)

    # Samples during settling (0.0 to 0.49s) must be discarded
    s_settling = RawGazeSample(
        timestamp_s=t0 + 0.2,
        left_iris_x=0.5, left_iris_y=0.0,
        right_iris_x=0.5, right_iris_y=0.0,
        head_yaw=0.0, head_pitch=0.0, head_roll=0.0,
        confidence=0.9, is_blink=False, face_detected=True,
    )
    assert not collector.add_sample(s_settling)

    # Samples during active fixation (0.5s to 1.5s) must be accepted
    s_valid = RawGazeSample(
        timestamp_s=t0 + 0.6,
        left_iris_x=0.5, left_iris_y=0.0,
        right_iris_x=0.5, right_iris_y=0.0,
        head_yaw=0.0, head_pitch=0.0, head_roll=0.0,
        confidence=0.9, is_blink=False, face_detected=True,
    )
    assert collector.add_sample(s_valid)


def test_collector_blink_and_face_loss_rejection():
    collector = TargetSampleCollector(
        target_index=0, target_name="center",
        target_norm_x=0.5, target_norm_y=0.5,
        settling_duration_s=0.1, collection_duration_s=1.0,
    )
    t0 = 100.0
    collector.start(t0)

    # Blink contaminated sample -> rejected
    s_blink = RawGazeSample(
        timestamp_s=t0 + 0.2, left_iris_x=0.5, left_iris_y=0.0,
        right_iris_x=0.5, right_iris_y=0.0,
        head_yaw=0.0, head_pitch=0.0, head_roll=0.0,
        confidence=0.9, is_blink=True, face_detected=True,
    )
    assert not collector.add_sample(s_blink)

    # Face loss sample -> rejected
    s_face_loss = RawGazeSample(
        timestamp_s=t0 + 0.3, left_iris_x=0.5, left_iris_y=0.0,
        right_iris_x=0.5, right_iris_y=0.0,
        head_yaw=0.0, head_pitch=0.0, head_roll=0.0,
        confidence=0.0, is_blink=False, face_detected=False,
    )
    assert not collector.add_sample(s_face_loss)


def test_collector_mad_outlier_rejection():
    collector = TargetSampleCollector(
        target_index=0, target_name="center",
        target_norm_x=0.5, target_norm_y=0.5,
        settling_duration_s=0.0, collection_duration_s=1.0,
        min_required_samples=10, outlier_mad_threshold=2.5,
    )
    collector.start(0.0)

    # Add 20 inlier samples around 0.5
    for i in range(20):
        s = RawGazeSample(
            timestamp_s=0.1 + i * 0.03,
            left_iris_x=0.50 + np.random.normal(0, 0.005),
            left_iris_y=0.00 + np.random.normal(0, 0.005),
            right_iris_x=0.50 + np.random.normal(0, 0.005),
            right_iris_y=0.00 + np.random.normal(0, 0.005),
            head_yaw=0.0, head_pitch=0.0, head_roll=0.0,
            confidence=0.95, is_blink=False, face_detected=True,
        )
        collector.add_sample(s)

    # Add 2 extreme outlier samples (sudden glance away)
    for i in range(2):
        s_outlier = RawGazeSample(
            timestamp_s=0.8 + i * 0.03,
            left_iris_x=0.99,  # Glanced to edge
            left_iris_y=0.99,
            right_iris_x=0.99,
            right_iris_y=0.99,
            head_yaw=15.0, head_pitch=-10.0, head_roll=0.0,
            confidence=0.95, is_blink=False, face_detected=True,
        )
        collector.add_sample(s_outlier)

    aggregate = collector.aggregate()
    assert aggregate.is_valid
    assert aggregate.outliers_rejected >= 2
    assert abs(aggregate.feature_median[0] - 0.50) < 0.02


# --- Mathematical Regression & Model Comparison Tests ---

def test_ridge_regression_closed_form():
    # Linear function: y = 2*x1 - 3*x2 + 0.5
    np.random.seed(42)
    X_raw = np.random.uniform(-1, 1, (50, 2))
    X_design = np.hstack([np.ones((50, 1)), X_raw])
    W_true = np.array([[0.5], [2.0], [-3.0]])
    Y = X_design @ W_true + np.random.normal(0, 0.01, (50, 1))

    W_est = solve_ridge_regression(X_design, Y, alpha=0.001)
    np.testing.assert_allclose(W_est, W_true, atol=0.05)


def test_affine_and_polynomial_model_fitting(macbook_air_screen):
    screen = macbook_air_screen
    calib_targets = generate_13_point_calibration_grid()
    val_targets = generate_9_point_validation_grid()

    # Synthetic eye-screen mapping function
    # u = 0.5 + 0.6 * (left_x - 0.5) + 0.01 * yaw
    # v = 0.5 + 0.6 * left_y - 0.01 * pitch
    np.random.seed(123)

    def synthetic_features(target_u, target_v):
        # Invert linear relation plus small quadratic curvature
        lx = 0.5 + (target_u - 0.5) / 0.6 + 0.05 * (target_u - 0.5)**2
        ly = (target_v - 0.5) / 0.6 + 0.05 * (target_v - 0.5)**2
        rx = lx + 0.01
        ry = ly
        yaw = (target_u - 0.5) * 10.0
        pitch = -(target_v - 0.5) * 8.0
        roll = 0.5
        return np.array([lx, ly, rx, ry, yaw, pitch, roll])

    # Build calibration dataset
    X_train = np.array([synthetic_features(t.norm_x, t.norm_y) for t in calib_targets])
    Y_train = np.array([[t.norm_x, t.norm_y] for t in calib_targets])

    # Build held-out validation dataset
    X_val = np.array([synthetic_features(t.norm_x, t.norm_y) for t in val_targets])

    # Fit Affine Model
    aff = AffineGazeModel(alpha=0.1).fit(X_train, Y_train)
    evaluator = CalibrationEvaluator(screen)
    aff_report = evaluator.evaluate(aff, X_val, val_targets)

    # Fit Polynomial Model
    poly = PolynomialGazeModel(alpha=0.1).fit(X_train, Y_train)
    poly_report = evaluator.evaluate(poly, X_val, val_targets)

    # Both models should achieve valid quality on smooth synthetic data
    assert aff_report.is_valid_for_use
    assert poly_report.is_valid_for_use

    # Both models achieve excellent sub-1% diagonal error on 2940x1912 display
    assert aff_report.mean_pixel_error < 5.0
    assert poly_report.mean_pixel_error < 30.0

    # Test Physically-Informed Model
    from src.vision_engine.calibration.models import PhysicallyInformedGazeModel
    phys = PhysicallyInformedGazeModel(alpha=0.001).fit(X_train, Y_train)
    phys_report = evaluator.evaluate(phys, X_val, val_targets)
    assert phys_report.is_valid_for_use
    assert phys_report.quality_grade == "EXCELLENT"

    # Verify quality grades
    assert aff_report.quality_grade == "EXCELLENT"
    assert poly_report.quality_grade == "EXCELLENT"
    assert poly_report.median_error_pct_diagonal < 2.5


def test_cross_validation_hyperparameter_selection():
    np.random.seed(42)
    X = np.random.randn(13, 7)
    Y = np.random.uniform(0.1, 0.9, (13, 2))

    best_alpha, min_rmse = cross_validate_model(
        AffineGazeModel, X, Y, [0.001, 0.01, 0.1, 1.0, 10.0]
    )
    assert best_alpha in [0.001, 0.01, 0.1, 1.0, 10.0]
    assert min_rmse > 0.0


# --- Profile Persistence & Compatibility Tests ---

def test_profile_serialization_and_compatibility(macbook_air_screen):
    screen = macbook_air_screen
    calib_targets = generate_13_point_calibration_grid()
    val_targets = generate_9_point_validation_grid()

    # Synthetic realistic linear data ensuring valid profile
    X_train = np.array([[t.norm_x, t.norm_y, t.norm_x, t.norm_y, 0.0, 0.0, 0.0] for t in calib_targets])
    Y_train = np.array([[t.norm_x, t.norm_y] for t in calib_targets])
    X_val = np.array([[t.norm_x, t.norm_y, t.norm_x, t.norm_y, 0.0, 0.0, 0.0] for t in val_targets])

    aff = AffineGazeModel(alpha=0.01).fit(X_train, Y_train)
    evaluator = CalibrationEvaluator(screen)
    report = evaluator.evaluate(aff, X_val, val_targets)
    assert report.is_valid_for_use

    profile = create_profile(
        model=aff,
        validation_report=report,
        screen=screen,
        camera_id="BuiltInCamera-001",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        profile_path = Path(tmpdir) / "test_profile.json"
        saved_path = profile.save_to_file(profile_path)
        assert saved_path.exists()

        # Reload
        loaded = CalibrationProfile.load_from_file(saved_path)
        assert loaded.camera_unique_id == "BuiltInCamera-001"
        assert loaded.model_name == "affine_linear"
        assert loaded.screen_width_points == 1470.0

        # Check compatibility
        compat, reason = loaded.is_compatible("BuiltInCamera-001", screen)
        assert compat
        assert reason is None

        # Camera mismatch triggers incompatibility
        compat_diff, reason_diff = loaded.is_compatible("ExternalCamera-999", screen)
        assert not compat_diff
        assert "Camera mismatch" in reason_diff

        # Screen resize triggers incompatibility
        diff_screen = ScreenGeometry(width_points=1920.0, height_points=1080.0, backing_scale=1.0)
        compat_res, reason_res = loaded.is_compatible("BuiltInCamera-001", diff_screen)
        assert not compat_res
        assert "Display" in reason_res


# --- Calibrated Gaze Estimator Real-Time Inference Tests ---

def test_calibrated_gaze_estimator_inference(macbook_air_screen):
    screen = macbook_air_screen
    calib_targets = generate_13_point_calibration_grid()
    X_train = np.tile(np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]), (13, 1))
    Y_train = np.array([[t.norm_x, t.norm_y] for t in calib_targets])

    model = AffineGazeModel().fit(X_train, Y_train)
    estimator = CalibratedGazeEstimator(model=model)

    assert estimator.is_calibrated

    # Construct mock EyeMetrics
    norm_left = NormalizedEyeCoords(iris_x=0.5, iris_y=0.0, openness=0.25, ear=0.3)
    norm_right = NormalizedEyeCoords(iris_x=0.5, iris_y=0.0, openness=0.25, ear=0.3)
    p2d = Point2D(0.0, 0.0)

    eye_metrics = EyeMetrics(
        left_corner_inner=p2d, left_corner_outer=p2d, left_iris_center=p2d, left_normalized=norm_left,
        right_corner_inner=p2d, right_corner_outer=p2d, right_iris_center=p2d, right_normalized=norm_right,
        gaze_feature_vector=np.zeros(4),
    )
    head_pose = HeadPose(yaw=0.0, pitch=0.0, roll=0.0, tx=0.0, ty=0.0, tz=50.0)

    estimate = estimator.estimate(eye_metrics, head_pose, timestamp_s=1.0)

    assert estimate.is_valid
    assert 0.0 <= estimate.raw_x <= 1.0
    assert 0.0 <= estimate.raw_y <= 1.0
    assert 0.0 <= estimate.filtered_x <= 1.0
    assert 0.0 <= estimate.filtered_y <= 1.0
    assert estimate.confidence > 0.8
