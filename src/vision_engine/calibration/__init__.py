"""EyeMouse eye-screen calibration package."""

from .geometry import (
    ScreenGeometry,
    TargetPoint,
    generate_13_point_calibration_grid,
    generate_9_point_validation_grid,
    generate_32_point_validation_grid,
)
from .collector import (
    RawGazeSample,
    TargetSampleAggregate,
    TargetSampleCollector,
)
from .models import (
    BaseGazeModel,
    AffineGazeModel,
    PolynomialGazeModel,
    PhysicallyInformedGazeModel,
    solve_ridge_regression,
    cross_validate_model,
)
from .adaptive_model import (
    AdaptiveGazeModel,
    get_base_prior_weights,
    get_default_lambda_diag,
    FEATURE_DIM,
)
from .evaluator import (
    CalibrationEvaluator,
    CalibrationValidationReport,
    TargetPredictionResult,
)
from .profile import (
    CalibrationProfile,
    create_profile,
    DEFAULT_PROFILE_DIR,
    DEFAULT_PROFILE_NAME,
)
from .estimator import CalibratedGazeEstimator
from .session import CalibrationSession

__all__ = [
    "ScreenGeometry",
    "TargetPoint",
    "generate_13_point_calibration_grid",
    "generate_9_point_validation_grid",
    "generate_32_point_validation_grid",
    "RawGazeSample",
    "TargetSampleAggregate",
    "TargetSampleCollector",
    "BaseGazeModel",
    "AffineGazeModel",
    "PolynomialGazeModel",
    "PhysicallyInformedGazeModel",
    "AdaptiveGazeModel",
    "solve_ridge_regression",
    "cross_validate_model",
    "CalibrationEvaluator",
    "CalibrationValidationReport",
    "TargetPredictionResult",
    "CalibrationProfile",
    "create_profile",
    "DEFAULT_PROFILE_DIR",
    "DEFAULT_PROFILE_NAME",
    "CalibratedGazeEstimator",
    "CalibrationSession",
]

