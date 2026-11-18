"""Calibration session coordinator orchestrating target presentation, model training, and validation."""

from __future__ import annotations

import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .geometry import (
    ScreenGeometry,
    TargetPoint,
    generate_13_point_calibration_grid,
    generate_9_point_validation_grid,
)
from .collector import TargetSampleCollector, TargetSampleAggregate, RawGazeSample
from .models import (
    BaseGazeModel,
    AffineGazeModel,
    PolynomialGazeModel,
    PhysicallyInformedGazeModel,
    cross_validate_model,
)
from .evaluator import CalibrationEvaluator, CalibrationValidationReport
from .profile import CalibrationProfile, create_profile


class CalibrationSession:
    """Coordinates a full 13-point calibration and 9-point held-out validation session."""

    def __init__(
        self,
        screen: ScreenGeometry,
        camera_id: str = "MacBook Air Camera",
        randomize_order: bool = True,
        settling_duration_s: float = 0.5,
        collection_duration_s: float = 1.0,
    ) -> None:
        self.screen = screen
        self.camera_id = camera_id
        self.randomize_order = randomize_order
        self.settling_duration_s = settling_duration_s
        self.collection_duration_s = collection_duration_s

        self.calibration_targets: List[TargetPoint] = generate_13_point_calibration_grid()
        self.validation_targets: List[TargetPoint] = generate_9_point_validation_grid()

        if self.randomize_order:
            # Keep center as first target, randomize remaining
            first = self.calibration_targets[0]
            rest = self.calibration_targets[1:]
            random.seed(42)  # Deterministic seed for reproducible testing
            random.shuffle(rest)
            self.ordered_calibration_targets = [first] + rest
        else:
            self.ordered_calibration_targets = list(self.calibration_targets)

        self.evaluator = CalibrationEvaluator(self.screen)

    def fit_and_compare_models(
        self,
        calib_aggregates: List[TargetSampleAggregate],
        val_aggregates: List[TargetSampleAggregate],
    ) -> Tuple[BaseGazeModel, CalibrationValidationReport, Dict[str, Any]]:
        """Fit Affine and Polynomial models, cross-validate, and evaluate against held-out targets.
        
        Returns:
            (best_model, validation_report, comparison_summary)
        """
        # Validate aggregates
        valid_calib = [a for a in calib_aggregates if a.is_valid]
        valid_val = [a for a in val_aggregates if a.is_valid]

        if len(valid_calib) < 9:
            raise ValueError(f"Insufficient valid calibration targets: {len(valid_calib)} < 9")
        if len(valid_val) < 6:
            raise ValueError(f"Insufficient valid validation targets: {len(valid_val)} < 6")

        # Training matrices
        X_train = np.array([a.feature_median for a in valid_calib], dtype=np.float64)  # (N, 7)
        Y_train = np.array([[a.target_norm_x, a.target_norm_y] for a in valid_calib], dtype=np.float64)  # (N, 2)

        # Validation matrices (held-out)
        X_val = np.array([a.feature_median for a in valid_val], dtype=np.float64)      # (M, 7)
        val_target_points = [
            TargetPoint(
                index=a.target_index,
                name=a.target_name,
                norm_x=a.target_norm_x,
                norm_y=a.target_norm_y,
                is_center=a.is_center or (0.30 <= a.target_norm_x <= 0.70 and 0.22 <= a.target_norm_y <= 0.78),
                is_corner=a.is_corner,
                is_edge=a.is_edge,
            )
            for a in valid_val
        ]

        alpha_candidates = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

        # 1. Fit Affine Baseline
        best_alpha_aff, cv_aff = cross_validate_model(
            AffineGazeModel, X_train, Y_train, alpha_candidates
        )
        affine_model = AffineGazeModel(alpha=best_alpha_aff)
        affine_model.fit(X_train, Y_train)
        affine_report = self.evaluator.evaluate(affine_model, X_val, val_target_points)

        # 2. Fit Physically-Informed Ridge
        best_alpha_phys, cv_phys = cross_validate_model(
            PhysicallyInformedGazeModel, X_train, Y_train, alpha_candidates
        )
        phys_model = PhysicallyInformedGazeModel(alpha=best_alpha_phys)
        phys_model.fit(X_train, Y_train)
        phys_report = self.evaluator.evaluate(phys_model, X_val, val_target_points)

        # 3. Fit Polynomial Degree-2 Ridge
        best_alpha_poly, cv_poly = cross_validate_model(
            PolynomialGazeModel, X_train, Y_train, alpha_candidates
        )
        poly_model = PolynomialGazeModel(alpha=best_alpha_poly)
        poly_model.fit(X_train, Y_train)
        poly_report = self.evaluator.evaluate(poly_model, X_val, val_target_points)

        # 4. Model selection based on held-out validation error
        candidates = [
            (phys_model, phys_report),
            (affine_model, affine_report),
            (poly_model, poly_report),
        ]
        # Sort primarily by median validation error, preferring valid models
        candidates.sort(key=lambda item: (not item[1].is_valid_for_use, item[1].median_pixel_error))
        best_model, best_report = candidates[0]

        comparison = {
            "affine": {
                "alpha": best_alpha_aff,
                "cv_rmse": round(cv_aff, 4),
                "median_pixel_error": round(affine_report.median_pixel_error, 2),
                "p95_pixel_error": round(affine_report.p95_pixel_error, 2),
                "quality": affine_report.quality_grade,
            },
            "physically_informed": {
                "alpha": best_alpha_phys,
                "cv_rmse": round(cv_phys, 4),
                "median_pixel_error": round(phys_report.median_pixel_error, 2),
                "p95_pixel_error": round(phys_report.p95_pixel_error, 2),
                "quality": phys_report.quality_grade,
            },
            "polynomial": {
                "alpha": best_alpha_poly,
                "cv_rmse": round(cv_poly, 4),
                "median_pixel_error": round(poly_report.median_pixel_error, 2),
                "p95_pixel_error": round(poly_report.p95_pixel_error, 2),
                "quality": poly_report.quality_grade,
            },
            "selected_model": best_model.model_name,
        }

        return best_model, best_report, comparison

