"""Held-out validation and calibration quality assessment engine.

Calculates comprehensive empirical accuracy metrics against independent targets:
- Mean, Median, and P95 Euclidean pixel errors.
- Independent horizontal (dx) and vertical (dy) errors.
- Regional breakdown (Center vs Corners vs Edges).
- Visual angular error (degrees visual angle).
- Quality scoring grounded in held-out validation results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .geometry import ScreenGeometry, TargetPoint
from .models import BaseGazeModel


@dataclass(slots=True)
class TargetPredictionResult:
    """Individual target prediction error assessment."""
    target_index: int
    target_name: str
    is_center: bool
    is_corner: bool
    is_edge: bool
    ground_truth_norm: Tuple[float, float]
    predicted_norm: Tuple[float, float]
    error_norm: float
    error_norm_x: float
    error_norm_y: float
    error_pixels: float
    error_pixels_x: float
    error_pixels_y: float
    error_visual_degrees: float


@dataclass(slots=True)
class CalibrationValidationReport:
    """Complete validation scorecard evaluating generalization to held-out targets."""
    model_name: str
    total_validation_targets: int
    mean_pixel_error: float
    median_pixel_error: float
    p95_pixel_error: float
    max_pixel_error: float
    min_pixel_error: float
    mean_norm_error: float
    median_norm_error: float
    mean_pixel_error_x: float
    mean_pixel_error_y: float
    center_mean_pixel_error: float
    corner_mean_pixel_error: float
    edge_mean_pixel_error: float
    mean_visual_degrees: float
    median_visual_degrees: float
    p95_visual_degrees: float
    median_error_pct_diagonal: float
    p95_error_pct_diagonal: float
    quality_grade: str              # "EXCELLENT", "GOOD", "FAIR", "POOR"
    is_valid_for_use: bool
    problematic_regions: List[str] = field(default_factory=list)
    target_results: List[TargetPredictionResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_validation_targets": self.total_validation_targets,
            "mean_pixel_error": round(self.mean_pixel_error, 2),
            "median_pixel_error": round(self.median_pixel_error, 2),
            "p95_pixel_error": round(self.p95_pixel_error, 2),
            "max_pixel_error": round(self.max_pixel_error, 2),
            "min_pixel_error": round(self.min_pixel_error, 2),
            "mean_norm_error": round(self.mean_norm_error, 4),
            "median_norm_error": round(self.median_norm_error, 4),
            "mean_pixel_error_x": round(self.mean_pixel_error_x, 2),
            "mean_pixel_error_y": round(self.mean_pixel_error_y, 2),
            "center_mean_pixel_error": round(self.center_mean_pixel_error, 2),
            "corner_mean_pixel_error": round(self.corner_mean_pixel_error, 2),
            "edge_mean_pixel_error": round(self.edge_mean_pixel_error, 2),
            "mean_visual_degrees": round(self.mean_visual_degrees, 2),
            "median_visual_degrees": round(self.median_visual_degrees, 2),
            "p95_visual_degrees": round(self.p95_visual_degrees, 2),
            "median_error_pct_diagonal": round(self.median_error_pct_diagonal, 2),
            "p95_error_pct_diagonal": round(self.p95_error_pct_diagonal, 2),
            "quality_grade": self.quality_grade,
            "is_valid_for_use": self.is_valid_for_use,
            "problematic_regions": self.problematic_regions,
            "target_results": [
                {
                    "target_index": r.target_index,
                    "target_name": r.target_name,
                    "ground_truth_norm": list(r.ground_truth_norm),
                    "predicted_norm": list(r.predicted_norm),
                    "error_pixels": round(r.error_pixels, 2),
                    "error_pixels_x": round(r.error_pixels_x, 2),
                    "error_pixels_y": round(r.error_pixels_y, 2),
                    "error_visual_degrees": round(r.error_visual_degrees, 2),
                }
                for r in self.target_results
            ],
        }


class CalibrationEvaluator:
    """Evaluates gaze models against independent validation observations."""

    def __init__(self, screen_geometry: ScreenGeometry) -> None:
        self.screen = screen_geometry

    def evaluate(
        self,
        model: BaseGazeModel,
        validation_features: np.ndarray,          # (M, 7)
        validation_targets: List[TargetPoint],     # M targets
    ) -> CalibrationValidationReport:
        """Evaluate fitted model against held-out validation points."""
        M = len(validation_targets)
        assert validation_features.shape[0] == M, f"Feature count {validation_features.shape[0]} != target count {M}"

        # 1. Generate predictions in normalized space
        preds_norm = model.predict(validation_features)  # (M, 2)

        results: List[TargetPredictionResult] = []
        diag_pixels = self.screen.diagonal_pixels

        for i, target in enumerate(validation_targets):
            gt_u, gt_v = target.norm_x, target.norm_y
            pred_u, pred_v = float(preds_norm[i, 0]), float(preds_norm[i, 1])

            # Normalized coordinate error
            du = pred_u - gt_u
            dv = pred_v - gt_v
            norm_err = math.hypot(du, dv)

            # Convert to physical Retina pixels
            gt_px_x = gt_u * self.screen.width_pixels
            gt_px_y = gt_v * self.screen.height_pixels
            pred_px_x = pred_u * self.screen.width_pixels
            pred_px_y = pred_v * self.screen.height_pixels

            dx_px = abs(pred_px_x - gt_px_x)
            dy_px = abs(pred_px_y - gt_px_y)
            pixel_err = math.hypot(dx_px, dy_px)

            # Angular visual degree error
            vis_deg = self.screen.pixel_error_to_visual_degrees(pixel_err)

            results.append(
                TargetPredictionResult(
                    target_index=target.index,
                    target_name=target.name,
                    is_center=target.is_center,
                    is_corner=target.is_corner,
                    is_edge=target.is_edge,
                    ground_truth_norm=(gt_u, gt_v),
                    predicted_norm=(pred_u, pred_v),
                    error_norm=norm_err,
                    error_norm_x=abs(du),
                    error_norm_y=abs(dv),
                    error_pixels=pixel_err,
                    error_pixels_x=dx_px,
                    error_pixels_y=dy_px,
                    error_visual_degrees=vis_deg,
                )
            )

        # Aggregate metrics
        pixel_errors = np.array([r.error_pixels for r in results])
        pixel_errors_x = np.array([r.error_pixels_x for r in results])
        pixel_errors_y = np.array([r.error_pixels_y for r in results])
        norm_errors = np.array([r.error_norm for r in results])
        vis_deg_errors = np.array([r.error_visual_degrees for r in results])

        mean_px = float(np.mean(pixel_errors))
        median_px = float(np.median(pixel_errors))
        p95_px = float(np.percentile(pixel_errors, 95))
        max_px = float(np.max(pixel_errors))
        min_px = float(np.min(pixel_errors))

        median_pct_diag = (median_px / diag_pixels) * 100.0
        p95_pct_diag = (p95_px / diag_pixels) * 100.0

        # Regional breakdowns
        center_errs = [r.error_pixels for r in results if r.is_center]
        corner_errs = [r.error_pixels for r in results if r.is_corner]
        edge_errs = [r.error_pixels for r in results if r.is_edge]

        center_mean = float(np.mean(center_errs)) if center_errs else median_px
        corner_mean = float(np.mean(corner_errs)) if corner_errs else median_px * 1.3
        edge_mean = float(np.mean(edge_errs)) if edge_errs else median_px * 1.1

        # Problematic regions check
        problematic: List[str] = []
        if corner_mean > 1.8 * center_mean and corner_mean > 80.0:
            problematic.append("Display corners exhibit increased error")
        if np.mean(pixel_errors_y) > 1.5 * np.mean(pixel_errors_x):
            problematic.append("Vertical axis tracking has higher variance than horizontal")
        elif np.mean(pixel_errors_x) > 1.5 * np.mean(pixel_errors_y):
            problematic.append("Horizontal axis tracking has higher variance than vertical")

        # Quality scoring (target: median <= 2.5% diagonal, p95 <= 6.0% diagonal)
        if median_pct_diag <= 2.0 and p95_pct_diag <= 5.0:
            quality = "EXCELLENT"
            is_valid = True
        elif median_pct_diag <= 2.5 and p95_pct_diag <= 6.0:
            quality = "GOOD"
            is_valid = True
        elif median_pct_diag <= 4.0 and p95_pct_diag <= 8.5:
            quality = "FAIR"
            is_valid = True
        else:
            quality = "POOR"
            is_valid = False
            problematic.append(f"Validation error ({median_pct_diag:.1f}% diag) exceeds 4.0% threshold")

        return CalibrationValidationReport(
            model_name=model.model_name,
            total_validation_targets=M,
            mean_pixel_error=mean_px,
            median_pixel_error=median_px,
            p95_pixel_error=p95_px,
            max_pixel_error=max_px,
            min_pixel_error=min_px,
            mean_norm_error=float(np.mean(norm_errors)),
            median_norm_error=float(np.median(norm_errors)),
            mean_pixel_error_x=float(np.mean(pixel_errors_x)),
            mean_pixel_error_y=float(np.mean(pixel_errors_y)),
            center_mean_pixel_error=center_mean,
            corner_mean_pixel_error=corner_mean,
            edge_mean_pixel_error=edge_mean,
            mean_visual_degrees=float(np.mean(vis_deg_errors)),
            median_visual_degrees=float(np.median(vis_deg_errors)),
            p95_visual_degrees=float(np.percentile(vis_deg_errors, 95)),
            median_error_pct_diagonal=median_pct_diag,
            p95_error_pct_diagonal=p95_pct_diag,
            quality_grade=quality,
            is_valid_for_use=is_valid,
            problematic_regions=problematic,
            target_results=results,
        )

