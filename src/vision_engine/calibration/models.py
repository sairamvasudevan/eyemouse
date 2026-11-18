"""Mathematical regression models for gaze-to-screen coordinate mapping.

Implements:
1. Affine / Linear Regression Baseline (with Ridge shrinkage).
2. Regularized Polynomial Degree-2 Regression (with cross-validated L2 penalty).
3. Leave-One-Out / K-Fold Cross-Validation for hyperparameter tuning and model selection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import numpy as np


def solve_ridge_regression(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 1.0,
    penalize_intercept: bool = False,
) -> np.ndarray:
    """Solve regularized Ridge regression (W) minimizing ||X W - Y||^2 + alpha * ||W||^2.
    
    Args:
        X: (N, D) design matrix (first column is usually 1.0 for bias).
        Y: (N, K) target screen coordinates [u, v].
        alpha: Regularization strength (L2 weight decay).
        penalize_intercept: Whether to penalize the first (bias) column.
    
    Returns:
        W: (D, K) coefficient matrix.
    """
    N, D = X.shape
    reg_diag = np.ones(D, dtype=np.float64) * alpha
    if not penalize_intercept and D > 0:
        reg_diag[0] = 0.0  # Do not shrink the bias term

    reg_matrix = np.diag(reg_diag)
    XTX = X.T @ X + reg_matrix
    XTY = X.T @ Y

    # Solve XTX @ W = XTY using numerically stable SVD/lstsq or solve
    try:
        W = np.linalg.solve(XTX, XTY)
    except np.linalg.LinAlgError:
        W, _, _, _ = np.linalg.lstsq(XTX, XTY, rcond=None)
    return W


class BaseGazeModel(ABC):
    """Abstract base class for gaze regression models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier name of the model architecture."""
        ...

    @abstractmethod
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Expand raw 7-dimensional gaze features into the model's design matrix."""
        ...

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> BaseGazeModel:
        """Fit model weights to training samples."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict screen coordinates (u, v) in [0..1] range."""
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model weights and metadata to dictionary."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaseGazeModel:
        """Instantiate model from serialized dictionary."""
        ...


class AffineGazeModel(BaseGazeModel):
    """Affine / Linear Gaze Regression Baseline.
    
    Maps [1, left_x, left_y, right_x, right_y, yaw, pitch, roll] linearly to [u, v].
    """

    def __init__(self, alpha: float = 0.1, weights: Optional[np.ndarray] = None) -> None:
        self.alpha = float(alpha)
        self.weights = weights  # Shape: (8, 2)

    @property
    def model_name(self) -> str:
        return "affine_linear"

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Expand (N, 7) or (7,) into (N, 8) with bias term."""
        X_arr = np.atleast_2d(X)
        N = X_arr.shape[0]
        bias = np.ones((N, 1), dtype=np.float64)
        return np.hstack([bias, X_arr])

    def fit(self, X: np.ndarray, Y: np.ndarray, alpha: Optional[float] = None) -> AffineGazeModel:
        if alpha is not None:
            self.alpha = float(alpha)
        X_design = self.transform_features(X)
        Y_arr = np.atleast_2d(Y)
        self.weights = solve_ridge_regression(X_design, Y_arr, alpha=self.alpha, penalize_intercept=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fitted.")
        X_design = self.transform_features(X)
        preds = X_design @ self.weights
        return np.clip(preds, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "alpha": self.alpha,
            "weights": self.weights.tolist() if self.weights is not None else [],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AffineGazeModel:
        weights = np.array(data["weights"], dtype=np.float64) if data.get("weights") else None
        return cls(alpha=data.get("alpha", 0.1), weights=weights)


class PolynomialGazeModel(BaseGazeModel):
    """Regularized Degree-2 Polynomial Gaze Regression.
    
    Includes linear terms, quadratic terms, and cross-eye/pose interaction terms.
    Total features: 19.
    """

    def __init__(self, alpha: float = 1.0, weights: Optional[np.ndarray] = None) -> None:
        self.alpha = float(alpha)
        self.weights = weights  # Shape: (19, 2)

    @property
    def model_name(self) -> str:
        return "polynomial_deg2_ridge"

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Expand raw features:
        [1,
         lx, ly, rx, ry, yaw, pitch, roll,          (7 linear)
         lx^2, ly^2, rx^2, ry^2, yaw^2, pitch^2, roll^2, (7 quadratic)
         lx*ly, rx*ry, lx*yaw, ly*pitch]             (4 interaction)
        """
        X_arr = np.atleast_2d(X)
        N = X_arr.shape[0]

        lx = X_arr[:, 0:1]
        ly = X_arr[:, 1:2]
        rx = X_arr[:, 2:3]
        ry = X_arr[:, 3:4]
        # Scale head pose angles to [-1.0, 1.0] order of magnitude to prevent ill-conditioning
        yaw = X_arr[:, 4:5] / 30.0
        pitch = X_arr[:, 5:6] / 25.0
        roll = X_arr[:, 6:7] / 20.0

        bias = np.ones((N, 1), dtype=np.float64)

        linear = np.hstack([lx, ly, rx, ry, yaw, pitch, roll])
        quadratic = np.hstack([lx**2, ly**2, rx**2, ry**2, yaw**2, pitch**2, roll**2])
        interactions = np.hstack([lx * ly, rx * ry, lx * yaw, ly * pitch])

        return np.hstack([bias, linear, quadratic, interactions])

    def fit(self, X: np.ndarray, Y: np.ndarray, alpha: Optional[float] = None) -> PolynomialGazeModel:
        if alpha is not None:
            self.alpha = float(alpha)
        X_design = self.transform_features(X)
        Y_arr = np.atleast_2d(Y)
        self.weights = solve_ridge_regression(X_design, Y_arr, alpha=self.alpha, penalize_intercept=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fitted.")
        X_design = self.transform_features(X)
        preds = X_design @ self.weights
        return np.clip(preds, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "alpha": self.alpha,
            "weights": self.weights.tolist() if self.weights is not None else [],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PolynomialGazeModel:
        weights = np.array(data["weights"], dtype=np.float64) if data.get("weights") else None
        return cls(alpha=data.get("alpha", 1.0), weights=weights)


class PhysicallyInformedGazeModel(BaseGazeModel):
    """Physically-Informed 12-Parameter Regularized Ridge Regression.
    
    Incorporates:
    - Conjugate iris center: avg_x = (lx + rx)/2, avg_y = (ly + ry)/2
    - Inter-ocular horizontal disparity: disp_x = rx - lx (distance/vergence signal)
    - Normalized head pose angles: yaw/30, pitch/25, roll/20
    - Quadratic gaze curvature terms: avg_x^2, avg_y^2
    - Head-eye coupling and vertical pitch parallax interaction: avg_x*yaw, avg_y*pitch, avg_x*avg_y
    """

    def __init__(self, alpha: float = 0.05, weights: Optional[np.ndarray] = None) -> None:
        self.alpha = float(alpha)
        self.weights = weights  # Shape: (12, 2)

    @property
    def model_name(self) -> str:
        return "physically_informed_ridge"

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.atleast_2d(X)
        N = X_arr.shape[0]

        lx = X_arr[:, 0:1]
        ly = X_arr[:, 1:2]
        rx = X_arr[:, 2:3]
        ry = X_arr[:, 3:4]
        yaw = X_arr[:, 4:5] / 30.0
        pitch = X_arr[:, 5:6] / 25.0
        roll = X_arr[:, 6:7] / 20.0

        avg_x = (lx + rx) / 2.0 - 0.5
        avg_y = (ly + ry) / 2.0
        disp_x = rx - lx

        terms = [
            np.ones((N, 1), dtype=np.float64),
            avg_x,
            avg_y,
            disp_x,
            yaw,
            pitch,
            roll,
            avg_x**2,
            avg_y**2,
            avg_x * yaw,
            avg_y * pitch,
            avg_x * avg_y,
        ]
        return np.hstack(terms)

    def fit(self, X: np.ndarray, Y: np.ndarray, alpha: Optional[float] = None) -> PhysicallyInformedGazeModel:
        if alpha is not None:
            self.alpha = float(alpha)
        X_design = self.transform_features(X)
        Y_arr = np.atleast_2d(Y)
        self.weights = solve_ridge_regression(X_design, Y_arr, alpha=self.alpha, penalize_intercept=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fitted.")
        X_design = self.transform_features(X)
        preds = X_design @ self.weights
        return np.clip(preds, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "alpha": self.alpha,
            "weights": self.weights.tolist() if self.weights is not None else [],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PhysicallyInformedGazeModel:
        weights = np.array(data["weights"], dtype=np.float64) if data.get("weights") else None
        return cls(alpha=data.get("alpha", 0.05), weights=weights)


def cross_validate_model(
    model_factory,
    X: np.ndarray,
    Y: np.ndarray,
    alpha_candidates: List[float],
) -> Tuple[float, float]:
    """Perform Leave-One-Out Cross-Validation (LOOCV) to select the optimal alpha.
    
    Returns:
        (best_alpha, min_cv_rmse)
    """
    N = X.shape[0]
    best_alpha = alpha_candidates[0]
    min_cv_rmse = float("inf")

    for alpha in alpha_candidates:
        errors = []
        for i in range(N):
            # Split LOO
            train_idx = [j for j in range(N) if j != i]
            val_idx = [i]

            X_tr, Y_tr = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]

            m = model_factory(alpha=alpha)
            m.fit(X_tr, Y_tr)
            y_pred = m.predict(X_val)

            err = np.sqrt(np.sum((y_pred - Y_val) ** 2))
            errors.append(err)

        cv_rmse = float(np.mean(errors))
        if cv_rmse < min_cv_rmse:
            min_cv_rmse = cv_rmse
            best_alpha = alpha

    return best_alpha, min_cv_rmse

