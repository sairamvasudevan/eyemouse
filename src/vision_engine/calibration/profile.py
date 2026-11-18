"""Calibration profile persistence, validation, and metadata management."""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .geometry import ScreenGeometry
from .models import (
    BaseGazeModel,
    AffineGazeModel,
    PolynomialGazeModel,
    PhysicallyInformedGazeModel,
)
from .evaluator import CalibrationValidationReport


DEFAULT_PROFILE_DIR = Path.home() / ".eyemouse" / "profiles"
DEFAULT_PROFILE_NAME = "active_profile.json"


@dataclass
class CalibrationProfile:
    """Persistent personalized gaze calibration profile."""
    version: int
    created_at_iso: str
    camera_unique_id: str
    camera_name: str
    camera_resolution: str
    pixel_format: str
    screen_width_points: float
    screen_height_points: float
    backing_scale_factor: float
    model_name: str
    model_dict: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    quality_grade: str
    is_valid: bool

    def instantiate_model(self) -> BaseGazeModel:
        """Instantiate the trained model from stored parameters."""
        if self.model_name == "affine_linear":
            return AffineGazeModel.from_dict(self.model_dict)
        elif self.model_name == "polynomial_deg2_ridge":
            return PolynomialGazeModel.from_dict(self.model_dict)
        elif self.model_name == "physically_informed_ridge":
            return PhysicallyInformedGazeModel.from_dict(self.model_dict)
        else:
            raise ValueError(f"Unknown model architecture: {self.model_name}")

    def is_compatible(
        self,
        current_camera_id: str,
        current_screen: ScreenGeometry,
    ) -> Tuple[bool, Optional[str]]:
        """Verify whether profile matches current physical hardware and display setup."""
        if not self.is_valid:
            return False, "Profile marked invalid"

        if self.camera_unique_id and current_camera_id:
            if self.camera_unique_id != current_camera_id:
                return False, f"Camera mismatch: profile={self.camera_unique_id}, current={current_camera_id}"

        # Display geometry tolerance (e.g. resolution change, scaling change)
        if abs(self.screen_width_points - current_screen.width_points) > 10.0:
            return False, f"Display width changed: {self.screen_width_points} vs {current_screen.width_points}"
        if abs(self.screen_height_points - current_screen.height_points) > 10.0:
            return False, f"Display height changed: {self.screen_height_points} vs {current_screen.height_points}"
        if abs(self.backing_scale_factor - current_screen.backing_scale) > 0.1:
            return False, f"Display scale changed: {self.backing_scale_factor} vs {current_screen.backing_scale}"

        return True, None

    def save_to_file(self, filepath: Optional[Path] = None) -> Path:
        """Save profile to JSON file."""
        path = filepath or (DEFAULT_PROFILE_DIR / DEFAULT_PROFILE_NAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "created_at_iso": self.created_at_iso,
            "camera_unique_id": self.camera_unique_id,
            "camera_name": self.camera_name,
            "camera_resolution": self.camera_resolution,
            "pixel_format": self.pixel_format,
            "screen_width_points": self.screen_width_points,
            "screen_height_points": self.screen_height_points,
            "backing_scale_factor": self.backing_scale_factor,
            "model_name": self.model_name,
            "model_dict": self.model_dict,
            "validation_metrics": self.validation_metrics,
            "quality_grade": self.quality_grade,
            "is_valid": self.is_valid,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    @classmethod
    def load_from_file(cls, filepath: Optional[Path] = None) -> CalibrationProfile:
        """Load and parse profile from JSON file."""
        path = filepath or (DEFAULT_PROFILE_DIR / DEFAULT_PROFILE_NAME)
        if not path.exists():
            raise FileNotFoundError(f"Calibration profile not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            version=data.get("version", 1),
            created_at_iso=data.get("created_at_iso", ""),
            camera_unique_id=data.get("camera_unique_id", ""),
            camera_name=data.get("camera_name", "Unknown Camera"),
            camera_resolution=data.get("camera_resolution", "640x480"),
            pixel_format=data.get("pixel_format", "BGRA"),
            screen_width_points=float(data.get("screen_width_points", 1470.0)),
            screen_height_points=float(data.get("screen_height_points", 956.0)),
            backing_scale_factor=float(data.get("backing_scale_factor", 2.0)),
            model_name=data.get("model_name", "polynomial_deg2_ridge"),
            model_dict=data.get("model_dict", {}),
            validation_metrics=data.get("validation_metrics", {}),
            quality_grade=data.get("quality_grade", "FAIR"),
            is_valid=bool(data.get("is_valid", True)),
        )


def create_profile(
    model: BaseGazeModel,
    validation_report: CalibrationValidationReport,
    screen: ScreenGeometry,
    camera_id: str,
    camera_name: str = "MacBook Air Camera",
    camera_resolution: str = "640x480",
    pixel_format: str = "BGRA",
) -> CalibrationProfile:
    """Factory helper to construct a verified CalibrationProfile."""
    return CalibrationProfile(
        version=1,
        created_at_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        camera_unique_id=camera_id,
        camera_name=camera_name,
        camera_resolution=camera_resolution,
        pixel_format=pixel_format,
        screen_width_points=screen.width_points,
        screen_height_points=screen.height_points,
        backing_scale_factor=screen.backing_scale,
        model_name=model.model_name,
        model_dict=model.to_dict(),
        validation_metrics=validation_report.to_dict(),
        quality_grade=validation_report.quality_grade,
        is_valid=validation_report.is_valid_for_use,
    )

