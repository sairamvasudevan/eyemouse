"""CLI script for processing interactive calibration samples, model fitting, and validation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np

from src.vision_engine.calibration.geometry import (
    ScreenGeometry,
    TargetPoint,
)
from src.vision_engine.calibration.collector import TargetSampleAggregate
from src.vision_engine.calibration.session import CalibrationSession
from src.vision_engine.calibration.profile import (
    create_profile,
    DEFAULT_PROFILE_DIR,
    DEFAULT_PROFILE_NAME,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and validate gaze calibration models")
    parser.add_argument("--input", required=True, help="Path to input session samples JSON")
    parser.add_argument("--output-profile", default=str(DEFAULT_PROFILE_DIR / DEFAULT_PROFILE_NAME))
    parser.add_argument("--output-report", required=True, help="Path to save validation scorecard JSON")
    parser.add_argument("--camera-id", default="MacBook Air Camera")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found", file=sys.stderr)
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Parse Screen Geometry
    s_info = data.get("screen", {})
    screen = ScreenGeometry(
        width_points=float(s_info.get("width_points", 1470.0)),
        height_points=float(s_info.get("height_points", 956.0)),
        backing_scale=float(s_info.get("backing_scale", 2.0)),
    )

    # 2. Parse Calibration Aggregates
    calib_list = data.get("calibration_targets", [])
    calib_aggregates = []
    for item in calib_list:
        med = np.array(item["feature_median"], dtype=np.float64)
        std = np.array(item.get("feature_std", [0.0]*7), dtype=np.float64)
        calib_aggregates.append(
            TargetSampleAggregate(
                target_index=int(item["target_index"]),
                target_name=str(item["target_name"]),
                target_norm_x=float(item["target_norm_x"]),
                target_norm_y=float(item["target_norm_y"]),
                feature_median=med,
                feature_std=std,
                total_samples_collected=int(item.get("total_samples", 30)),
                valid_samples_accepted=int(item.get("valid_samples", 25)),
                outliers_rejected=int(item.get("outliers_rejected", 2)),
                blinks_rejected=int(item.get("blinks_rejected", 0)),
                is_valid=bool(item.get("is_valid", True)),
                is_center=bool(item.get("is_center", False)),
                is_corner=bool(item.get("is_corner", False)),
                is_edge=bool(item.get("is_edge", False)),
            )
        )

    # 3. Parse Validation Aggregates
    val_list = data.get("validation_targets", [])
    val_aggregates = []
    for item in val_list:
        med = np.array(item["feature_median"], dtype=np.float64)
        std = np.array(item.get("feature_std", [0.0]*7), dtype=np.float64)
        val_aggregates.append(
            TargetSampleAggregate(
                target_index=int(item["target_index"]),
                target_name=str(item["target_name"]),
                target_norm_x=float(item["target_norm_x"]),
                target_norm_y=float(item["target_norm_y"]),
                feature_median=med,
                feature_std=std,
                total_samples_collected=int(item.get("total_samples", 30)),
                valid_samples_accepted=int(item.get("valid_samples", 25)),
                outliers_rejected=int(item.get("outliers_rejected", 2)),
                blinks_rejected=int(item.get("blinks_rejected", 0)),
                is_valid=bool(item.get("is_valid", True)),
                is_center=bool(item.get("is_center", False)),
                is_corner=bool(item.get("is_corner", False)),
                is_edge=bool(item.get("is_edge", False)),
            )
        )

    session = CalibrationSession(screen=screen, camera_id=args.camera_id)
    best_model, report, comparison = session.fit_and_compare_models(
        calib_aggregates, val_aggregates
    )

    # 4. Create and save profile
    profile = create_profile(
        model=best_model,
        validation_report=report,
        screen=screen,
        camera_id=args.camera_id,
    )
    profile_path = profile.save_to_file(Path(args.output_profile))
    print(f"Saved calibration profile to {profile_path}")

    # 5. Save validation scorecard
    scorecard = {
        "report": report.to_dict(),
        "comparison": comparison,
        "profile_path": str(profile_path),
    }
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"Saved scorecard to {args.output_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
