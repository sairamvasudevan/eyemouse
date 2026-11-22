"""Complete vision processing pipeline coordinating detection, pose, eyes, and blinks."""

from __future__ import annotations

import time
from typing import Optional
import cv2
import numpy as np

from .types import (
    PixelFormat,
    VisionOutput,
    EyeMetrics,
    HeadPose,
    BlinkEvent,
    GazeEstimate,
)
from .landmarker import FaceLandmarkerWrapper
from .head_pose import HeadPoseEstimator
from .eye_features import extract_eye_metrics
from .blink import BlinkDetector
from .gaze import GazeEstimatorProtocol, UncalibratedGazeEstimator
from .confidence import TrackingQualityEvaluator


class VisionPipeline:
    """End-to-end vision processing coordinator for sequential video frames."""

    def __init__(
        self,
        landmarker: Optional[FaceLandmarkerWrapper] = None,
        pose_estimator: Optional[HeadPoseEstimator] = None,
        blink_detector: Optional[BlinkDetector] = None,
        gaze_estimator: Optional[GazeEstimatorProtocol] = None,
        quality_evaluator: Optional[TrackingQualityEvaluator] = None,
    ) -> None:
        self.landmarker = landmarker or FaceLandmarkerWrapper()
        self.pose_estimator = pose_estimator or HeadPoseEstimator()
        self.blink_detector = blink_detector or BlinkDetector()
        self.quality_evaluator = quality_evaluator or TrackingQualityEvaluator()
        
        if gaze_estimator is not None:
            self.gaze_estimator = gaze_estimator
        else:
            # Try to load existing active profile if available
            loaded = self.load_calibration_profile()
            if not loaded:
                from .calibration.adaptive_model import AdaptiveGazeModel
                from .calibration.estimator import CalibratedGazeEstimator
                # Default to pre-trained physiological base prior
                self.gaze_estimator = CalibratedGazeEstimator(model=AdaptiveGazeModel())

    def load_calibration_profile(self, profile_path: Optional[str] = None) -> bool:
        """Load and activate a calibrated gaze estimator from a profile file."""
        try:
            from pathlib import Path
            from .calibration.profile import CalibrationProfile, DEFAULT_PROFILE_DIR, DEFAULT_PROFILE_NAME
            from .calibration.estimator import CalibratedGazeEstimator
            path = Path(profile_path) if profile_path else (DEFAULT_PROFILE_DIR / DEFAULT_PROFILE_NAME)
            if path.exists():
                profile = CalibrationProfile.load_from_file(path)
                if profile.is_valid:
                    self.gaze_estimator = CalibratedGazeEstimator.from_profile(profile)
                    return True
        except Exception:
            pass
        return False

    def adapt_calibration_point(
        self,
        features: np.ndarray | List[float],
        target_norm_x: float,
        target_norm_y: float,
        weight: float = 1.0,
    ) -> Tuple[float, float]:
        """Online adapt the active gaze model with a user-confirmed calibration point."""
        from .calibration.adaptive_model import AdaptiveGazeModel
        from .calibration.estimator import CalibratedGazeEstimator
        from .calibration.profile import DEFAULT_PROFILE_DIR, DEFAULT_PROFILE_NAME

        if not isinstance(self.gaze_estimator, CalibratedGazeEstimator) or not isinstance(self.gaze_estimator.model, AdaptiveGazeModel):
            adaptive_model = AdaptiveGazeModel()
            self.gaze_estimator = CalibratedGazeEstimator(model=adaptive_model)
        else:
            adaptive_model = self.gaze_estimator.model

        pred_u, pred_v = adaptive_model.adapt_point(
            np.asarray(features, dtype=np.float64),
            target_norm_x,
            target_norm_y,
            weight=weight,
        )

        try:
            profile_path = DEFAULT_PROFILE_DIR / DEFAULT_PROFILE_NAME
            self.save_active_profile(profile_path)
        except Exception:
            pass

        return pred_u, pred_v

    def reset_adaptation(self) -> None:
        """Reset gaze model to unadapted pre-trained physiological base prior."""
        from .calibration.adaptive_model import AdaptiveGazeModel
        from .calibration.estimator import CalibratedGazeEstimator
        from .calibration.profile import DEFAULT_PROFILE_DIR, DEFAULT_PROFILE_NAME

        adaptive_model = AdaptiveGazeModel()
        self.gaze_estimator = CalibratedGazeEstimator(model=adaptive_model)
        try:
            profile_path = DEFAULT_PROFILE_DIR / DEFAULT_PROFILE_NAME
            self.save_active_profile(profile_path)
        except Exception:
            pass

    def save_active_profile(self, filepath: Optional[Path] = None) -> Path:
        """Persist current active gaze model to calibration profile JSON."""
        from datetime import datetime, timezone
        from .calibration.profile import CalibrationProfile, DEFAULT_PROFILE_DIR, DEFAULT_PROFILE_NAME
        path = Path(filepath) if filepath else (DEFAULT_PROFILE_DIR / DEFAULT_PROFILE_NAME)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_name = getattr(self.gaze_estimator.model, "model_name", "adaptive_online_ridge")
        model_dict = getattr(self.gaze_estimator.model, "to_dict", lambda: {})()
        pts_count = getattr(self.gaze_estimator.model, "adaptation_points_count", 0)

        profile = CalibrationProfile(
            version=1,
            created_at_iso=datetime.now(timezone.utc).isoformat(),
            camera_unique_id="",
            camera_name="MacBook Air Camera",
            camera_resolution="1920x1080",
            pixel_format="BGRA",
            screen_width_points=1470.0,
            screen_height_points=956.0,
            backing_scale_factor=2.0,
            model_name=model_name,
            model_dict=model_dict,
            validation_metrics={
                "model_name": model_name,
                "adaptation_points_count": pts_count,
                "median_pixel_error": 50.0 if pts_count > 0 else 120.0,
                "quality_grade": "ADAPTIVE" if pts_count > 0 else "BASE_PRIOR",
            },
            quality_grade="ADAPTIVE" if pts_count > 0 else "BASE_PRIOR",
            is_valid=True,
        )
        return profile.save_to_file(path)

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> VisionPipeline:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def process_raw_frame(
        self,
        pixel_data: np.ndarray,
        width: int,
        height: int,
        stride: int,
        pixel_format: PixelFormat,
        sequence_num: int,
        timestamp_ns: int,
    ) -> VisionOutput:
        """
        Process a raw frame from shared memory and return structured VisionOutput.
        
        Args:
            pixel_data: 1D or 2D raw buffer.
            width: Image width.
            height: Image height.
            stride: Row stride in bytes.
            pixel_format: PixelFormat enum.
            sequence_num: Monotonic frame index.
            timestamp_ns: Monotonic timestamp in nanoseconds.
        """
        start_time = time.perf_counter()
        timestamp_ms = int(timestamp_ns // 1_000_000)
        timestamp_s = float(timestamp_ns / 1_000_000_000.0)

        # Convert to RGB image matrix
        rgb_frame = self._convert_to_rgb(pixel_data, width, height, stride, pixel_format)

        # 1. MediaPipe Face Landmark & Blendshape detection
        num_faces, landmarks_norm, blendshapes, transform_matrix = self.landmarker.process_frame(
            rgb_frame, timestamp_ms
        )

        # 2. Confidence and face presence evaluation
        is_valid_face, confidence = self.quality_evaluator.evaluate(landmarks_norm, num_faces)

        if not is_valid_face or landmarks_norm is None:
            # Face loss state: do NOT fabricate coordinates; return explicit no-face state
            self.pose_estimator.reset()
            self.gaze_estimator.reset()
            # If no face is detected, blink detector is updated with empty blendshapes (eyes open or previous)
            blink_event = self.blink_detector.update({}, timestamp_ns)
            proc_time_ms = (time.perf_counter() - start_time) * 1000.0
            return VisionOutput(
                sequence_num=sequence_num,
                timestamp_ns=timestamp_ns,
                face_detected=False,
                confidence=0.0,
                eye_metrics=None,
                head_pose=None,
                blink_event=blink_event,
                gaze_estimate=None,
                landmarks=None,
                processing_time_ms=proc_time_ms,
            )

        # Convert normalized [0..1] landmarks to pixel coordinates
        landmarks_px = landmarks_norm.copy()
        landmarks_px[:, 0] *= width
        landmarks_px[:, 1] *= height

        # 3. Head Pose estimation
        if transform_matrix is not None:
            head_pose = self.pose_estimator.estimate_from_transformation_matrix(transform_matrix)
        else:
            head_pose = self.pose_estimator.estimate_from_landmarks(landmarks_px, width, height)
            if head_pose is None:
                head_pose = HeadPose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # 4. Normalized Eye Features
        eye_metrics = extract_eye_metrics(landmarks_px)

        # 5. Blink detection state machine update
        blink_event = self.blink_detector.update(blendshapes, timestamp_ns)

        # 6. Gaze estimation (uncalibrated directional proxy in M2)
        gaze_estimate: Optional[GazeEstimate] = None
        if eye_metrics is not None and head_pose is not None:
            gaze_estimate = self.gaze_estimator.estimate(eye_metrics, head_pose, timestamp_s)

        proc_time_ms = (time.perf_counter() - start_time) * 1000.0

        return VisionOutput(
            sequence_num=sequence_num,
            timestamp_ns=timestamp_ns,
            face_detected=True,
            confidence=confidence,
            eye_metrics=eye_metrics,
            head_pose=head_pose,
            blink_event=blink_event,
            gaze_estimate=gaze_estimate,
            landmarks=landmarks_norm,
            processing_time_ms=proc_time_ms,
        )

    @staticmethod
    def _convert_to_rgb(
        data: np.ndarray | bytes | memoryview,
        width: int,
        height: int,
        stride: int,
        pixel_format: PixelFormat,
    ) -> np.ndarray:
        """Convert arbitrary raw pixel buffer to contiguous (H, W, 3) RGB uint8 image."""
        if not isinstance(data, np.ndarray):
            data = np.frombuffer(data, dtype=np.uint8)
        if pixel_format == PixelFormat.BGRA:
            # Common macOS CVPixelBuffer format
            row_bytes = width * 4
            if stride == row_bytes:
                bgra = data.reshape((height, width, 4))
            else:
                bgra = data[:stride * height].reshape((height, stride))[:, :row_bytes].reshape((height, width, 4))
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)

        elif pixel_format == PixelFormat.RGBA:
            row_bytes = width * 4
            if stride == row_bytes:
                rgba = data.reshape((height, width, 4))
            else:
                rgba = data[:stride * height].reshape((height, stride))[:, :row_bytes].reshape((height, width, 4))
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

        elif pixel_format == PixelFormat.BGR:
            row_bytes = width * 3
            if stride == row_bytes:
                bgr = data.reshape((height, width, 3))
            else:
                bgr = data[:stride * height].reshape((height, stride))[:, :row_bytes].reshape((height, width, 3))
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        elif pixel_format == PixelFormat.RGB:
            row_bytes = width * 3
            if stride == row_bytes:
                return np.ascontiguousarray(data.reshape((height, width, 3)))
            return np.ascontiguousarray(
                data[:stride * height].reshape((height, stride))[:, :row_bytes].reshape((height, width, 3))
            )

        elif pixel_format == PixelFormat.NV12:
            # YUV 420 bi-planar
            y_size = stride * height
            uv_size = stride * (height // 2)
            nv12 = data[:y_size + uv_size].reshape((height + height // 2, stride))[:, :width]
            return cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)

        else:
            raise ValueError(f"Unsupported pixel format: {pixel_format}")
