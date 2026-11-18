"""Core data structures and types for the EyeMouse vision engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Optional, Sequence
import numpy as np


class PixelFormat(IntEnum):
    """Supported raw pixel buffer formats in POSIX shared memory."""
    UNKNOWN = 0
    BGRA = 1    # 32-bit BGRA (macOS native CVPixelBuffer format kCVPixelFormatType_32BGRA)
    RGBA = 2    # 32-bit RGBA
    BGR = 3     # 24-bit BGR (OpenCV default)
    RGB = 4     # 24-bit RGB (MediaPipe default)
    NV12 = 5    # Bi-planar YUV 4:2:0 (Apple Silicon hardware camera format)


class BlinkPhase(Enum):
    """Discrete phases of the blink detection state machine."""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    REOPENING = "reopening"
    SUSTAINED_CLOSED = "sustained_closed"


@dataclass(frozen=True, slots=True)
class Point2D:
    x: float
    y: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class Point3D:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class NormalizedEyeCoords:
    """Iris position normalized within the eye socket coordinate frame.
    
    Coordinate frame:
      x: horizontal relative position along the eye corner axis (0.0 = inner corner, 1.0 = outer corner or vice versa).
      y: vertical relative position perpendicular to the eye corner axis (positive = looking down, negative = looking up).
      openness: vertical distance between upper and lower eyelids divided by eye width.
    """
    iris_x: float
    iris_y: float
    openness: float
    ear: float  # Eye Aspect Ratio


@dataclass(frozen=True, slots=True)
class EyeMetrics:
    """Comprehensive eye geometry measurements for both eyes."""
    left_corner_inner: Point2D
    left_corner_outer: Point2D
    left_iris_center: Point2D
    left_normalized: NormalizedEyeCoords

    right_corner_inner: Point2D
    right_corner_outer: Point2D
    right_iris_center: Point2D
    right_normalized: NormalizedEyeCoords

    # Combined / averaged metrics
    gaze_feature_vector: np.ndarray  # [left_norm_x, left_norm_y, right_norm_x, right_norm_y]


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Rigid 3D head pose orientation and translation."""
    yaw: float      # Rotation around Y axis (degrees, negative = looking right, positive = looking left)
    pitch: float    # Rotation around X axis (degrees, negative = looking down, positive = looking up)
    roll: float     # Rotation around Z axis (degrees, tilt)
    tx: float       # Translation X (centimeters or normalized units)
    ty: float       # Translation Y
    tz: float       # Translation Z (distance from camera)
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class BlinkEvent:
    """State of eye closures for intentional gesture analysis."""
    phase: BlinkPhase
    left_blink_score: float   # 0.0 (wide open) to 1.0 (fully closed) from blendshape
    right_blink_score: float  # 0.0 (wide open) to 1.0 (fully closed) from blendshape
    closure_duration_ms: float
    is_intentional_sustained: bool


@dataclass(frozen=True, slots=True)
class GazeEstimate:
    """Gaze prediction output from the replaceable gaze estimator."""
    raw_x: float
    raw_y: float
    filtered_x: float
    filtered_y: float
    confidence: float
    is_valid: bool


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Binary header metadata associated with each video frame slot."""
    sequence_num: int
    timestamp_ns: int
    width: int
    height: int
    stride: int
    pixel_format: PixelFormat


@dataclass(frozen=True, slots=True)
class VisionOutput:
    """Complete structured output from a single frame vision inference."""
    sequence_num: int
    timestamp_ns: int
    face_detected: bool
    confidence: float
    eye_metrics: Optional[EyeMetrics] = None
    head_pose: Optional[HeadPose] = None
    blink_event: Optional[BlinkEvent] = None
    gaze_estimate: Optional[GazeEstimate] = None
    landmarks: Optional[np.ndarray] = None  # (478, 3) landmarks
    processing_time_ms: float = 0.0

