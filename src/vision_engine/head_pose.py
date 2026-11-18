"""3D Head pose estimation using canonical face geometry and MediaPipe facial matrices."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import cv2
import numpy as np

from .types import HeadPose
from .face_geometry import FaceGeometry, load_face_geometry
from .resources import get_geometry_metadata_path


def pose_from_rotation_matrix(rotation: np.ndarray, translation: np.ndarray) -> HeadPose:
    """
    Extract yaw, pitch, roll angles and translation vectors from a 3x3 rotation matrix.
    
    Conventions:
      yaw: degrees, negative = turned right, positive = turned left.
      pitch: degrees, negative = tilted down, positive = tilted up.
      roll: degrees, negative = tilted clockwise, positive = tilted counter-clockwise.
    """
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    nose = -r[:, 2]  # Unit forward direction of the face
    yaw = math.degrees(math.atan2(nose[0], -nose[2]))
    pitch = math.degrees(math.atan2(nose[1], math.hypot(nose[0], nose[2])))
    roll = math.degrees(math.atan2(-r[1, 0], r[0, 0]))
    t = np.asarray(translation, dtype=np.float64).ravel()
    tx = float(t[0]) if len(t) > 0 else 0.0
    ty = float(t[1]) if len(t) > 1 else 0.0
    tz = float(t[2]) if len(t) > 2 else 0.0
    return HeadPose(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        tx=tx,
        ty=ty,
        tz=tz,
        rotation_matrix=r,
    )


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Approximate pinhole camera intrinsic matrix."""
    f = float(max(width, height))
    return np.array([
        [f, 0.0, width / 2.0],
        [0.0, f, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


class HeadPoseEstimator:
    """Estimate 3D head pose from facial landmarks using canonical 3D mesh alignment."""

    def __init__(self, geometry: Optional[FaceGeometry] = None) -> None:
        if geometry is None:
            geom_path = get_geometry_metadata_path()
            geometry = load_face_geometry(geom_path)
        self.geometry = geometry
        self._ids = np.array(geometry.basis_ids, dtype=np.intp)
        # Convert canonical head-centered coords (x right, y up, z to camera)
        # to OpenCV camera object frame (x right, y down, z away)
        self._object_points = np.ascontiguousarray(geometry.vertices[self._ids] * (1.0, -1.0, -1.0), dtype=np.float64)
        self._rvec: Optional[np.ndarray] = None
        self._tvec: Optional[np.ndarray] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        self._camera_matrix: np.ndarray = camera_matrix(640, 480)

    def reset(self) -> None:
        """Reset tracking state for a new face or scene break."""
        self._rvec = None
        self._tvec = None

    def estimate_from_transformation_matrix(self, matrix_4x4: np.ndarray) -> HeadPose:
        """Derive HeadPose directly from MediaPipe's 4x4 facial transformation matrix."""
        mat = np.asarray(matrix_4x4, dtype=np.float64).reshape(4, 4)
        rot_3x3 = mat[:3, :3]
        trans_3 = mat[:3, 3]
        return pose_from_rotation_matrix(rot_3x3, trans_3)

    def estimate_from_landmarks(
        self,
        landmarks_px: np.ndarray,
        width: int,
        height: int,
    ) -> Optional[HeadPose]:
        """
        Estimate HeadPose from 2D pixel landmark positions using SolvePnP on the Procrustes basis.
        
        Args:
            landmarks_px: (478, 2) or (478, 3) pixel positions.
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        if (width, height) != self._frame_size:
            self._frame_size = (width, height)
            self._camera_matrix = camera_matrix(width, height)
            self.reset()

        if len(landmarks_px) < MESH_VERTICES:
            return None

        image_points = np.ascontiguousarray(landmarks_px[self._ids, :2], dtype=np.float64)
        if not np.all(np.isfinite(image_points)):
            self.reset()
            return None

        # SolvePnP: Use extrinsic guess if tracking sequentially, otherwise global SQPNP
        use_guess = self._rvec is not None and self._tvec is not None
        flags = cv2.SOLVEPNP_ITERATIVE if use_guess else cv2.SOLVEPNP_SQPNP

        try:
            ok, r, t = cv2.solvePnP(
                self._object_points,
                image_points,
                self._camera_matrix,
                None,
                self._rvec.copy() if use_guess else None,
                self._tvec.copy() if use_guess else None,
                useExtrinsicGuess=use_guess,
                flags=flags,
            )
        except cv2.error:
            self.reset()
            return None

        if not ok or not np.all(np.isfinite(r)) or not np.all(np.isfinite(t)):
            self.reset()
            return None

        self._rvec, self._tvec = r, t
        rotation_matrix, _ = cv2.Rodrigues(r)
        return pose_from_rotation_matrix(rotation_matrix, t)

