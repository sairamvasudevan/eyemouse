"""Automated tests for 3D head pose estimation and rotation conversions."""

import math
import numpy as np
import pytest

from src.vision_engine.head_pose import (
    HeadPoseEstimator,
    pose_from_rotation_matrix,
)


def test_pose_from_identity_rotation():
    """Verify identity rotation produces zero Euler angles."""
    rot = np.eye(3, dtype=np.float64)
    trans = np.array([0.0, 0.0, 50.0], dtype=np.float64)

    pose = pose_from_rotation_matrix(rot, trans)
    assert pytest.approx(pose.yaw, abs=1e-3) == 0.0
    assert pytest.approx(pose.pitch, abs=1e-3) == 0.0
    assert pytest.approx(pose.roll, abs=1e-3) == 0.0
    assert pytest.approx(pose.tz, abs=1e-3) == 50.0


def test_pose_from_yaw_rotation():
    """Verify yaw rotation around Y axis."""
    angle_deg = 30.0
    theta = math.radians(angle_deg)
    # Rotation around Y axis
    rot_y = np.array([
        [math.cos(theta), 0.0, math.sin(theta)],
        [0.0, 1.0, 0.0],
        [-math.sin(theta), 0.0, math.cos(theta)],
    ], dtype=np.float64)

    pose = pose_from_rotation_matrix(rot_y, np.zeros(3))
    assert pytest.approx(pose.yaw, abs=1e-2) == -angle_deg
    assert pytest.approx(pose.pitch, abs=1e-2) == 0.0


def test_pose_estimator_initialization():
    """Verify HeadPoseEstimator loads canonical geometry and basis landmarks correctly."""
    estimator = HeadPoseEstimator()
    assert estimator.geometry is not None
    assert len(estimator.geometry.basis_ids) >= 6
    assert len(estimator.geometry.vertices) == 468

