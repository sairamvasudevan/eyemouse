"""Display geometry and target coordinate definitions for eye-screen calibration.

Supports:
- Normalized screen space: [0.0, 1.0] x [0.0, 1.0] with (0,0) at top-left.
- macOS logical display points: [0, W_pt] x [0, H_pt] with CoreGraphics top-left origin.
- Cocoa coordinates: [0, W_pt] x [0, H_pt] with bottom-left origin.
- Physical Retina pixels: [0, W_px] x [0, H_px] = points * backing_scale_factor.
- Angular visual degree calculation given display PPI and viewing distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True, slots=True)
class ScreenGeometry:
    """Display dimensions and scaling properties."""
    width_points: float        # e.g. 1470.0
    height_points: float       # e.g. 956.0
    backing_scale: float       # e.g. 2.0 for Retina
    viewing_distance_cm: float = 55.0  # Assumed viewing distance: 55 cm
    diagonal_inches: float = 13.6      # MacBook Air M2/M3/M5 physical display diagonal

    @property
    def width_pixels(self) -> float:
        return self.width_points * self.backing_scale

    @property
    def height_pixels(self) -> float:
        return self.height_points * self.backing_scale

    @property
    def diagonal_points(self) -> float:
        return math.hypot(self.width_points, self.height_points)

    @property
    def diagonal_pixels(self) -> float:
        return math.hypot(self.width_pixels, self.height_pixels)

    @property
    def ppi(self) -> float:
        """Physical pixels per inch."""
        return self.diagonal_pixels / self.diagonal_inches

    @property
    def pixels_per_cm(self) -> float:
        return self.ppi / 2.54

    @property
    def points_per_cm(self) -> float:
        return self.pixels_per_cm / self.backing_scale

    def normalized_to_points(self, u: float, v: float) -> Tuple[float, float]:
        """Convert normalized screen (0..1, top-left) to logical points (CoreGraphics top-left)."""
        x = np.clip(u, 0.0, 1.0) * self.width_points
        y = np.clip(v, 0.0, 1.0) * self.height_points
        return float(x), float(y)

    def points_to_normalized(self, x: float, y: float) -> Tuple[float, float]:
        """Convert logical points to normalized screen coordinates."""
        u = np.clip(x / self.width_points, 0.0, 1.0)
        v = np.clip(y / self.height_points, 0.0, 1.0)
        return float(u), float(v)

    def points_to_cocoa(self, x: float, y: float) -> Tuple[float, float]:
        """Convert CoreGraphics points (top-left origin) to Cocoa points (bottom-left origin)."""
        return float(x), float(self.height_points - y)

    def cocoa_to_points(self, x_cocoa: float, y_cocoa: float) -> Tuple[float, float]:
        """Convert Cocoa points to CoreGraphics points."""
        return float(x_cocoa), float(self.height_points - y_cocoa)

    def points_to_pixels(self, x: float, y: float) -> Tuple[float, float]:
        """Convert logical points to physical Retina pixels."""
        return float(x * self.backing_scale), float(y * self.backing_scale)

    def pixel_error_to_visual_degrees(self, error_pixels: float) -> float:
        """Calculate visual angle in degrees for a given pixel error at assumed viewing distance.
        
        Formula: theta = 2 * arctan( (error_cm / 2) / distance_cm ) * (180 / pi)
        """
        if self.pixels_per_cm <= 0 or self.viewing_distance_cm <= 0:
            return 0.0
        error_cm = error_pixels / self.pixels_per_cm
        angle_rad = 2.0 * math.atan2(error_cm / 2.0, self.viewing_distance_cm)
        return math.degrees(angle_rad)


@dataclass(frozen=True, slots=True)
class TargetPoint:
    """A calibration or validation target location."""
    index: int
    name: str
    norm_x: float  # [0.0, 1.0]
    norm_y: float  # [0.0, 1.0]
    is_center: bool = False
    is_corner: bool = False
    is_edge: bool = False

    def point_coordinates(self, screen: ScreenGeometry) -> Tuple[float, float]:
        return screen.normalized_to_points(self.norm_x, self.norm_y)


def generate_13_point_calibration_grid(margin: float = 0.08) -> List[TargetPoint]:
    """Generate 13-point calibration layout covering center, corners, edges, and inner quadrants.
    
    Layout:
    - 1 Center point
    - 4 Outer Corners (with safety margin)
    - 4 Screen Edge midpoints
    - 4 Inner Quadrants
    """
    m = margin
    cx = 0.5
    cy = 0.5
    left = m
    right = 1.0 - m
    top = m
    bottom = 1.0 - m
    
    q_left = 0.25
    q_right = 0.75
    q_top = 0.25
    q_bottom = 0.75

    raw_targets = [
        # Center (0)
        (0, "center", cx, cy, True, False, False),
        # 4 Corners (1..4)
        (1, "top_left", left, top, False, True, False),
        (2, "top_right", right, top, False, True, False),
        (3, "bottom_left", left, bottom, False, True, False),
        (4, "bottom_right", right, bottom, False, True, False),
        # 4 Edges (5..8)
        (5, "top_center", cx, top, False, False, True),
        (6, "bottom_center", cx, bottom, False, False, True),
        (7, "left_center", left, cy, False, False, True),
        (8, "right_center", right, cy, False, False, True),
        # 4 Inner Quadrants (9..12)
        (9, "inner_top_left", q_left, q_top, False, False, False),
        (10, "inner_top_right", q_right, q_top, False, False, False),
        (11, "inner_bottom_left", q_left, q_bottom, False, False, False),
        (12, "inner_bottom_right", q_right, q_bottom, False, False, False),
    ]

    return [
        TargetPoint(
            index=t[0],
            name=t[1],
            norm_x=t[2],
            norm_y=t[3],
            is_center=t[4],
            is_corner=t[5],
            is_edge=t[6],
        )
        for t in raw_targets
    ]


def generate_9_point_validation_grid() -> List[TargetPoint]:
    """Generate 9 independent screen target locations NOT used in the 13-point calibration.
    
    Locations are interleaved at intermediate positions to objectively assess generalization.
    """
    raw_targets = [
        (0, "val_top_mid_left", 0.20, 0.12, False, False, False),
        (1, "val_top_mid_right", 0.80, 0.12, False, False, False),
        (2, "val_bottom_mid_left", 0.20, 0.88, False, False, False),
        (3, "val_bottom_mid_right", 0.80, 0.88, False, False, False),
        (4, "val_mid_left_inner", 0.35, 0.50, False, False, False),
        (5, "val_mid_right_inner", 0.65, 0.50, False, False, False),
        (6, "val_mid_top_inner", 0.50, 0.30, False, False, False),
        (7, "val_mid_bottom_inner", 0.50, 0.70, False, False, False),
        (8, "val_center_offset", 0.40, 0.40, True, False, False),
    ]

    return [
        TargetPoint(
            index=t[0],
            name=t[1],
            norm_x=t[2],
            norm_y=t[3],
            is_center=t[4],
            is_corner=t[5],
            is_edge=t[6],
        )
        for t in raw_targets
    ]


def generate_32_point_validation_grid() -> List[TargetPoint]:
    """Generate 32 independent, spatially distributed, previously unseen validation targets.
    
    Includes:
    - 12 Central Workspace Targets (u in [0.30, 0.70], v in [0.22, 0.78])
    - 4 Intermediate Quadrant Targets
    - 16 Peripheral Targets (Top, Bottom, Left Edge, Right Edge, and Outer Corners)
    
    Guaranteed zero collision (minimum separation > 0.04) with the 13 calibration targets
    and the legacy 9 validation targets.
    """
    raw_targets = [
        # --- Top Peripheral Zone (4 targets) ---
        (0, "val32_top_outer_l", 0.14, 0.05, False, True, False),
        (1, "val32_top_mid_l", 0.32, 0.05, False, False, True),
        (2, "val32_top_mid_r", 0.68, 0.05, False, False, True),
        (3, "val32_top_outer_r", 0.86, 0.05, False, True, False),
        
        # --- Upper Flank / Edge (2 targets) ---
        (4, "val32_upper_edge_l", 0.06, 0.20, False, False, True),
        (5, "val32_upper_edge_r", 0.94, 0.20, False, False, True),

        # --- Upper-Mid / Transition Zone (4 targets) ---
        (6, "val32_upmid_l", 0.18, 0.32, False, False, False),
        (7, "val32_upmid_cl", 0.38, 0.22, True, False, False),
        (8, "val32_upmid_cr", 0.62, 0.22, True, False, False),
        (9, "val32_upmid_r", 0.82, 0.32, False, False, False),

        # --- Mid Flank / Edge (2 targets) ---
        (10, "val32_mid_edge_l", 0.06, 0.38, False, False, True),
        (11, "val32_mid_edge_r", 0.94, 0.38, False, False, True),

        # --- Central Primary Workspace (8 targets) ---
        (12, "val32_center_w_nw", 0.32, 0.36, True, False, False),
        (13, "val32_center_w_ne", 0.68, 0.36, True, False, False),
        (14, "val32_center_w_w", 0.42, 0.50, True, False, False),
        (15, "val32_center_w_e", 0.58, 0.50, True, False, False),
        (16, "val32_center_w_sw", 0.32, 0.64, True, False, False),
        (17, "val32_center_w_se", 0.68, 0.64, True, False, False),
        (18, "val32_center_w_n", 0.50, 0.42, True, False, False),
        (19, "val32_center_w_s", 0.50, 0.58, True, False, False),

        # --- Lower-Mid / Transition Zone (4 targets) ---
        (20, "val32_lomid_l", 0.18, 0.68, False, False, False),
        (21, "val32_lomid_cl", 0.38, 0.78, True, False, False),
        (22, "val32_lomid_cr", 0.62, 0.78, True, False, False),
        (23, "val32_lomid_r", 0.82, 0.68, False, False, False),

        # --- Low Flank / Edge (2 targets) ---
        (24, "val32_low_edge_l", 0.06, 0.62, False, False, True),
        (25, "val32_low_edge_r", 0.94, 0.62, False, False, True),

        # --- Bottom Peripheral Zone (6 targets) ---
        (26, "val32_bot_outer_l", 0.14, 0.95, False, True, False),
        (27, "val32_bot_mid_l", 0.32, 0.95, False, False, True),
        (28, "val32_bot_mid_r", 0.68, 0.95, False, False, True),
        (29, "val32_bot_outer_r", 0.86, 0.95, False, True, False),
        (30, "val32_bot_edge_l", 0.06, 0.80, False, False, True),
        (31, "val32_bot_edge_r", 0.94, 0.80, False, False, True),
    ]

    return [
        TargetPoint(
            index=t[0],
            name=t[1],
            norm_x=t[2],
            norm_y=t[3],
            is_center=t[4],
            is_corner=t[5],
            is_edge=t[6],
        )
        for t in raw_targets
    ]

