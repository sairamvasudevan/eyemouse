"""Generate comprehensive Milestone 4.2 validation and repeatability analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np


def analyze_m4_2():
    session1_path = Path("build/m4_2_session_1.json")
    session2_path = Path("build/m4_2_session_2.json")
    results_path = Path("build/m4_2_results.json")
    jitter_path = Path("build/m4_2_resolution_jitter_results.json")

    with open(session1_path, "r") as f:
        s1 = json.load(f)
    with open(session2_path, "r") as f:
        s2 = json.load(f)
    with open(results_path, "r") as f:
        res = json.load(f)
    with open(jitter_path, "r") as f:
        jit = json.load(f)

    rep1 = res.get("report", {})
    targets1 = rep1.get("target_results", [])
    
    # Session 2 report
    with open("/tmp/m4_report_2.json", "r") as f:
        s2_rep = json.load(f).get("report", {})
    targets2 = s2_rep.get("target_results", [])

    w_px = 2940.0
    h_px = 1912.0
    diag_px = math.hypot(w_px, h_px)

    print("=== M4.2 Validation & Repeatability Analysis ===")
    print(f"Display: {w_px}x{h_px} px, Diagonal: {diag_px:.1f} px")
    print(f"Session 1 Targets: {len(targets1)}, Session 2 Targets: {len(targets2)}")

    # 1. Zone Classification for 32 targets
    # 12 Central Workspace Targets (u in [0.30, 0.70], v in [0.22, 0.78])
    center_indices = [7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22]
    transition_indices = [6, 9, 20, 23]
    corner_indices = [0, 3, 26, 29]
    edge_indices = [1, 2, 4, 5, 10, 11, 24, 25, 27, 28, 30, 31]

    # Verify counts
    assert len(center_indices) == 12
    assert len(transition_indices) == 4
    assert len(corner_indices) == 4
    assert len(edge_indices) == 12
    assert len(center_indices) + len(transition_indices) + len(corner_indices) + len(edge_indices) == 32

    # Session 1 errors
    errors_all = [t["error_pixels"] for t in targets1]
    errors_center = [targets1[i]["error_pixels"] for i in center_indices]
    errors_trans = [targets1[i]["error_pixels"] for i in transition_indices]
    errors_edge = [targets1[i]["error_pixels"] for i in edge_indices]
    errors_corner = [targets1[i]["error_pixels"] for i in corner_indices]

    vis_all = [t["error_visual_degrees"] for t in targets1]
    vis_center = [targets1[i]["error_visual_degrees"] for i in center_indices]

    # Repeatability deltas between Session 1 and Session 2
    repeatability_deltas = []
    repeatability_center = []
    repeatability_peripheral = []

    for i in range(len(targets1)):
        p1 = targets1[i]["predicted_norm"]
        p2 = targets2[i]["predicted_norm"]
        dx = abs(p1[0] - p2[0]) * w_px
        dy = abs(p1[1] - p2[1]) * h_px
        dist = math.hypot(dx, dy)
        repeatability_deltas.append({
            "index": i,
            "name": targets1[i]["target_name"],
            "p1": p1,
            "p2": p2,
            "dx": round(dx, 1),
            "dy": round(dy, 1),
            "dist": round(dist, 1),
            "is_center": i in center_indices,
            "zone": "Center" if i in center_indices else ("Transition" if i in transition_indices else ("Corner" if i in corner_indices else "Edge")),
        })
        if i in center_indices:
            repeatability_center.append(dist)
        else:
            repeatability_peripheral.append(dist)

    rep_dists = [r["dist"] for r in repeatability_deltas]

    analysis = {
        "screen": {
            "width_px": w_px,
            "height_px": h_px,
            "diagonal_px": round(diag_px, 1),
            "viewing_distance_cm": 55.0,
            "display_diagonal_inches": 13.6,
        },
        "session1_fullscreen": {
            "total_targets": len(targets1),
            "median_pixel_error": round(float(np.median(errors_all)), 2),
            "mean_pixel_error": round(float(np.mean(errors_all)), 2),
            "p95_pixel_error": round(float(np.percentile(errors_all, 95)), 2),
            "max_pixel_error": round(float(np.max(errors_all)), 2),
            "min_pixel_error": round(float(np.min(errors_all)), 2),
            "median_pct_diagonal": round((float(np.median(errors_all)) / diag_px) * 100.0, 2),
            "p95_pct_diagonal": round((float(np.percentile(errors_all, 95)) / diag_px) * 100.0, 2),
            "mean_visual_degrees": round(float(np.mean(vis_all)), 2),
            "median_visual_degrees": round(float(np.median(vis_all)), 2),
            "mean_pixel_error_x": round(float(np.mean([t["error_pixels_x"] for t in targets1])), 2),
            "mean_pixel_error_y": round(float(np.mean([t["error_pixels_y"] for t in targets1])), 2),
            "quality_grade": rep1.get("quality_grade", "POOR"),
            "is_valid_for_use": rep1.get("is_valid_for_use", False),
        },
        "session1_central_region": {
            "targets_count": len(center_indices),
            "bounds": {"norm_x": [0.30, 0.70], "norm_y": [0.22, 0.78]},
            "median_pixel_error": round(float(np.median(errors_center)), 2),
            "mean_pixel_error": round(float(np.mean(errors_center)), 2),
            "p95_pixel_error": round(float(np.percentile(errors_center, 95)), 2),
            "max_pixel_error": round(float(np.max(errors_center)), 2),
            "min_pixel_error": round(float(np.min(errors_center)), 2),
            "median_pct_diagonal": round((float(np.median(errors_center)) / diag_px) * 100.0, 2),
            "p95_pct_diagonal": round((float(np.percentile(errors_center, 95)) / diag_px) * 100.0, 2),
            "mean_visual_degrees": round(float(np.mean(vis_center)), 2),
            "median_visual_degrees": round(float(np.median(vis_center)), 2),
        },
        "regional_breakdown": {
            "center_mean_error_px": round(float(np.mean(errors_center)), 2),
            "transition_mean_error_px": round(float(np.mean(errors_trans)), 2),
            "edge_mean_error_px": round(float(np.mean(errors_edge)), 2),
            "corner_mean_error_px": round(float(np.mean(errors_corner)), 2),
        },
        "repeatability": {
            "total_targets": len(repeatability_deltas),
            "mean_repeatability_delta_px": round(float(np.mean(rep_dists)), 2),
            "median_repeatability_delta_px": round(float(np.median(rep_dists)), 2),
            "p95_repeatability_delta_px": round(float(np.percentile(rep_dists, 95)), 2),
            "max_repeatability_delta_px": round(float(np.max(rep_dists)), 2),
            "min_repeatability_delta_px": round(float(np.min(rep_dists)), 2),
            "center_mean_repeatability_px": round(float(np.mean(repeatability_center)), 2),
            "center_median_repeatability_px": round(float(np.median(repeatability_center)), 2),
            "peripheral_mean_repeatability_px": round(float(np.mean(repeatability_peripheral)), 2),
            "peripheral_median_repeatability_px": round(float(np.median(repeatability_peripheral)), 2),
            "per_target_deltas": repeatability_deltas,
        },
        "resolution_jitter_benchmark": jit,
    }

    out_file = Path("build/m4_2_final_evaluation_metrics.json")
    with open(out_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved complete evaluation metrics to {out_file}")


if __name__ == "__main__":
    analyze_m4_2()

