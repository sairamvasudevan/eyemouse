"""Benchmark raw eye landmark coordinate variance / jitter across resolutions (1080p, 720p, 480p).

Uses the exact same set of physical video frames captured from the MacBook Air Camera to
isolate the effect of resolution on landmark noise, pixel quantization, and inference latency.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.vision_engine.landmarker import FaceLandmarkerWrapper
from src.vision_engine.eye_features import extract_eye_metrics


def run_benchmark():
    raw_path = Path("/tmp/eyemouse_jitter_frames.raw")
    if not raw_path.exists():
        print(f"Error: {raw_path} not found. Run scripts/capture_frames_for_jitter_test.swift first.")
        return 1

    orig_w, orig_h = 1920, 1080
    frame_bytes = orig_w * orig_h * 4
    total_size = raw_path.stat().st_size
    num_frames = total_size // frame_bytes
    print(f"Loading {num_frames} frames ({orig_w}x{orig_h} BGRA) from {raw_path}...")

    with open(raw_path, "rb") as f:
        raw_data = f.read()

    # Read BGRA frames and convert to RGB
    rgb_frames_1080p = []
    for i in range(num_frames):
        offset = i * frame_bytes
        frame_bgra = np.frombuffer(raw_data, dtype=np.uint8, count=frame_bytes, offset=offset).reshape((orig_h, orig_w, 4))
        # Convert BGRA to contiguous RGB
        frame_rgb = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2RGB)
        rgb_frames_1080p.append(frame_rgb)

    resolutions = [
        ("1080p", 1920, 1080),
        ("720p", 1280, 720),
        ("480p", 640, 480),
    ]

    all_results = {}

    for name, w, h in resolutions:
        print(f"\n--- Testing Resolution: {name} ({w}x{h}) ---")
        
        # Prepare frames at resolution
        if (w, h) == (orig_w, orig_h):
            test_frames = rgb_frames_1080p
        else:
            test_frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in rgb_frames_1080p]

        # Initialize fresh FaceLandmarkerWrapper for VIDEO mode
        wrapper = FaceLandmarkerWrapper()

        # Warm up
        for warmup_i in range(5):
            wrapper.process_frame(test_frames[0], warmup_i * 33)

        latencies_ms = []
        iris_left_raw = []
        iris_right_raw = []
        eye_features_list = []
        detected_count = 0

        for frame_idx, frame in enumerate(test_frames):
            ts_ms = (frame_idx + 5) * 33
            t0 = time.perf_counter()
            num_faces, landmarks, blendshapes, matrix = wrapper.process_frame(frame, ts_ms)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt)

            if num_faces > 0 and landmarks is not None:
                detected_count += 1
                
                # MediaPipe iris centers: 473 = Left Iris Center, 468 = Right Iris Center
                left_iris = landmarks[473]
                right_iris = landmarks[468]
                
                iris_left_raw.append([float(left_iris[0]), float(left_iris[1]), float(left_iris[2])])
                iris_right_raw.append([float(right_iris[0]), float(right_iris[1]), float(right_iris[2])])

                # Extract normalized eye features (lx, ly, rx, ry)
                eye_feat = extract_eye_metrics(landmarks)
                if eye_feat is not None:
                    eye_features_list.append([
                        eye_feat.left_normalized.iris_x,
                        eye_feat.left_normalized.iris_y,
                        eye_feat.right_normalized.iris_x,
                        eye_feat.right_normalized.iris_y,
                    ])

        wrapper.close()

        print(f"  Detected faces in {detected_count}/{num_frames} frames.")
        print(f"  Latency: Mean={np.mean(latencies_ms):.2f}ms, P95={np.percentile(latencies_ms, 95):.2f}ms")

        if detected_count < 10:
            print(f"  Warning: Insufficient detected frames for resolution {name}")
            all_results[name] = {"detected_count": detected_count, "error": "insufficient_detections"}
            continue

        arr_feat = np.array(eye_features_list, dtype=np.float64)  # (N, 4): [lx, ly, rx, ry]
        arr_iris_l = np.array(iris_left_raw, dtype=np.float64)   # (N, 3): [x, y, z]
        arr_iris_r = np.array(iris_right_raw, dtype=np.float64)  # (N, 3): [x, y, z]

        # Frame-to-frame delta (high-frequency jitter)
        feat_diffs = np.linalg.norm(np.diff(arr_feat, axis=0), axis=1)
        mean_jitter_feat = float(np.mean(feat_diffs))

        iris_l_diffs = np.linalg.norm(np.diff(arr_iris_l[:, :2], axis=0), axis=1)
        mean_jitter_iris_l = float(np.mean(iris_l_diffs))

        # Variances and Standard Deviations
        std_feat = np.std(arr_feat, axis=0) # [std_lx, std_ly, std_rx, std_ry]
        std_iris_l = np.std(arr_iris_l[:, :2], axis=0) # [std_x, std_y]
        std_iris_r = np.std(arr_iris_r[:, :2], axis=0)

        # Scale normalized iris standard deviation to physical screen pixels (Retina: 2940x1912)
        # Note: on Retina 2940x1912, eye feature norm scale to screen
        screen_w_px = 2940.0
        screen_h_px = 1912.0
        # Gaze sensitivity factor ~4.0x-5.0x optical leverage
        est_screen_jitter_px = mean_jitter_feat * screen_w_px * 0.25

        res_summary = {
            "name": name,
            "width": w,
            "height": h,
            "total_frames": num_frames,
            "detected_frames": detected_count,
            "detection_rate_pct": round((detected_count / num_frames) * 100.0, 1),
            "latency_mean_ms": round(float(np.mean(latencies_ms)), 2),
            "latency_median_ms": round(float(np.median(latencies_ms)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
            "iris_left_center_std_norm_x": float(std_iris_l[0]),
            "iris_left_center_std_norm_y": float(std_iris_l[1]),
            "iris_right_center_std_norm_x": float(std_iris_r[0]),
            "iris_right_center_std_norm_y": float(std_iris_r[1]),
            "eye_feature_std": {
                "std_lx": float(std_feat[0]),
                "std_ly": float(std_feat[1]),
                "std_rx": float(std_feat[2]),
                "std_ry": float(std_feat[3]),
                "mean_norm_std": float(np.mean(std_feat)),
            },
            "frame_to_frame_jitter": {
                "mean_jitter_normalized_features": float(mean_jitter_feat),
                "mean_jitter_iris_landmark": float(mean_jitter_iris_l),
                "estimated_screen_jitter_px": round(float(est_screen_jitter_px), 2),
            },
        }

        print(f"  Feature Standard Deviation (MAD/Noise):")
        print(f"    Left X:  {std_feat[0]:.6f}")
        print(f"    Left Y:  {std_feat[1]:.6f}")
        print(f"    Right X: {std_feat[2]:.6f}")
        print(f"    Right Y: {std_feat[3]:.6f}")
        print(f"    Mean Feature Std: {np.mean(std_feat):.6f}")
        print(f"  Frame-to-Frame Jitter: {mean_jitter_feat:.6f} norm units (~{est_screen_jitter_px:.1f} px)")

        all_results[name] = res_summary

    output_path = Path("build/m4_2_resolution_jitter_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved resolution jitter comparison results to {output_path}")

    # Clean up temp file
    if raw_path.exists():
        raw_path.unlink()
        print(f"Cleaned up temporary raw video file {raw_path}")

    return 0


if __name__ == "__main__":
    sys.exit(run_benchmark())
