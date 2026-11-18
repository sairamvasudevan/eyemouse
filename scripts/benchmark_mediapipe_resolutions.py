import time
import sys
from pathlib import Path
import numpy as np
import json

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.vision_engine.landmarker import FaceLandmarkerWrapper

def benchmark_resolution(width: int, height: int, num_frames: int = 100):
    wrapper = FaceLandmarkerWrapper()
    
    # Create a realistic test image with face-like geometry
    # MediaPipe needs reasonable contrast / face-like oval to detect face landmarks
    rng = np.random.RandomState(42)
    # Generate background
    frame = np.full((height, width, 4), 180, dtype=np.uint8)
    
    # Draw simple face oval
    cy, cx = height // 2, width // 2
    r_y, r_x = int(height * 0.35), int(width * 0.20)
    y, x = np.ogrid[:height, :width]
    mask = ((x - cx)**2) / (r_x**2) + ((y - cy)**2) / (r_y**2) <= 1.0
    frame[mask] = [210, 200, 190, 255] # skin tone
    
    # Eyes
    eye_y = int(cy - r_y * 0.2)
    l_eye_x = int(cx + r_x * 0.4)
    r_eye_x = int(cx - r_x * 0.4)
    eye_r = int(r_x * 0.15)
    
    for ex in [l_eye_x, r_eye_x]:
        eye_mask = ((x - ex)**2 + (y - eye_y)**2) <= eye_r**2
        frame[eye_mask] = [255, 255, 255, 255] # sclera
        iris_mask = ((x - ex)**2 + (y - eye_y)**2) <= (eye_r * 0.5)**2
        frame[iris_mask] = [50, 40, 30, 255] # iris
        pupil_mask = ((x - ex)**2 + (y - eye_y)**2) <= (eye_r * 0.2)**2
        frame[pupil_mask] = [10, 10, 10, 255] # pupil

    # Convert to RGB (H, W, 3)
    rgb_frame = np.ascontiguousarray(frame[:, :, :3])

    # Warmup
    for i in range(10):
        wrapper.process_frame(rgb_frame, int(i * 33))
        
    latencies_ms = []
    for i in range(num_frames):
        ts = int((i + 10) * 33)
        t0 = time.perf_counter()
        res = wrapper.process_frame(rgb_frame, ts)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)
        
    wrapper.close()
    
    mean_lat = float(np.mean(latencies_ms))
    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    max_lat = float(np.max(latencies_ms))
    min_lat = float(np.min(latencies_ms))
    
    return {
        "width": width,
        "height": height,
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "max_fps_capacity": round(1000.0 / mean_lat, 1),
    }

def main():
    print("=== Benchmarking MediaPipe Face Landmarker Across Resolutions ===")
    resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080),
    ]
    results = []
    for w, h in resolutions:
        print(f"Testing {w}x{h}...")
        r = benchmark_resolution(w, h, num_frames=100)
        print(f"  Result: Median={r['median_latency_ms']}ms, P95={r['p95_latency_ms']}ms, Capacity={r['max_fps_capacity']} FPS")
        results.append(r)
        
    out_path = Path("build/m4_1_mediapipe_resolution_benchmarks.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {out_path}")

if __name__ == "__main__":
    main()
