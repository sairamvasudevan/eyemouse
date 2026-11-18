"""Comprehensive performance benchmarking suite for EyeMouse Vision Engine.

Measures:
- In-process frame inference latency (min, max, median, p95, mean, std) over 100 frames.
- Frame processing throughput (FPS).
- Memory resident set size (RSS) tracking before and after inference.
- Face presence accuracy on synthetic frames.
- Cross-process Swift-to-Python IPC latency and throughput.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Any

import cv2
import numpy as np

from src.vision_engine.pipeline import VisionPipeline
from src.vision_engine.types import PixelFormat


def get_rss_mb() -> float:
    """Return current process resident set size in megabytes (macOS returns bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is in bytes
    return usage.ru_maxrss / (1024.0 * 1024.0)


def create_synthetic_face_bgra(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a deterministic synthetic face frame in 32-bit BGRA format."""
    img = np.full((height, width, 4), (200, 205, 210, 255), dtype=np.uint8)
    cx, cy = width // 2, height // 2

    # Face oval
    cv2.ellipse(img, (cx, cy), (90, 120), 0, 0, 360, (180, 195, 235, 255), -1)
    # Eyes
    cv2.ellipse(img, (cx - 38, cy - 15), (20, 10), 0, 0, 360, (255, 255, 255, 255), -1)
    cv2.ellipse(img, (cx + 38, cy - 15), (20, 10), 0, 0, 360, (255, 255, 255, 255), -1)
    # Irises
    cv2.circle(img, (cx - 38, cy - 15), 5, (40, 60, 90, 255), -1)
    cv2.circle(img, (cx + 38, cy - 15), 5, (40, 60, 90, 255), -1)
    # Pupils
    cv2.circle(img, (cx - 38, cy - 15), 2, (10, 10, 10, 255), -1)
    cv2.circle(img, (cx + 38, cy - 15), 2, (10, 10, 10, 255), -1)
    # Eyebrows
    cv2.ellipse(img, (cx - 38, cy - 32), (22, 5), 0, 180, 360, (30, 40, 60, 255), 3)
    cv2.ellipse(img, (cx + 38, cy - 32), (22, 5), 0, 180, 360, (30, 40, 60, 255), 3)
    # Nose
    cv2.line(img, (cx, cy - 15), (cx, cy + 20), (140, 160, 200, 255), 2)
    cv2.line(img, (cx, cy + 20), (cx + 8, cy + 20), (140, 160, 200, 255), 2)
    # Mouth
    cv2.ellipse(img, (cx, cy + 55), (28, 8), 0, 0, 180, (100, 110, 180, 255), -1)
    cv2.GaussianBlur(img, (5, 5), 1.2, dst=img)
    return img


def run_pipeline_benchmark(num_frames: int = 100) -> Dict[str, Any]:
    """Benchmark sequential VisionPipeline frame processing."""
    print(f"\n--- Running VisionPipeline Benchmark ({num_frames} frames, 640x480 BGRA) ---")
    width, height = 640, 480
    frame = create_synthetic_face_bgra(width, height)
    raw_bytes = frame.tobytes()
    stride = width * 4

    rss_before = get_rss_mb()
    pipeline = VisionPipeline()

    # Warmup 10 frames
    base_ts = 1_000_000_000
    for i in range(10):
        pipeline.process_raw_frame(
            pixel_data=raw_bytes,
            width=width,
            height=height,
            stride=stride,
            pixel_format=PixelFormat.BGRA,
            sequence_num=i,
            timestamp_ns=base_ts + i * 33_333_333,
        )

    latencies_ms: List[float] = []
    face_detected_count = 0

    t_start = time.perf_counter()
    for seq in range(1, num_frames + 1):
        ts_ns = base_ts + (10 + seq) * 33_333_333
        t0 = time.perf_counter()
        output = pipeline.process_raw_frame(
            pixel_data=raw_bytes,
            width=width,
            height=height,
            stride=stride,
            pixel_format=PixelFormat.BGRA,
            sequence_num=seq,
            timestamp_ns=ts_ns,
        )
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        if output.face_detected:
            face_detected_count += 1

    t_total = time.perf_counter() - t_start
    pipeline.close()
    rss_after = get_rss_mb()

    # Percentiles
    sorted_lat = sorted(latencies_ms)
    p50 = sorted_lat[int(0.50 * len(sorted_lat))]
    p90 = sorted_lat[int(0.90 * len(sorted_lat))]
    p95 = sorted_lat[int(0.95 * len(sorted_lat))]
    p99 = sorted_lat[int(0.99 * len(sorted_lat))]

    results = {
        "num_frames": num_frames,
        "resolution": f"{width}x{height}",
        "format": "BGRA (32-bit)",
        "total_wall_time_s": t_total,
        "throughput_fps": num_frames / t_total,
        "min_latency_ms": min(latencies_ms),
        "mean_latency_ms": statistics.mean(latencies_ms),
        "median_latency_ms": p50,
        "p90_latency_ms": p90,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "max_latency_ms": max(latencies_ms),
        "std_latency_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "face_detection_rate": face_detected_count / num_frames,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_growth_mb": rss_after - rss_before,
    }

    print(f"Throughput:       {results['throughput_fps']:.1f} FPS")
    print(f"Mean Latency:     {results['mean_latency_ms']:.2f} ms")
    print(f"Median (p50):     {results['median_latency_ms']:.2f} ms")
    print(f"95th %-ile (p95): {results['p95_latency_ms']:.2f} ms")
    print(f"99th %-ile (p99): {results['p99_latency_ms']:.2f} ms")
    print(f"Max Latency:      {results['max_latency_ms']:.2f} ms")
    print(f"RSS Memory:       {results['rss_after_mb']:.1f} MB (Delta: {results['rss_growth_mb']:+.1f} MB)")
    print(f"Detection Rate:   {results['face_detection_rate'] * 100.0:.1f}%")
    return results


def run_cross_process_ipc_benchmark(num_frames: int = 60) -> Dict[str, Any]:
    """Run full Swift producer -> Python engine over shared memory IPC benchmark."""
    print(f"\n--- Running Swift-to-Python IPC Benchmark ({num_frames} frames) ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        sock_path = os.path.join(tmpdir, "bm_eyemouse.sock")
        # macOS POSIX shared memory name length limit is 31 chars
        shm_name = f"bm_shm_{int(time.time()) % 100000}"
        swift_bin = "build/swift_ipc_producer"

        if not os.path.exists(swift_bin):
            print("Compiling Swift IPC producer...")
            subprocess.run(["make", "build-swift-ipc"], check=True)

        engine_proc = subprocess.Popen(
            [sys.executable, "-m", "src.vision_engine.main", "--socket", sock_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait for socket
            for _ in range(50):
                if os.path.exists(sock_path):
                    break
                time.sleep(0.05)
            else:
                raise TimeoutError("VisionEngine did not create socket")

            # Run Swift producer: <socket_path> <shm_name> <num_frames>
            t0 = time.perf_counter()
            swift_result = subprocess.run(
                [swift_bin, sock_path, shm_name, str(num_frames)],
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            total_time = time.perf_counter() - t0

            if swift_result.returncode != 0:
                print("Swift producer stderr:", swift_result.stderr)
                raise RuntimeError(f"Swift producer exited with code {swift_result.returncode}")

            # Parse lines from Swift producer
            fps = 0.0
            avg_ms = 0.0
            for line in swift_result.stdout.splitlines():
                if "FPS:" in line:
                    parts = line.split("|")
                    for p in parts:
                        if "FPS:" in p:
                            fps = float(p.split(":")[1].strip())
                        elif "Avg engine time:" in p:
                            avg_ms = float(p.split(":")[1].replace("ms", "").strip())

            ipc_results = {
                "num_frames": num_frames,
                "total_time_s": total_time,
                "producer_fps": fps,
                "avg_engine_ms": avg_ms,
                "swift_stdout": swift_result.stdout.strip(),
            }
            print(f"Total IPC Elapsed: {total_time:.2f}s")
            print(f"Producer FPS:      {fps:.1f}")
            print(f"Avg Engine Time:   {avg_ms:.2f} ms")
            return ipc_results

        finally:
            engine_proc.terminate()
            try:
                engine_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                engine_proc.kill()


if __name__ == "__main__":
    p_results = run_pipeline_benchmark(100)
    ipc_results = run_cross_process_ipc_benchmark(60)

    # Dump benchmark summary json for documentation inclusion
    bm_summary = {
        "pipeline_benchmark": p_results,
        "ipc_benchmark": ipc_results,
    }
    with open("build/benchmark_results.json", "w") as f:
        json.dump(bm_summary, f, indent=2)
    print("\nBenchmark results saved to build/benchmark_results.json")
