# Milestone 2: Test Results and Empirical Benchmarks

## 1. Test Environment and Execution Summary

- **Target Machine**: Apple Silicon MacBook (ARM64)
- **macOS Version**: macOS 27.0.1 (Build 26A434)
- **Python Runtime**: Python 3.15.0 ARM64 (`.venv/bin/python`)
- **Swift Toolchain**: `/usr/bin/swiftc` with macOS SDK 26.5 (`arm64-apple-macos14.0`)
- **All M2 Automated Unit Tests**: 29 passed in 1.27s (`make test-python`)
- **AppKit Smoke Test**: Passed (`make test-swift`)
- **Cross-Language Swift-to-Python IPC Test**: Passed (`make test-ipc`)
- **Comprehensive Benchmarks**: Passed (`make benchmark`)

---

## 2. In-Process Vision Pipeline Performance (100 Sequential Frames)

Executed via `tests/benchmarks.py` using a deterministic 640x480 BGRA synthetic face frame across 100 sequential frames with monotonic timestamps:

| Metric | Measured Value | Target / Requirement | Status |
|---|---|---|---|
| **Resolution & Format** | `640 x 480` BGRA (32-bit) | Standard webcam feed | Verified |
| **Total Wall Time** | `0.298 s` | - | - |
| **Throughput (Inference FPS)** | **335.1 FPS** | $\ge 30\text{ FPS}$ | **EXCEEDED (11x)** |
| **Mean Latency** | **2.98 ms** | $\le 16.6\text{ ms}$ (60 FPS budget) | **EXCEEDED (5.5x)** |
| **Median Latency (p50)** | **2.98 ms** | - | - |
| **90th Percentile (p90)** | **3.04 ms** | - | - |
| **95th Percentile (p95)** | **3.06 ms** | $\le 20.0\text{ ms}$ | **EXCEEDED (6.5x)** |
| **99th Percentile (p99)** | **3.35 ms** | - | - |
| **Max Latency** | **3.35 ms** | $\le 33.3\text{ ms}$ | **EXCEEDED** |
| **Latency Std Dev** | **0.058 ms** | Low jitter | Extremely stable |
| **Face Detection Rate** | **100.0%** | $\ge 99\%$ | **PERFECT** |
| **Starting RSS Memory** | `112.6 MB` | - | - |
| **Final RSS Memory** | `173.0 MB` | $\le 500\text{ MB}$ | Bounded |
| **Memory Growth (100 frames)**| `+60.4 MB` (TFLite init) | No unbounded leak | Stable |

---

## 3. Cross-Language Swift-to-Python IPC Benchmark (60 Frames)

Executed via `build/swift_ipc_producer` sending 60 BGRA frames through POSIX shared memory to the background Python `VisionEngine`:

```
=== Swift POSIX Shared-Memory Test Producer ===
Socket Path: /var/folders/zd/.../bm_eyemouse.sock
SHM Name:    /bm_shm_84726
Shared memory allocated and initialized: 16777280 bytes across 4 slots.
Connected to Python Vision Engine control socket.
Handshake SUCCESS: Vision Engine is READY.
Frame [1/60] -> Slot 0 ACK: face=true, engine=16.45ms
Frame [2/60] -> Slot 1 ACK: face=true, engine=5.20ms
Frame [3/60] -> Slot 2 ACK: face=true, engine=4.71ms
Frame [4/60] -> Slot 3 ACK: face=true, engine=4.26ms
Frame [5/60] -> Slot 0 ACK: face=true, engine=3.94ms
Frame [10/60] -> Slot 1 ACK: face=true, engine=3.35ms
Frame [20/60] -> Slot 3 ACK: face=true, engine=2.85ms
Frame [30/60] -> Slot 1 ACK: face=true, engine=3.01ms
Frame [40/60] -> Slot 3 ACK: face=true, engine=2.93ms
Frame [50/60] -> Slot 1 ACK: face=true, engine=3.02ms
Frame [60/60] -> Slot 3 ACK: face=true, engine=2.93ms
=== Benchmark Summary ===
Frames: 60 | Elapsed: 0.217s | FPS: 275.9 | Avg engine time: 3.35ms
SWIFT_IPC_PRODUCER_SUCCESS
```

### IPC Key Findings
- **Initial Warmup Frame**: 16.45 ms (MediaPipe Graph initialization).
- **Steady-State Latency**: 2.85 ms to 3.02 ms per frame.
- **End-to-End Swift Producer Throughput**: **275.9 FPS** (average elapsed round-trip time: ~3.6 ms including socket serialization, shared-memory synchronization, zero-copy buffer view, inference, and ACK reply).

---

## 4. Frozen PyInstaller Packaging Verification

The vision engine was compiled into a standalone ARM64 application bundle using `packaging/eyemouse_engine.spec`:

- **Binary**: `dist_engine/eyemouse_engine/eyemouse_engine`
- **Architecture**: `Mach-O 64-bit executable arm64`
- **Verification Command**:
  ```bash
  # Launch frozen daemon
  ./dist_engine/eyemouse_engine/eyemouse_engine --socket /tmp/test_frozen.sock
  
  # Connect Swift test producer
  ./build/swift_ipc_producer /tmp/test_frozen.sock /shm_froz 5
  ```
- **Frozen Execution Log**:
  ```
  INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
  [INFO] vision_engine.ipc: ControlServer listening on /tmp/test_frozen.sock
  [INFO] vision_engine: VisionEngine ready. Waiting for host connection...
  [INFO] vision_engine.ipc: Control client connected.
  [INFO] vision_engine: Attached to shared memory: /shm_froz
  Frame [1/5] -> Slot 0 ACK: face=true, engine=10.49ms
  Frame [2/5] -> Slot 1 ACK: face=true, engine=3.97ms
  Frame [3/5] -> Slot 2 ACK: face=true, engine=3.74ms
  Frame [4/5] -> Slot 3 ACK: face=true, engine=3.55ms
  Frame [5/5] -> Slot 0 ACK: face=true, engine=3.46ms
  [INFO] vision_engine: Received SHUTDOWN request from host.
  [INFO] vision_engine: VisionEngine stopped cleanly.
  ```
- **Result**: Complete success. The standalone executable requires zero external Python, boots cleanly, loads native Apple Silicon libraries, extracts bundled models, performs inference, and shuts down gracefully with exit code 0.

---

## 5. Automated Unit Test Suite Breakdown (29 Tests)

| Test Module | Tests | Description | Result |
|---|---|---|---|
| `test_blink_detector.py` | 3 | Initial state, hysteresis thresholds (0.55/0.35), sustained vs reflex blinks | **PASSED** |
| `test_eye_features.py` | 4 | Iris normalization centered, roll invariance, scale invariance, full mesh | **PASSED** |
| `test_filters.py` | 5 | One Euro filter initialization, jitter smoothing, saccade adaptation, NaN/Inf safety, 2D | **PASSED** |
| `test_head_pose.py` | 3 | Identity rotation, yaw rotation, solvePnP initialization | **PASSED** |
| `test_ipc_protocol.py` | 7 | Header pack/unpack, validation errors, slot lifecycle, stale rejection, socket communication | **PASSED** |
| `test_model_loading.py` | 3 | FaceLandmarker model file exists, geometry file exists, missing model error | **PASSED** |
| `test_vision_bootstrap.py` | 1 | MediaPipe FaceLandmarker bootstrap inference | **PASSED** |
| `test_vision_pipeline.py` | 3 | Face present, face absent explicit invalid state, invalid pixel format handling | **PASSED** |
| **TOTAL** | **29** | **Full core coverage** | **100% PASSED** |

---

## 6. Engineering Problems Identified and Resolved During M2

1. **Darwin `shm_open` Variadic ABI Incompatibility**:
   - *Issue*: `shm_open` is declared variadic in Darwin. Swift calls corrupted the mode argument on Apple Silicon ARM64, yielding `[Errno 13] Permission denied`.
   - *Fix*: Added `tests/swift_ipc_producer/shm_bridge.c` exposing non-variadic `eyemouse_shm_open`.
2. **Darwin Shared Memory Syscall Limitations**:
   - *Issue*: Calling `read()` or `write()` on shared memory file descriptors returned `[Errno 6] Device not configured`.
   - *Fix*: Strictly sized shared memory with `ftruncate()` and mapped with `mmap()`.
3. **Darwin Shared Memory Name Length Limit (`PSHNAMLEN`)**:
   - *Issue*: Names longer than 31 characters triggered `[Errno 63] File name too long`.
   - *Fix*: Enforced compact names strictly $\le 31$ bytes (`/emt_<pid>_<seq>`).
4. **PyInstaller Top-Level Relative Import in Frozen Script**:
   - *Issue*: `from .engine import VisionEngine` failed in `main.py` when frozen.
   - *Fix*: Handled dual import paths for module and top-level script execution.
5. **MediaPipe Drawing Utils Matplotlib Dependency**:
   - *Issue*: `mediapipe.tasks.python.vision` imports `drawing_styles`, which expects `matplotlib`. Excluding `matplotlib` caused a frozen import error.
   - *Fix*: Removed artificial exclusions from `packaging/eyemouse_engine.spec`.

