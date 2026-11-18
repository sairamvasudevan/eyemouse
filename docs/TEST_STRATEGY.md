# EyeMouse Quality Assurance & Test Strategy

**Version:** 1.0.0  
**Test Harnesses:** Pytest (Vision Engine) & Swift XCTest (Host Application)  
**Safety Mandate:** Zero Uncommanded Cursor Actions, Sub-5ms Key Release Latency  

---

## 1. Test Architecture & Multi-Tier Matrix

EyeMouse employs a four-tier testing hierarchy to guarantee algorithmic accuracy, native system stability, and fail-safe operation:

```
┌────────────────────────────────────────────────────────┐
│ Tier 4: End-to-End Hardware & Safety Verification      │
│ (Real camera capture, global shortcut hold, latency)   │
├────────────────────────────────────────────────────────┤
│ Tier 3: Inter-Process Communication & Lifecycle Tests   │
│ (Unix Domain Socket streaming, child process crash)    │
├────────────────────────────────────────────────────────┤
│ Tier 2: Component Integration Tests                    │
│ (MediaPipe -> HeadPose -> OneEuro -> IPC packet)       │
├────────────────────────────────────────────────────────┤
│ Tier 1: Unit Tests (Python & Swift)                    │
│ (Math, filters, state machines, protocol serialization)│
└────────────────────────────────────────────────────────┘
```

---

## 2. Tier 1: Unit Testing

### 2.1 Python Vision Engine (`tests/vision_engine/`)
1. **Filter Tests (`test_filters.py`):**
   - Verify `OneEuroFilter` numerical stability under:
     - Rapid step inputs (saccades): cut-off frequency must scale instantaneously above $15\text{ Hz}$.
     - Static inputs (fixations): cut-off frequency must settle to $0.3\text{ Hz}$ to eliminate jitter.
     - Corrupted inputs (`NaN`, `Inf`): must retain previous valid value and not poison internal state.
2. **Head Pose & Geometry Tests (`test_head_pose.py`):**
   - Validate Procrustes landmark alignment against canonical 3D mesh.
   - Verify invariance of normalized iris vectors when head undergoes pitch ($\pm 25^\circ$), yaw ($\pm 35^\circ$), and roll ($\pm 20^\circ$).
3. **Adaptive Blink Detector Tests (`test_blink.py`):**
   - Test rolling median baseline calculation over varied eye openness distributions.
   - Confirm intentional long blinks ($> 350\ \text{ms}$) trigger a click event.
   - Confirm natural involuntary blinks ($< 200\ \text{ms}$) and downward gaze do not trigger false clicks.
4. **Calibration Math Tests (`test_calibration.py`):**
   - Validate polynomial and Ridge regression mapping from normalized eye features to pixel coordinates.
   - Test leave-one-out cross-validation accuracy metrics.

### 2.2 Native Swift Host (`tests/native_host/`)
1. **Shortcut Parser Tests (`ShortcutParserTests.swift`):**
   - Validate string parsing for shortcut expressions (e.g., `Fn + Option`, `Ctrl + Shift + Space`).
   - Validate virtual key code and modifier bitmask translation.
2. **Key State Machine Tests (`KeyStateMachineTests.swift`):**
   - Test concurrent key press/release events.
   - Verify that releasing *any* single required key transitions state to `Inactive` immediately.
3. **IPC Protocol Serialization Tests (`IPCProtocolTests.swift`):**
   - Test JSON-lines decoding and encoding throughput ($> 50,000\ \text{messages/sec}$).
   - Test handling of partial packet chunks and socket buffer fragmentation.

---

## 3. Tier 2 & 3: IPC & Subprocess Lifecycle Integration Tests

1. **Socket Handshake & Recovery:**
   - Verify host establishes local Unix Domain Socket at startup and successfully receives heartbeat pings from engine.
   - Test recovery when engine process is killed via `SIGKILL`:
     - Host must detect socket closure within $100\ \text{ms}$.
     - Host must immediately set `SafetyGate` to inactive.
     - Host must log the event and attempt controlled engine restart.
2. **Bidirectional Command Dispatch:**
   - Host sends `start_calibration(points=9)` -> verifies engine transitions into calibration state and streams `calibration_target` packets.
   - Host sends `set_smoothing(0.8)` -> verifies engine updates filter parameters dynamically without frame drops.

---

## 4. Tier 4: Non-Negotiable Safety & Fail-Safe Verification

These tests enforce the critical safety requirements specified in the project charter:

| Safety Test | Test Procedure | Pass Criteria |
| :--- | :--- | :--- |
| **ST-01: Default Inactive** | Launch application; stream high-velocity gaze data through engine. | **Zero** mouse events posted. Cursor position remains identical to system rest position. |
| **ST-02: Sub-5ms Key Release** | Simulate continuous key hold with active gaze streaming. Issue synthetic `keyUp` event. | Host safety gate drops subsequent gaze coordinates in $< 5\ \text{ms}$. No further mouse movement occurs. |
| **ST-03: No Inactive Clicks** | Trigger blink click detection in vision engine while activation shortcut is **not** held. | Click packet is rejected at the host safety gate. WindowServer receives **zero** mouse click events. |
| **ST-04: Button Stuck Release** | Initiate synthetic drag/click (`mouseDown`). Abruptly issue `keyUp` or terminate vision engine. | Host immediately dispatches `CGEvent(mouseType: .leftMouseUp)` to WindowServer within $2\ \text{ms}$, preventing stuck drag state. |
| **ST-05: Tap Failure Safe State** | Simulate `kCGEventTapDisabledByTimeout` or permission revocation. | EyeMouse suspends cursor injection instantly and transitions to `Error` status. |
| **ST-06: Face Loss Freeze** | Obscure camera lens while shortcut is actively held. | Vision engine emits `face_detected: false`. Host **freezes** cursor at last known coordinate (does not jump to 0,0 or drift). |
| **ST-07: Privacy & Zero Storage** | Run 10-minute continuous tracking session. Perform forensic file scan across `/tmp`, `~/Library`, and workspace. | **Zero** frame files, video clips, or uncompressed facial data written to disk. |

---

## 5. Performance & Latency Benchmarks

Automated benchmark scripts evaluate system efficiency on Apple Silicon hardware:

| Metric | Target Specification | Measurement Method |
| :--- | :--- | :--- |
| **Tracking Pipeline Rate** | $\ge 30\ \text{FPS}$ (Target $60\ \text{FPS}$) | Moving average over 1000 frames via `PerformanceMetrics` |
| **Inference Latency** | $< 18\ \text{ms}$ per frame | High-resolution timestamp `mach_absolute_time()` around MediaPipe call |
| **IPC Roundtrip Latency** | $< 1.5\ \text{ms}$ | Timestamp difference between Python emission and Swift receipt |
| **Host Key Release Latency** | $< 5.0\ \text{ms}$ | Hardware key event timestamp to CGEventPost suspension |
| **Apple Silicon CPU Load** | $< 15\%$ single core | `ps -p <pid> -o %cpu` sampled under active tracking |
| **Host Memory (RSS)** | $< 120\ \text{MB}$ | Host Swift process memory usage |
| **Engine Memory (RSS)** | $< 250\ \text{MB}$ | Python runtime + MediaPipe model memory usage |

---

## 6. Continuous Verification Script (`scripts/run_safety_audit.sh`)

A dedicated script will be packaged into the repository to execute the automated test battery:
```bash
#!/bin/bash
set -euo pipefail

echo "==> Running Vision Engine Unit Tests..."
.venv/bin/pytest tests/vision_engine/ -v

echo "==> Running IPC & Protocol Tests..."
.venv/bin/pytest tests/ipc/ -v

echo "==> Running Swift Host Unit & Safety Tests..."
make test-swift

echo "==> Running Privacy & Zero-Storage Forensic Audit..."
python3 scripts/audit_zero_storage.py

echo "==> ALL SAFETY GATES PASSED."
```

---

## 7. Milestone 3 Realized Verification Battery

Milestone 3 added native test runners orchestrating safety state and real webcam hardware verification:

### 7.1 Automated Safety State & Shortcut Matching (`build/test_m3_safety_state`)
- **Initial State**: Validates default `INACTIVE` state and `canProcessFrames == false`.
- **Activation Lifecycle**: Tests `handleShortcutPressed()` $\to$ `STARTING` $\to$ `ACTIVE`.
- **Synchronous Gate Cutoff**: Tests that `handleShortcutReleased()` synchronously forces `canProcessFrames = false` and `isKeyHeld = false` in memory.
- **Startup Cancellation**: Simulates releasing activation shortcut before camera/engine completes startup; verifies state aborts cleanly to `INACTIVE`.
- **System Sleep / Lock Recovery**: Tests that simulated sleep notification immediately drops active tracking.
- **Shortcut Matching Matrix**: Tests `ShortcutConfig` against exact key codes and modifier flags, verifying rejection of partial chords and extra modifiers.

### 7.2 Real Webcam Live Pipeline Verification (`build/test_m3_live_pipeline`)
- **Hardware Ingestion**: Captures live frames from the built-in MacBook Air camera via `AVCaptureSession`.
- **Zero-Copy Transport**: Delivers frames across POSIX shared memory to the standalone frozen vision engine.
- **Real-World Live Benchmark**: Measures live capture rate (19.8 FPS), inference latency (3.69 ms mean, 3.49 ms p50, 4.65 ms p95), and face presence.
- **Zero Leak Verification**: Verifies that 0 frames are published or processed after the shortcut release event.

---

## 8. Milestone 4 Realized Verification Battery

Milestone 4 added comprehensive unit, integration, and live hardware tests for personalized gaze calibration and held-out validation:

### 8.1 Python Calibration Suite (`tests/test_calibration.py` - 11 Unit Tests)
- **Geometry & Scaling**: Tests dynamic `ScreenGeometry` dimensions, Retina scale ($2.0\times$), CoreGraphics-to-Cocoa origin flip, bounds clipping, and visual angle ($\theta$) conversion.
- **Target Independence**: Proves that the 13 calibration points and 9 validation points have zero coordinate overlap.
- **Filtering & MAD Outlier Rejection**: Tests the 400ms settling window, blink rejection, face-loss dropping, and MAD outlier filtering on synthetic distributions.
- **Model Fitting & LOOCV**: Validates closed-form Ridge regression with unpenalized intercept and Leave-One-Out Cross-Validation parameter selection.
- **Profile Persistence**: Validates JSON serialization, deserialization, and compatibility checks against screen resizing and camera switching.
- **Real-Time Calibrated Inference**: Tests sub-millisecond evaluation and OneEuroFilter adaptive smoothing in `CalibratedGazeEstimator`.

### 8.2 End-to-End Live Hardware Calibration (`build/test_m4_live_calibration`)
- **Hardware Calibration & Held-Out Validation**: Captures live camera frames across 13 calibration targets and 9 independent validation targets, fitting Affine and Polynomial models.
- **Session Repeatability**: Compares consecutive calibration sessions on the same user, measuring repeatability and error delta.
- **Fixation Stability**: Measures variance and standard deviation across 50 stationary fixation frames ($\sigma < 0.010$).
- **Head Rotation Decoupling**: Verifies that rotating head yaw $\pm 15^\circ$ does not induce false gaze drift.
- **Safety & Zero Cursor Injection**: Verifies that diagnostic markers update without injecting any `CGEvent` mouse movements or clicks.

