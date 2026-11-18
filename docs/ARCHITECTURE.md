# EyeMouse System Architecture

**Version:** 1.0.0  
**Target Platform:** Apple Silicon macOS (macOS 14.0+)  
**Security & Privacy:** 100% Offline, Zero Cloud Dependencies, Strict TCC Compliance  

---

## 1. Architectural Overview & Design Philosophy

EyeMouse is engineered as a hybrid native application combining high-performance Apple Silicon operating-system integration with a modular Python computer-vision engine.

```mermaid
flowchart TD
    subgraph HostProcess ["Native macOS Host (EyeMouse.app - Swift / AppKit / SwiftUI)"]
        MenuBar["NSStatusItem Menu-Bar UI"]
        PermMgr["macOS Permissions Manager<br>(AVFoundation, TCC, AX, Input Monitoring)"]
        AVCam["AVFoundation Camera Capture<br>(Owns Video Session)"]
        KeyTap["CGEventTap Key Monitor<br>(Physical Key-Hold State Machine)"]
        SafetyGate["Atomic Safety Gate<br>(Active ONLY while shortcut is held)"]
        MouseInjector["CGEvent Native Mouse Controller<br>(Move, Left/Right Click, Release)"]
        IPCHost["Local IPC Host Controller<br>(UDS Control + POSIX Shared Memory)"]
    end

    subgraph VisionEngineProcess ["Vision Engine Subprocess (eyemouse-engine - Standalone Python)"]
        IPCEngine["Local IPC Engine Controller"]
        SHMReader["Shared-Memory Frame Reader<br>(Zero-Copy Pixel Buffer)"]
        FaceDetector["MediaPipe Face Landmarker<br>(478 Landmarks & Irises)"]
        HeadPoseComp["3D Head Pose Estimator<br>(Procrustes Analysis)"]
        GazeEst["Replaceable Gaze Projection Model<br>(Calibration Matrix / Ridge)"]
        OneEuro["One-Euro Filter<br>(Adaptive Speed-Based Smoothing)"]
        BlinkDet["Adaptive Baseline Blink Detector"]
    end

    subgraph OSInterfaces ["macOS Kernel & WindowServer"]
        HardwareKeyboard["Hardware Keyboard Events"]
        HardwareCamera["Built-in / USB Camera"]
        WindowServer["macOS WindowServer Cursor & Clicks"]
    end

    HardwareCamera -->|Raw Video Frames| AVCam
    AVCam -->|Write Frame Buffer| IPCHost
    IPCHost -->|POSIX shm (Zero-Copy)| SHMReader
    SHMReader --> FaceDetector

    HardwareKeyboard -->|kCGEventKeyDown / Up| KeyTap
    KeyTap -->|Atomic KeyState updates| SafetyGate

    FaceDetector --> HeadPoseComp
    HeadPoseComp --> GazeEst
    GazeEst --> OneEuro
    FaceDetector --> BlinkDet

    OneEuro -->|GazeCoordinates (x, y)| IPCEngine
    BlinkDet -->|ClickEvent (left, right)| IPCEngine
    IPCEngine -->|IPC Stream (UDS)| IPCHost

    IPCHost --> SafetyGate
    SafetyGate -->|Allowed ONLY if key held| MouseInjector
    MouseInjector -->|CGEventPost| WindowServer
```

---

## 2. Core Architectural Components

### 2.1 Native Host Process (`EyeMouse.app`)
The host process is written in Swift (AppKit & SwiftUI) and serves as the primary system bridge and process manager:
1. **Menu-Bar Integration (`NSStatusItem`):**
   - Configured with `LSUIElement = true` (no Dock icon, unobtrusive system accessory).
   - Provides live menu-bar status icon (Idle, Active/Tracking, Warning/Permissions Missing).
   - Dropdown menu for:
     - Shortcut Configuration (e.g. `Fn + Option`, `Ctrl + Cmd`, or custom keys).
     - Calibration Trigger & Status.
     - Sensitivity & Smoothing adjustments.
     - Permission Health Check & Settings Shortcuts.
     - Quit EyeMouse.
2. **Global Key Monitoring (`CGEventTap`):**
   - Created using `CGEventTapCreate(kCGSessionEventTap, kCGHeadInsertEventTap, kCGEventTapOptionListenOnly, ...)`.
   - Listens to:
     - `kCGEventKeyDown`
     - `kCGEventKeyUp`
     - `kCGEventFlagsChanged` (critical for tracking modifier keys like Option, Control, Command, Shift, Fn).
   - Maintains an atomic `KeyHoldState`: Eye control is **active if and only if** all configured shortcut keys are physically depressed.
   - **Zero Inference in Callback:** The CGEventTap callback executes in $< 100\ \mu\text{s}$, simply flipping atomic bitmasks. It performs no image processing, file I/O, or blocking IPC.
3. **Native Cursor Injection (`CGEvent`):**
   - Uses `CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: pt, mouseButton: .left)` followed by `CGEventPost(kCGHIDEventTap, event)`.
   - Bypasses slow third-party Python wrappers (such as PyAutoGUI), delivering sub-millisecond cursor warping.
   - Handles mouse-down, mouse-up, and single/double clicks natively.

### 2.2 Vision Engine Process (`eyemouse-engine`)
The vision engine is an isolated child process packaged with its own embedded ARM64 runtime and native dependencies:
1. **Headless Ingestion (No OpenCV Camera Capture):**
   - The native Swift host owns the camera session via AVFoundation and writes raw pixel buffers into POSIX shared memory (`/eyemouse_shm_<session>`).
   - The vision engine reads raw frames directly from shared memory with zero memory copies. Python does not open or control camera hardware directly.
   - Enforces zero frame persistence: frames exist in memory buffers only during landmark inference and are immediately overwritten.
2. **Landmark & Iris Detection:**
   - MediaPipe Face Landmarker task running on Apple Silicon CPU/Neural Engine.
   - Generates 478 3D facial landmarks including inner/outer eye corners and 10 iris center landmarks.
3. **3D Head-Pose Compensation:**
   - Employs Procrustes analysis on rigid facial landmarks against canonical 3D mesh vertices (`face_geometry.py`).
   - Normalizes iris vectors with respect to head rotation (yaw, pitch, roll) and distance (translation z), preventing gaze drift during head motion.
4. **Adaptive Filtering (`OneEuroFilter`):**
   - Dynamic low-pass filtering based on instantaneous velocity.
   - Low cutoff ($0.3\text{ Hz}$) during stationary fixation prevents jitter.
   - High cutoff ($15\text{--}30\text{ Hz}$) during rapid eye movements prevents lag.
5. **Replaceable Gaze Estimation Model:**
   - Gaze mapper interface (`GazeEstimatorProtocol`) allowing plug-and-play models:
     - Default: Multi-point Polynomial / Ridge Regression mapping.
     - Extensible: Deep gaze neural network (e.g. ETH-XGaze / L2CS-Net ONNX models).

### 2.3 Local Inter-Process Communication (IPC)
The host and engine communicate strictly over local IPC (no networking, no open ports):
- **Transport:** Unix Domain Socket (UDS) located at:
  `$TMPDIR/eyemouse_<session_uuid>.sock` with file permissions `0600` (restricted strictly to the current user).
- **Fallback:** High-throughput bi-directional POSIX pipe (`stdin`/`stdout`).
- **Framing Protocol:** High-speed JSON-Lines or binary fixed-length struct packets:
  ```json
  {"type": "gaze", "x": 1420.5, "y": 680.2, "t": 1730718500.123, "conf": 0.94}
  {"type": "click", "btn": "left", "t": 1730718500.250}
  {"type": "status", "face_detected": true, "fps": 31.2}
  ```
- **Control Commands (Host -> Engine):**
  ```json
  {"cmd": "start_calibration", "points": 9}
  {"cmd": "set_smoothing", "value": 0.75}
  {"cmd": "pause"}
  {"cmd": "resume"}
  ```

---

## 3. Strict Safety & Fail-Safe Architecture

The non-negotiable safety requirements are enforced across multiple architectural layers:

```
[Camera / Inference Frame] -> [Gaze (x, y)] 
                                   |
                                   v
             +-------------------------------------------+
             |   SAFETY GATE (Native Swift Host)         |
             |                                           |
             |   1. Is Physical Key Shortcut Held?       |
             |      [ NO ] ---> DROP EVENT & FREEZE      |
             |      [ YES ]                              |
             |         |                                 |
             |   2. Is Tracking State Active?            |
             |      [ NO ] ---> DROP EVENT & FREEZE      |
             |      [ YES ]                              |
             |         |                                 |
             |   3. Is Face Tracking Valid & Confident?  |
             |      [ NO ] ---> HOLD LAST CURSOR POS     |
             |      [ YES ]                              |
             +-------------------------------------------+
                                   |
                                   v
                      [Post CGEvent to WindowServer]
```

### Safety Invariants
1. **Default Inactive:** Mouse control is completely dormant upon application launch.
2. **Continuous Physical Hold Required:** The moment any required activation key is released, `KeyHoldState` is cleared within the `CGEventTap` callback, and the safety gate drops all subsequent gaze and click coordinates immediately.
3. **Emergency Mouse Button Release:** If the key is released while a synthetic drag or click is in progress, the host immediately dispatches `CGEvent(mouseType: .leftMouseUp)` to prevent stuck cursor dragging.
4. **Failure Isolation:**
   - If the `CGEventTap` is disabled by the OS, EyeMouse immediately transitions to an error state and suspends all mouse injection.
   - If the Vision Engine crashes or stops responding, the host watchdog resets state, releases buttons, and freezes the cursor.
   - If camera tracking loses the face, the cursor position freezes at the last valid coordinate rather than jumping to (0,0) or drifting.
5. **Zero Frame Storage:** Neither camera frames, landmark arrays, nor facial representations are ever written to disk or transmitted across networks.

---

## 4. Standalone Bundle Structure (`EyeMouse.app`)

EyeMouse is packaged as a completely self-contained macOS Application Bundle that requires no pre-installed Python, Xcode, Antigravity, or terminal intervention:

```text
EyeMouse.app/
├── Contents/
│   ├── Info.plist                     # LSUIElement=1, NSCameraUsageDescription, Bundle ID
│   ├── MacOS/
│   │   ├── EyeMouse                   # Native Swift AppKit executable (Host & Menu Bar)
│   │   ├── eyemouse-engine            # Standalone PyInstaller Mach-O executable
│   │   └── _internal/                 # Bundled Python 3.15 runtime, dylibs, MediaPipe tasks
│   │       ├── Python                 # Python dynamic library
│   │       ├── cv2/                   # OpenCV native dylibs
│   │       ├── mediapipe/             # MediaPipe C++ graph runtimes
│   │       └── models/
│   │           └── face_landmarker.task  # Pinned offline MediaPipe task model
│   └── Resources/
│       ├── AppIcon.icns               # Application icon
│       └── Assets.car                 # Menu bar glyphs and UI assets
```

---

## 5. Performance Targets & Latency Budgets

| Pipeline Stage | Implementation | Latency Target |
| :--- | :--- | :--- |
| **Camera Acquisition** | AVFoundation capture @ 60 FPS / 720p | $8\text{--}16\ \text{ms}$ |
| **Landmark Inference** | MediaPipe Face Landmarker (ARM64 NEON) | $12\text{--}16\ \text{ms}$ |
| **Pose & Filter Compute** | Procrustes + One-Euro Filter (NumPy) | $< 1.5\ \text{ms}$ |
| **IPC Transfer** | Unix Domain Socket (Local Memory) | $< 0.8\ \text{ms}$ |
| **Key Tap Verification** | Atomic bitmask check (Swift) | $< 0.05\ \text{ms}$ |
| **Cursor Post** | Native `CGEventPost` | $< 0.5\ \text{ms}$ |
| **Total End-to-End Latency** | **Camera photon to Cursor motion** | **$< 35\ \text{ms}$** |

---

## 6. Milestone 2 Implementation Realization

Milestone 2 fully realized the headless vision engine core, the zero-copy shared memory frame transport, and the standalone frozen bundle:

### 6.1 Validated IPC Realization
- **Frame Data Plane**: Multi-slot POSIX shared-memory ring buffer (default 4 slots $\times$ 4 MiB) governed by 64-byte Little-Endian headers (`EMSH`, `EYEM`).
- **Control Plane**: AF_UNIX stream socket carrying line-delimited JSON commands (`HANDSHAKE`, `FRAME`, `ACK`, `SHUTDOWN`).
- **Darwin C-Bridge**: Variadic promotion in Darwin's `shm_open` resolved via `eyemouse_shm_open` C wrapper; memory-mapped pointers strictly used for all frame reads/writes.
- **Darwin `PSHNAMLEN`**: Segment naming strictly clamped to $\le 31$ bytes.

### 6.2 Empirical Benchmark Performance (Apple Silicon M-Series)
- **Pipeline Latency**: Mean **2.98 ms** (median: 2.98 ms, p95: 3.06 ms, max: 3.35 ms) on 640x480 BGRA frames.
- **Pipeline Throughput**: **335.1 FPS** (11x headroom over 30 FPS webcam baseline).
- **Cross-Language IPC Throughput**: **275.9 FPS** with 3.35 ms average engine processing time across Swift producer and Python consumer processes.
- **Process Memory**: 173.0 MB RSS bounded resident set size.
- **Frozen Packaging**: Built and verified standalone ARM64 binary `dist_engine/eyemouse_engine/eyemouse_engine` with zero external Python dependency.

---

## 7. Milestone 3 Implementation Realization

Milestone 3 realized the complete native macOS host application, the global hold-key control subsystem, and live AVFoundation camera integration:

### 7.1 Native AppKit & Menu Bar Subsystem
- **Application Bundle**: Assembled and codesigned `build/EyeMouse.app` with `LSUIElement = true`, stable bundle ID `com.eyemouse.EyeMouse`, and embedded `packaging/Info.plist`.
- **Menu Bar Control**: `MenuBarController` manages an `NSStatusItem` reflecting live safety states (`INACTIVE`, `STARTING`, `ACTIVE`, `STOPPING`, `ERROR`), with dynamic camera device selection submenus and shortcut display.
- **Live Diagnostics**: SwiftUI `DiagnosticsView` presents live camera active indicators, face presence badges, normalized iris coordinates, blink dynamics, 3D head pose, FPS, and latency metrics with an explicit `UNCALIBRATED OPTICAL PROXY` notice.

### 7.2 Global Hold Shortcut & Safety Gate
- **Core Graphics Event Tap**: `GlobalEventTap` intercepts `keyDown`, `keyUp`, and `flagsChanged` globally via `.defaultTap`.
- **Keystroke Suppression**: Suppresses trigger keystrokes so holding `Control + Option + E` does not leak characters into Terminal, Chrome, VS Code, or Finder.
- **Synchronous Safety Cutoff**: The moment any chord key is released, `SafetyStateManager` clears `canProcessFrames` synchronously in memory. Real-world testing verified **0 leaked frames** after release.
- **Edge-Case Resilience**: Traps `tapDisabledByTimeout` and `tapDisabledByUserInput` for automatic re-enablement, and hooks `NSWorkspace.willSleepNotification` to force immediate deactivation upon system sleep or screen lock.

### 7.3 Real Webcam Integration
- **AVFoundation Ownership**: `CameraManager` and `CameraCaptureService` discover and manage physical built-in, external, and Continuity cameras, streaming 640x480 BGRA frames on a dedicated serial queue.
- **Row Stride Integrity**: Respects actual `CVPixelBufferGetBytesPerRow` values, preventing row shearing across arbitrary hardware alignments.
- **Real-World Live Benchmark**: Measured **3.69 ms** mean engine latency (median: 3.49 ms, p95: 4.65 ms) on live frames delivered by the MacBook Air Camera at ~20 FPS with 0 dropped frames.

---

## 8. Milestone 4 Implementation Realization

Milestone 4 realized personalized gaze-to-screen coordinate calibration and objective accuracy validation against independent target locations:

### 8.1 Dedicated Calibration Package (`src/vision_engine/calibration/`)
- **Geometry & Target Grids (`geometry.py`)**: Defines physical Retina and logical display scaling dynamically from `NSScreen.main`, providing a 13-point calibration layout (center, 4 corners, 4 edges, 4 inner quadrants) and a strictly independent 9-point held-out validation grid.
- **Fixation Dynamics & MAD Filtering (`collector.py`)**: Implements a 400 ms saccadic settling window followed by a 1000 ms steady fixation window. Filters blinks, face losses, and outliers exceeding $2.5 \times \text{MAD}$ (Median Absolute Deviation), extracting a robust median feature vector per target.
- **Closed-Form Dual Regression Models (`models.py`)**:
  - **Affine / Linear Baseline**: Maps 8 design features linearly to normalized screen space $[0.0, 1.0]$.
  - **Polynomial Degree-2 Ridge Regression**: Expands features to 19 terms (linear, quadratic, cross-eye and head pose interactions) regularized by an unpenalized-intercept $\ell_2$ penalty matrix.
  - **Leave-One-Out Cross-Validation (LOOCV)**: Optimizes regularization strength $\lambda \in [10^{-4}, 10.0]$ across calibration folds.
- **Held-Out Validation Engine (`evaluator.py`)**: Evaluates predictions exclusively against 9 unseen locations, reporting Euclidean pixel error (mean, median, P95), normalized coordinate error, visual angle in degrees ($\theta$), horizontal vs vertical error, center vs corner accuracy, and qualitative grading (`EXCELLENT`, `GOOD`, `FAIR`, `POOR`).
- **Profile Persistence (`profile.py`)**: Stores model weights, $\lambda$, camera ID, resolution, and validation statistics in `~/.eyemouse/profiles/active_profile.json`. Incompatible profiles (due to display resizing or camera switching) are cleanly invalidated.
- **Real-Time Calibrated Estimator (`estimator.py`)**: Evaluates calibrated gaze coordinates in $< 0.1\text{ ms}$ with adaptive OneEuroFilter smoothing, bound safely to display dimensions.

### 8.2 Native Host Calibration Subsystem (`src/native_host/Calibration/`)
- **Full-Screen Calibration Wizard (`CalibrationWindowController.swift`)**: Native SwiftUI full-screen overlay guiding attention across 13 calibration targets and 9 validation targets with animated shrinking fixation rings, progress counters, and cancellation via Escape key.
- **Diagnostic Gaze Marker Overlay (`LiveGazePreviewOverlay.swift`)**: Click-through floating reticle (`ignoresMouseEvents = true`) displaying real-time calibrated gaze position, confidence halo, and tracking status.
- **Zero OS Cursor Injections**: Operating system cursor movement (`CGEventPost`), clicking, dwell, and dragging remain **strictly unexecuted** throughout Milestone 4.



