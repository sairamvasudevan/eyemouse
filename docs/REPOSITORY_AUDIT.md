# Repository Audit: Comparative Analysis of Eye-Tracking Codebases

**Audit Date:** November 4, 2026  
**Auditor:** Lead macOS Application Engineer & Computer-Vision Engineer  
**Repositories Cloned in `research/`:**
1. `research/eye-control-mouse` (Author: Sohaib Basharat, commit `1a2e32b`)
2. `research/eye-tracker` (Author: Buğra Şıkel, tag `v0.3.0`, commit `9644527`)

---

## 1. Executive Summary & Comparison Matrix

Both repositories address aspects of webcam-based gaze tracking, but their engineering maturity, architectural goals, and platform support diverge significantly.

| Audit Dimension | `eye-control-mouse` (Sohaib Basharat) | `eye-tracker` (Buğra Şıkel) | EyeMouse Recommendation |
| :--- | :--- | :--- | :--- |
| **LICENSE** | MIT License (2025) | MIT License (2026) | Full license compatibility for code adaptation and reuse with proper attribution. |
| **macOS Support** | Unofficial / Minimal. Relies on PyAutoGUI and pynput fallbacks. No menu bar or TCC handling. | High / Production-Grade. Deep AppKit, Quartz, ApplicationServices, AVFoundation, App Nap inhibition. | Adopt `eye-tracker`'s macOS system engineering architecture; replace its UI with native Swift menu-bar app. |
| **Application Structure** | Monolithic PyQt6 desktop application with single worker thread. | Multi-tier modular architecture (engine, vision backends, platform, IPC, tray UI). | Adopt decoupled architecture: Native Swift Host + Headless Python Vision Engine. |
| **Gaze Estimation** | 2D iris position relative to eye corners (`iris_in_eye_v1`). Drifts severely with head motion. | 3D head pose (Procrustes fitting) + normalized iris vector relative to eye socket geometry. | Reuse `eye-tracker`'s 3D head-pose compensated feature pipeline. |
| **Calibration** | 9/12/16-point 2D grid using affine, polynomial, or RBF mapping. | State machine collector (`settle` -> `collect`), leave-one-out cross-validation, Ridge regression. | Port `eye-tracker`'s `CalibrationCollector` and cross-validation model. |
| **Head-Pose Compensation** | None. Assumes completely static head. | Protobuf-free extraction of MediaPipe canonical 3D mesh (`face_geometry.py`) with rigid pose fitting. | **Direct Reusable Asset:** Extract `face_geometry.py` for head-pose compensation. |
| **Cursor Filtering** | Basic SimpleKalmanFilter or EWMA. Suffers from lag on saccades or jitter when stationary. | Time-aware adaptive `OneEuroFilter` (Casiez et al., CHI 2012). | **Direct Reusable Asset:** Extract `filters.py` (`OneEuroFilter`). |
| **Blink Recognition** | Fixed Eye Aspect Ratio (EAR = 0.21) threshold. High false-positive rate across facial structures. | Rolling median adaptive baseline openness with safety floor. | Adapt `eye-tracker`'s rolling baseline blink detector. |
| **Global Hotkey** | `pynput` listener. Key-down toggle only. Cannot detect sustained physical hold. | Carbon `RegisterEventHotKey`. Key-down event only. Cannot track key-release or modifier hold. | **Reject both.** Mandatory requirement: Native Swift `CGEventTap` tracking physical key hold/release. |
| **Camera Permissions** | None. Crashes or opens black frame if ungranted. | Queries `AVCaptureDevice authorizationStatusForMediaType:` via pyobjc. | Native Swift `AVCaptureDevice.requestAccess(for: .video)` in host UI. |
| **Accessibility / TCC** | None. Relies on PyAutoGUI silently failing. | Checks `AXIsProcessTrustedWithOptions`, detects ad-hoc cdhash invalidation after builds. | Native Swift `AXIsProcessTrustedWithOptions` + `CGRequestListenEventAccess` in host UI. |
| **Packaging & DMG** | Windows Inno Setup focused (`build_installer.py`). | PyInstaller spec (`eye-tracker.spec`) + custom `make_dmg.sh` with `hdiutil` and `codesign`. | Adapt `make_dmg.sh` and PyInstaller bundling patterns for `EyeMouse.app`. |
| **Test Coverage** | 9 test files, basic unit tests for math and clicker. | 60+ test suites with fakes for Quartz, AX, AVFoundation, camera, and scheduler. | Establish hybrid test suite: Swift XCTest + pytest. |

---

## 2. In-Depth Repository Inspection: `eye-control-mouse`

### 2.1 License & Provenance
- **License File:** `research/eye-control-mouse/LICENSE`
- **Terms:** Standard MIT License.
- **Copyright:** Copyright (c) 2025 Eye Control Mouse Contributors.
- **Suitability:** Permissive; allows adaptation and sublicensing with attribution.

### 2.2 Application Structure
- Package root: `src/eyemouse/`
- Key files:
  - `app.py`: CLI parser and bootstrap.
  - `gui.py`: PyQt6 monolithic window combining camera preview, calibration trigger, settings sliders, and processing timer.
  - `detector.py`: MediaPipe Face Landmarker wrapper.
  - `tracker.py`: GazeTracker wrapping Kalman/EWMA filters and polynomial/affine mapper.
  - `calibration.py`: Grid-based calibration manager.
  - `clicker.py`: PyAutoGUI click dispatcher with fixed EAR blink detection.
  - `hotkey.py`: `pynput.keyboard.GlobalHotKeys` listener.

### 2.3 Gaze Estimation & Head-Pose Compensation
- **Feature Extraction (`detector.py`):**
  Uses MediaPipe Face Landmarker tasks. Computes `_iris_in_eye`:
  ```python
  rel = iris - a
  x = float(rel @ axis) / width_sq
  y = float(axis[0] * rel[1] - axis[1] * rel[0]) / width_sq
  ```
  This computes the 2D offset of iris center relative to inner/outer eye corners.
- **Fatal Flaw:** It lacks 3D head pose estimation. As soon as the user rotates, pitches, or translates their head, the eye corners shift in perspective, causing the predicted gaze coordinates to drift off-screen.

### 2.4 Cursor Filtering & Click Detection
- **Filtering (`utils.py`):**
  Implements a rudimentary 1D Kalman filter and EWMA filter. The EWMA filter introduces unacceptable latency when moving between distant targets, while the Kalman filter with constant measurement variance exhibits jitter during fixations.
- **Click Detection (`clicker.py`):**
  Fixed EAR threshold (`0.21`). If a user has narrower eyes or ptosis, the detector either registers continuous blinks or fails completely.
- **Mouse Injection (`clicker.py`):**
  Uses `pyautogui.moveTo()` and `pyautogui.click()`. PyAutoGUI introduces unavoidable latency, does not provide atomic event timing, and cannot intercept native event streams.

### 2.5 Hotkey Handling & Permissions
- Uses `pynput` (`src/eyemouse/hotkey.py`).
- **Limitation:** Listens only for key triggers (`_on_activate`). It cannot detect whether a key is *currently held down* or *released*.
- Zero permission verification. If Accessibility or Camera access is missing, it logs an exception or fails silently.

---

## 3. In-Depth Repository Inspection: `eye-tracker`

### 3.1 License & Provenance
- **License File:** `research/eye-tracker/LICENSE`
- **Terms:** Standard MIT License.
- **Copyright:** Copyright (c) 2026 Buğra Şıkel.
- **Suitability:** Permissive; excellent candidate for extracting architectural modules.

### 3.2 Application Structure
- Package root: `src/eye_tracker/`
- Highly decoupled, enterprise-grade architecture:
  - `platform/macos.py`: Deep native macOS integration using `pyobjc` (Quartz, AppKit, ApplicationServices, AVFoundation).
  - `vision/backends/face_geometry.py`: Canonical 3D face mesh and Procrustes analysis for rigid head pose.
  - `vision/backends/facemesh_backend.py`: High-precision landmark metrics and adaptive blink baseline.
  - `gaze/filters.py`: One-Euro filter implementation with velocity-dependent frequency cutoff.
  - `gaze/calibration.py`: Settle/collect state machine, target generator, and cross-validation reports.
  - `gaze/model.py`: Polynomial and Ridge regression screen projection models.
  - `engine/guard.py`: Failsafe cursor controller ensuring cursor is never moved erroneously.
  - `packaging/macos/make_dmg.sh`: Standalone DMG staging, ad-hoc codesigning, and validation.

### 3.3 Gaze Estimation & Head-Pose Compensation
- **`face_geometry.py`:**
  A remarkable implementation: decodes MediaPipe's `geometry_pipeline_metadata_landmarks.binarypb` wire-format protocol buffer directly without requiring the heavy `protobuf` library.
  It extracts 468 canonical 3D vertices in centimeter space, identifies rigid Procrustes landmarks (forehead, cheekbones, nose bridge), and calculates the 3D rotation matrix and translation vector of the head.
- **Gaze Vector:**
  Combines the 3D head orientation vector with the normalized 3D eyeball optical axis, completely decoupling user gaze from natural head movement.

### 3.4 Cursor Filtering & Blink Recognition
- **`filters.py`:**
  Implements the academic standard `OneEuroFilter` (Casiez, Roussel, Vogel, ACM CHI 2012):
  $$\alpha = \frac{1}{1 + \frac{\tau}{T_e}}, \quad \tau = \frac{1}{2\pi f_c}, \quad f_c = f_{c,\text{min}} + \beta |\dot{x}|$$
  When gaze is steady, $f_c$ drops to $0.3\text{ Hz}$, eliminating jitter completely. When the user executes a saccade ($|\dot{x}| > 2000\text{ px/s}$), $f_c$ scales instantaneously, providing zero perceptible lag.
- **Blink Recognition (`facemesh_backend.py`):**
  Uses an adaptive baseline: tracks a rolling median of open-eye distances. A blink is declared when eye openness drops below $0.5 \times \text{baseline}$, with a hard safety minimum. This eliminates false positives across differing facial morphologies.

### 3.5 macOS Platform Layer & Permissions
- **`platform/macos.py`:**
  - Implements `MacPlatform.accessibility_status()`: Detects when macOS has invalidated an ad-hoc signed build's Accessibility trust after an update (the "stale permission" bug).
  - Uses `NSProcessInfo.processInfo().beginActivityWithOptions_reason_()` to prevent App Nap from throttling background vision timers.
  - Queries `AVCaptureDevice.authorizationStatusForMediaType_()`.
  - Warps mouse via CoreGraphics `CGWarpMouseCursorPosition` and posts synthetic events.
- **Hotkey Limitation:**
  In `platform/hotkeys.py`, `MacHotkeyManager` uses Carbon `RegisterEventHotKey`. Carbon global hotkeys only notify on `kEventHotKeyPressed` (press); they do not generate release notifications or state tracking for modifier keys.

---

## 4. Reusable Modules and Source Code Assets

The following files from `research/` have been audited and selected for adaptation into EyeMouse:

| Source File in `research/` | Reusable Concept / Module | Purpose in EyeMouse |
| :--- | :--- | :--- |
| `eye-tracker/src/eye_tracker/gaze/filters.py` | `OneEuroFilter` | Jitter-free, low-latency cursor filtering. |
| `eye-tracker/src/eye_tracker/vision/backends/face_geometry.py` | `FaceGeometry`, canonical mesh, Procrustes alignment | Robust 3D head-pose compensation. |
| `eye-tracker/src/eye_tracker/vision/backends/facemesh_backend.py` | Landmark extraction & adaptive blink baseline | High-precision iris tracking and blink click detection. |
| `eye-tracker/src/eye_tracker/gaze/calibration.py` | `CalibrationCollector`, `CalibrationTarget`, leave-one-out cv | Accurate screen calibration mapping. |
| `eye-tracker/packaging/macos/make_dmg.sh` | `hdiutil` staging, signing verification, UDZO compression | Production DMG build script. |
| `eye-tracker/packaging/pyinstaller/eye-tracker.spec` | Dependency pruning, Mach-O bundling | PyInstaller recipe for standalone vision engine binary. |

---

## 5. Architectural Deficiencies in Both Repositories vs EyeMouse Requirements

Neither repository fulfills the core requirements of EyeMouse off-the-shelf:

1. **Physical Key-Hold Requirement:**
   - *Requirement:* "Eye control must only be active while a configurable keyboard shortcut is physically held. Releasing any required key must immediately stop all eye-generated cursor and mouse actions."
   - *Audit Finding:* `eye-control-mouse` uses `pynput` (toggle only). `eye-tracker` uses Carbon `RegisterEventHotKey` (press only).
   - *Engineering Solution:* Implement a native Swift `CGEventTap` that listens to `kCGEventKeyDown`, `kCGEventKeyUp`, and `kCGEventFlagsChanged`, maintaining an atomic state of pressed physical keys.

2. **Native Menu-Bar OS Integration:**
   - *Requirement:* Standalone native macOS SwiftUI/AppKit menu-bar application.
   - *Audit Finding:* Both repositories rely on heavy Python GUI frameworks (PyQt6 and PySide6), resulting in large bundle sizes, slow startup, and non-native UI behavior.
   - *Engineering Solution:* Native Swift AppKit/SwiftUI `NSStatusItem` menu-bar host that manages permissions, displays live status, and controls a headless Python vision engine subprocess via IPC.

3. **Permission Workflow & Input Monitoring:**
   - *Requirement:* Explicit macOS permission checking and user guidance without bypassing security.
   - *Audit Finding:* Neither codebase properly requests or guides the user through `Input Monitoring` (`kTCCServiceListenEvent`), which macOS Catalina+ strictly requires for `CGEventTap`.
   - *Engineering Solution:* Dedicated Swift permissions manager utilizing `CGRequestListenEventAccess()` and `AXIsProcessTrustedWithOptions()`.

