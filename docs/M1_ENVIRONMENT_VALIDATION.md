# Milestone 1: Environment Bootstrap and Dependency Validation

**Execution Date:** November 4, 2026  
**Status:** **PASSED / COMPLETE**  
**Auditor:** Lead macOS Application Engineer & Computer-Vision Engineer  
**Workspace:** `/Users/sairamv/Projects/Eye Mouse`  

---

## 1. Executive Summary

Milestone 1 (M1) establishes the verified, reproducible environment foundation for **EyeMouse**. All Python dependencies, offline machine-learning assets, frozen binary packaging recipes, and native Swift toolchains have been validated using real automated tests.

No system Python installations or permissions were modified, no global packages were touched, and all work was conducted inside an isolated project virtual environment (`.venv`).

---

## 2. Python Environment Bootstrap (Task A)

### 2.1 Python Interpreter Audit
* **Installed System Interpreters:**
  * Primary: `/Library/Frameworks/Python.framework/Versions/3.15/bin/python3` (Python 3.15.0 ARM64 universal binary, Clang 21.0.0).
  * Secondary: `/usr/bin/python3` (Python 3.9.6, Apple legacy system runtime).
* **Patch Check for 3.15.2:**
  * Python 3.15.2 is available on python.org as a package (`.pkg`), but installing it requires root installer privileges that would overwrite system frameworks in `/Library/Frameworks`.
  * In strict compliance with the project rule (*"Do not uninstall or modify the system Python; Ensure all installations happen inside .venv"*), the stable installed Python 3.15.0 ARM64 was selected as the base interpreter.
* **Virtual Environment Initialization:**
  ```bash
  /Library/Frameworks/Python.framework/Versions/3.15/bin/python3 -m venv .venv
  source .venv/bin/activate
  ```
  * Active Interpreter: `/Users/sairamv/Projects/Eye Mouse/.venv/bin/python`
  * Active Pip: Version 26.2.1

---

## 3. Dependency Evaluation & Pinning (Task B)

### 3.1 Primary Dependencies Evaluated
* **`mediapipe==1.2.2`:** Pinned. Provides Google's high-speed Face Landmarker tasks running natively on Apple Silicon CPU/Neural Engine.
* **`numpy==2.5.4`:** Pinned. High-performance numerical linear algebra supporting landmark normalization and filter math.
* **`opencv-contrib-python==5.0.0.93`:** Pinned. Pulled in by MediaPipe. Exposes native `cv2` image manipulation functions. (`opencv-python` was not installed separately to prevent binary collisions).
* **`pyinstaller==6.22.3`:** Pinned. Required for freezing the standalone vision engine.
* **`pytest==9.2.0`:** Pinned. Automated test runner.
* **`scipy` Evaluation:**
  * Audited source code in `research/eye-tracker`: All rigid head pose alignment (Procrustes) is implemented using pure NumPy singular value decomposition (`numpy.linalg.svd`), and the `OneEuroFilter` uses standard Python `math`.
  * `scipy` is **not required** by the actual vision pipeline. Omitting it eliminated ~100MB of unnecessary dependency bloat.

### 3.2 Manifests Created
* High-level manifest: [`requirements.txt`](file:///Users/sairamv/Projects/Eye%20Mouse/requirements.txt)
* Complete transitive lockfile: [`requirements.lock`](file:///Users/sairamv/Projects/Eye%20Mouse/requirements.lock)

---

## 4. Model Asset Acquisition & Offline Verification (Task C)

* **Canonical Source:** Google Cloud Storage MediaPipe repository:
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
* **Local Storage Path:** `assets/models/face_landmarker.task` (Size: 3.6 MB)
* **License:** Apache License 2.0 (saved in `assets/models/LICENSE.txt`).
* **SHA-256 Checksums Verified:**
  * `assets/models/face_landmarker.task`:
    `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
  * Extracted `face_landmarks_detector.tflite`:
    `c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5`
  * Extracted `geometry_pipeline_metadata_landmarks.binarypb`:
    `bdbcda96dfcb7da883da124aaa2c55dee49770d934f0fcc71747f8c21bdc75b4`
* **Zero Runtime Downloads:** The model is bundled directly into the repository and app bundle.

---

## 5. Actual Test Validation (Task D)

### 5.1 Python Import & Architecture Verification
Executed script verifying library versions and architecture:
```text
=== Python Environment Validation ===
Python Version: 3.15.0 (v3.15.0:d220409a13b, Oct  9 2026, 08:17:44) [Clang 21.0.0]
Platform Machine: arm64
NumPy Version: 2.5.4
OpenCV Version: 5.0.0
MediaPipe Version: 1.2.2
```
All native libraries (`.so`, `.dylib`, `libmediapipe.dylib`) were verified using `file` as native `Mach-O 64-bit arm64`.

### 5.2 Automated Inference Test (`tests/test_vision_bootstrap.py`)
Run command: `.venv/bin/pytest tests/test_vision_bootstrap.py -v`
```text
============================= test session starts ==============================
platform darwin -- Python 3.15.0, pytest-9.2.0, pluggy-1.7.0
rootdir: /Users/sairamv/Projects/Eye Mouse
collecting ... collected 1 item

tests/test_vision_bootstrap.py::test_mediapipe_landmarker_inference PASSED [100%]

============================== 1 passed in 0.84s ===============================
```
* **Results:** Successfully initialized `FaceLandmarker` with `face_landmarker.task`, processed a deterministic synthetic face image, and verified all 478 landmarks (including left iris index 468 and right iris index 473).

### 5.3 PyInstaller Frozen Standalone Verification
1. **Initial Freeze & Failure Discovery:**
   * Build command: `pyinstaller --onedir tests/smoke_engine.py`
   * Failure on launch:
     ```text
     ModuleNotFoundError: No module named 'mediapipe.tasks.c'.
     ```
   * *Root Cause Identified:* MediaPipe 1.2.2 loads its C runtime dynamically via `importlib.resources.files('mediapipe.tasks.c').joinpath('libmediapipe.dylib')`. PyInstaller's default import graph analyzer missed this package.
2. **Resolution & Rebuild:**
   * Updated PyInstaller command with `--collect-all mediapipe`:
     ```bash
     .venv/bin/pyinstaller --onedir --clean --noconfirm \
         --collect-all mediapipe \
         --add-data "assets/models/face_landmarker.task:assets/models" \
         --distpath dist_smoke --workpath build_smoke \
         tests/smoke_engine.py
     ```
3. **Execution of Standalone Frozen Binary:**
   * Command: `./dist_smoke/smoke_engine/smoke_engine`
   * Execution Output:
     ```text
     === Frozen Vision Engine Smoke Test ===
     Python Executable: /Users/sairamv/Projects/Eye Mouse/dist_smoke/smoke_engine/smoke_engine
     Is Frozen: True
     NumPy Version: 2.5.4
     MediaPipe Version: 1.2.2
     OpenCV Version: 5.0.0
     Loading Model from: .../dist_smoke/smoke_engine/_internal/assets/models/face_landmarker.task
     INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
     SUCCESS: Face detected with 478 landmarks.
     Left Iris: (0.4410, 0.4727)
     Right Iris: (0.5604, 0.4742)
     FROZEN_VISION_SMOKE_TEST_SUCCESS
     ```
   * Exit Code: `0`

---

## 6. Swift Toolchain Validation (Task E)

### 6.1 SDK Flag & Toolchain Configuration
Direct `swiftc` compilation was configured using the root [`Makefile`](file:///Users/sairamv/Projects/Eye%20Mouse/Makefile):
```makefile
SDK_PATH ?= /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
TARGET_ARCH ?= arm64-apple-macos14.0
SWIFTC ?= /usr/bin/swiftc
SWIFT_FLAGS = -sdk $(SDK_PATH) -target $(TARGET_ARCH) -O
```

### 6.2 AppKit Smoke Test Execution
* Source file: [`tests/swift_smoke/main.swift`](file:///Users/sairamv/Projects/Eye%20Mouse/tests/swift_smoke/main.swift)
* Build & Run command: `make test-swift`
* Output:
  ```text
  mkdir -p build
  /usr/bin/swiftc -sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk -target arm64-apple-macos14.0 -O -o build/swift_smoke tests/swift_smoke/main.swift
  ./build/swift_smoke
  === Swift & AppKit Toolchain Smoke Test ===
  Process Name: swift_smoke
  Operating System Version: Version 27.0.1 (Build 26A434)
  Target Architecture: arm64 (Apple Silicon) - VERIFIED
  AppKit NSApplication initialized with .accessory activation policy.
  Discovered Displays: 1
    Display [0]: 1470x956 at origin (0, 0)
  Discovered Video Input Devices: 3
    Camera: MacBook Air Camera (ID: 6C707041-05AC-0011-0003-000000000001)
    Camera: Sharing Camera(Virtual) (ID: A8926A26-32E1-42B8-AFE2-AAFBA66827D8)
    Camera: Sai’s Iphone Camera (ID: F42F60FB-66F5-414B-84CC-816900000001)
  SWIFT_APPKIT_SMOKE_TEST_SUCCESS
  ```
* Exit Code: `0` (Zero compiler warnings, zero runtime errors).

---

## 7. Acceptance Criteria Verification

| Acceptance Criterion | Result | Evidence |
| :--- | :--- | :--- |
| **Isolated ARM64 Python environment works** | **PASS** | `.venv` created with Python 3.15.0 ARM64 |
| **All required imports work** | **PASS** | `numpy`, `cv2`, `mediapipe` imported cleanly |
| **MediaPipe model loads and processes an image** | **PASS** | `pytest tests/test_vision_bootstrap.py` passed (0.84s, 478 landmarks) |
| **PyInstaller frozen smoke test runs** | **PASS** | `./dist_smoke/smoke_engine/smoke_engine` executed with code 0 |
| **Swift AppKit smoke test runs** | **PASS** | `./build/swift_smoke` executed with code 0 |
| **Dependencies are pinned and reproducible** | **PASS** | `requirements.txt` and `requirements.lock` generated |
| **No global Python dependencies modified** | **PASS** | Global frameworks untouched; `.venv` isolated |
| **No camera permission requested from Terminal** | **PASS** | `AVCaptureDevice.DiscoverySession` enumerated devices passively |
| **No full application implementation started** | **PASS** | Only validation harnesses and build scaffolding created |

---

## 8. Conclusion & M2 Readiness

Milestone 1 is complete. All prerequisites for **Milestone 2 (Headless Vision Engine Core & Shared-Memory Frame IPC)** are satisfied.

