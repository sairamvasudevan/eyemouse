# Environment Audit: EyeMouse macOS Target Machine

**Audit Date:** November 4, 2026  
**Auditor:** Lead macOS Application Engineer & Computer-Vision Engineer  
**Workspace:** `/Users/sairamv/Projects/Eye Mouse`

---

## 1. Executive Summary

This environment audit evaluates the host workstation for the development, native compilation, computer-vision pipeline integration, and standalone packaging of **EyeMouse**.

All hardware and operating system parameters were inspected without altering system permissions or dependencies. Native arm64 capabilities are verified for both Swift compilation and Python scientific/vision libraries.

| Component | Status | Finding | Action / Mitigation |
| :--- | :--- | :--- | :--- |
| **CPU Architecture** | **Ready** | Apple Silicon (`arm64`), M-series | Target native arm64 binaries exclusively |
| **Operating System** | **Ready** | macOS 27.0.1 (Build 26A434, Darwin 27.0.0) | High-level macOS API support (AppKit, AVFoundation, CGEvent) |
| **Apple Toolchain** | **Ready with flag** | CommandLineTools 21.0.0, Swift 6.3.3 | Must pass `-sdk .../MacOSX26.5.sdk` to `swiftc` |
| **Full Xcode.app** | **Not Present** | No `/Applications/Xcode.app` (`xcodebuild` unavailable) | Use `swiftc` + Makefile directly (Xcode not required) |
| **Swift Package Manager** | **Blocked** | `swift-package` dyld symbol mismatch in SDK | Use direct `swiftc` build pipeline for `EyeMouse.app` |
| **Python Environment** | **Ready** | Python 3.15.0 ARM64 Universal (`/Library/Frameworks/...`) | Create workspace virtual environment (`.venv`) |
| **Vision Libraries** | **Ready** | `mediapipe` 1.2.2, `opencv-python` 5.0.0, `numpy` 2.5.4 | Pre-built arm64 wheels resolved and verified |
| **Packaging Tools** | **Ready** | `hdiutil`, `codesign`, `make` present; PyInstaller verified | PyInstaller 6.22.3 bundles standalone Python engine |
| **Package Manager (uv)** | **Not Installed** | `uv not found` | Use standard `python3 -m venv` and `pip` |

---

## 2. Host Machine & Operating System Inspection

### 2.1 Hardware Architecture (`uname -m`)
```bash
$ uname -m
arm64
```
The host is an Apple Silicon Mac executing natively in 64-bit ARM architecture. All compilation targets, dynamic libraries, and Python C-extensions must target `arm64-apple-macos`. No Rosetta 2 emulation is required or recommended.

### 2.2 macOS Version (`sw_vers`)
```bash
$ sw_vers
ProductName:    macOS
ProductVersion: 27.0.1
BuildVersion:   26A434
```
The system runs macOS 27.0.1. Modern macOS privacy enforcement (TCC - Transparency, Consent, and Control) strictly mandates runtime user consent for:
- **Camera Access** (`kTCCServiceCamera` via AVFoundation)
- **Accessibility** (`kTCCServiceAccessibility` via ApplicationServices)
- **Input Monitoring** (`kTCCServiceListenEvent` via CoreGraphics `CGEventTapCreate`)

---

## 3. Developer Tools & Swift Compilation Audit

### 3.1 Active Developer Directory (`xcode-select -p`)
```bash
$ xcode-select -p
/Library/Developer/CommandLineTools
```

### 3.2 Full Xcode Availability (`xcodebuild -version`)
```bash
$ xcodebuild -version
xcode-select: error: tool 'xcodebuild' requires Xcode, but active developer directory '/Library/Developer/CommandLineTools' is a command line tools instance
```
`Xcode.app` is not installed on this machine. Full Xcode project builds (`.xcodeproj` / `xcodebuild`) cannot be used.

### 3.3 Swift & Clang Compilers
```bash
$ swift --version
swift-driver version: 1.148.6 Apple Swift version 6.3.3 (swiftlang-6.3.3.1.3 clang-2100.1.1.101)
Target: arm64-apple-macosx28.0

$ clang --version
Apple clang version 21.0.0 (clang-2100.1.1.101)
Target: arm64-apple-darwin27.0.0
InstalledDir: /Library/Developer/CommandLineTools/usr/bin
```

### 3.4 Crucial Toolchain Nuance & Discovery
When invoking `swiftc` without explicit SDK arguments or invoking `swift package`, the following errors occur:
1. `swift package`:
   ```text
   dyld: Symbol not found: _$s19BuildServerProtocol019SourceKitInitializeA12ResponseDataV14encodeToLSPAny08LanguagebC00K0OyF
   Referenced from: .../usr/bin/swift-package
   ```
2. `swiftc` default:
   ```text
   MacOSX27.0.sdk/usr/lib/swift/Swift.swiftmodule/arm64e-apple-macos.swiftinterface:
   failed to build module 'Swift'; this SDK is not supported by the compiler
   (the SDK is built with 'Apple Swift version 6.4', while this compiler is 'Apple Swift version 6.3.3').
   ```

**The Solution:**
Inspecting `/Library/Developer/CommandLineTools/SDKs/` reveals:
- `MacOSX15.4.sdk`
- `MacOSX26.5.sdk` (aliased by `MacOSX.sdk`)
- `MacOSX27.0.sdk`

Compiling explicitly against `MacOSX26.5.sdk` resolves all interface mismatches:
```bash
swiftc -sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
       -target arm64-apple-macos14.0 \
       -O -o EyeMouseApp <sources...>
```
This was verified with a live test compilation and confirmed 100% operational for native AppKit and Foundation code generation.

---

## 4. Python Environment & Computer Vision Wheels Audit

### 4.1 Installed Python Binaries
1. `/Library/Frameworks/Python.framework/Versions/3.15/bin/python3`:
   - Version: **Python 3.15.0**
   - Format: Mach-O universal binary (arm64 + x86_64)
   - Pip: Version 26.2.1
2. `/usr/bin/python3`:
   - Version: **Python 3.9.6** (legacy Apple system python)

### 4.2 Dependency Availability Check (Dry-Run Verification)
To confirm wheel compatibility with ARM64 macOS without polluting system frameworks, a dry-run resolution was performed:
```bash
pip3 install --dry-run opencv-python mediapipe numpy pyinstaller
```
**Results:**
- `opencv-python`: Resolved `opencv_python-5.0.0.93-cp37-abi3-macosx_13_0_arm64.whl` (Native ARM64 ABI3)
- `mediapipe`: Resolved `mediapipe-1.2.2-py3-none-macosx_11_0_arm64.whl` (Native ARM64)
- `numpy`: Resolved `numpy-2.5.4-cp315-cp315-macosx_14_0_arm64.whl` (Native ARM64 Python 3.15)
- `pyinstaller`: Resolved `pyinstaller-6.22.3-py3-none-macosx_10_13_universal2.whl`
- Sub-dependencies: `flatbuffers`, `sounddevice`, `pillow`, `matplotlib` all resolve to universal2/arm64 wheels.

### 4.3 Package Manager (`uv`) Status
- `uv` is not currently present in PATH (`command not found`).
- Standard Python 3.15 `venv` (`python3 -m venv .venv`) and `pip` are fully functional and will be used for environment isolation.

---

## 5. Packaging & Disk Image Utility Audit

| Tool | Path | Version/Capability | Status |
| :--- | :--- | :--- | :--- |
| `make` | `/usr/bin/make` | GNU Make 3.81 | Ready |
| `hdiutil` | `/usr/bin/hdiutil` | macOS Disk Image Utility | Ready (supports UDZO compressed DMG) |
| `codesign` | `/usr/bin/codesign` | Apple Code Signing Utility | Ready (supports ad-hoc `-` and identity signing) |
| `ditto` | `/usr/bin/ditto` | Apple Archive / Copy Utility | Ready (preserves Mach-O code signatures and symlinks) |
| `git` | `/usr/bin/git` | Apple Git 2.45.1 | Ready |

---

## 6. Standalone Bundling Architecture Feasibility

The non-negotiable project requirements mandate:
1. Deliverable must be a standalone `EyeMouse.app` and `EyeMouse.dmg`.
2. Final application must **not** require Terminal, Antigravity, or a separately installed Python environment on the user's Mac.

**Engineered Bundling Strategy:**
1. The **Vision Engine** (`eyemouse-engine`) will be compiled into a standalone self-contained binary bundle using PyInstaller:
   ```bash
   pyinstaller --clean --noconfirm \
               --name eyemouse-engine \
               --onedir \
               --contents-directory _internal \
               src/vision_engine/main.py
   ```
2. The **Native Host Application** (`EyeMouse.app`) is compiled via `swiftc`.
3. The packaged `eyemouse-engine` folder and its `_internal` shared libraries are placed inside:
   ```text
   EyeMouse.app/Contents/MacOS/
   ├── EyeMouse            (Native Swift AppKit executable)
   ├── eyemouse-engine     (Vision Engine standalone executable)
   └── _internal/          (Bundled Python runtime, OpenCV dylibs, MediaPipe graphs)
   ```
4. When `EyeMouse` launches, it forks and manages `eyemouse-engine` as a child process using local Unix Domain Sockets (`/tmp/eyemouse_<uuid>.sock`).
5. `codesign --deep` signs the entire bundle.
6. `hdiutil create` formats the DMG with an `/Applications` symlink for one-drag installation.

---

## 7. M1 Bootstrap & Validation Status (Executed November 4, 2026)

All environment bootstrap objectives have been executed and validated with real automated tests:

1. **Python Environment Bootstrap:**
   - Isolated project virtual environment initialized at `.venv` using Python 3.15.0 ARM64 (`/Library/Frameworks/Python.framework/Versions/3.15/bin/python3`).
   - Global Python frameworks were completely untouched.
   - Pinned manifests generated: [`requirements.txt`](file:///Users/sairamv/Projects/Eye%20Mouse/requirements.txt) and [`requirements.lock`](file:///Users/sairamv/Projects/Eye%20Mouse/requirements.lock).
2. **Native ARM64 Package Verification:**
   - `mediapipe==1.2.2` (Apple Silicon ARM64 native)
   - `numpy==2.5.4` (ARM64 native)
   - `opencv-contrib-python==5.0.0.93` (ARM64 native ABI3)
   - `pyinstaller==6.22.3` (ARM64 native)
   - `pytest==9.2.0`
3. **Offline Model Asset Provisioned:**
   - `assets/models/face_landmarker.task` downloaded from official Google Cloud Storage and verified against pinned SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
4. **PyInstaller Frozen Packaging Validated:**
   - Identified and resolved the `mediapipe.tasks.c` dynamic library bundling requirement (`--collect-all mediapipe`).
   - Standalone frozen binary `./dist_smoke/smoke_engine/smoke_engine` executed without errors, detecting all 478 landmarks.
5. **Swift Toolchain Validated:**
   - Direct compilation using `swiftc -sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk -target arm64-apple-macos14.0 -O` verified.
   - Swift AppKit test binary successfully enumerated displays and discovered AVFoundation video devices without requesting camera permissions from Terminal.

---

## 8. Audit Conclusions & Next Steps

1. **Prerequisites are satisfied:** The machine possesses full native ARM64 capabilities for both Python computer vision and native Swift compilation.
2. **Identified Toolchain Workaround:** `swiftc` must be invoked with `-sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk` due to the CommandLineTools SDK mismatch on macOS 27.0.1.
3. **No System Alterations Made:** No root changes or permission overrides were performed during this audit.
4. **M1 Complete:** Environment is 100% verified and ready for Milestone 2 (Vision Engine Core & Shared-Memory IPC).

