# Milestone 3: Native macOS Host Application Specification

## 1. Executive Summary

Milestone 3 (M3) delivers the native macOS menu-bar desktop application (`EyeMouse.app`). The application integrates operating system facilities via AppKit, SwiftUI, AVFoundation, and Core Graphics to manage real webcam acquisition, global hold-to-control keyboard shortcut monitoring, safety gating, and inter-process communication with the headless Python vision engine.

EyeMouse operates strictly as a background menu-bar accessory utility (`LSUIElement = true`). Video capture and vision inference are completely dormant while inactive; holding the configured global shortcut physically opens the safety gate and starts live capture and processing. Releasing any required key immediately cuts the vision pipeline.

---

## 2. Application Bundle Structure

`EyeMouse.app` is organized as a standalone, self-contained macOS Application Bundle signed with a stable development identity (`com.eyemouse.EyeMouse`):

```text
build/EyeMouse.app/
└── Contents/
    ├── Info.plist                     # Bundle metadata, LSUIElement=true, NSCameraUsageDescription
    ├── MacOS/
    │   └── EyeMouse                   # Native Swift AppKit executable (Thin ARM64 Mach-O)
    └── Resources/
        ├── AppIcon.icns               # Multi-resolution application icon
        └── engine/                    # Standalone frozen vision engine (PyInstaller Mach-O)
            ├── eyemouse_engine        # Frozen Python binary
            └── _internal/             # Bundled Python 3.15 dylibs, MediaPipe graphs, offline models
```

### 2.1 Bundle Properties (`Info.plist`)
- **`CFBundleIdentifier`**: `com.eyemouse.EyeMouse` (Stable across all development builds)
- **`CFBundleName` / `CFBundleExecutable`**: `EyeMouse`
- **`CFBundleShortVersionString`**: `0.3.0` (Milestone 3)
- **`LSUIElement`**: `true` (Runs strictly as an NSStatusItem accessory without Dock clutter or application window on launch)
- **`NSCameraUsageDescription`**: Explicit, localized purpose string explaining local in-memory eye tracking with zero storage or cloud upload.

---

## 3. Native Architecture and Subsystem Coordination

```mermaid
flowchart TD
    subgraph Host ["EyeMouse.app (Swift Native Host)"]
        AppDelegate["AppDelegate<br>(App Lifecycle & Coordination)"]
        MenuBar["MenuBarController<br>(NSStatusItem UI)"]
        EventTap["GlobalEventTap<br>(CGEventTap Global Key Monitor)"]
        SafetyGate["SafetyStateManager<br>(Atomic Gate State Machine)"]
        CamService["CameraCaptureService<br>(AVCaptureSession 640x480 BGRA)"]
        CamMgr["CameraManager<br>(Device Discovery & Selection)"]
        DiagStore["DiagnosticsStore<br>(Observable Live Telemetry)"]
        DiagUI["DiagnosticsWindowController<br>(Live SwiftUI Telemetry)"]
        SettingsUI["SettingsWindowController<br>(Shortcut & Camera Config)"]
        ProcMgr["EngineProcessManager<br>(Subprocess Supervisor)"]
        ShmProd["ShmProducer<br>(POSIX SHM 4-Slot Buffer)"]
        CtrlClient["ControlClient<br>(Unix Domain Socket Client)"]
    end

    subgraph VisionEngine ["Subprocess: eyemouse_engine (Frozen Python)"]
        IPCSrv["ControlServer (UDS)"]
        SHMCons["ShmConsumer (Zero-Copy)"]
        Pipeline["VisionPipeline (MediaPipe VIDEO Mode)"]
    end

    EventTap -->|Key Down Chord| SafetyGate
    EventTap -->|Key Up (Any)| SafetyGate
    SafetyGate -->|Open Gate| CamService
    SafetyGate -->|Start / Ensure Running| ProcMgr
    CamService -->|Raw CVPixelBuffer| ShmProd
    ShmProd -->|Write Frame Slot| SHMCons
    CamService -->|FRAME Notification| CtrlClient
    CtrlClient -->|Socket Msg| IPCSrv
    IPCSrv --> Pipeline
    Pipeline -->|ACK Telemetry| CtrlClient
    CtrlClient --> DiagStore
    DiagStore --> DiagUI
    MenuBar --> SettingsUI
    MenuBar --> DiagUI
```

---

## 4. Menu Bar User Interface

Managed by `MenuBarController.swift` via `NSStatusBar.system.statusItem`:

### 4.1 Status Indicator States
| State | Menu Bar Title / Icon | Description |
|---|---|---|
| **`INACTIVE`** | `👁️ EyeMouse` | Idle. Hardware camera and vision engine frame processing dormant. |
| **`STARTING`** | `👁️‍🗨️ Starting...` | Shortcut held; camera session and vision engine initializing. |
| **`ACTIVE`** | `👁️ [Active]` | Live tracking active. Real frames being processed by vision engine. |
| **`STOPPING`** | `👁️ Stopping...` | Shortcut released; frame gate closed synchronously, camera stopping. |
| **`ERROR`** | `⚠️ Error` | Camera disconnected, permission denied, or engine process crash. |

### 4.2 Menu Items Hierarchy
1. `EyeMouse: <Status>` (Disabled header reflecting live safety state)
2. `---` (Separator)
3. `Selected Camera: <Name>` (Submenu enumerating Built-in, Virtual, and Continuity cameras with radio selection)
4. `Activation Shortcut: <Chord> (Hold)` (Displays configured trigger chord, e.g. `⌃⌥E`)
5. `---` (Separator)
6. `Tracking Diagnostics...` (Opens real-time tracking diagnostics window)
7. `Settings...` (Opens configuration window for shortcut and camera preferences)
8. `---` (Separator)
9. `Quit EyeMouse` (`⌘Q`)

---

## 5. Live Diagnostics and Telemetry Window

Accessible from the menu bar via `Tracking Diagnostics...`, `DiagnosticsWindowController` hosts a SwiftUI telemetry monitor:

- **Caution Banner**: `⚠️ UNCALIBRATED OPTICAL PROXY — MILESTONE 3` (Explicit notice that OS mouse movement and clicks are not active).
- **Camera Acquisition**: Live status indicator (pulsing green circle when camera captures, grey when idle), selected device name, and capture FPS.
- **Face Presence**: High-visibility status badge (`[FACE DETECTED]` green vs `[NO FACE DETECTED]` red) and confidence percentage.
- **Normalized Iris Features**: Real-time $(x, y)$ coordinates for left and right irises in the local eye coordinate frame, plus filtered optical gaze proxy.
- **Blink Dynamics**: Eyelid phase (`OPEN`, `CLOSING`, `CLOSED`, `REOPENING`), closure duration in milliseconds, and sustained blink detection indicator.
- **3D Head Pose**: Real-time yaw, pitch, and roll in degrees from OpenCV SolvePnP.
- **System Telemetry**: Live vision engine processing rate (FPS), mean and p95 inference latency, published vs dropped frame counters, and subprocess PID/status.

