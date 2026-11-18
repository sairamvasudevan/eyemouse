# Milestone 3: macOS Security and Permission Validation

## 1. Permission Architecture Overview

EyeMouse interacts with three distinct macOS TCC privacy domains:

| Domain | Subsystem | Purpose in EyeMouse | Failure Behavior |
|---|---|---|---|
| **Camera** | `kTCCServiceCamera` | Capture webcam video frames for local gaze analysis | App disables video pipeline; status shows `Camera permission denied` |
| **Accessibility** | `kTCCServiceAccessibility` | Active modifying event tap to suppress trigger keystrokes | Fallback to listen-only tap; shortcut triggers work but keystrokes reach active app |
| **Input Monitoring** | `kTCCServiceListenEvent` | Passive global event tap to monitor shortcut key chords | App cannot detect shortcut holds; status shows `Permission required` |

---

## 2. Camera Authorization (`PermissionsManager.swift`)

### 2.1 Bundle Requirement
The application bundle's `Contents/Info.plist` contains the mandatory description:
```xml
<key>NSCameraUsageDescription</key>
<string>EyeMouse uses your camera locally to track eye gaze and head orientation. Video frames are processed in real-time in memory and are never recorded, saved, or uploaded.</string>
```

### 2.2 Authorization Verification
- **API**: `AVCaptureDevice.authorizationStatus(for: .video)`
- **States Tested**:
  - `.notDetermined (0)`: Prompts user via `AVCaptureDevice.requestAccess(for: .video)`.
  - `.authorized (3)`: Verified on test system (`Camera Auth: 3`). Camera session starts immediately on shortcut hold.
  - `.denied (2)` / `.restricted (1)`: Camera session does not attempt initialization. Safety state transitions to `ERROR`, and the user is directed to System Settings.

### 2.3 System Settings Direct URL Navigation
When permissions are denied, the UI provides one-click buttons linking directly to macOS System Settings:
- **Camera**: `x-apple.systempreferences:com.apple.preference.security?Privacy_Camera`
- **Accessibility**: `x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility`
- **Input Monitoring**: `x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent`

---

## 3. Application vs Terminal Permission Isolation

> [!IMPORTANT]
> **Prompting from EyeMouse.app vs Terminal:**  
> If an application is launched directly from Terminal via shell commands, macOS may attribute permission requests to `Terminal.app` or `zsh` instead of the application bundle.
>
> EyeMouse solves this by:
> 1. Compiling and sealing `EyeMouse.app` with an embedded `Info.plist` and stable identifier `com.eyemouse.EyeMouse`.
> 2. Ad-hoc codesigning the sealed bundle (`codesign -s - --force --identifier "com.eyemouse.EyeMouse"`).
> 3. Running as a native app bundle using `open build/EyeMouse.app` or direct bundle executable execution, guaranteeing that macOS TCC prompts attribute the request to `EyeMouse.app`.

---

## 4. Permission Test Matrix and Results

| Test Scenario | Procedure | Expected Result | Actual Result |
|---|---|---|---|
| **Initial Camera Query** | Query `AVCaptureDevice.authorizationStatus(for: .video)` | Returns valid status enum | **PASSED** (Returned `3 = authorized`) |
| **Camera Disconnect** | Unplug or disable active camera device | Triggers `wasDisconnectedNotification`; forces `INACTIVE` state | **PASSED** (Verified in unit test & manager) |
| **Active Tap with Accessibility** | `CGEvent.tapCreate` with `.defaultTap` | Creates active tap and suppresses trigger keystrokes | **PASSED** |
| **Passive Tap Fallback** | Fallback to `.listenOnly` if Accessibility denied | Intercepts key events without crashing | **PASSED** |
| **System Sleep During Capture** | Post `willSleepNotification` | Closes safety gate immediately; disables camera | **PASSED** |
| **Screen Lock During Capture** | Post `screensDidSleepNotification` | Forces `INACTIVE` state; prevents background camera capture | **PASSED** |

