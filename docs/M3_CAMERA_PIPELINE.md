# Milestone 3: Real Camera Pipeline and Shared-Memory Integration

## 1. AVFoundation Architecture and Ownership

In EyeMouse, the native macOS host application (`EyeMouse.app`) is the **sole and exclusive owner** of hardware camera capture. The Python vision engine operates completely headlessly and never opens a video capture device or invokes `cv2.VideoCapture`.

---

## 2. Device Discovery and Classification (`CameraManager.swift`)

Using `AVCaptureDevice.DiscoverySession`, EyeMouse discovers all available camera sources on the system and categorizes them:

```swift
var deviceTypes: [AVCaptureDevice.DeviceType] = [
    .builtInWideAngleCamera,
    .external,
    .continuityCamera
]
```

### 2.1 Device Categorization
1. **Built-in Webcam**: Devices matching `.builtInWideAngleCamera` or localized names containing `"MacBook"` / `"FaceTime HD"`. Default selection.
2. **External USB/Thunderbolt**: Physical plug-and-play webcams (`.external`).
3. **Continuity Camera**: Wireless iPhone / iPad video sources (`.continuityCamera`).
4. **Virtual Camera**: Software video feeds (e.g. `"Sharing Camera(Virtual)"`, OBS Virtual Camera, Camo).

### 2.2 Persistence and Hot-Plugging
- Selected camera unique ID is persisted in `UserDefaults` under `com.eyemouse.selectedCameraUniqueID`.
- Observers for `AVCaptureDevice.wasConnectedNotification` and `wasDisconnectedNotification` dynamically update the menu-bar device list.
- If the actively streaming camera is unplugged or disconnected, `CameraManager` immediately signals `SafetyStateManager.shared.forceInactive(reason: "Camera disconnected")`.

---

## 3. Capture Session and Buffer Configuration (`CameraCaptureService.swift`)

```swift
session.beginConfiguration()
defer { session.commitConfiguration() }

// Attach device input
let input = try AVCaptureDeviceInput(device: selectedDevice)
if session.canAddInput(input) { session.addInput(input) }

// Attach video data output
videoOutput.alwaysDiscardsLateVideoFrames = true
videoOutput.videoSettings = [
    kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
]
videoOutput.setSampleBufferDelegate(self, queue: frameQueue)
if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

// Set optimal preset after inputs and outputs are attached
if session.canSetSessionPreset(.vga640x480) {
    session.sessionPreset = .vga640x480
} else if session.canSetSessionPreset(.hd1280x720) {
    session.sessionPreset = .hd1280x720
}
```

### 3.1 Timing and Queue Isolation
- **Session Control Queue (`com.eyemouse.camera.session`)**: Serial queue for starting and stopping the `AVCaptureSession` asynchronously, preventing main thread stalls.
- **Frame Delivery Queue (`com.eyemouse.camera.output`)**: Dedicated high-priority serial queue receiving raw `CMSampleBuffer` frames from WindowServer.
- **Frame Discard Policy**: `alwaysDiscardsLateVideoFrames = true` ensures that if inference experiences temporary jitter, the video subsystem drops late frames immediately rather than creating an unbounded capture backlog.

---

## 4. Shared-Memory Frame Publishing (`ShmProducer.swift`)

Raw frames extracted from `CMSampleBufferGetImageBuffer(sampleBuffer)` as `CVPixelBuffer` are written directly into mapped POSIX shared-memory slots:

```
+--------------------------------------------------------------------------+
| CVPixelBuffer (Locked Read-Only Base Address)                            |
| Width: 640 | Height: 480 | BytesPerRow: Actual row stride from OS        |
+--------------------------------------------------------------------------+
                                    |
                            memcpy(slotPayload)
                                    v
+--------------------------------------------------------------------------+
| POSIX Shared Memory Slot N:                                              |
| [Slot Header (64 bytes)] | [Raw Pixel Payload (BytesPerRow * Height)]    |
+--------------------------------------------------------------------------+
```

### 4.1 Row Stride Integrity (`bytesPerRow`)
> [!IMPORTANT]
> **Never Assume `bytesPerRow == width * 4`:**  
> On Apple Silicon Macs, the hardware camera pipeline frequently aligns row strides to 16-byte, 32-byte, or 64-byte boundaries for GPU/NEON efficiency. EyeMouse queries `CVPixelBufferGetBytesPerRow(pixelBuffer)` directly and writes the actual stride into byte offset 40 of the slot header. The Python `VisionPipeline` unpacks rows using this exact stride:
> ```python
> bgra = data[:stride * height].reshape((height, stride))[:, :width*4].reshape((height, width, 4))
> ```

### 4.2 Timestamps and Coordinate Conventions
- **Timestamps**: Monotonic nanosecond timestamp acquired using Apple's `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`. Monotonically increments even across NTP clock adjustments.
- **Orientation**: Standard camera frame layout with top-left origin $(0, 0)$. Subject's left eye appears on the image right. MediaPipe landmark indices strictly respect this optical orientation (`33` = subject right eye / image left, `263` = subject left eye / image right).

