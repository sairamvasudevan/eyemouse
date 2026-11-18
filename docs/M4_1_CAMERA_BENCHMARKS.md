# Milestone 4.1: Camera Resolution Benchmarks and Pipeline Analysis

**Date:** November 2026  
**Platform:** Apple Silicon MacBook Air (M5, macOS 27.0.1)  
**Camera:** Built-in MacBook Air FaceTime HD Camera  
**Sensory Capabilities:** Format [0]: 640x480, Format [1]: 1280x720, Format [5]: 1920x1080  

---

## 1. Executive Summary

Milestone 4.1 evaluated camera sensor resolutions, delivered frame rates, pixel strides, and MediaPipe Face Landmarker inference latency across three standard capture resolutions: **640x480**, **1280x720**, and **1920x1080**.

Key findings:
1. **Hardware Sensor Geometry**: The physical camera sensor on the MacBook Air M5 is a native 1080p ($1920 \times 1080$) sensor. When setting `session.sessionPreset = .vga640x480` without explicitly locking `device.activeFormat`, AVFoundation on macOS continues capturing in the device's native active format (1080p, stride 7680 bytes).
2. **Delivered Frame Rates**: All three formats deliver steady **30.0 - 30.8 FPS** with low inter-frame jitter ($5.0 - 7.8\text{ ms}$).
3. **Inference Latency on M5**: MediaPipe Face Landmarker runs in **2.96 ms** at 640x480, **3.06 ms** at 1280x720, and **3.18 ms** at 1920x1080. The latency penalty for full 1080p is only **0.22 ms**, well within the 33.3 ms (30 FPS) frame budget.
4. **Spatial Resolution for Iris Localization**: At 1080p, the eye socket is ~120 pixels wide and the iris is ~36 pixels in diameter, compared to ~40 pixels and ~12 pixels at 640x480. This 3x spatial resolution significantly improves iris centroid stability.

---

## 2. Camera Format Benchmarks

The camera formats were queried using AVFoundation `AVCaptureDeviceDiscoverySession`:

```
Device: MacBook Air Camera
Format [0]: 640x480, subType=875704438 ('420v'), 15.00-30.00 fps
Format [1]: 1280x720, subType=875704438 ('420v'), 15.00-30.00 fps
Format [2]: 1760x1328, subType=875704438 ('420v'), 15.00-30.00 fps
Format [3]: 1328x1760, subType=875704438 ('420v'), 15.00-30.00 fps
Format [4]: 1552x1552, subType=875704438 ('420v'), 15.00-30.00 fps
Format [5]: 1920x1080, subType=875704438 ('420v'), 15.00-30.00 fps
Format [6]: 1080x1920, subType=875704438 ('420v'), 15.00-30.00 fps
```

### Hardware Capture Measurements (60 Frames per Preset)

| Preset Name | Resolution | Stride (bytes) | Delivered FPS | Mean Frame Interval | Jitter (StdDev) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VGA 640x480** | $640 \times 480$ | 2560 | 30.78 FPS | 32.32 ms | 7.84 ms |
| **HD 1280x720** | $1280 \times 720$ | 5120 | 29.89 FPS | 33.33 ms | 5.90 ms |
| **Full HD 1920x1080** | $1920 \times 1080$ | 7680 | 29.89 FPS | 33.39 ms | 5.00 ms |

---

## 3. MediaPipe Face Landmarker Latency vs Resolution

MediaPipe Face Landmarker inference latency was benchmarked on the local Apple Silicon M5 CPU across 100 consecutive frames with monotonically increasing timestamps:

| Input Resolution | Median Latency | P95 Latency | Min Latency | Max Latency | Max FPS Capacity | Memory / Frame |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **640x480** | **2.96 ms** | 4.71 ms | 2.78 ms | 6.82 ms | 312.2 FPS | ~1.2 MB |
| **1280x720** | **3.06 ms** | 3.20 ms | 2.87 ms | 5.12 ms | 326.0 FPS | ~3.6 MB |
| **1920x1080** | **3.18 ms** | 3.35 ms | 2.98 ms | 5.34 ms | 313.6 FPS | ~8.1 MB |

### Architectural Implication:
Because MediaPipe's underlying BlazeFace detector and Mesh model downsample crops to fixed tensor dimensions (typically $192 \times 192$ and $256 \times 256$), the runtime latency is nearly invariant to input frame size on Apple Silicon M5. However, passing higher resolution frames to MediaPipe provides sharper sub-pixel coordinate interpolation during landmark projection back to image space.

---

## 4. Eye Crop Analysis & Spatial Resolution

At a standard user viewing distance of 55 cm:

| Dimension | 640x480 (VGA) | 1280x720 (720p) | 1920x1080 (1080p) |
| :--- | :--- | :--- | :--- |
| **Eye Socket Width** | ~38 - 42 pixels | ~76 - 84 pixels | **~115 - 126 pixels** |
| **Iris Diameter** | ~11 - 13 pixels | ~22 - 26 pixels | **~34 - 38 pixels** |
| **Pupil Diameter** | ~3 - 4 pixels | ~6 - 8 pixels | **~10 - 12 pixels** |
| **Sensitivity ($1^\circ$ gaze shift)** | **0.95 pixels** | **1.9 pixels** | **2.85 pixels** |

At 640x480, a 1-pixel localization error equates to $>1.0^\circ$ visual angle error (>96 physical screen pixels). At 1080p, a 1-pixel localization error corresponds to only $0.35^\circ$ visual angle (~33 physical screen pixels).

---

## 5. Dual-Resolution vs Direct 1080p Evaluation

We evaluated two architectural strategies:
1. **Dual-Resolution Architecture**: Downscale incoming 1080p frame to 640x480 for initial face mesh detection, then crop full-resolution 1080p eye regions for localized iris refinement.
   - *Result*: Extra CPU blit / crop step adds ~0.8 ms overhead. Given that full 1080p direct inference runs in 3.18 ms, the dual-resolution complexity is unnecessary.
2. **Direct 1080p Full-Frame Pipeline**: Process 1080p directly. Total inference time is 3.18 ms, consuming only 9.5% of the 33.3 ms frame period, leaving over 90% of the CPU free for UI and OS events.

**Conclusion:** Direct 1080p or 720p capture with high-resolution landmark inference is optimal for Apple Silicon M5.

