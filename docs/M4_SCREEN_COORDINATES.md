# EyeMouse Screen Coordinate System & Transformations

## 1. Overview and Mathematical Architecture

Accurate eye-gaze tracking requires unambiguous coordinate conversions across six distinct reference frames:
1. **Camera Sensor Space**: Physical pixel array from `AVCaptureSession` ($640 \times 480$, top-left origin).
2. **MediaPipe Normalized Space**: Optical UV plane $[0.0, 1.0] \times [0.0, 1.0]$ with $(0, 0)$ at top-left.
3. **Normalized Eye Frame**: Scale-invariant and roll-invariant iris coordinates $(x_{\text{norm}}, y_{\text{norm}})$ centered at the inter-canthal midpoint, oriented along the eye corner axis.
4. **Normalized Screen Space**: Dimensionless display coordinates $(u, v) \in [0.0, 1.0] \times [0.0, 1.0]$ with $(0, 0)$ at top-left, $(1, 1)$ at bottom-right.
5. **macOS Logical Display Space (CoreGraphics Points)**: System desktop coordinates $(X_{\text{pt}}, Y_{\text{pt}}) \in [0, W_{\text{pt}}] \times [0, H_{\text{pt}}]$ with $(0, 0)$ at top-left.
6. **Cocoa Application Space (`NSScreen` / `NSWindow`)**: Application window coordinates with $(0, 0)$ at the **bottom-left** of the primary display.
7. **Physical Retina Pixel Space**: Native display hardware pixels $(X_{\text{px}}, Y_{\text{px}}) = (X_{\text{pt}} \cdot S, Y_{\text{pt}} \cdot S)$ where $S = \text{backingScaleFactor}$.

---

## 2. Dynamic Display Geometry Discovery

EyeMouse **never** assumes hard-coded display dimensions (e.g. $1470 \times 956$). Display properties are queried dynamically from AppKit:

```swift
let screen = NSScreen.main ?? NSScreen.screens[0]
let widthPoints = screen.frame.width          // 1470.0 pt
let heightPoints = screen.frame.height        // 956.0 pt
let backingScale = screen.backingScaleFactor  // 2.0x (Retina)
let widthPixels = widthPoints * backingScale  // 2940.0 px
let heightPixels = heightPoints * backingScale// 1912.0 px
```

### 2.1 Hardware Metrics on Target Machine (MacBook Air Apple Silicon)

| Property | Logical Value | Physical Retina Value | Units |
| :--- | :--- | :--- | :--- |
| **Display Width** | `1470.0` | `2940.0` | Points / Pixels |
| **Display Height** | `956.0` | `1912.0` | Points / Pixels |
| **Visible Height** (Excluding Dock / Menu Bar) | `861.0` | `1722.0` | Points / Pixels |
| **Display Diagonal** | `1753.5` | `3507.0` | Points / Pixels |
| **Backing Scale Factor** | `2.0` | — | Multiplier |
| **Physical Display Diagonal** | `13.6` | — | Inches |
| **Pixel Density (PPI)** | `257.8` | — | Pixels / Inch |
| **Viewing Distance (Nominal)** | `55.0` | — | Centimeters |

---

## 3. Reference Frame Transformation Pipeline

```
[Camera Sensor: 640x480 BGRA]
       |
       v (MediaPipe FaceLandmarker)
[Normalized Landmarks: (u, v) in [0..1]]
       |
       v (Inter-Canthal Normalization & SolvePnP)
[Eye Feature Vector: (lx, ly, rx, ry, yaw, pitch, roll) in R^7]
       |
       v (Calibrated Regression Model: Affine or Polynomial)
[Normalized Screen Space: (u, v) in [0..1] x [0..1], Origin Top-Left]
       |
       +---------------------------------------------+
       |                                             |
       v (Scale by widthPoints, heightPoints)        v (Flip Y-axis for Cocoa)
[CoreGraphics Points (X_pt, Y_pt)]          [Cocoa Points (X_cocoa, Y_cocoa)]
       |
       v (Multiply by backingScaleFactor = 2.0)
[Physical Retina Pixels (X_px, Y_px)]
```

---

## 4. Mathematical Conversion Equations

### 4.1 Normalized Screen to CoreGraphics Points
$$\begin{pmatrix} X_{\text{pt}} \\ Y_{\text{pt}} \end{pmatrix} = \begin{pmatrix} \text{clamp}(u, 0.0, 1.0) \cdot W_{\text{pt}} \\ \text{clamp}(v, 0.0, 1.0) \cdot H_{\text{pt}} \end{pmatrix}$$

### 4.2 CoreGraphics to Cocoa (Bottom-Left Origin)
$$\begin{pmatrix} X_{\text{cocoa}} \\ Y_{\text{cocoa}} \end{pmatrix} = \begin{pmatrix} X_{\text{pt}} \\ H_{\text{pt}} - Y_{\text{pt}} \end{pmatrix}$$

### 4.3 Logical Points to Physical Retina Pixels
$$\begin{pmatrix} X_{\text{px}} \\ Y_{\text{px}} \end{pmatrix} = \begin{pmatrix} X_{\text{pt}} \cdot S \\ Y_{\text{pt}} \cdot S \end{pmatrix}$$

### 4.4 Pixel Error to Angular Visual Degrees
For empirical error evaluation, Euclidean pixel errors are converted to visual degrees ($\theta$) based on viewing distance $D_{\text{cm}}$:
$$\Delta d_{\text{cm}} = \frac{\Delta d_{\text{px}}}{\text{PPI} / 2.54}$$
$$\theta = 2 \cdot \arctan\left(\frac{\Delta d_{\text{cm}}}{2 \cdot D_{\text{cm}}}\right) \cdot \frac{180^\circ}{\pi}$$
At nominal $55\text{ cm}$ viewing distance on the $257.8\text{ PPI}$ Liquid Retina display, **100 physical pixels** corresponds to **$1.02^\circ$ visual angle**.

---

## 5. Optical Mirroring and Orientation

1. **Horizontal Inversion (Camera Mirroring)**:  
   When the user glances toward the right side of their screen, their physical gaze shifts rightward (+X on screen). In the raw webcam image, looking right translates to the camera's image-left. The calibration regression matrices automatically absorb this inversion during model fitting:
   $$W_{x, \text{iris\_x}} < 0$$
   This guarantees that gazing toward the right side of the screen consistently increases screen X.
2. **Display Boundary Clipping**:  
   Gaze predictions are hard-clamped to $[0.0, 1.0]$ in normalized space prior to display point transformation. Out-of-bounds extrapolations are safely constrained to screen edges.

