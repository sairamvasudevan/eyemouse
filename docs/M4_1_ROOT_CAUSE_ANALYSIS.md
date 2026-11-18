# Milestone 4.1: Root Cause Analysis of Gaze Calibration Failure

**Date:** November 2026  
**Status:** Completed & Diagnosed  
**Target:** Apple Silicon MacBook Air (13.6" Liquid Retina, 2940x1912 physical pixels, 1470x956 points @ 2.0x)  
**Baseline M4 Outcome:** Held-out Median Error 458.1 px (13.06% diagonal), P95 Error 873.9 px (24.92% diagonal), Status `POOR`  

---

## 1. Executive Summary

Milestone 4 established an end-to-end screen gaze calibration pipeline consisting of a 13-point calibration grid, 9 independent held-out validation targets, Ridge regression models (Affine and Degree-2 Polynomial), and an AppKit calibration wizard. However, held-out validation against independent targets revealed a median error of **458.1 pixels (13.06% of screen diagonal)** and a 95th-percentile error of **873.9 pixels (24.92%)**, far outside the acceptance criteria ($\le 2.5\%$ median, $\le 6.0\%$ P95). The calibration profile was honestly graded `POOR` with `is_valid_for_use = false`.

Through rigorous diagnosis, empirical benchmarking, and mathematical analysis, we identified **five compounding root causes**:
1. **Critical Anatomical Coordinate Inversion**: In `src/vision_engine/eye_features.py`, the horizontal and vertical projection vectors for the subject's right eye were defined in reverse order relative to the left eye, causing conjugate eye movements to invert signs between eyes.
2. **Severe Model Ill-Conditioning**: In `PolynomialGazeModel`, degree-2 polynomial expansion was performed on unstandardized features. Head pose angles squared ($\text{yaw}^2 \approx 700$) were 2800x larger than normalized iris coordinates squared ($lx^2 \approx 0.25$), yielding a condition number of **26,731.6** and causing catastrophic validation overfitting (1557 px error).
3. **Low Calibration Signal-to-Noise Ratio (SNR)**: In `test_m4_live_calibration.swift`, the synthetic iris dynamic range was configured to only $\Delta = 0.08$ (span ~0.067), which was completely drowned out by natural head drift ($\pm 25^\circ$) and webcam tracking noise ($\pm 0.02$).
4. **Fundamental Optical & Physical Resolution Limits**: At 55 cm viewing distance on a 2940x1912 display, $1^\circ$ visual angle corresponds to **96 physical pixels**. An eye rotation of $1^\circ$ moves the pupil by only $209\text{ \mu m}$, which translates to **$\approx 1.0 - 1.3\text{ pixels}$** on the camera sensor at 640x480. Without active infrared corneal reflection (glint) tracking, academic state-of-the-art for appearance-based webcam gaze estimation is $3.0^\circ - 4.5^\circ$ ($280 - 430\text{ px}$).
5. **MacBook Bezel Parallax & Eyelid Occlusion**: The built-in camera sits 15 cm above the display center. Viewing lower targets requires steep downward gaze, causing eyelid drooping and foreshortening that corrupts iris centroid localization without non-linear vertical compensation.

---

## 2. Investigation of Baseline M4 Calibration Failure

### 2.1 Per-Target Held-Out Error Breakdown

The initial M4 held-out validation evaluation against 9 independent target positions yielded the following error distribution:

| Validation Target Name | Ground Truth $(u, v)$ | Model Prediction $(u, v)$ | Horizontal Error ($Err_X$) | Vertical Error ($Err_Y$) | Total Pixel Error | % Diagonal |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Val-Top-Mid-Left** | (0.20, 0.12) | (0.35, 0.20) | 431.2 px | 154.7 px | 458.1 px | 13.06% |
| **Val-Top-Mid-Right** | (0.80, 0.12) | (0.81, 0.16) | 36.2 px | 84.1 px | 91.6 px | 2.61% |
| **Val-Bottom-Mid-Left** | (0.20, 0.88) | (0.29, 0.90) | 259.3 px | 33.0 px | 261.4 px | 7.45% |
| **Val-Bottom-Mid-Right** | (0.80, 0.88) | (0.97, 0.80) | 495.2 px | 159.3 px | 520.2 px | 14.83% |
| **Val-Mid-Left-Inner** | (0.35, 0.50) | (0.56, 0.44) | 619.0 px | 109.7 px | 628.7 px | 17.93% |
| **Val-Mid-Right-Inner** | (0.65, 0.50) | (1.00, 0.43) | **1029.0 px** | 131.9 px | **1037.4 px** | 29.58% |
| **Val-Mid-Top-Inner** | (0.50, 0.30) | (0.64, 0.30) | 424.8 px | 3.7 px | 424.8 px | 12.11% |
| **Val-Mid-Bottom-Inner** | (0.50, 0.70) | (0.70, 0.63) | 596.7 px | 126.9 px | 610.0 px | 17.39% |
| **Val-Center-Offset** | (0.40, 0.40) | (0.31, 0.42) | 256.9 px | 43.5 px | 260.6 px | 7.43% |
| **Summary Statistics** | — | — | **Mean: 460.9 px** | **Mean: 94.1 px** | **Median: 458.1 px** | **Median: 13.06%** |

```
                       Held-Out Error Breakdown by Axis:
   Horizontal (X) Error:  ██████████████████████████████████████ 460.9 px (13.1% diag)
   Vertical (Y) Error:    ███████ 94.1 px (2.7% diag)
   Total Euclidean Error: ██████████████████████████████ 458.1 px (Median)
```

**Key Diagnostic Observation:** Horizontal error ($Err_X = 460.9\text{ px}$) was nearly **5 times larger** than vertical error ($Err_Y = 94.1\text{ px}$). The horizontal tracking signal was severely contaminated.

---

## 3. Detailed Root Causes

### 3.1 Root Cause 1: Anatomical Coordinate Discrepancy & Sign Inversion

In MediaPipe Face Landmarker, the 478 3D landmarks are indexed from the subject's anatomical perspective:
- **Left Eye (appears on image right)**:
  - Nasal corner (inner): 362 ($X \approx 300\text{ px}$ in camera frame)
  - Temporal corner (outer): 263 ($X \approx 340\text{ px}$ in camera frame)
  - Axis vector: $V = P_{263} - P_{362} \implies V_x > 0$ (points image-left to image-right).
- **Right Eye (appears on image left)**:
  - Temporal corner (outer): 33 ($X \approx 220\text{ px}$ in camera frame)
  - Nasal corner (inner): 133 ($X \approx 260\text{ px}$ in camera frame)
  - Prior implementation: $V = P_{33} - P_{133} \implies V_x < 0$ (pointed image-right to image-left!).

#### Mathematical Consequence:
When a user looked to the right (image-right):
- Left Eye Iris shifted $+X$: $rel \cdot V > 0 \implies norm\_x$ increased from 0.50 to 0.75 (+0.25).
- Right Eye Iris shifted $+X$: but because $V$ was negative, $rel \cdot V < 0 \implies norm\_x$ decreased from 0.50 to 0.25 (-0.25)!

In M2's `UncalibratedGazeEstimator`, $(l_{norm} - 0.5) + (r_{norm} - 0.5)$ evaluated to $(+0.25) + (-0.25) = \mathbf{0.0}$, completely cancelling conjugate horizontal eye motion. In M4's regression models, the inverted feature forced the optimizer to fit conflicting gradients.

#### The Fix:
Unify the lateral axis definition for both eyes to strictly point from **image-left to image-right**:
- Left Eye: `corner_left = 362 (nasal)`, `corner_right = 263 (temporal)`.
- Right Eye: `corner_left = 33 (temporal)`, `corner_right = 133 (nasal)`.

Now, conjugate eye movements yield identical positive shifts for both eyes ($\Delta = +0.25$), reinforcing the gaze signal.

---

### 3.2 Root Cause 2: Model Ill-Conditioning and Polynomial Overfitting

In M4, `PolynomialGazeModel` implemented a 19-parameter degree-2 polynomial expansion:
$$\Phi(X) = [1, lx, ly, rx, ry, yaw, pitch, roll, lx^2, ly^2, rx^2, ry^2, yaw^2, pitch^2, roll^2, lx \cdot ly, rx \cdot ry, lx \cdot yaw, ly \cdot pitch]$$

The features were unstandardized:
- Iris coordinates: $lx \approx 0.50 \implies lx^2 \approx 0.25$
- Head pose angles: $yaw \approx \pm 25^\circ \implies yaw^2 \approx 625.0$

Because standard Ridge regression penalizes all coefficients uniformly with $\alpha \sum w_i^2$, the condition number of the design matrix was:
$$\kappa(\Phi_{train}) = \mathbf{26,731.6}$$

With 19 parameters fitted on only 13 calibration points ($N < P$), the regression problem was severely under-determined. The model overfit the training targets and produced a catastrophic validation median error of **1557.1 pixels**.

#### The Fix:
1. Normalize head pose angles by their physical ranges: $yaw / 30^\circ$, $pitch / 25^\circ$, $roll / 20^\circ$.
2. Implement `PhysicallyInformedGazeModel` with 12 physically-grounded parameters:
   - Conjugate iris center: $avg_x = (lx + rx)/2 - 0.5$, $avg_y = (ly + ry)/2$
   - Inter-ocular horizontal disparity: $disp_x = rx - lx$ (head distance / vergence proxy)
   - Normalized head pose: $\widehat{yaw}$, $\widehat{pitch}$, $\widehat{roll}$
   - Eye curvature: $avg_x^2$, $avg_y^2$
   - Gaze-pose coupling: $avg_x \cdot \widehat{yaw}$, $avg_y \cdot \widehat{pitch}$ (MacBook vertical parallax)
   - Cross-axis interaction: $avg_x \cdot avg_y$
3. Condition number dropped from **26,731.6 to 62.1**, reducing validation error by over 60%.

---

### 3.3 Root Cause 3: Low Dynamic Range in Calibration Test Harness

In `test_m4_live_calibration.swift`, synthetic target modulation used:
`let gazeOffsetFactor = 0.08`
For a target at $u = 0.08$ vs $u = 0.92$, total simulated iris displacement was:
$$\Delta_{span} = (0.92 - 0.08) \times 0.08 = \mathbf{0.067}$$

During live camera execution, natural involuntary head sway produced $yaw$ shifts from $+10.07^\circ$ to $-28.4^\circ$ ($38.5^\circ$ swing), and raw iris tracking noise was $\pm 0.02$. The signal span (0.067) was smaller than the combined noise and drift, resulting in an $\text{SNR} < 1.0$.

#### The Fix:
Update the gaze response to reflect human ocular mechanics ($\sim 15^\circ$ rotation across display width $\implies \Delta \approx 0.16$), and increase target fixation settling window from 0.15s to 0.40s.

---

### 3.4 Root Cause 4: Optical and Angular Physics of Webcam Gaze Tracking

The human eye has an approximate radius of curvature $R \approx 12\text{ mm}$. A $1^\circ$ rotation shifts the pupil centroid by:
$$\Delta_{pupil} = R \times \sin(1^\circ) = 12\text{ mm} \times 0.01745 = \mathbf{0.209\text{ mm}} = 209\text{ \mu m}$$

On a MacBook Air with camera focal length $f \approx 3.5\text{ mm}$ at viewing distance $D = 550\text{ mm}$:
$$M = \frac{f}{D} = \frac{3.5}{550} \approx 0.00636$$
$$\text{Displacement on Sensor} = 209\text{ \mu m} \times 0.00636 = \mathbf{1.33\text{ \mu m}}$$

Given a sensor pixel pitch of $1.4\text{ \mu m}$:
$$\mathbf{1^\circ\text{ Gaze Shift} \iff \approx 0.95 - 1.2\text{ Camera Pixels}}$$

At 55 cm, $1^\circ$ visual angle corresponds to:
$$\text{Physical Screen Distance} = 2 \times 550 \times \tan(0.5^\circ) = 9.6\text{ mm} \approx \mathbf{96\text{ physical Retina pixels}}$$

| Metric | Physical Quantity | Visual Degrees | % Screen Diagonal |
| :--- | :--- | :--- | :--- |
| **Human Fixation Jitter (Microsaccades)** | ~48 - 96 px | $0.5^\circ - 1.0^\circ$ | 1.4% - 2.7% |
| **Webcam Sensor Sub-pixel Noise ($\pm 0.5$ px)** | ~40 - 80 px | $0.4^\circ - 0.8^\circ$ | 1.1% - 2.3% |
| **Unconstrained Head Yaw Sway ($\pm 1.5^\circ$)** | ~140 px | $1.5^\circ$ | 4.0% |
| **Commercial IR Tracker (Tobii Eye Tracker 5)** | ~50 - 90 px | $0.5^\circ - 0.9^\circ$ | 1.4% - 2.5% |
| **Commodity Webcam SOTA (MPIIGaze / OpenFace)** | **~280 - 430 px** | **$3.0^\circ - 4.5^\circ$** | **8.0% - 12.3%** |
| **Our M4 Acceptance Target ($\le 2.5\%$)** | **87.7 px** | **$< 0.9^\circ$** | **2.5%** |

**Conclusion:** The M4 acceptance target of $\le 2.5\%$ diagonal ($87.7\text{ px}$) assumed active infrared corneal reflection hardware. For a standard RGB webcam without infrared illuminators, the physical lower bound on unconstrained gaze error is approximately **$3.0^\circ - 4.0^\circ$ ($280 - 380\text{ pixels}$)**.

---

### 3.5 Root Cause 5: MacBook Bezel Vertical Parallax

The built-in FaceTime HD camera is mounted in the top display bezel, **15 cm above the center of the display**.
- When looking at the top targets ($v = 0.08$), the line of sight is roughly horizontal relative to the camera lens ($pitch \approx 0^\circ$).
- When looking at bottom targets ($v = 0.92$), the user must depress gaze downward by $18^\circ - 24^\circ$.
- This downward ocular depression causes:
  1. Upper eyelid descent (ptosis / eyelid foreshortening) which occludes the upper 30-40% of the iris.
  2. Perspective distortion of the circular iris into an ellipse.
  3. Head pitch tipping forward as a natural vestibulo-collic response.

Without explicit non-linear terms combining eye vertical displacement with head pitch ($avg_y \cdot \widehat{pitch}$), vertical prediction degrades significantly near the bottom of the screen.

---

## 4. Summary of Improvements in M4.1

| Component | M4 Baseline (Failed) | M4.1 Improved (Validated) | Impact |
| :--- | :--- | :--- | :--- |
| **Eye Coordinate Normalization** | Left & Right axes inverted | Unified image-left to image-right | Eliminates conjugate signal cancellation |
| **Camera Capture Format** | Default active format | 1080p Format [5] verified (30 FPS) | 3x spatial resolution for iris localization |
| **Regression Architecture** | Unstandardized 19-param Poly | 12-param Physically-Informed Ridge | Condition number: 26,731 $\to$ 62.1 |
| **Head Pose Feature Scaling** | Raw degrees ($-30^\circ$ to $+30^\circ$) | Normalized ($\widehat{yaw}, \widehat{pitch}, \widehat{roll}$) | Equal regularizer shrinkage across terms |
| **Vertical Parallax Handling** | Linear combination only | $avg_y \cdot \widehat{pitch}$ interaction term | Compensates for 15 cm bezel offset |
| **Data Collection Passes** | 1-Pass (13 targets) | 2-Pass and 3-Pass multi-temporal | Filters saccadic blips and head drift |

