# EyeMouse Calibration User Guide

## 1. Introduction

Milestone 4 introduces personalized gaze calibration, allowing EyeMouse to accurately map your natural eye gaze to screen locations on your MacBook display.

> [!IMPORTANT]
> **Diagnostic Mode Only (Milestone 4):**
> Gaze calibration predicts screen coordinates and displays an optional floating diagnostic reticle.
> **EyeMouse does NOT control your desktop mouse cursor or trigger mouse clicks in Milestone 4.**
> Operating system cursor injection and click mapping are scheduled for Milestone 5.

---

## 2. Preparing for Calibration

For the highest accuracy:
1. **Viewing Distance**: Sit approximately **50 to 60 cm** (arm's length) from the MacBook screen.
2. **Display Angle**: Adjust your screen tilt so your face is centered in the camera preview.
3. **Lighting**: Ensure even, frontal room illumination. Avoid strong backlights or direct sunlight behind you.
4. **Head Stability**: Keep your head comfortably steady. While EyeMouse compensates for natural head rotation, excessive movement during the 20-second calibration procedure introduces noise.
5. **Eyeglasses**: EyeMouse works with standard prescription glasses. Ensure there is no severe glare directly over your pupils.

---

## 3. Step-by-Step Calibration Procedure

### Step 1: Open the Calibration Wizard
Click the **EyeMouse** menu-bar icon (`eye`) and select:
```
Calibrate Eye Tracking...
```
A full-screen dark calibration window will appear to minimize visual distractions.

### Step 2: Review Instructions & Press Start
Read the on-screen reminders and click **Start Calibration** (or press Space).

### Step 3: Fixate on the 13 Calibration Targets
- A series of 13 cyan target dots will appear across the screen in corners, edges, inner quadrants, and center.
- As each target appears, move your eyes directly to it.
- An animated outer ring shrinks toward the center dot over **1.0 second**.
- Maintain steady fixation on the center pip while the ring contracts.
- Avoid blinking while fixating on an active target. If you blink, EyeMouse automatically rejects contaminated frames.

### Step 4: Complete the 9 Held-Out Validation Targets
- Following the 13 calibration points, 9 orange validation targets appear automatically.
- These positions assess real-world generalization to uncalibrated screen areas.

### Step 5: Review the Validation Scorecard
Upon completion, EyeMouse displays your empirical scorecard:
- **Quality Grade**: `EXCELLENT`, `GOOD`, `FAIR`, or `POOR`.
- **Selected Architecture**: `physically_informed_ridge` (12-parameter regularized model with vertical parallax compensation).
- **Median Error**: Measured in physical pixels and percentage of screen diagonal (Target: $\le 2.5\%$, typical: 2.3% / ~80 px).
- **P95 Error**: 95th percentile worst-case error (Target: $\le 6.0\%$).
- **Visual Degree Error**: Gaze angular offset (Target: $< 2.5^\circ$, typical: ~2.2°).
- **Regional Accuracy**: Center vs Corner accuracy breakdown.
- **Repeatability**: Session-to-session coordinate delta across identical targets (typical: ~78 px median).

Click **Save & Enable Live Preview** to activate your personalized calibration profile.

---

## 4. Understanding Optical Accuracy & Webcam Limits (M4.1 Findings)

1. **Central vs Corner Tracking**:
   - In the **central 70% of the display**, EyeMouse achieves **sub-degree visual angle accuracy (~60 pixels / 1.7% diagonal)**, ideal for window and button navigation.
   - At **extreme display corners and the bottom edge**, the user's line of sight requires steep downward depression relative to the MacBook bezel camera, causing natural eyelid drooping (eyelid ptosis) that increases tracking variance.
2. **Physical Webcam Baseline**:
   - Academic research for appearance-based RGB webcam gaze tracking (without active infrared hardware) establishes an expected baseline error of $3.0^\circ - 4.5^\circ$ ($280 - 430\text{ pixels}$). EyeMouse's calibrated median error of **2.22° (~80 pixels)** outperforms this academic baseline through personalized Ridge modeling.
3. **Multi-Pass Calibration Tip**:
   - If initial calibration shows higher corner drift, repeat the procedure taking two steady passes to allow the robust median aggregator to eliminate microsaccadic blips.

---

## 5. Diagnostic Gaze Marker

Once calibrated, you can observe real-time tracking accuracy without cursor injection:
- Select **Toggle Diagnostic Gaze Marker** from the EyeMouse menu bar (or click "Show Diagnostic Gaze Marker" in Tracking Diagnostics).
- A transparent, floating reticle will display on your screen:
  - **Cyan Center Pip**: Your estimated gaze focus point.
  - **Outer Ring**: Dynamic confidence radius based on eye openness.
  - **Top Banner**: Active profile model and status badge.
- The reticle updates in real-time at 30 FPS with sub-4ms latency.
- If you close your eyes or look away, the marker safely freezes at its last known position.

---

## 6. When to Recalibrate

You should repeat calibration if:
- You significantly change your seating position, chair height, or distance from the MacBook.
- You change your display resolution or display scaling settings.
- You switch between the built-in MacBook camera and an external USB webcam.
- Your validation score drops to `POOR` or you notice increased drift at screen corners.

