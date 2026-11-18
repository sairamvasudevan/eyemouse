# EyeMouse Personalized Calibration Engine

## 1. Architectural Philosophy and Requirements

Standard optical gaze tracking from a webcam cannot deduce screen fixation coordinates out-of-the-box. Physical camera placement (top bezel of MacBook), eye socket depth, pupil-corneal geometry, display angle, and viewing distance vary significantly across individuals.

The EyeMouse Calibration Engine (`src/vision_engine/calibration/`) bridges normalized facial mesh geometry to physical macOS display pixels through a rigorous, personalized calibration pipeline:

1. **Decoupled Optical Geometry**: Separates eye movement (normalized iris UV relative to lateral and medial canthi) from rigid head rotation (Euler angles $\text{yaw}, \text{pitch}, \text{roll}$ derived via PnP).
2. **Statistically Sound Aggregation**: Fixation frames are collected over a steady 1.0s window following a 0.4s settling period. Hundreds of correlated frames are **not** treated as independent observations; instead, robust median aggregation and Median Absolute Deviation (MAD) outlier filtering extract a single high-fidelity feature vector per target.
3. **Dual Model Architecture**: Fits both an **Affine/Linear Ridge Regression** baseline and a **Regularized Polynomial Degree-2 Ridge Regression** model.
4. **Generalization Over Fitting**: Hyperparameters are optimized via Leave-One-Out Cross-Validation (LOOCV), and final model selection is determined strictly by held-out performance on 9 independent validation locations.

---

## 2. Package Architecture (`src/vision_engine/calibration/`)

```
src/vision_engine/calibration/
├── __init__.py           # Unified exports
├── geometry.py           # ScreenGeometry, TargetPoint, 13-point & 9-point grids
├── collector.py          # RawGazeSample, TargetSampleCollector, MAD outlier rejection
├── models.py             # AffineGazeModel, PolynomialGazeModel, solve_ridge_regression, LOOCV
├── evaluator.py          # CalibrationEvaluator, CalibrationValidationReport
├── profile.py            # CalibrationProfile JSON persistence & hardware compatibility checks
├── estimator.py          # CalibratedGazeEstimator with OneEuroFilter real-time inference
└── session.py            # CalibrationSession coordinator
```

---

## 3. Mathematical Feature Representation

For each video frame, the vision engine extracts a 7-dimensional normalized feature vector:

$$\mathbf{f} = \begin{bmatrix} x_{\text{iris}, L} & y_{\text{iris}, L} & x_{\text{iris}, R} & y_{\text{iris}, R} & \theta_{\text{yaw}} & \theta_{\text{pitch}} & \theta_{\text{roll}} \end{bmatrix}^T \in \mathbb{R}^7$$

### Feature Definition:
- **$x_{\text{iris}, L}, y_{\text{iris}, L}$**: Left eye iris center normalized relative to inner and outer eye corners ($[0.0, 1.0]$, centered at $0.5$).
- **$x_{\text{iris}, R}, y_{\text{iris}, R}$**: Right eye iris center normalized relative to corners.
- **$\theta_{\text{yaw}}, \theta_{\text{pitch}}, \theta_{\text{roll}}$**: Rigid 3D head rotation in degrees computed from non-deformable facial landmarks (`nose tip`, `chin`, `nasion`, `temples`).

---

## 4. Fixation Dynamics & Robust Sample Filtering (`collector.py`)

During each target presentation:
1. **Saccadic Settling Window ($t \in [0.0, 0.40]\text{ s}$)**:  
   When a new calibration target appears, the user's eyes execute a ballistic saccade to locate the target. All frames within the first 400 ms are unconditionally dropped.
2. **Active Fixation Window ($t \in [0.40, 1.40]\text{ s}$)**:  
   Frames are accepted into the target pool only if:
   - `face_detected == true`
   - `confidence >= 0.60`
   - `blink_phase != CLOSED` and `left_blink_score < 0.25` and `right_blink_score < 0.25`
3. **Median Absolute Deviation (MAD) Outlier Rejection**:  
   Given $N$ accepted feature vectors $\mathbf{f}_i$:
   $$\mathbf{m} = \text{median}(\{\mathbf{f}_i\}_{i=1}^N)$$
   $$\text{MAD}_j = \text{median}(\{|f_{i, j} - m_j|\}_{i=1}^N)$$
   Any sample exceeding $2.5 \times \text{MAD}$ along any dimension is purged as an outlier.
4. **Target Representation**:  
   The representative observation for target $k$ is the robust median:
   $$\mathbf{x}_k = \text{median}(\mathbf{f}_{\text{inliers}})$$

---

## 5. Privacy and Security Non-Negotiables

> [!IMPORTANT]
> **No Video or Face Storage:**
> EyeMouse calibration profiles persist **only** mathematical model weights ($W$), regularizer hyperparameters ($\lambda$), camera unique ID, display geometry, and validation error statistics.
> **Zero raw pixel frames, eye crops, facial landmarks, or biometric templates are ever written to disk or transmitted across IPC.**

