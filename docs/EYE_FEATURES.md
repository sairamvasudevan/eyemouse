# Eye Feature Extraction and Head Pose Specification

## 1. Landmark Index Map and Topology

EyeMouse utilizes MediaPipe's 478-point canonical face mesh topology (468 facial surface mesh landmarks + 10 refined iris landmarks).

### 1.1 Critical Landmark Indices

| Anatomical Landmark | Subject Left Eye (Image Right) | Subject Right Eye (Image Left) |
|---|---|---|
| **Outer Eye Corner (Lateral Canthus)** | `263` | `33` |
| **Inner Eye Corner (Medial Canthus)** | `362` | `133` |
| **Upper Eyelid Superior** | `386` | `159` |
| **Lower Eyelid Inferior** | `374` | `145` |
| **Iris Center** | `473` | `468` |
| **Iris Perimeter (Top, Bottom, Left, Right)** | `474, 475, 476, 477` | `469, 470, 471, 472` |

### 1.2 Rigid Facial Anchor Indices (Head Pose SolvePnP)

To solve head pose via Perspective-n-Point without contamination from facial expressions (smiling, blinking, talking), rigid skeletal anchor points are selected:

- **Nose Tip**: `1`
- **Chin (Menton)**: `152`
- **Nasion (Forehead / Glabella)**: `10`
- **Left Temple / Zygomatic Arch**: `454`
- **Right Temple / Zygomatic Arch**: `234`
- **Subnasale (Base of Nose)**: `2`

---

## 2. Coordinate Systems

```
+-------------------------------------------------------------+
| Image Coordinate Space (Normalized UV):                     |
| Origin (0,0) at top-left, (1.0, 1.0) at bottom-right        |
|                                                             |
| Face Coordinate Space (Local Eye Frame):                    |
| Origin (0,0) at eye midpoint                                |
| X-axis: parallel to inter-canthal line (inner to outer)      |
| Y-axis: perpendicular to inter-canthal line                 |
| Range: [-1.0, +1.0]                                         |
|                                                             |
| Camera Coordinate Space (3D Metric):                        |
| X: Right (+), Y: Down (+), Z: Forward (+ out of camera)     |
+-------------------------------------------------------------+
```

---

## 3. Iris Normalization Mathematics

Absolute pixel coordinates of the iris vary wildly with camera resolution, user distance, head tilt, and head roll. EyeMouse transforms raw iris points into a scale-invariant, roll-invariant local eye coordinate system:

1. **Inter-Canthal Vector**:
   $$\vec{u}_x = \frac{\mathbf{P}_{\text{outer}} - \mathbf{P}_{\text{inner}}}{\|\mathbf{P}_{\text{outer}} - \mathbf{P}_{\text{inner}}\|}$$
2. **Orthogonal Eyelid Vector**:
   $$\vec{u}_y = \begin{pmatrix} -\vec{u}_{x, y} \\ \vec{u}_{x, x} \end{pmatrix}$$
3. **Eye Center**:
   $$\mathbf{C}_{\text{eye}} = \frac{\mathbf{P}_{\text{inner}} + \mathbf{P}_{\text{outer}}}{2}$$
4. **Normalized Iris Offset**:
   Let $\vec{d} = \mathbf{P}_{\text{iris}} - \mathbf{C}_{\text{eye}}$, and half-width $w_{\text{eye}} = \frac{1}{2}\|\mathbf{P}_{\text{outer}} - \mathbf{P}_{\text{inner}}\|$:
   $$x_{\text{norm}} = \frac{\vec{d} \cdot \vec{u}_x}{w_{\text{eye}}}, \quad y_{\text{norm}} = \frac{\vec{d} \cdot \vec{u}_y}{w_{\text{eye}}}$$

By projecting the iris displacement onto the orthonormal basis $(\vec{u}_x, \vec{u}_y)$, the resulting normalized coordinates $(x_{\text{norm}}, y_{\text{norm}})$ remain strictly invariant when the user rotates their head or moves closer/further from the lens.

---

## 4. Head Pose Estimation and Decoupling

Head movement must not be mistaken for eye gaze movement. EyeMouse isolates 3D head rotation:

1. **Canonical 3D Model**: Based on the canonical face mesh model (`assets/models/geometry_pipeline_metadata_landmarks.binarypb`), 3D metric coordinates $(X, Y, Z)$ for rigid anchors are referenced.
2. **SolvePnP Optimization**: OpenCV's `solvePnP` computes the rotation vector $\mathbf{r}_{\text{vec}}$ and translation vector $\mathbf{t}_{\text{vec}}$ minimizing reprojection error between the 3D model and 2D observed landmarks.
3. **Euler Angles**: Using Rodrigues rotation matrices:
   - **Yaw ($\theta_y$)**: Head turn left/right (degrees).
   - **Pitch ($\theta_x$)**: Head tilt up/down (degrees).
   - **Roll ($\theta_z$)**: Head tilt sideways (degrees).

### 4.1 Decoupling in Gaze Estimation
The optical proxy estimator calculates:
$$\text{Gaze}_x = \text{Iris}_x + \kappa_{\text{yaw}} \cdot \text{Yaw}$$
$$\text{Gaze}_y = \text{Iris}_y + \kappa_{\text{pitch}} \cdot \text{Pitch}$$
where $\kappa_{\text{yaw}}$ and $\kappa_{\text{pitch}}$ are head-pose compensation coefficients. This allows the user to hold their eye gaze stationary on a target while naturally rotating their head, without causing erratic cursor movement.

---

## 5. Temporal Filtering (One Euro Filter)

Eye gaze signals suffer from two competing constraints:
1. When fixing on a button or link, micro-tremors and sensor noise cause cursor jitter.
2. When scanning across the screen (saccades), heavy smoothing causes lag and overshoot.

EyeMouse implements the adaptive velocity-based One Euro filter (Casiez, Roussel, Vogel, CHI 2012) in `src/vision_engine/filters.py`:
- **Low Speed (Stationary)**: Cutoff frequency reduces to $f_c = 0.5\text{ Hz}$ to eliminate all jitter.
- **High Speed (Saccades)**: Velocity derivative $\dot{x}$ scales cutoff frequency $f_c = f_{\text{min}} + \beta |\dot{x}|$ ($\beta = 0.005$) to eliminate latency during rapid shifts.
- **NaN/Inf Safety**: If tracking drops or numerical issues occur, invalid values are rejected, preserving filter state without poisoning.

---

## 6. Gaze Calibration Notice

> [!WARNING]
> **M2 Does NOT Implement Calibrated Screen Gaze**:
> The coordinates produced in Milestone 2 represent an uncalibrated optical proxy $([-1, 1] \times [-1, 1])$, not screen pixel positions $(X_{\text{screen}}, Y_{\text{screen}})$.
>
> True screen gaze requires:
> 1. User eye-screen geometry calibration (user distance, screen size, camera FOV).
> 2. Polynomial regression or multi-point calibration mapping.
> 3. These capabilities are architected under the `GazeEstimatorProtocol` and will be implemented in Milestone 4.

