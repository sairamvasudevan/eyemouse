# Gaze Calibration Model Comparison: Affine vs. Polynomial Ridge Regression

## 1. Overview and Problem Formulation

Mapping high-dimensional optical eye landmarks to planar screen coordinates is an ill-posed regression problem:
1. **Limited Training Samples**: User calibration must remain fast and effortless; asking a user to fixate on more than 13 targets induces fatigue and blink artifacts.
2. **Non-Linear Optical Curvature**: Iris displacement along the eyeball's spherical geometry exhibits mild quadratic curvature near the corners of the eye socket.
3. **Risk of Overfitting**: Unconstrained high-degree polynomials (degree 3+) perfectly interpolate 13 training targets but suffer severe Runge-phenomenon oscillations between targets, degrading generalization to independent screen locations.

To establish an empirically grounded gaze engine, EyeMouse implements, cross-validates, and compares two models:
- **Baseline**: Affine / Linear Ridge Regression
- **Candidate**: Regularized Polynomial Degree-2 Ridge Regression

---

## 2. Mathematical Formulations

### 2.1 Model 1: Affine / Linear Regression Baseline

The linear model maps the 7-dimensional gaze feature vector linearly to normalized screen coordinates $(u, v) \in [0.0, 1.0]$:

$$\begin{pmatrix} \hat{u} \\ \hat{v} \end{pmatrix} = W^T \mathbf{x}_{\text{affine}}$$

where:
$$\mathbf{x}_{\text{affine}} = \begin{bmatrix} 1 & x_{L} & y_{L} & x_{R} & y_{R} & \theta_{\text{yaw}} & \theta_{\text{pitch}} & \theta_{\text{roll}} \end{bmatrix}^T \in \mathbb{R}^8$$
$$W \in \mathbb{R}^{8 \times 2}$$

#### Properties:
- **Parameter Count**: $8 \times 2 = 16$ weights.
- **Monotonicity**: Guaranteed monotonic mapping; zero risk of high-frequency oscillations.
- **Limitation**: Cannot model spherical eyeball curvature or cross-axis perspective distortions.

---

### 2.2 Model 2: Regularized Polynomial Degree-2 Ridge Regression

The polynomial model expands the feature vector into linear, quadratic, and selected cross-interaction terms:

$$\mathbf{x}_{\text{poly}} = \begin{bmatrix} 1 \\ \mathbf{x}_{\text{linear}} \\ \mathbf{x}_{\text{quadratic}} \\ \mathbf{x}_{\text{interactions}} \end{bmatrix} \in \mathbb{R}^{19}$$

where:
$$\mathbf{x}_{\text{linear}} = \begin{bmatrix} x_L & y_L & x_R & y_R & \theta_{\text{yaw}} & \theta_{\text{pitch}} & \theta_{\text{roll}} \end{bmatrix}^T \quad (7\text{ terms})$$
$$\mathbf{x}_{\text{quadratic}} = \begin{bmatrix} x_L^2 & y_L^2 & x_R^2 & y_R^2 & \theta_{\text{yaw}}^2 & \theta_{\text{pitch}}^2 & \theta_{\text{roll}}^2 \end{bmatrix}^T \quad (7\text{ terms})$$
$$\mathbf{x}_{\text{interactions}} = \begin{bmatrix} x_L \cdot y_L & x_R \cdot y_R & x_L \cdot \theta_{\text{yaw}} & y_L \cdot \theta_{\text{pitch}} \end{bmatrix}^T \quad (4\text{ terms})$$

#### Properties:
- **Parameter Count**: $19 \times 2 = 38$ weights.
- **Curvature Modeling**: Accurately bends coordinate lines to match corneal projection on flat displays.
- **Overfitting Prevention**: Regularized using an $\ell_2$ Tikhonov shrinkage penalty.

---

## 3. Regularization & Closed-Form Optimization

For design matrix $X \in \mathbb{R}^{N \times D}$ and ground-truth targets $Y \in \mathbb{R}^{N \times 2}$:

$$\min_{W} \|X W - Y\|_F^2 + \lambda \|R W\|_F^2$$

where $R = \text{diag}(0, 1, 1, \dots, 1) \in \mathbb{R}^{D \times D}$.  
Crucially, **the intercept (column 0) is unpenalized**, allowing the screen coordinate offset to adjust freely while higher-order weights are shrunk toward zero.

Closed-form solution:
$$W = (X^T X + \lambda R)^{-1} X^T Y$$

---

## 4. Hyperparameter Tuning via Leave-One-Out Cross-Validation (LOOCV)

Rather than fixing $\lambda$ arbitrarily, EyeMouse runs Leave-One-Out Cross-Validation across candidate values:
$$\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 0.5, 1.0, 5.0, 10.0\}$$

For each candidate $\lambda$:
1. Withhold target $k \in \{1, \dots, 13\}$.
2. Train model on the remaining 12 targets.
3. Predict target $k$ and compute Euclidean error.
4. Average error across all 13 folds: $\text{RMSE}_{\text{CV}}(\lambda)$.
5. Select $\lambda^* = \arg\min \text{RMSE}_{\text{CV}}(\lambda)$.

---

## 5. Decision Rule for Model Selection

The calibration engine evaluates **both** models on the 9 held-out validation targets:
1. If $\text{MedianError}_{\text{poly}} \le \text{MedianError}_{\text{affine}}$ and polynomial model satisfies validation criteria, **Polynomial Degree-2** is selected.
2. If the polynomial model overfits (yielding higher validation error than affine), **Affine Baseline** is automatically retained.
3. This guarantees that model complexity is only adopted when it demonstrably improves generalization on unseen points.

