import sys
import json
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.vision_engine.calibration.geometry import (
    ScreenGeometry,
    generate_13_point_calibration_grid,
    generate_9_point_validation_grid,
)
from src.vision_engine.calibration.models import solve_ridge_regression

class ModelComparisonBenchmark:
    def __init__(self, screen: ScreenGeometry):
        self.screen = screen
        self.w_px = screen.width_pixels
        self.h_px = screen.height_pixels
        self.diag_px = screen.diagonal_pixels

    def generate_realistic_data(self, noise_std: float = 0.003, head_yaw_drift: float = 2.0, num_passes: int = 1):
        """Generate realistic calibration and held-out validation data with true conjugate eye movement and vertical parallax."""
        calib_targets = generate_13_point_calibration_grid()
        val_targets = generate_9_point_validation_grid()

        def simulate_features(tx: float, ty: float, pass_idx: int = 0):
            # Target offset from center (-0.5 to +0.5)
            dx = tx - 0.5
            dy = ty - 0.5

            # Eye iris response: ~0.15 normalized offset for full screen span
            # Both eyes now move conjugate (image-left to image-right basis)
            lx = 0.50 + dx * 0.16 + np.random.normal(0, noise_std)
            rx = 0.50 + dx * 0.16 + np.random.normal(0, noise_std)

            # Vertical iris response: looking down moves iris down (+Y), eyelids droop
            # Parallax: MacBook camera is 15cm above screen center. User must look down ~15-25 degrees to see bottom.
            ly = 0.00 + dy * 0.12 + 0.02 * (dy**2) + np.random.normal(0, noise_std)
            ry = 0.00 + dy * 0.12 + 0.02 * (dy**2) + np.random.normal(0, noise_std)

            # Head pose response: slight natural head turn with target, plus pass jitter
            yaw = dx * 8.0 + np.random.normal(0, head_yaw_drift)
            pitch = -dy * 10.0 + 3.0 + np.random.normal(0, head_yaw_drift * 0.5) # slightly tilted down
            roll = np.random.normal(0, 0.5)

            # Openness (EAR): droops when looking down
            openness = 0.26 - dy * 0.06 + np.random.normal(0, 0.005)

            return np.array([lx, ly, rx, ry, yaw, pitch, roll, openness], dtype=np.float64)

        # Multi-pass calibration training set
        X_train_list, Y_train_list = [], []
        for p in range(num_passes):
            for t in calib_targets:
                feat = simulate_features(t.norm_x, t.norm_y, pass_idx=p)
                X_train_list.append(feat)
                Y_train_list.append([t.norm_x, t.norm_y])

        X_train = np.array(X_train_list)
        Y_train = np.array(Y_train_list)

        # Held-out validation set (independent 9 points)
        X_val = np.array([simulate_features(t.norm_x, t.norm_y) for t in val_targets])
        Y_val = np.array([[t.norm_x, t.norm_y] for t in val_targets])

        return X_train, Y_train, X_val, Y_val, val_targets

    def evaluate_model(self, name: str, X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, transform_fn, alpha: float = 0.1):
        Phi_train = transform_fn(X_train)
        Phi_val = transform_fn(X_val)

        # Condition number of design matrix
        cond_num = float(np.linalg.cond(Phi_train))

        # Solve Ridge
        W = solve_ridge_regression(Phi_train, Y_train, alpha=alpha, penalize_intercept=False)

        # Predictions
        preds = np.clip(Phi_val @ W, 0.0, 1.0)

        # Calculate errors
        errs_x = np.abs(preds[:, 0] - Y_val[:, 0]) * self.w_px
        errs_y = np.abs(preds[:, 1] - Y_val[:, 1]) * self.h_px
        tot_err = np.sqrt(errs_x**2 + errs_y**2)

        med_err = float(np.median(tot_err))
        p95_err = float(np.percentile(tot_err, 95))
        mean_err_x = float(np.mean(errs_x))
        mean_err_y = float(np.mean(errs_y))
        med_pct = (med_err / self.diag_px) * 100.0
        p95_pct = (p95_err / self.diag_px) * 100.0

        return {
            "name": name,
            "features_dim": Phi_train.shape[1],
            "cond_number": round(cond_num, 1),
            "median_px": round(med_err, 1),
            "p95_px": round(p95_err, 1),
            "mean_x_px": round(mean_err_x, 1),
            "mean_y_px": round(mean_err_y, 1),
            "median_pct_diag": round(med_pct, 2),
            "p95_pct_diag": round(p95_pct, 2),
            "meets_acceptance": med_pct <= 2.5 and p95_pct <= 6.0,
        }

def run_benchmarks():
    screen = ScreenGeometry(width_points=1470.0, height_points=956.0, backing_scale=2.0)
    bench = ModelComparisonBenchmark(screen)

    # 1. Feature Transform Definitions
    def transform_affine(X):
        # [1, lx, ly, rx, ry, yaw, pitch, roll] (8)
        N = X.shape[0]
        return np.hstack([np.ones((N, 1)), X[:, :7]])

    def transform_poly_raw(X):
        # Unstandardized polynomial (M4 implementation)
        N = X.shape[0]
        lx, ly, rx, ry = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]
        yaw, pitch, roll = X[:, 4:5], X[:, 5:6], X[:, 6:7]
        linear = np.hstack([lx, ly, rx, ry, yaw, pitch, roll])
        quad = np.hstack([lx**2, ly**2, rx**2, ry**2, yaw**2, pitch**2, roll**2])
        inter = np.hstack([lx * ly, rx * ry, lx * yaw, ly * pitch])
        return np.hstack([np.ones((N, 1)), linear, quad, inter])

    def transform_poly_standardized(X):
        # Standardize features before polynomial expansion
        N = X.shape[0]
        mu = np.mean(X[:, :7], axis=0, keepdims=True)
        sigma = np.std(X[:, :7], axis=0, keepdims=True) + 1e-6
        X_norm = (X[:, :7] - mu) / sigma
        lx, ly, rx, ry = X_norm[:, 0:1], X_norm[:, 1:2], X_norm[:, 2:3], X_norm[:, 3:4]
        yaw, pitch, roll = X_norm[:, 4:5], X_norm[:, 5:6], X_norm[:, 6:7]
        linear = np.hstack([lx, ly, rx, ry, yaw, pitch, roll])
        quad = np.hstack([lx**2, ly**2, rx**2, ry**2, yaw**2, pitch**2, roll**2])
        inter = np.hstack([lx * ly, rx * ry, lx * yaw, ly * pitch])
        return np.hstack([np.ones((N, 1)), linear, quad, inter])

    def transform_physically_informed(X):
        # Physically informed features:
        # 1. Conjugate iris center: avg_x = (lx + rx)/2, avg_y = (ly + ry)/2
        # 2. Scaled pose: yaw/30, pitch/25, roll/20
        # 3. Eye aspect ratio (openness)
        # 4. Vertical parallax interaction: avg_y * (pitch/25)
        # 5. Quadratic terms for gaze: avg_x^2, avg_y^2
        N = X.shape[0]
        lx, ly, rx, ry = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]
        yaw, pitch, roll = X[:, 4:5], X[:, 5:6], X[:, 6:7]
        ear = X[:, 7:8] if X.shape[1] > 7 else np.zeros((N, 1))

        avg_x = (lx + rx) / 2.0 - 0.5
        avg_y = (ly + ry) / 2.0
        disp_x = (rx - lx)
        norm_yaw = yaw / 30.0
        norm_pitch = pitch / 25.0
        norm_roll = roll / 20.0

        terms = [
            np.ones((N, 1)),
            avg_x, avg_y,
            disp_x,
            norm_yaw, norm_pitch, norm_roll,
            ear,
            avg_x**2, avg_y**2,
            avg_x * norm_yaw,
            avg_y * norm_pitch,
        ]
        return np.hstack(terms)

    # 2. Benchmark Single-Pass vs Multi-Pass
    print("=== MODEL COMPARISON BENCHMARK (Single-Pass Calibration) ===")
    X_train1, Y_train1, X_val, Y_val, _ = bench.generate_realistic_data(noise_std=0.003, num_passes=1)

    m1 = bench.evaluate_model("1. Affine Ridge (Unstandardized)", X_train1, Y_train1, X_val, Y_val, transform_affine, alpha=0.1)
    m2 = bench.evaluate_model("2. Polynomial Deg-2 (Unstandardized)", X_train1, Y_train1, X_val, Y_val, transform_poly_raw, alpha=0.1)
    m3 = bench.evaluate_model("3. Polynomial Deg-2 (Standardized)", X_train1, Y_train1, X_val, Y_val, transform_poly_standardized, alpha=0.1)
    m4 = bench.evaluate_model("4. Physically-Informed Ridge", X_train1, Y_train1, X_val, Y_val, transform_physically_informed, alpha=0.05)

    models_pass1 = [m1, m2, m3, m4]
    for m in models_pass1:
        print(f"{m['name']:36s} | Dims={m['features_dim']:2d} | Cond={m['cond_number']:8.1f} | Med={m['median_px']:5.1f}px ({m['median_pct_diag']:4.2f}%) | P95={m['p95_px']:5.1f}px ({m['p95_pct_diag']:4.2f}%) | Meets={m['meets_acceptance']}")

    print("\n=== DATA COLLECTION BENCHMARK: 1-Pass vs 2-Pass vs 3-Pass ===")
    pass_results = {}
    for passes in [1, 2, 3]:
        X_tr, Y_tr, _, _, _ = bench.generate_realistic_data(noise_std=0.003, num_passes=passes)
        res = bench.evaluate_model(f"Physically-Informed ({passes}-Pass)", X_tr, Y_tr, X_val, Y_val, transform_physically_informed, alpha=0.05)
        pass_results[passes] = res
        print(f"{passes}-Pass Collection ({passes*13:2d} points) | Median Error: {res['median_px']:5.1f} px ({res['median_pct_diag']:4.2f}%) | P95: {res['p95_px']:5.1f} px ({res['p95_pct_diag']:4.2f}%) | Meets={res['meets_acceptance']}")

    # Save benchmark results
    out_file = Path("build/m4_1_model_benchmarks.json")
    with open(out_file, "w") as f:
        json.dump({
            "models_single_pass": models_pass1,
            "multi_pass_comparison": pass_results,
        }, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")

if __name__ == "__main__":
    run_benchmarks()

