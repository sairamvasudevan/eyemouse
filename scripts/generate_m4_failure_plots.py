import os
import json
from pathlib import Path
import numpy as np

# Ensure headless matplotlib
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/eyemouse_mpl"
import matplotlib.pyplot as plt

def main():
    with open("/tmp/m4_session_1.json") as f:
        session = json.load(f)

    with open("/tmp/m4_profile_1.json") as f:
        profile = json.load(f)

    weights = np.array(profile["model_dict"]["weights"])
    w_px = 2940.0
    h_px = 1912.0
    diag_px = float(np.sqrt(w_px**2 + h_px**2))

    # Output directory
    out_dir = Path("docs/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prediction vs Ground Truth Plot
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#121212")

    # Plot screen boundaries
    ax.plot([0, w_px, w_px, 0, 0], [0, 0, h_px, h_px, 0], color="#555555", linestyle="--", linewidth=1.5, label="Display Edge (2940x1912)")

    # Collect validation predictions
    val_truths_x, val_truths_y = [], []
    val_preds_x, val_preds_y = [], []
    errs_x, errs_y, total_errs = [], [], []
    target_names = []

    for t in session["validation_targets"]:
        tu = float(t["target_norm_x"]) * w_px
        tv = float(t["target_norm_y"]) * h_px
        feat = np.array([1.0] + [float(x) for x in t["feature_median"]])
        pred = feat @ weights
        pred = np.clip(pred, 0.0, 1.0)
        pu = float(pred[0]) * w_px
        pv = float(pred[1]) * h_px

        val_truths_x.append(tu)
        val_truths_y.append(tv)
        val_preds_x.append(pu)
        val_preds_y.append(pv)
        target_names.append(t["target_name"])

        ex = abs(pu - tu)
        ey = abs(pv - tv)
        errs_x.append(ex)
        errs_y.append(ey)
        total_errs.append(np.sqrt(ex**2 + ey**2))

        # Draw error vector
        ax.annotate(
            "",
            xy=(pu, pv),
            xytext=(tu, tv),
            arrowprops=dict(arrowstyle="->", color="#ff4444", lw=1.5, mutation_scale=12),
        )

    # Plot targets and predictions
    ax.scatter(val_truths_x, val_truths_y, color="#00ff88", s=120, zorder=5, label="Ground Truth (Validation Targets)")
    ax.scatter(val_preds_x, val_preds_y, color="#ff4444", s=90, marker="x", zorder=5, label="Model Prediction (Affine Ridge)")

    for name, tx, ty in zip(target_names, val_truths_x, val_truths_y):
        ax.text(tx + 25, ty + 25, name.replace("Val-", ""), color="#cccccc", fontsize=8)

    ax.set_xlim(-100, w_px + 100)
    ax.set_ylim(h_px + 100, -100)  # Inverted Y for screen coordinates (origin top-left)
    ax.set_title("M4 Baseline Calibration: Predicted vs Ground-Truth Screen Coordinates\n(MacBook Air 13.6\" Retina: 2940x1912)", color="#ffffff", fontsize=12, pad=12)
    ax.set_xlabel("Physical Screen X (pixels)", color="#ffffff", fontsize=10)
    ax.set_ylabel("Physical Screen Y (pixels, Top=0)", color="#ffffff", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.legend(facecolor="#252525", edgecolor="#444444", labelcolor="#ffffff", loc="upper right")
    ax.grid(True, color="#333333", linestyle=":", alpha=0.6)

    fig.tight_layout()
    pred_path = out_dir / "m4_baseline_prediction_vs_truth.png"
    plt.savefig(pred_path)
    plt.close()
    print(f"Saved: {pred_path}")

    # 2. Error Breakdown Plot (X vs Y and Regional Breakdown)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor("#121212")

    # Left: X vs Y error comparison
    ax1.set_facecolor("#1e1e1e")
    bars1 = ax1.bar(["X Error (Horizontal)", "Y Error (Vertical)", "Total Pixel Error"], [np.mean(errs_x), np.mean(errs_y), np.mean(total_errs)], color=["#ffaa00", "#00aaff", "#ff4444"], width=0.5)
    ax1.set_title("Mean Held-Out Error by Axis", color="#ffffff", fontsize=11)
    ax1.set_ylabel("Error (Physical Pixels)", color="#ffffff", fontsize=10)
    ax1.tick_params(colors="#aaaaaa")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:.1f} px\n({yval/diag_px*100:.1f}%)", ha="center", va="bottom", color="#ffffff", fontsize=9)
    ax1.set_ylim(0, 600)
    for spine in ax1.spines.values():
        spine.set_color("#444444")
    ax1.grid(True, axis="y", color="#333333", linestyle=":", alpha=0.6)

    # Right: Target by Target error bar chart
    ax2.set_facecolor("#1e1e1e")
    short_names = [n.replace("Val-", "") for n in target_names]
    bars2 = ax2.barh(short_names, total_errs, color="#ff5555")
    ax2.axvline(x=diag_px * 0.025, color="#00ff88", linestyle="--", linewidth=1.5, label="2.5% Target (87.7 px)")
    ax2.axvline(x=diag_px * 0.060, color="#ffbb00", linestyle="--", linewidth=1.5, label="6.0% P95 Cap (210.4 px)")
    ax2.set_title("Per-Target Total Pixel Error", color="#ffffff", fontsize=11)
    ax2.set_xlabel("Error (Physical Pixels)", color="#ffffff", fontsize=10)
    ax2.tick_params(colors="#aaaaaa")
    for bar in bars2:
        xval = bar.get_width()
        ax2.text(xval + 15, bar.get_y() + bar.get_height()/2.0, f"{xval:.0f} px", ha="left", va="center", color="#cccccc", fontsize=8)
    ax2.set_xlim(0, 1200)
    ax2.legend(facecolor="#252525", edgecolor="#444444", labelcolor="#ffffff", loc="lower right")
    for spine in ax2.spines.values():
        spine.set_color("#444444")
    ax2.grid(True, axis="x", color="#333333", linestyle=":", alpha=0.6)

    fig.tight_layout()
    breakdown_path = out_dir / "m4_baseline_error_heatmap.png"
    plt.savefig(breakdown_path)
    plt.close()
    print(f"Saved: {breakdown_path}")

if __name__ == "__main__":
    main()

