import os
import json
from pathlib import Path
import numpy as np

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/eyemouse_mpl"
import matplotlib.pyplot as plt

def main():
    with open("build/m4_calibration_results.json") as f:
        res = json.load(f)

    rep = res["report"]
    targets = rep["target_results"]

    w_px = 2940.0
    h_px = 1912.0
    diag_px = float(np.sqrt(w_px**2 + h_px**2))

    out_dir = Path("docs/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prediction vs Ground Truth Plot for M4.1
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#121212")

    ax.plot([0, w_px, w_px, 0, 0], [0, 0, h_px, h_px, 0], color="#555555", linestyle="--", linewidth=1.5, label="Display Edge (2940x1912)")

    truths_x, truths_y = [], []
    preds_x, preds_y = [], []
    total_errs = []
    names = []

    for t in targets:
        gt_x = t["ground_truth_norm"][0] * w_px
        gt_y = t["ground_truth_norm"][1] * h_px
        pr_x = t["predicted_norm"][0] * w_px
        pr_y = t["predicted_norm"][1] * h_px

        truths_x.append(gt_x)
        truths_y.append(gt_y)
        preds_x.append(pr_x)
        preds_y.append(pr_y)
        names.append(t["target_name"].replace("Val-", ""))
        total_errs.append(t["error_pixels"])

        # Error vector
        ax.annotate(
            "",
            xy=(pr_x, pr_y),
            xytext=(gt_x, gt_y),
            arrowprops=dict(arrowstyle="->", color="#00ffcc", lw=1.5, mutation_scale=10),
        )

    ax.scatter(truths_x, truths_y, color="#00ff88", s=130, zorder=5, label="Ground Truth (Validation Targets)")
    ax.scatter(preds_x, preds_y, color="#00ccff", s=100, marker="o", facecolors="none", edgecolors="#00ccff", linewidth=2, zorder=5, label="M4.1 Prediction (Physically-Informed Ridge)")

    for name, tx, ty in zip(names, truths_x, truths_y):
        ax.text(tx + 25, ty + 25, name, color="#dddddd", fontsize=8)

    ax.set_xlim(-100, w_px + 100)
    ax.set_ylim(h_px + 100, -100)
    ax.set_title(f"M4.1 Improved Calibration: Predicted vs Ground-Truth Screen Coordinates\nModel: {rep['model_name']} | Median Error: {rep['median_pixel_error']} px ({rep['median_error_pct_diagonal']}%)", color="#ffffff", fontsize=12, pad=12)
    ax.set_xlabel("Physical Screen X (pixels)", color="#ffffff", fontsize=10)
    ax.set_ylabel("Physical Screen Y (pixels, Top=0)", color="#ffffff", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.legend(facecolor="#252525", edgecolor="#444444", labelcolor="#ffffff", loc="upper right")
    ax.grid(True, color="#333333", linestyle=":", alpha=0.6)

    fig.tight_layout()
    pred_path = out_dir / "m4_1_prediction_vs_truth.png"
    plt.savefig(pred_path)
    plt.close()
    print(f"Saved: {pred_path}")

    # 2. Before vs After Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#121212")

    metrics_labels = ["Median Error (px)", "Mean Error X (px)", "Mean Error Y (px)", "Visual Degrees (°)"]
    m4_baseline = [458.1, 460.9, 94.1, 4.89]
    m4_1_improved = [rep["median_pixel_error"], rep["mean_pixel_error_x"], rep["mean_pixel_error_y"], rep["mean_visual_degrees"]]

    x = np.arange(len(metrics_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, m4_baseline, width, label="M4 Baseline (Affine Ridge)", color="#ff5555")
    rects2 = ax.bar(x + width/2, m4_1_improved, width, label="M4.1 Improved (Physically-Informed)", color="#00ff88")

    ax.set_title("Calibration Accuracy Improvement: M4 Baseline vs M4.1 Improved", color="#ffffff", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, color="#ffffff", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    ax.legend(facecolor="#252525", edgecolor="#444444", labelcolor="#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(True, axis="y", color="#333333", linestyle=":", alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", color="#ffffff", fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    comp_path = out_dir / "m4_1_error_comparison_baseline_vs_improved.png"
    plt.savefig(comp_path)
    plt.close()
    print(f"Saved: {comp_path}")

if __name__ == "__main__":
    main()

