import json
import numpy as np

def main():
    with open("/tmp/m4_session_1.json") as f:
        session = json.load(f)

    with open("/tmp/m4_profile_1.json") as f:
        profile = json.load(f)

    weights = np.array(profile["model_dict"]["weights"])
    w_points = 1470.0
    h_points = 956.0
    scale = 2.0
    w_px = w_points * scale
    h_px = h_points * scale

    diag_px = float(np.sqrt(w_px**2 + h_px**2))
    errs = []
    errs_x = []
    errs_y = []

    header = f"{'Target Name':26s} | {'Truth (u,v)':13s} | {'Pred (u,v)':13s} | {'Err X (px)':10s} | {'Err Y (px)':10s} | {'Err Total':10s} | {'% Diag':6s}"
    print(header)
    print("-" * len(header))

    for t in session["validation_targets"]:
        name = t["target_name"]
        tu, tv = float(t["target_norm_x"]), float(t["target_norm_y"])
        feat = np.array([1.0] + [float(x) for x in t["feature_median"]])
        pred = feat @ weights
        pred = np.clip(pred, 0.0, 1.0)
        pu, pv = float(pred[0]), float(pred[1])
        
        err_x_px = abs(pu - tu) * w_px
        err_y_px = abs(pv - tv) * h_px
        tot_err_px = float(np.sqrt(err_x_px**2 + err_y_px**2))
        err_pct = (tot_err_px / diag_px) * 100.0
        errs.append(tot_err_px)
        errs_x.append(err_x_px)
        errs_y.append(err_y_px)
        print(f"{name:26s} | ({tu:.2f}, {tv:.2f})     | ({pu:.2f}, {pv:.2f})     | {err_x_px:10.1f} | {err_y_px:10.1f} | {tot_err_px:10.1f} | {err_pct:5.2f}%")

    print("-" * len(header))
    print(f"Mean Error X:   {np.mean(errs_x):.1f} px | Mean Error Y: {np.mean(errs_y):.1f} px")
    print(f"Median Error:   {np.median(errs):.1f} px ({np.median(errs)/diag_px*100:.2f}%)")
    print(f"P95 Error:      {np.percentile(errs, 95):.1f} px ({np.percentile(errs, 95)/diag_px*100:.2f}%)")

if __name__ == "__main__":
    main()

