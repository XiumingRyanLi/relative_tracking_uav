#!/usr/bin/env python3
"""
Evaluate the DOPE target estimate against ground truth from a
relative_position_controller CSV (relative_pid_<timestamp>.csv).

Usage:
    python3 scripts/plot_dope_evaluation.py                 # newest CSV in logs/ (or cwd)
    python3 scripts/plot_dope_evaluation.py path/to/file.csv
    python3 scripts/plot_dope_evaluation.py file.csv --max-age 0.3 --no-show

Reads the evaluation columns the controller appends to every row:
    drone_x/y/z, drone_yaw_deg, drone_lat/lon/alt
    target_x/y/z            DOPE target in the world frame (via gimbal/drone pose)
    dope_dist_m             DOPE straight-line camera->car range
    dope_age_s              time since that detection was captured
    dope_target_yaw_deg     car heading from DOPE's orientation (world ENU)
    car_gt_x/y/z, car_gt_yaw_deg, car_gt_lat/lon   Gazebo truth
    gt_dist_m, dope_dist_err_m, dope_pos_err_m, dope_yaw_err_deg

Rows are only scored while a detection is fresh (dope_age_s <= --max-age);
a stale detection compared against a moving car would inflate the error.

Writes <csv basename>_dope_eval.png next to the CSV and prints a summary.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

REQUIRED = ["gt_dist_m", "dope_dist_m", "dope_age_s", "dope_pos_err_m", "dope_yaw_err_deg",
            "car_gt_x", "car_gt_y", "target_x", "target_y", "drone_x", "drone_y"]


def find_latest_csv():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    for d in (os.path.join(here, "..", "logs"), os.getcwd()):
        candidates += glob.glob(os.path.join(d, "relative_pid_*.csv"))
    if not candidates:
        sys.exit("No relative_pid_*.csv found in logs/ or the current directory.")
    return max(candidates, key=os.path.getmtime)


def summarise(name, values, unit):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        print(f"  {name:28s} no data")
        return
    print(f"  {name:28s} n={len(v):5d}  mean {v.mean():+7.2f} {unit}  "
          f"|mean| {np.abs(v).mean():6.2f}  std {v.std():6.2f}  p95 |err| {np.percentile(np.abs(v), 95):6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="controller CSV (default: newest in logs/)")
    ap.add_argument("--max-age", type=float, default=0.25,
                    help="only score rows whose detection is at most this old, seconds (default 0.25)")
    ap.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = ap.parse_args()

    path = args.csv or find_latest_csv()
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"{path} is missing evaluation columns {missing}; "
                 "it was written by a controller version without ground-truth logging.")

    t0 = df["time"].iloc[0]
    df["t"] = df["time"] - t0
    have_truth = df["gt_dist_m"].notna()
    fresh = have_truth & df["dope_dist_m"].notna() & (df["dope_age_s"] <= args.max_age)
    ev = df[fresh].copy()

    print(f"CSV: {path}")
    print(f"rows: {len(df)} total, {int(have_truth.sum())} with car truth, "
          f"{len(ev)} scored (detection age <= {args.max_age:.2f} s)")
    if len(ev) == 0:
        sys.exit("Nothing to score: no rows with both a fresh detection and car truth.")
    if "stage" in df.columns:
        print("stages scored:", ", ".join(f"{k}={v}" for k, v in ev["stage"].value_counts().items()))
    print("errors (DOPE minus truth):")
    summarise("range error", ev["dope_dist_err_m"], "m")
    summarise("range error, relative", 100.0 * ev["dope_dist_err_m"] / ev["gt_dist_m"], "%")
    summarise("world position error (3D)", ev["dope_pos_err_m"], "m")
    summarise("heading error", ev["dope_yaw_err_deg"], "deg")
    det_frac = fresh[have_truth].mean() if have_truth.any() else float("nan")
    print(f"  fresh detection available on {100 * det_frac:.0f}% of rows with truth")

    if args.no_show:
        matplotlib.use("Agg")
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"DOPE vs ground truth  ({os.path.basename(path)}, detections <= {args.max_age:.2f} s old)")

    a = ax[0, 0]
    a.plot(df.loc[have_truth, "t"], df.loc[have_truth, "gt_dist_m"], "k-", lw=1, label="truth range")
    a.plot(ev["t"], ev["dope_dist_m"], "g.", ms=3, label="DOPE range")
    a.set_xlabel("time [s]"); a.set_ylabel("range [m]"); a.set_title("Range: DOPE vs truth"); a.legend(); a.grid(alpha=0.3)

    # One dot per scored row: how far DOPE's range was off (DOPE - truth,
    # + = DOPE says too far) at that true distance. A car parked at a
    # fixed distance stacks all its rows into one vertical stripe.
    a = ax[0, 1]
    sc = a.scatter(ev["gt_dist_m"], ev["dope_dist_err_m"], s=6, c=ev["t"], cmap="viridis")
    fig.colorbar(sc, ax=a, label="time [s]")
    a.axhline(0, color="k", lw=0.8)
    r = np.linspace(0, max(1.0, float(ev["gt_dist_m"].max())), 50)
    a.plot(r, 0.05 * r, "k:", lw=0.8, label="±5 % of range")
    a.plot(r, -0.05 * r, "k:", lw=0.8)
    a.legend(loc="upper left")
    a.set_xlabel("truth range [m]"); a.set_ylabel("DOPE - truth range [m]"); a.set_title("Range error vs range"); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(ev["t"], ev["dope_pos_err_m"], "r.", ms=3)
    a.set_xlabel("time [s]"); a.set_ylabel("3D position error [m]"); a.set_title("World-frame target position error"); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(ev["t"], ev["dope_yaw_err_deg"], "b.", ms=3)
    a.axhline(0, color="k", lw=0.8)
    a.set_ylim(-180, 180)
    a.set_xlabel("time [s]"); a.set_ylabel("heading error [deg]"); a.set_title("Car heading error (DOPE - truth)"); a.grid(alpha=0.3)

    # Where things were, seen from above. Grey lines join each DOPE
    # estimate to where the car really was at that moment, so their length
    # is the horizontal position error.
    a = ax[2, 0]
    for _, row in ev.iloc[::5].iterrows():
        a.plot([row["target_x"], row["car_gt_x"]], [row["target_y"], row["car_gt_y"]], "-", color="0.8", lw=0.5, zorder=1)
    a.plot(df["drone_x"], df["drone_y"], "c-", lw=1, label="drone")
    a.plot(df["drone_x"].iloc[-1], df["drone_y"].iloc[-1], "c^", ms=9, label="drone (end)")
    gt = df.loc[have_truth]
    a.plot(gt["car_gt_x"], gt["car_gt_y"], "k-", lw=1.5, label="car truth")
    a.plot(gt["car_gt_x"].iloc[0], gt["car_gt_y"].iloc[0], "ko", ms=6, mfc="w", label="car start")
    a.plot(gt["car_gt_x"].iloc[-1], gt["car_gt_y"].iloc[-1], "ks", ms=6, label="car end")
    a.plot(ev["target_x"], ev["target_y"], "g.", ms=3, label="DOPE target", zorder=3)
    a.set_aspect("equal", adjustable="datalim")
    a.set_xlabel("east [m]"); a.set_ylabel("north [m]"); a.set_title("Top-down tracks (world frame)"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[2, 1]
    err = ev["dope_dist_err_m"].to_numpy()
    a.hist(err[np.isfinite(err)], bins=40, color="g", alpha=0.7)
    a.axvline(0, color="k", lw=0.8)
    a.set_xlabel("range error [m]"); a.set_ylabel("rows"); a.set_title("Range error distribution"); a.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.splitext(path)[0] + "_dope_eval.png"
    fig.savefig(out, dpi=120)
    print(f"figure saved: {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
