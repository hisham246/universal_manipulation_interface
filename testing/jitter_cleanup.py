#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import argparse
import matplotlib.pyplot as plt
import re, glob, os

def prompt_save():
    while True:
        resp = input("Save this episode? [y = save / n = discard / q = quit]: ").strip().lower()
        if resp in ["y", "n", "q"]:
            return resp
        print("Invalid input. Please enter y, n, or q.")

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def mad(x):
    return np.median(np.abs(x - np.median(x)))


def remove_tracking_jitter(
    timestamps,
    positions,
    vel_k=6.0,
    acc_k=6.0,
    max_burst_len=5,
    direction_angle_deg=120.0,
    interpolate=True,
):
    t = timestamps
    p = positions
    N = len(t)

    # Velocity
    dt = np.diff(t)
    dp = np.diff(p, axis=0)
    v = np.linalg.norm(dp, axis=1) / dt

    # Acceleration
    acc = np.abs(np.diff(v)) / dt[1:]

    # Robust thresholds
    v_th = np.median(v) + vel_k * mad(v)
    a_th = np.median(acc) + acc_k * mad(acc)

    vel_out = np.where(v > v_th)[0]
    acc_out = np.where(acc > a_th)[0] + 1

    candidate = set(vel_out) | set(acc_out)

    # Direction flip test
    ang_th = np.deg2rad(direction_angle_deg)
    for i in range(1, N - 1):
        d1 = p[i] - p[i - 1]
        d2 = p[i + 1] - p[i]
        if np.linalg.norm(d1) < 1e-6 or np.linalg.norm(d2) < 1e-6:
            continue
        cosang = np.dot(d1, d2) / (
            np.linalg.norm(d1) * np.linalg.norm(d2)
        )
        cosang = np.clip(cosang, -1.0, 1.0)
        if np.arccos(cosang) > ang_th:
            candidate.add(i)

    candidate = sorted(candidate)

    # Group into bursts
    bursts = []
    if candidate:
        cur = [candidate[0]]
        for i in candidate[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                bursts.append(cur)
                cur = [i]
        bursts.append(cur)

    # Remove only short bursts
    remove_idx = set()
    for b in bursts:
        if len(b) <= max_burst_len:
            for i in b:
                remove_idx.add(i)
                remove_idx.add(i + 1)

    remove_idx = sorted(i for i in remove_idx if i < N)

    mask = np.ones(N, dtype=bool)
    mask[remove_idx] = False

    # Interpolation
    if interpolate and np.any(~mask):
        p_clean = p.copy()
        for d in range(3):
            f = interp1d(
                t[mask],
                p[mask, d],
                kind="linear",
                fill_value="extrapolate",
            )
            p_clean[:, d] = f(t)
    else:
        p_clean = p[mask]

    return mask, p_clean, remove_idx



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_dir", help="Input ORB-SLAM CSV")
    parser.add_argument("--out_dir", required=True, help="Output cleaned CSV")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()


    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")), key=natural_key)

    for csv_file in csv_files:

        df = pd.read_csv(csv_file)

        # Required columns
        for c in ["timestamp", "robot0_eef_pos_0", "robot0_eef_pos_1", "robot0_eef_pos_2"]:
            if c not in df.columns:
                raise ValueError(f"Missing column: {c}")

        t = df["timestamp"].to_numpy()
        p = df[["robot0_eef_pos_0", "robot0_eef_pos_1", "robot0_eef_pos_2"]].to_numpy()

        mask, p_clean, removed = remove_tracking_jitter(
            t,
            p,
            vel_k=6,
            acc_k=6,
            max_burst_len=4,
            interpolate=True,
        )

        df_out = df.copy()
        df_out[["x", "y", "z"]] = p_clean
        df_out["jitter_removed"] = ~mask

        filename = os.path.basename(csv_file)
        out_path = os.path.join(args.out_dir, filename)

        # df_out.to_csv(out_path, index=False)

        # print(f"Removed {len(removed)} jittery samples")
        # print(f"Saved cleaned trajectory to {out_path}")

        decision = "y"  # default if no plotting

        if args.plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(p[:, 0], p[:, 1], p[:, 2], "r--", alpha=0.5, label="Original")
            ax.plot(p_clean[:, 0], p_clean[:, 1], p_clean[:, 2], "b", label="Cleaned")
            ax.legend()
            ax.set_title(f"Trajectory: {os.path.basename(csv_file)}")
            plt.show()

            decision = prompt_save()


        if decision == "y":
            df_out.to_csv(out_path, index=False)
            print(f"Saved cleaned trajectory to {out_path}")

        elif decision == "n":
            print("Discarded episode, moving to next.")

        elif decision == "q":
            print("User requested exit. Stopping.")
            break

if __name__ == "__main__":
    main()