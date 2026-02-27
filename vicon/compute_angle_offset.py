#!/usr/bin/env python3
"""
Compute (per timestep) rotation about global Y and final orientation angle w.r.t global Y
for every episode CSV in a directory.

For each CSV:
- reads axis-angle rotation vector columns (default: robot0_eef_rot_axis_angle_0/1/2)
- computes:
  1) euler_y_deg_xyz: Y component of rot.as_euler('xyz', degrees=True)
  2) angle_between_frame_y_and_global_y_deg: angle between frame's +Y axis and global +Y
     (0 deg = aligned, 180 deg = anti-aligned)

Outputs:
- <out_dir>/angles_all_episodes.csv  (per-timestep, all files)
- <out_dir>/summary_final_angles.csv (one row per file: final angles)
"""

import argparse
import os
import glob
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(s))]


def find_rotvec_cols(df: pd.DataFrame, base: Optional[str] = None) -> Tuple[str, str, str]:
    cols = df.columns

    if base is not None:
        c0, c1, c2 = f"{base}_0", f"{base}_1", f"{base}_2"
        if all(c in cols for c in (c0, c1, c2)):
            return c0, c1, c2
        raise ValueError(f"Requested base '{base}' not found as {c0},{c1},{c2} in columns.")

    # common default in your files
    default = ("robot0_eef_rot_axis_angle_0", "robot0_eef_rot_axis_angle_1", "robot0_eef_rot_axis_angle_2")
    if all(c in cols for c in default):
        return default

    # try a few common alternatives
    patterns = [
        ("eef_rot_axis_angle_0", "eef_rot_axis_angle_1", "eef_rot_axis_angle_2"),
        ("rot_axis_angle_0", "rot_axis_angle_1", "rot_axis_angle_2"),
        ("rotvec_0", "rotvec_1", "rotvec_2"),
    ]
    for c0, c1, c2 in patterns:
        if all(c in cols for c in (c0, c1, c2)):
            return c0, c1, c2

    # suffix inference
    candidates0 = [c for c in cols if c.lower().endswith("_0") and ("rot" in c.lower())]
    for c0 in candidates0:
        base2 = c0[:-2]
        c1, c2 = base2 + "_1", base2 + "_2"
        if c1 in cols and c2 in cols:
            return c0, c1, c2

    raise ValueError(
        "Could not find axis-angle/rotvec columns. "
        f"First 50 columns: {list(cols)[:50]}"
    )


def pick_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("timestamp", "Timestamp", "time", "t"):
        if c in df.columns:
            return c
    return None


def compute_angles(df: pd.DataFrame, rotvec_cols: Tuple[str, str, str]) -> pd.DataFrame:
    c0, c1, c2 = rotvec_cols
    rv = df[[c0, c1, c2]].to_numpy(dtype=float)

    rot = R.from_rotvec(rv)

    # (1) Euler y-angle using intrinsic xyz convention
    eul_xyz = rot.as_euler("xyz", degrees=True)
    euler_y_deg = eul_xyz[:, 1]

    # (2) angle between frame y-axis and global y-axis
    mats = rot.as_matrix()            # (N,3,3)
    y_axis_world = mats[:, :, 1]      # frame's y axis expressed in world
    cosang = np.clip(y_axis_world[:, 1], -1.0, 1.0)  # dot with [0,1,0]
    y_align_deg = np.degrees(np.arccos(cosang))

    out = df.copy()
    out["euler_y_deg_xyz"] = euler_y_deg
    out["angle_between_frame_y_and_global_y_deg"] = y_align_deg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory containing episode_*.csv files")
    ap.add_argument("--glob", default="*.csv", help="Glob pattern inside in_dir (default: *.csv)")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: <in_dir>/angles_out)")
    ap.add_argument(
        "--rotvec_base",
        default=None,
        help="Optional base name for rotvec columns, e.g. robot0_eef_rot_axis_angle (will use *_0/1/2)",
    )
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(in_dir, "angles_out")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, args.glob)), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No files matched: {os.path.join(in_dir, args.glob)}")

    all_rows = []
    summary_rows = []

    for f in files:
        df = pd.read_csv(f)
        rotvec_cols = find_rotvec_cols(df, base=args.rotvec_base)
        time_col = pick_time_col(df)

        out = compute_angles(df, rotvec_cols)
        out.insert(0, "file", os.path.basename(f))

        keep = ["file"]
        if time_col:
            keep.append(time_col)
        keep += ["euler_y_deg_xyz", "angle_between_frame_y_and_global_y_deg"]
        per_ts = out[keep].copy()
        all_rows.append(per_ts)

        final = out.iloc[-1]
        summary_rows.append(
            {
                "file": os.path.basename(f),
                "n_timesteps": len(out),
                "rotvec_cols_used": ",".join(rotvec_cols),
                "final_euler_y_deg_xyz": float(final["euler_y_deg_xyz"]),
                "final_angle_frame_y_vs_global_y_deg": float(final["angle_between_frame_y_and_global_y_deg"]),
                "final_time": float(final[time_col]) if time_col else np.nan,
            }
        )

        print(
            f"{os.path.basename(f)} | N={len(out)} | "
            f"final Euler y={summary_rows[-1]['final_euler_y_deg_xyz']:.3f} deg | "
            f"final angle(frame y, global y)={summary_rows[-1]['final_angle_frame_y_vs_global_y_deg']:.3f} deg"
        )

    all_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    all_path = os.path.join(out_dir, "angles_all_episodes.csv")
    summary_path = os.path.join(out_dir, "summary_final_angles.csv")
    all_df.to_csv(all_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nWrote:")
    print(f"  {all_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()