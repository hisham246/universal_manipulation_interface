#!/usr/bin/env python3
import pathlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
CSV_PATH = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/gopro_to_tcp_transform.csv"  # change if running elsewhere
POS_SCALE = 1e-3   # mm -> m (your file shows "mm" in the units row)

# This CSV layout (like your file) has two pose blocks:
#   Block A: RX RY RZ RW TX TY TZ   (GoPro)
#   Block B: RX.1 RY.1 RZ.1 RW.1 TX.1 TY.1 TZ.1 (TCP)
CAM_SUFFIX = ""     # "" uses RX..TZ
TCP_SUFFIX = ".1"   # ".1" uses RX.1..TZ.1
# If you ever need to swap them, swap these suffixes.

# Outlier rejection
MAD_K = 3.5         # bigger = more permissive
MAD_SCALE = 1.4826  # to make MAD comparable to std under Gaussian noise


def read_vicon_two_body_csv(csv_path: str) -> pd.DataFrame:
    p = pathlib.Path(csv_path)
    lines = p.read_text(errors="ignore").splitlines()

    header_idx = None
    for i, ln in enumerate(lines):
        low = ln.strip().lower()
        if low.startswith("frame,") and ("tx" in low) and ("rw" in low):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"Could not find header row starting with 'Frame,' in: {csv_path}")

    df = pd.read_csv(p, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]

    # Drop the units row (has NaN Frame + 'mm' strings) and any other junk rows
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df = df.dropna(subset=["Frame"]).copy()

    return df


def get_pos_quat(df: pd.DataFrame, suffix: str, pos_scale: float):
    rx, ry, rz, rw = f"RX{suffix}", f"RY{suffix}", f"RZ{suffix}", f"RW{suffix}"
    tx, ty, tz = f"TX{suffix}", f"TY{suffix}", f"TZ{suffix}"

    for c in [tx, ty, tz]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[rx, ry, rz, rw, tx, ty, tz]).copy()

    pos = df[[tx, ty, tz]].to_numpy(np.float64) * pos_scale
    quat = df[[rx, ry, rz, rw]].to_numpy(np.float64)
    quat /= (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12)
    return pos, quat


def robust_mean_translation(t: np.ndarray):
    # Robust center by median, reject by MAD on Euclidean residual
    med = np.median(t, axis=0)
    resid = np.linalg.norm(t - med, axis=1)

    med_r = np.median(resid)
    mad_r = np.median(np.abs(resid - med_r)) + 1e-12
    thr = med_r + MAD_K * MAD_SCALE * mad_r

    inliers = resid < thr
    t_in = t[inliers]
    mean = t_in.mean(axis=0)
    std = t_in.std(axis=0)

    return mean, std, inliers


def main():
    df = read_vicon_two_body_csv(CSV_PATH)

    pos_cam, quat_cam = get_pos_quat(df, CAM_SUFFIX, POS_SCALE)
    pos_tcp, quat_tcp = get_pos_quat(df, TCP_SUFFIX, POS_SCALE)

    N = min(len(pos_cam), len(pos_tcp))
    pos_cam, quat_cam = pos_cam[:N], quat_cam[:N]
    pos_tcp, quat_tcp = pos_tcp[:N], quat_tcp[:N]

    r_cam = Rotation.from_quat(quat_cam)

    # Translation of tcp expressed in camera frame:
    # t_cam_tcp_i = R_cam_i^T (p_tcp_i - p_cam_i)
    t_cam_tcp = r_cam.inv().apply(pos_tcp - pos_cam)

    mean, std, inliers = robust_mean_translation(t_cam_tcp)

    print("\n=== Estimated camera -> tcp translation (expressed in camera frame) ===")
    print(f"Samples: {N} | Inliers: {inliers.sum()} ({inliers.mean()*100:.1f}%)")
    print(f"t_cam_tcp mean [m]: {mean}")
    print(f"t_cam_tcp std  [m]: {std}")

    print("\nPaste into your main script as:")
    print(f"pose_cam_tcp = np.array([{mean[0]: .8f}, {mean[1]: .8f}, {mean[2]: .8f}, 0.0, 0.0, 0.0], dtype=np.float64)")
    print("T_cam_tcp = pose_to_mat(pose_cam_tcp)")

    # Optional quick sanity: compare to your previous assumption (x=0, y=0.086, z=0.21965)
    prev = np.array([0.0, 0.086, 0.01465 + 0.205])
    print("\nDelta vs your old guess [m] (new - old):", mean - prev)


if __name__ == "__main__":
    main()