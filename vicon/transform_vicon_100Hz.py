#!/usr/bin/env python3
import os, re, glob
import numpy as np
import pandas as pd
from io import StringIO
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import butter, filtfilt

# ----------------------------
# Paths
# ----------------------------
in_dir  = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_trimmed_3/"  # 100 Hz camera
out_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_tcp_60hz_aa/"     # output
os.makedirs(out_dir, exist_ok=True)

# ----------------------------
# Transform cam -> tcp
# ----------------------------
t_cam_tcp_m = np.array([-0.01935615, -0.19549504, -0.09199475], dtype=np.float64)  # meters
R_cam_tcp = Rotation.identity()  # keep identity unless you have measured rotation offset

POS_IN_SCALE  = 1e-3  # mm -> m
POS_OUT_SCALE = 1e3   # m -> mm

# ----------------------------
# Resample target
# ----------------------------
OUT_HZ = 60.0

# ----------------------------
# Anti-alias filter (zero-phase)
# ----------------------------
# choose <= ~15 Hz to reduce jitter without killing motion dynamics.
CUTOFF_HZ = 12.0
FILTER_ORDER = 4

POS_DECIMALS = 6
QUAT_DECIMALS = 10

# ----------------------------
# Helpers
# ----------------------------
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def find_header_idx(lines):
    for i, ln in enumerate(lines):
        low = ln.lower()
        if ("frame" in low) and ("tx" in low) and ("ty" in low) and ("tz" in low) and ("rw" in low):
            return i
    raise RuntimeError("Could not find header line containing Frame, TX, TY, TZ, RX, RY, RZ, RW")

def load_raw_preserve_exact(csv_path):
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    hdr_idx = find_header_idx(lines)
    data_start = hdr_idx + 2  # header + units row
    header_lines = lines[:data_start]
    colnames = [c.strip() for c in lines[hdr_idx].strip().split(",")]
    df = pd.read_csv(StringIO("".join(lines[data_start:])), names=colnames, header=None, dtype=str)
    return header_lines, df

def to_float(col):
    return pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)

def quat_fix_continuity(q):
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i-1], q[i]) < 0:
            q[i] *= -1.0
    return q

def butter_filtfilt(x, fs, cutoff, order):
    if len(x) < (order * 6 + 10):  # avoid filtfilt blowing up on very short seq
        return x
    nyq = 0.5 * fs
    wn = min(0.99, cutoff / nyq)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x, axis=0)

def fmt(x, d):
    if np.isnan(x): return ""
    return f"{x:.{d}f}"

def build_time_seconds(df):
    # Prefer Frame/Sub Frame if present
    if "Frame" in df.columns:
        frame = to_float(df["Frame"])
        sub = to_float(df["Sub Frame"]) if "Sub Frame" in df.columns else np.zeros_like(frame)
        # Vicon typically uses Sub Frame = 0..?? within a frame; if it's always 0, fine.
        # Use just (frame + sub) as monotonically increasing sample index.
        idx = frame + sub
        # Convert to seconds using median dt estimate (robust to occasional skips)
        di = np.diff(idx)
        di = di[np.isfinite(di) & (di > 0)]
        if len(di) == 0:
            raise RuntimeError("Could not infer time: Frame is not increasing.")
        # If your data is truly 100 Hz, each step in idx should be ~1.0
        # We map idx units to seconds by assuming 100 Hz when step≈1.
        # More robust: estimate Hz from median step assuming original rate=100:
        # t = (idx - idx[0]) / 100.0
        t = (idx - idx[0]) / 100.0
        return t.astype(np.float64)

    raise RuntimeError("No Frame column; need a time base.")

def robust_drop_nans_keep_time(t, pos, quat):
    good = np.isfinite(t) & np.isfinite(pos).all(axis=1) & np.isfinite(quat).all(axis=1)
    # critical: keep original time spacing, don’t “compress” if there are gaps:
    # we will only use good samples, but t remains actual timestamps.
    t2 = t[good]
    pos2 = pos[good]
    quat2 = quat[good]
    return t2, pos2, quat2

# ----------------------------
# Main
# ----------------------------
csv_files = sorted(glob.glob(os.path.join(in_dir, "*.csv")), key=natural_key)

for csv_path in csv_files:
    base = os.path.basename(csv_path)
    out_path = os.path.join(out_dir, base)

    header_lines, df = load_raw_preserve_exact(csv_path)

    # time base
    t = build_time_seconds(df)

    # cam pose
    pos_cam_m = np.stack([to_float(df["TX"]), to_float(df["TY"]), to_float(df["TZ"])], axis=1) * POS_IN_SCALE
    quat_cam = np.stack([to_float(df["RX"]), to_float(df["RY"]), to_float(df["RZ"]), to_float(df["RW"])], axis=1)

    # remove NaNs but keep correct time axis (no compression artifacts)
    t, pos_cam_m, quat_cam = robust_drop_nans_keep_time(t, pos_cam_m, quat_cam)

    if len(t) < 5:
        print(f"[WARN] {base}: too short after NaN removal.")
        continue

    # normalize + continuity
    quat_cam /= (np.linalg.norm(quat_cam, axis=1, keepdims=True) + 1e-12)
    quat_cam = quat_fix_continuity(quat_cam)

    # cam rotation
    R_world_cam = Rotation.from_quat(quat_cam)

    # tcp at native timestamps
    R_world_tcp = R_world_cam * R_cam_tcp
    pos_tcp_m = pos_cam_m + R_world_cam.apply(t_cam_tcp_m)

    # estimate native sampling rate from time deltas (for filtering)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    fs = 1.0 / np.median(dt)

    # zero-phase anti-alias filter (no lag)
    pos_tcp_m_f = butter_filtfilt(pos_tcp_m, fs, CUTOFF_HZ, FILTER_ORDER)

    # filter rotation by filtering incremental rotvec (angular velocity) then re-integrating
    dR = R_world_tcp[:-1].inv() * R_world_tcp[1:]
    dtheta = dR.as_rotvec()  # radians per step (in SO(3) log map)
    # convert to angular velocity (rad/s) using actual dt per step
    dt_full = np.diff(t)
    omega = dtheta / dt_full[:, None]

    omega_f = butter_filtfilt(omega, fs, CUTOFF_HZ, FILTER_ORDER)

    # integrate back to filtered rotations using true dt
    Rf = [R_world_tcp[0]]
    for k in range(len(omega_f)):
        Rf.append(Rf[-1] * Rotation.from_rotvec(omega_f[k] * dt_full[k]))
    R_world_tcp_f = Rotation.concatenate(Rf)

    quat_tcp_f = R_world_tcp_f.as_quat()
    quat_tcp_f = quat_fix_continuity(quat_tcp_f)

    # resample to 60 Hz
    t0, t1 = t[0], t[-1]
    t_out = np.arange(t0, t1 + 1e-9, 1.0 / OUT_HZ)

    # position interp
    pos_out_m = np.stack([np.interp(t_out, t, pos_tcp_m_f[:, i]) for i in range(3)], axis=1)

    # rotation slerp
    slerp = Slerp(t, Rotation.from_quat(quat_tcp_f))
    R_out = slerp(t_out)
    quat_out = quat_fix_continuity(R_out.as_quat())

    # write back as Vicon-like CSV (mm + quat)
    df_out = pd.DataFrame(columns=df.columns, dtype=object)
    df_out = df_out.reindex(range(len(t_out)))

    if "Frame" in df_out.columns:
        df_out["Frame"] = [str(i) for i in range(len(t_out))]
    if "Sub Frame" in df_out.columns:
        df_out["Sub Frame"] = ["0"] * len(t_out)

    pos_out_mm = pos_out_m * POS_OUT_SCALE
    for i, col in enumerate(["TX", "TY", "TZ"]):
        df_out[col] = [fmt(v, POS_DECIMALS) for v in pos_out_mm[:, i]]
    for i, col in enumerate(["RX", "RY", "RZ", "RW"]):
        df_out[col] = [fmt(v, QUAT_DECIMALS) for v in quat_out[:, i]]

    with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
        for ln in header_lines:
            f.write(ln)
        df_out.to_csv(f, index=False, header=False)

    print(f"Saved: {out_path}  (fs~{fs:.2f} Hz -> {OUT_HZ} Hz, cutoff={CUTOFF_HZ} Hz)")
