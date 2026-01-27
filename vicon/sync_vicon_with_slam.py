#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd

# ============================================================
# Helpers: file sorting / pairing
# ============================================================
def natural_key(path):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", os.path.basename(path))]

def extract_index(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None

def build_index_map(files):
    mp = {}
    for f in files:
        idx = extract_index(f)
        if idx is None:
            continue
        mp[idx] = f
    return mp

# ============================================================
# Quaternion utilities
# ============================================================
def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n

def quat_slerp(q0, q1, u):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        q = q0 + u * (q1 - q0)
        return quat_normalize(q)

    theta = np.arccos(dot)
    s0 = np.sin((1.0 - u) * theta) / (np.sin(theta) + 1e-12)
    s1 = np.sin(u * theta) / (np.sin(theta) + 1e-12)
    return s0 * q0 + s1 * q1

# ============================================================
# Vicon reader + writer (preserve preamble)
# ============================================================
def read_vicon_csv(path):
    lines = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()

    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Frame,Sub Frame"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find Vicon header line 'Frame,Sub Frame,...' in {path}")

    df = pd.read_csv(path, skiprows=header_idx)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Frame", "TX", "TY", "TZ", "RW", "RX", "RY", "RZ"]).reset_index(drop=True)

    # FPS: try to read from first few lines, else default 100
    fps = 100.0
    for ln in lines[:5]:
        ln2 = ln.strip().strip(",")
        if ln2.isdigit():
            fps = float(ln2)
            break

    t = (df["Frame"].to_numpy() - df["Frame"].iloc[0]) / fps
    p = df[["TX", "TY", "TZ"]].to_numpy() / 1000.0  # mm -> m
    q = df[["RX", "RY", "RZ", "RW"]].to_numpy()      # xyzw
    q = quat_normalize(q)

    return t, p, q, fps, df, lines, header_idx

def write_vicon_like_original(out_path, original_lines, header_idx, new_df):
    preamble = original_lines[:header_idx]

    with open(out_path, "w", encoding="utf-8") as f:
        for ln in preamble:
            f.write(ln.rstrip("\n") + "\n")

        f.write(",".join(new_df.columns) + "\n")

        for _, row in new_df.iterrows():
            vals = []
            for c in new_df.columns:
                v = row[c]
                if c in ["Frame", "Sub Frame"]:
                    vals.append(str(int(round(v))))
                else:
                    vals.append(f"{float(v):.10f}".rstrip("0").rstrip("."))
            f.write(",".join(vals) + "\n")

# ============================================================
# Resampling (no synchronization / matching)
# ============================================================
def resample_vicon_to_N(t_v, p_v, q_v, N, mode, t_s=None):
    """
    mode:
      - "linspace": query times are linspace(t_v[0], t_v[-1], N)
      - "slam_scaled": use SLAM relative time shape but scaled to Vicon time span
                      t_query = t_v[0] + (t_s - t_s[0]) * (t_v[-1]-t_v[0])/(t_s[-1]-t_s[0])
                      Requires t_s.
    Returns: p_out (N,3), q_out (N,4), t_query (N,)
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if len(t_v) < 2:
        raise ValueError("Vicon must have at least 2 samples to resample.")
    if not np.all(np.diff(t_v) > 0):
        # enforce strictly increasing for searchsorted/interp
        order = np.argsort(t_v)
        t_v = t_v[order]
        p_v = p_v[order]
        q_v = q_v[order]

    if N == 1:
        t_query = np.array([t_v[0]], dtype=np.float64)
    else:
        if mode == "linspace":
            t_query = np.linspace(t_v[0], t_v[-1], N, dtype=np.float64)
        elif mode == "slam_scaled":
            if t_s is None or len(t_s) < 2:
                raise ValueError("slam_scaled mode requires SLAM timestamps with at least 2 samples.")
            ts = np.asarray(t_s, dtype=np.float64)
            ts = ts - ts[0]
            dur_s = float(ts[-1])
            dur_v = float(t_v[-1] - t_v[0])
            if dur_s <= 1e-12:
                # degenerate SLAM time: fall back to linspace
                t_query = np.linspace(t_v[0], t_v[-1], N, dtype=np.float64)
            else:
                t_query = t_v[0] + ts * (dur_v / dur_s)
                # ensure same length N as SLAM
                if len(t_query) != N:
                    t_query = np.linspace(t_v[0], t_v[-1], N, dtype=np.float64)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # position interp (meters)
    p_out = np.vstack([np.interp(t_query, t_v, p_v[:, d]) for d in range(3)]).T

    # quaternion slerp
    q_out = np.zeros((len(t_query), 4), dtype=np.float64)
    for i, tq in enumerate(t_query):
        j = np.searchsorted(t_v, tq)
        if j <= 0:
            q_out[i] = q_v[0]
        elif j >= len(t_v):
            q_out[i] = q_v[-1]
        else:
            tL, tR = t_v[j - 1], t_v[j]
            u = 0.0 if (tR - tL) < 1e-12 else (tq - tL) / (tR - tL)
            q_out[i] = quat_slerp(q_v[j - 1], q_v[j], float(u))
    q_out = quat_normalize(q_out)

    return p_out, q_out, t_query

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Inputs
    slam_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/slam_segmented"
    vicon_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_trimmed_3"

    # Output
    out_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_resampled_to_slam_3/"
    os.makedirs(out_dir, exist_ok=True)

    # Resample mode:
    #   "linspace"    -> ignores SLAM timestamps, only matches sample count
    #   "slam_scaled" -> uses SLAM time shape but scaled into Vicon time span (still no onset/corr sync)
    MODE = "slam_scaled"

    slam_files = sorted(glob.glob(os.path.join(slam_dir, "*.csv")), key=natural_key)
    vicon_files = sorted(glob.glob(os.path.join(vicon_dir, "*.csv")), key=natural_key)

    slam_map = build_index_map(slam_files)
    vicon_map = build_index_map(vicon_files)

    common_idxs = sorted(set(slam_map.keys()) & set(vicon_map.keys()))
    print(f"Found {len(common_idxs)} paired episodes.")

    saved_count = 0
    skipped = []

    for idx in common_idxs:
        slam_csv = slam_map[idx]
        vicon_csv = vicon_map[idx]

        # SLAM: only need N (and optionally timestamps for slam_scaled)
        slam = pd.read_csv(slam_csv)
        if "timestamp" not in slam.columns:
            skipped.append((idx, "no timestamp in SLAM"))
            continue
        t_s = slam["timestamp"].to_numpy()
        if len(t_s) < 1:
            skipped.append((idx, "empty SLAM"))
            continue
        N = int(len(t_s))

        # Vicon
        try:
            t_v, p_v, q_v, fps_v, vicon_df, vicon_lines, header_idx = read_vicon_csv(vicon_csv)
        except Exception as e:
            skipped.append((idx, f"vicon read failed: {e}"))
            continue

        if len(t_v) < 2:
            skipped.append((idx, "vicon too short"))
            continue

        try:
            p_rs, q_rs, t_query = resample_vicon_to_N(t_v, p_v, q_v, N, mode=MODE, t_s=t_s)
        except Exception as e:
            skipped.append((idx, f"resample failed: {e}"))
            continue

        # Build output DF in original-like format
        frame0 = int(vicon_df["Frame"].iloc[0])
        sub0 = int(vicon_df["Sub Frame"].iloc[0]) if "Sub Frame" in vicon_df.columns else 0

        # Create frame numbers consistent with Vicon fps and chosen query times
        frame_numbers = frame0 + np.round(t_query * fps_v).astype(int)

        new_df = pd.DataFrame({
            "Frame": frame_numbers,
            "Sub Frame": np.full_like(frame_numbers, sub0),
            "TX": p_rs[:, 0] * 1000.0,  # m -> mm
            "TY": p_rs[:, 1] * 1000.0,
            "TZ": p_rs[:, 2] * 1000.0,
            "RX": q_rs[:, 0],
            "RY": q_rs[:, 1],
            "RZ": q_rs[:, 2],
            "RW": q_rs[:, 3],
        })

        out_path = os.path.join(out_dir, os.path.basename(vicon_csv))
        write_vicon_like_original(out_path, vicon_lines, header_idx, new_df)

        saved_count += 1
        print(f"[{idx}] Saved: {out_path} (N={N}, mode={MODE})")

    print(f"Saved {saved_count}/{len(common_idxs)}")
    if skipped:
        print("First 20 skipped:")
        for k, why in skipped[:20]:
            print(k, why)