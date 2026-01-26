import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Helpers: file sorting / pairing
# ============================================================
def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", os.path.basename(path))]

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
# Quaternion utilities (same as yours)
# ============================================================
def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n

def quat_slerp(q0, q1, u):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    dot = np.dot(q0, q1)
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
# Vicon reader (same as yours)
# ============================================================
def robust_speed(t, p):
    dt = np.diff(t)
    v = np.diff(p, axis=0) / dt[:, None]
    s = np.linalg.norm(v, axis=1)
    ts = 0.5 * (t[:-1] + t[1:])
    return ts, s

def find_motion_onset_time(t, p, hold_sec=0.15, thresh_k=6.0):
    """
    Returns onset time based on when speed persistently rises above a robust threshold.
    No smoothing/filtering; uses MAD for a robust threshold.
    """
    ts, s = robust_speed(t, p)

    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-12
    thresh = med + thresh_k * 1.4826 * mad  # robust "std" estimate

    # require speed above threshold for a short continuous duration
    dt_est = np.median(np.diff(ts)) if len(ts) > 1 else 1e-2
    hold_n = max(1, int(round(hold_sec / dt_est)))

    above = s > thresh
    # find first index where we have hold_n consecutive Trues
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= hold_n:
            onset_idx = i - hold_n + 1
            return float(ts[onset_idx])

    # fallback: if we never exceed, just return start
    return float(ts[0]) if len(ts) else float(t[0])

def plot_raw_translation_2d_and_time(idx, t_s, p_s, t_v, p_v, slam_name="", vicon_name=""):
    """
    Raw comparison only:
      - No alignment
      - No trimming
      - No resampling
      - Only unit conversion (Vicon mm->m already done in read_vicon_csv)
    Produces:
      (1) XY trajectory overlay
      (2) X/Y/Z vs time (separate subplots)
    """

    # --- 2D XY overlay (top-down) ---
    plt.figure(figsize=(8, 7))
    plt.plot(p_s[:, 0], p_s[:, 1], label="SLAM XY (raw)", linewidth=2)
    plt.plot(p_v[:, 0], p_v[:, 1], label="Vicon XY (raw)", linewidth=1)
    plt.scatter(p_s[0, 0], p_s[0, 1], s=40, label="SLAM start", marker="o")
    plt.scatter(p_v[0, 0], p_v[0, 1], s=40, label="Vicon start", marker="x")
    plt.title(f"[{idx}] Raw XY overlay (no alignment)\n{slam_name}\n{vicon_name}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    # --- XYZ vs time ---
    fig, axs = plt.subplots(3, 1, sharex=False, figsize=(12, 8))

    labels = ["X", "Y", "Z"]
    for i in range(3):
        axs[i].plot(t_s, p_s[:, i], label=f"SLAM {labels[i]} (raw)", linewidth=2)
        axs[i].plot(t_v, p_v[:, i], label=f"Vicon {labels[i]} (raw)", linewidth=1)
        axs[i].set_ylabel(f"{labels[i]} (m)")
        axs[i].grid(True)
        axs[i].legend()

    axs[0].set_title(f"[{idx}] Raw translation vs time (no alignment)")
    axs[-1].set_xlabel("time (s)")

    plt.tight_layout()
    plt.show()


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

# ============================================================
# Matching + resampling (same logic as yours)
# ============================================================
def speed_signature(t, p):
    dt = np.diff(t)
    v = np.diff(p, axis=0) / dt[:, None]
    s = np.linalg.norm(v, axis=1)
    ts = 0.5 * (t[:-1] + t[1:])
    return ts, s

def find_best_vicon_window(t_v, p_v, t_s, p_s, fps_v):
    tsv, sv = speed_signature(t_v, p_v)
    tss, ss = speed_signature(t_s, p_s)

    t_grid = np.arange(0.0, t_s[-1], 1.0 / fps_v)
    ss_100 = np.interp(t_grid, tss, ss)

    a = (ss_100 - ss_100.mean()) / (ss_100.std() + 1e-12)
    b = (sv - sv.mean()) / (sv.std() + 1e-12)

    N = len(a)
    if len(b) < N:
        raise ValueError("Vicon signal shorter than SLAM; cannot match.")

    corr = np.correlate(b, a, mode="valid")
    k = int(np.argmax(corr))
    score = float(corr[k] / N)

    t0 = float(tsv[k])
    t1 = t0 + float(t_s[-1])
    return t0, t1, score

def resample_vicon_to_slam(t_v, p_v, q_v, t0, t_slam):
    t_query = t0 + t_slam

    # position interp (meters)
    p_out = np.vstack([np.interp(t_query, t_v, p_v[:, d]) for d in range(3)]).T
    # position interp (meters) with clamping
    # p_out = np.vstack([
    #     np.interp(t_query, t_v, p_v[:, d], left=p_v[0, d], right=p_v[-1, d])
    #     for d in range(3)
    # ]).T

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
# Plot helpers (same as yours, but packaged)
# ============================================================
def interp_pos(t_src, p_src, t_query):
    return np.vstack([np.interp(t_query, t_src, p_src[:, d]) for d in range(3)]).T

def speed_on_grid(t, p, t_grid):
    p_g = interp_pos(t, p, t_grid)
    dt = np.diff(t_grid)
    v = np.diff(p_g, axis=0) / dt[:, None]
    s = np.linalg.norm(v, axis=1)
    t_mid = 0.5 * (t_grid[:-1] + t_grid[1:])
    return t_mid, s

# ============================================================
# Writer: save new Vicon CSV in ORIGINAL FORMAT (XYZ + QUAT)
# - Keep header/preamble lines exactly
# - Keep columns: Frame,Sub Frame,TX,TY,TZ,RX,RY,RZ,RW
# - Units: TX/TY/TZ in mm (as original), quats same
# ============================================================
def write_vicon_like_original(out_path, original_lines, header_idx, new_df):
    """
    original_lines: list of lines from the original file
    header_idx: line index where "Frame,Sub Frame,..." starts
    new_df: dataframe containing numeric columns with same names as original
    """
    # Keep preamble (everything before header line)
    preamble = original_lines[:header_idx]

    # Write preamble + header + data using CSV formatting
    with open(out_path, "w", encoding="utf-8") as f:
        for ln in preamble:
            f.write(ln.rstrip("\n") + "\n")

        # Header line (exactly as in original file)
        f.write(",".join(new_df.columns) + "\n")

        # Data rows
        # Preserve typical Vicon formatting: Frame/Sub Frame as ints, the rest as floats
        for _, row in new_df.iterrows():
            vals = []
            for c in new_df.columns:
                v = row[c]
                if c in ["Frame", "Sub Frame"]:
                    vals.append(str(int(round(v))))
                else:
                    # Enough precision; adjust if you want
                    vals.append(f"{float(v):.10f}".rstrip("0").rstrip("."))
            f.write(",".join(vals) + "\n")

def find_best_vicon_window_early(t_v, p_v, t_s, p_s, fps_v, max_start_sec=2.0):
    tsv, sv = speed_signature(t_v, p_v)
    tss, ss = speed_signature(t_s, p_s)

    t_grid = np.arange(0.0, t_s[-1], 1.0 / fps_v)
    ss_100 = np.interp(t_grid, tss, ss)
    a = (ss_100 - ss_100.mean()) / (ss_100.std() + 1e-12)

    b = (sv - sv.mean()) / (sv.std() + 1e-12)

    N = len(a)
    if len(b) < N:
        raise ValueError("Vicon signal shorter than SLAM; cannot match.")

    corr = np.correlate(b, a, mode="valid")

    # only allow offsets whose start time is within the first max_start_sec
    valid_k = []
    for k in range(len(corr)):
        t0 = float(tsv[k])
        if t0 <= float(tsv[0]) + max_start_sec:
            valid_k.append(k)

    if not valid_k:
        raise ValueError("No valid early offsets found.")

    k_best = max(valid_k, key=lambda k: corr[k])
    score = float(corr[k_best] / N)
    t0 = float(tsv[k_best])
    t1 = t0 + float(t_s[-1])
    return t0, t1, score

# ============================================================
# Main loop over all files
# ============================================================
if __name__ == "__main__":
    slam_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/slam_segmented"
    vicon_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_trimmed"
    out_dir  = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_resampled_to_slam_2"
    os.makedirs(out_dir, exist_ok=True)

    slam_files = sorted(glob.glob(os.path.join(slam_dir, "*.csv")), key=natural_key)
    vicon_files = sorted(glob.glob(os.path.join(vicon_dir, "*.csv")), key=natural_key)

    slam_map = build_index_map(slam_files)
    vicon_map = build_index_map(vicon_files)

    common_idxs = sorted(set(slam_map.keys()) & set(vicon_map.keys()))
    missing_slam = sorted(set(vicon_map.keys()) - set(slam_map.keys()))
    missing_vicon = sorted(set(slam_map.keys()) - set(vicon_map.keys()))

    print(f"Found {len(common_idxs)} paired episodes.")
    if missing_slam:
        print(f"Warning: missing SLAM for indices: {missing_slam[:20]}{'...' if len(missing_slam)>20 else ''}")
    if missing_vicon:
        print(f"Warning: missing Vicon for indices: {missing_vicon[:20]}{'...' if len(missing_vicon)>20 else ''}")

    # for idx in common_idxs:
    #     slam_csv = slam_map[idx]
    #     vicon_csv = vicon_map[idx]

    #     # Load SLAM
    #     slam = pd.read_csv(slam_csv)
    #     if "timestamp" not in slam.columns:
    #         print(f"[{idx}] Skip: no timestamp in {slam_csv}")
    #         continue

    #     t_s = slam["timestamp"].to_numpy()
    #     t_s = t_s - t_s[0]

    #     pos_cols = ["robot0_eef_pos_0", "robot0_eef_pos_1", "robot0_eef_pos_2"]
    #     if not all(c in slam.columns for c in pos_cols):
    #         print(f"[{idx}] Skip: missing SLAM pos cols in {slam_csv}")
    #         continue
    #     p_s = slam[pos_cols].to_numpy()

    #     # Load Vicon
    #     t_v, p_v, q_v, fps_v, vicon_df, vicon_lines, header_idx = read_vicon_csv(vicon_csv)

    #     # Onset-based t0 (preferred)
    #     t_on_s = find_motion_onset_time(t_s, p_s, hold_sec=0.15, thresh_k=6.0)
    #     t_on_v = find_motion_onset_time(t_v, p_v, hold_sec=0.15, thresh_k=6.0)

    #     t0 = t_on_v - t_on_s
    #     dur = float(t_s[-1])

    #     print(f"[{idx}] onset_s={t_on_s:.3f}s onset_v={t_on_v:.3f}s => t0={t0:.3f}s  dur={dur:.3f}s")

    saved_count = 0
    skipped = []

    for idx in common_idxs:
        slam_csv = slam_map[idx]
        vicon_csv = vicon_map[idx]

        slam = pd.read_csv(slam_csv)
        if "timestamp" not in slam.columns:
            skipped.append((idx, "no timestamp"))
            continue

        pos_cols = ["robot0_eef_pos_0","robot0_eef_pos_1","robot0_eef_pos_2"]
        if not all(c in slam.columns for c in pos_cols):
            skipped.append((idx, "missing pos cols"))
            continue

        t_s = slam["timestamp"].to_numpy()
        t_s = t_s - t_s[0]
        p_s = slam[pos_cols].to_numpy()

        t_v, p_v, q_v, fps_v, vicon_df, vicon_lines, header_idx = read_vicon_csv(vicon_csv)

        t_on_s = find_motion_onset_time(t_s, p_s)
        t_on_v = find_motion_onset_time(t_v, p_v)
        t0 = t_on_v - t_on_s

        eps = 2.0 / fps_v
        if t0 < 0 and abs(t0) <= eps:
            t0 = 0.0

        p_rs, q_rs, t_query = resample_vicon_to_slam(t_v, p_v, q_v, t0, t_s)

        # if you want to allow small out-of-range due to eps, use this:
        if (t_query[0] < t_v[0] - eps) or (t_query[-1] > t_v[-1] + eps):
            skipped.append((idx, f"window out of range t0={t0:.4f}"))
            continue

        # build output df
        frame0 = int(vicon_df["Frame"].iloc[0])
        sub0 = int(vicon_df["Sub Frame"].iloc[0]) if "Sub Frame" in vicon_df.columns else 0
        frame_numbers = frame0 + np.round(t_query * fps_v).astype(int)

        new_df = pd.DataFrame({
            "Frame": frame_numbers,
            "Sub Frame": np.full_like(frame_numbers, sub0),
            "TX": p_rs[:, 0] * 1000.0,
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
        print(f"[{idx}] Saved: {out_path} (rows={len(new_df)})")

    print(f"Saved {saved_count}/{len(common_idxs)}")
    if skipped:
        print("First 20 skipped:")
        for k, why in skipped[:20]:
            print(k, why)



        # # --- RAW PLOTS (before any processing) ---
        # plot_raw_translation_2d_and_time(
        #     idx,
        #     t_s, p_s,
        #     t_v, p_v,
        #     slam_name=os.path.basename(slam_csv),
        #     vicon_name=os.path.basename(vicon_csv),
        # )

        # # Match window
        # try:
        #     t0, t1, score = find_best_vicon_window(t_v, p_v, t_s, p_s, fps_v)
        # except Exception as e:
        #     print(f"[{idx}] Match failed: {e}")
        #     continue

        # print(f"[{idx}] t0={t0:.3f}s dur={t_s[-1]:.3f}s corr={score:.3f}")

        # # Resample Vicon at SLAM timestamps (this guarantees SAME LENGTH)
        # p_rs, q_rs, t_query = resample_vicon_to_slam(t_v, p_v, q_v, t0, t_s)

        # # Safety: ensure coverage
        # if t_query[0] < t_v[0] or t_query[-1] > t_v[-1]:
        #     print(f"[{idx}] Skip: t_query out of Vicon range")
        #     continue

        # try:
        #     # primary: onset-based
        #     t0 = t_on_v - t_on_s
        #     p_rs, q_rs, t_query = resample_vicon_to_slam(t_v, p_v, q_v, t0, t_s)
        #     if t_query[0] < t_v[0] or t_query[-1] > t_v[-1]:
        #         raise ValueError("onset window out of range")
        #     score = np.nan
        # except Exception:
        #     # fallback: early constrained correlation
        #     t0, t1, score = find_best_vicon_window_early(t_v, p_v, t_s, p_s, fps_v, max_start_sec=2.0)
        #     p_rs, q_rs, t_query = resample_vicon_to_slam(t_v, p_v, q_v, t0, t_s)

        # # primary: onset-based
        # t0_onset = float(t_on_v - t_on_s)

        # # if onset suggests a big offset, try early correlation as fallback
        # if abs(t0_onset) > 1.0:
        #     t0, t1, score = find_best_vicon_window_early(t_v, p_v, t_s, p_s, fps_v, max_start_sec=0.5)
        # else:
        #     t0, score = t0_onset, np.nan

        # p_rs, q_rs, t_query = resample_vicon_to_slam(t_v, p_v, q_v, t0, t_s)
        # if t_query[0] < t_v[0] or t_query[-1] > t_v[-1]:
        #     print(f"[{idx}] Skip: window out of range (t0={t0:.3f})")
        #     continue

        # # Resample Vicon at SLAM timestamps using onset-based t0
        # p_rs, q_rs, t_query = resample_vicon_to_slam(t_v, p_v, q_v, t0, t_s)

        # # enforce window coverage: anything outside Vicon range is invalid
        # if t_query[0] < t_v[0] or t_query[-1] > t_v[-1]:
        #     print(f"[{idx}] Skip: onset-based window out of Vicon range")
        #     continue


        # # =========================
        # # PLOTS (block until you close them)
        # # =========================
        # dur = float(t_s[-1])
        # dtv = 1.0 / fps_v
        # t_grid_slam = np.arange(0.0, dur, dtv)
        # t_grid_vwin = t0 + t_grid_slam

        # t_mid_s, s_s = speed_on_grid(t_s, p_s, t_grid_slam)
        # t_mid_v, s_v = speed_on_grid(t_v, p_v, t_grid_vwin)

        # s_s_n = (s_s - s_s.mean()) / (s_s.std() + 1e-12)
        # s_v_n = (s_v - s_v.mean()) / (s_v.std() + 1e-12)

        # plt.figure()
        # plt.plot(t_mid_s, s_s_n, label="SLAM speed (norm)")
        # plt.plot(t_mid_s, s_v_n, label="Vicon-window speed (norm)")
        # plt.title(f"[{idx}] Speed overlay (normalized)")
        # plt.xlabel("time (s) (SLAM-relative)")
        # plt.ylabel("normalized speed")
        # plt.legend()
        # plt.grid(True)

        # fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        # labels = ["x", "y", "z"]
        # for i in range(3):
        #     axs[i].plot(t_s, p_s[:, i], label=f"SLAM pos {labels[i]}")
        #     axs[i].plot(t_s, p_rs[:, i], label=f"Vicon(resampled) pos {labels[i]}")
        #     axs[i].set_ylabel(labels[i])
        #     axs[i].grid(True)
        #     axs[i].legend()
        # axs[-1].set_xlabel("time (s)")
        # fig.suptitle(f"[{idx}] Position overlay (no frame alignment)")

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # ax.plot(p_s[:, 0], p_s[:, 1], p_s[:, 2], label="SLAM traj")
        # ax.plot(p_rs[:, 0], p_rs[:, 1], p_rs[:, 2], label="Vicon(resampled) traj")
        # ax.set_title(f"[{idx}] 3D overlay (no frame alignment)")
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # ax.legend()

        # plt.show()  # <-- when you close plots, we proceed to saving

        # =========================
        # SAVE: new Vicon CSV in original format (Frame/SubFrame + TX/TY/TZ + RX/RY/RZ/RW)
        # Keep same number of rows as SLAM (exact)
        # =========================
        out_path = os.path.join(out_dir, os.path.basename(vicon_csv))

        # Build new dataframe with the same columns order as original Vicon file
        # Use the original first Frame/Sub Frame as a base and create a new frame sequence.
        cols = list(vicon_df.columns)

        # If original contains extra columns, you can keep them, but you asked xyz+quat,
        # so we will only keep the standard ones if present.
        keep_cols = [c for c in cols if c in ["Frame", "Sub Frame", "TX", "TY", "TZ", "RX", "RY", "RZ", "RW"]]
        if keep_cols != ["Frame", "Sub Frame", "TX", "TY", "TZ", "RX", "RY", "RZ", "RW"]:
            # Force canonical order if some are missing or reordered
            keep_cols = ["Frame", "Sub Frame", "TX", "TY", "TZ", "RX", "RY", "RZ", "RW"]

        # Create frames at Vicon fps that correspond to the SLAM timestamps (nearest)
        # Frame numbers are not critical for pose values, but keep them consistent.
        frame0 = int(vicon_df["Frame"].iloc[0])
        sub0 = int(vicon_df["Sub Frame"].iloc[0]) if "Sub Frame" in vicon_df.columns else 0

        # Map query times to "frame numbers" (relative to vicon start)
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

        # Save with original preamble preserved
        write_vicon_like_original(out_path, vicon_lines, header_idx, new_df)

        # Quick sanity
        assert len(new_df) == len(slam), f"[{idx}] Saved length != SLAM length"
        print(f"[{idx}] Saved: {out_path}  (rows={len(new_df)})")
