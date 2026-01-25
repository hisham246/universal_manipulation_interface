import numpy as np
import pandas as pd

# ----------------------------
# Quaternion utilities
# ----------------------------
def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n

def quat_slerp(q0, q1, u):
    """
    Slerp between unit quaternions q0, q1 (shape (4,)), u in [0,1].
    Returns shape (4,).
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    dot = np.dot(q0, q1)
    # Avoid long-path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        # Nearly linear
        q = q0 + u * (q1 - q0)
        return quat_normalize(q)

    theta = np.arccos(dot)
    s0 = np.sin((1.0 - u) * theta) / (np.sin(theta) + 1e-12)
    s1 = np.sin(u * theta) / (np.sin(theta) + 1e-12)
    return s0 * q0 + s1 * q1

def quat_to_rotmat(q):
    # q = [x,y,z,w]
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=np.float64)

def rotmat_to_axis_angle(R):
    # Returns axis-angle vector (axis * angle), angle in radians
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    w = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ], dtype=np.float64) / (2.0 * np.sin(theta) + 1e-12)
    return w * theta

def apply_T_to_pose(T, p, q_xyzw):
    """
    Apply 4x4 transform T to pose (p,q).
    p in R^3, q in xyzw describing rotation of body in source frame.
    If T maps source frame -> target frame, then:
      p' = R_T p + t_T
      R' = R_T R
    """
    R_T = T[:3,:3]
    t_T = T[:3, 3]
    p2 = R_T @ p + t_T
    R = quat_to_rotmat(q_xyzw)
    R2 = R_T @ R
    # Convert R2 back to quaternion by a simple robust method
    # (you can replace with scipy if you prefer)
    q2 = rotmat_to_quat_xyzw(R2)
    return p2, q2

def rotmat_to_quat_xyzw(R):
    # Robust conversion
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=np.float64)
    return quat_normalize(q)

# ----------------------------
# Vicon reader for your CSV format
# ----------------------------
def read_vicon_csv(path):
    lines = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    # Find the header line that starts with "Frame,Sub Frame"
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Frame,Sub Frame"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find Vicon header line 'Frame,Sub Frame,...'")

    df = pd.read_csv(path, skiprows=header_idx)
    # First row after header is usually units (mm, mm, ...)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Frame", "TX", "TY", "TZ", "RW", "RX", "RY", "RZ"]).reset_index(drop=True)

    # Infer Vicon fps: in your file it’s the second line "100"
    # If missing, assume 100
    fps = 100.0
    for ln in lines[:5]:
        ln2 = ln.strip().strip(",")
        if ln2.isdigit():
            fps = float(ln2)
            break

    t = (df["Frame"].to_numpy() - df["Frame"].iloc[0]) / fps
    p = df[["TX","TY","TZ"]].to_numpy() / 1000.0  # mm -> m
    q = df[["RX","RY","RZ","RW"]].to_numpy()       # xyzw
    q = quat_normalize(q)

    return t, p, q, fps

# ----------------------------
# Matching + resampling
# ----------------------------
def speed_signature(t, p):
    dt = np.diff(t)
    v = np.diff(p, axis=0) / dt[:,None]
    s = np.linalg.norm(v, axis=1)
    # center timestamps for speed samples
    ts = 0.5 * (t[:-1] + t[1:])
    return ts, s

def find_best_vicon_window(t_v, p_v, t_s, p_s, fps_v):
    # Build speed signatures
    tsv, sv = speed_signature(t_v, p_v)
    tss, ss = speed_signature(t_s, p_s)

    # Interpolate SLAM speed onto a 100 Hz grid to compare apples-to-apples
    t_grid = np.arange(0.0, t_s[-1], 1.0/fps_v)
    ss_100 = np.interp(t_grid, tss, ss)

    # Z-normalize
    a = (ss_100 - ss_100.mean()) / (ss_100.std() + 1e-12)   # length N
    b = (sv - sv.mean()) / (sv.std() + 1e-12)               # length M

    N = len(a)
    if len(b) < N:
        raise ValueError("Vicon signal shorter than SLAM; cannot match.")

    # Correlate (valid windows only)
    corr = np.correlate(b, a, mode="valid")
    k = int(np.argmax(corr))
    score = float(corr[k] / N)

    # Map speed-index k to a start time. sv is between positions, so use tsv[k]
    t0 = float(tsv[k])
    t1 = t0 + float(t_s[-1])  # target duration

    return t0, t1, score

def resample_vicon_to_slam(t_v, p_v, q_v, t0, t_slam):
    """
    Extract vicon segment starting at t0 and resample onto t_slam (relative to its own 0).
    """
    t_query = t0 + t_slam

    # Position interp
    p_out = np.vstack([np.interp(t_query, t_v, p_v[:,d]) for d in range(3)]).T

    # Quaternion slerp
    # For each query time, find the bracketing indices
    q_out = np.zeros((len(t_query), 4), dtype=np.float64)
    for i, tq in enumerate(t_query):
        j = np.searchsorted(t_v, tq)
        if j <= 0:
            q_out[i] = q_v[0]
        elif j >= len(t_v):
            q_out[i] = q_v[-1]
        else:
            tL, tR = t_v[j-1], t_v[j]
            u = 0.0 if (tR - tL) < 1e-12 else (tq - tL) / (tR - tL)
            q_out[i] = quat_slerp(q_v[j-1], q_v[j], float(u))
    q_out = quat_normalize(q_out)
    return p_out, q_out

import matplotlib.pyplot as plt

def interp_pos(t_src, p_src, t_query):
    return np.vstack([np.interp(t_query, t_src, p_src[:, d]) for d in range(3)]).T

def speed_on_grid(t, p, t_grid):
    """
    Compute speed on a given time grid by interpolating position to grid then differencing.
    Returns t_mid (len-1) and speed (len-1).
    """
    p_g = interp_pos(t, p, t_grid)
    dt = np.diff(t_grid)
    v = np.diff(p_g, axis=0) / dt[:, None]
    s = np.linalg.norm(v, axis=1)
    t_mid = 0.5 * (t_grid[:-1] + t_grid[1:])
    return t_mid, s

if __name__ == "__main__":
    slam_csv = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/slam_segmented/episode_200.csv"
    vicon_csv = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_trimmed/peg_umi_quat 200.csv"

    slam = pd.read_csv(slam_csv)
    t_s = slam["timestamp"].to_numpy()
    t_s = t_s - t_s[0]
    p_s = slam[["robot0_eef_pos_0","robot0_eef_pos_1","robot0_eef_pos_2"]].to_numpy()

    t_v, p_v, q_v, fps_v = read_vicon_csv(vicon_csv)

    # Find best match window using your speed-correlation method
    t0, t1, score = find_best_vicon_window(t_v, p_v, t_s, p_s, fps_v)
    print(f"Best Vicon window start t0={t0:.3f}s, duration={t_s[-1]:.3f}s, corr={score:.3f}")

    # Resample Vicon to SLAM timestamps (for position overlay)
    p60, q60 = resample_vicon_to_slam(t_v, p_v, q_v, t0, t_s)

    # ----------------------------
    # 1) Overlay speed signatures (the thing you matched on)
    # ----------------------------
    # Build a Vicon-rate grid for the SLAM interval and for the matched Vicon window
    dur = float(t_s[-1])
    dtv = 1.0 / fps_v

    t_grid_slam = np.arange(0.0, dur, dtv)                  # 0..dur at 100 Hz
    t_grid_vwin = t0 + t_grid_slam                          # align to Vicon absolute time

    # Speed on same grid (apples-to-apples)
    t_mid_s, s_s = speed_on_grid(t_s, p_s, t_grid_slam)
    t_mid_v, s_v = speed_on_grid(t_v, p_v, t_grid_vwin)

    # Optional normalization for visual comparison
    s_s_n = (s_s - s_s.mean()) / (s_s.std() + 1e-12)
    s_v_n = (s_v - s_v.mean()) / (s_v.std() + 1e-12)

    plt.figure()
    plt.plot(t_mid_s, s_s_n, label="SLAM speed (norm)")
    plt.plot(t_mid_s, s_v_n, label="Vicon-window speed (norm)")
    plt.title("Speed signature overlay (normalized)")
    plt.xlabel("time (s) (SLAM-relative)")
    plt.ylabel("normalized speed")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(t_mid_s, s_s, label="SLAM speed")
    plt.plot(t_mid_s, s_v, label="Vicon-window speed")
    plt.title("Speed signature overlay (raw)")
    plt.xlabel("time (s) (SLAM-relative)")
    plt.ylabel("speed (units depend on position units)")
    plt.legend()
    plt.grid(True)

    # ----------------------------
    # 2) Overlay positions per axis (shape match; offsets are expected)
    # ----------------------------
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    labels = ["x", "y", "z"]
    for i in range(3):
        axs[i].plot(t_s, p_s[:, i], label=f"SLAM pos {labels[i]}")
        axs[i].plot(t_s, p60[:, i], label=f"Vicon(resampled) pos {labels[i]}")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        axs[i].legend()
    axs[-1].set_xlabel("time (s)")
    fig.suptitle("Position overlay (no frame alignment)")

    # ----------------------------
    # 3) 3D trajectory overlay (shape)
    # ----------------------------
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(p_s[:,0], p_s[:,1], p_s[:,2], label="SLAM traj")
    ax.plot(p60[:,0], p60[:,1], p60[:,2], label="Vicon(resampled) traj")
    ax.set_title("3D trajectory overlay (no frame alignment)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

    plt.show()

