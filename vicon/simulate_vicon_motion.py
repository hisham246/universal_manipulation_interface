#!/usr/bin/env python3
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# File utilities
# ----------------------------
def natural_key(path):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", os.path.basename(path))]

def find_header_idx(lines):
    # Find the header row that starts with Frame,Sub Frame,...
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("frame,"):
            return i
    return None

def load_vicon_quat_csv(path):
    """
    Loads a Vicon CSV like your sample:
      - has a few metadata lines
      - header line: Frame,Sub Frame,RX,RY,RZ,RW,TX,TY,TZ
      - next line may contain units (mm,mm,mm) with empty Frame
    Returns a clean dataframe with numeric columns.
    """
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    h = find_header_idx(lines)
    if h is None:
        # fallback: let pandas try
        df = pd.read_csv(path)
    else:
        from io import StringIO
        df = pd.read_csv(StringIO("".join(lines[h:])), engine="python")

    # Drop the units row or any non-data rows: Frame should be numeric
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df = df.dropna(subset=["Frame"]).copy()

    # Ensure numeric for all expected columns if present
    expected = ["Frame", "Sub Frame", "RX", "RY", "RZ", "RW", "TX", "TY", "TZ"]
    for c in expected:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["TX", "TY", "TZ", "RX", "RY", "RZ", "RW"]).reset_index(drop=True)

    return df

def guess_pos_scale_to_meters(pos_xyz):
    """
    Heuristic:
      - if typical magnitude is tens/hundreds -> likely mm, convert to meters
      - if typical magnitude is < ~10 -> likely already meters
    """
    med = np.nanmedian(np.linalg.norm(pos_xyz, axis=1))
    if med > 10.0:
        return 1.0 / 1000.0  # mm -> m
    return 1.0              # already meters (likely)

# ----------------------------
# Quaternion + frame drawing
# ----------------------------
def quat_to_rotmat_xyzw(q):
    """
    q: (x,y,z,w)
    returns 3x3 rotation matrix
    """
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n

    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s

    R = np.array([
        [1.0 - (yy + zz),       xy - wz,         xz + wy],
        [xy + wz,               1.0 - (xx + zz), yz - wx],
        [xz - wy,               yz + wx,         1.0 - (xx + yy)]
    ], dtype=float)
    return R

def set_3d_equal(ax, X, Y, Z, pad=0.05):
    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    zmin, zmax = np.min(Z), np.max(Z)
    cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
    r = max(xmax-xmin, ymax-ymin, zmax-zmin) / 2
    r = r * (1 + pad)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)

# ----------------------------
# Plot + animation
# ----------------------------
def plot_overlay_3d(orig_df, proc_df, title="Trajectory overlay (original vs processed)"):
    o_pos = orig_df[["TX","TY","TZ"]].to_numpy(dtype=float)
    p_pos = proc_df[["TX","TY","TZ"]].to_numpy(dtype=float)

    o_s = guess_pos_scale_to_meters(o_pos)
    p_s = guess_pos_scale_to_meters(p_pos)
    o_pos = o_pos * o_s
    p_pos = p_pos * p_s

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(o_pos[:,0], o_pos[:,1], o_pos[:,2], label="original", linewidth=1.0)
    ax.plot(p_pos[:,0], p_pos[:,1], p_pos[:,2], label="processed", linewidth=2.0)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()

    all_pos = np.vstack([o_pos, p_pos])
    set_3d_equal(ax, all_pos[:,0], all_pos[:,1], all_pos[:,2])
    plt.tight_layout()
    plt.show()

def animate_overlay_frames(orig_df, proc_df, stride=1, axis_len=0.03, interval_ms=30, save_path=None):
    """
    Animates both original + processed:
      - draws both full trajectories
      - draws moving coordinate frames (x,y,z axes) at current pose for both
    Mapping between the two sequences:
      - uses normalized progress (0..1) -> index mapping
    """
    o_pos = orig_df[["TX","TY","TZ"]].to_numpy(dtype=float)
    p_pos = proc_df[["TX","TY","TZ"]].to_numpy(dtype=float)

    o_quat = orig_df[["RX","RY","RZ","RW"]].to_numpy(dtype=float)  # xyzw
    p_quat = proc_df[["RX","RY","RZ","RW"]].to_numpy(dtype=float)

    o_s = guess_pos_scale_to_meters(o_pos)
    p_s = guess_pos_scale_to_meters(p_pos)
    o_pos = o_pos * o_s
    p_pos = p_pos * p_s

    # Stride down for speed
    o_pos_s = o_pos[::stride]
    p_pos_s = p_pos[::stride]
    o_quat_s = o_quat[::stride]
    p_quat_s = p_quat[::stride]

    nO = len(o_pos_s)
    nP = len(p_pos_s)
    if nO < 2 or nP < 2:
        raise ValueError("Not enough points to animate.")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # full trajectories
    ax.plot(o_pos_s[:,0], o_pos_s[:,1], o_pos_s[:,2], label="original", linewidth=1.0)
    ax.plot(p_pos_s[:,0], p_pos_s[:,1], p_pos_s[:,2], label="processed", linewidth=2.0)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()

    all_pos = np.vstack([o_pos_s, p_pos_s])
    set_3d_equal(ax, all_pos[:,0], all_pos[:,1], all_pos[:,2])

    # 3D "frame" lines: we’ll keep 3 lines per trajectory (x,y,z axes)
    # original
    o_x_line, = ax.plot([], [], [], linewidth=2)
    o_y_line, = ax.plot([], [], [], linewidth=2)
    o_z_line, = ax.plot([], [], [], linewidth=2)
    # processed
    p_x_line, = ax.plot([], [], [], linewidth=2)
    p_y_line, = ax.plot([], [], [], linewidth=2)
    p_z_line, = ax.plot([], [], [], linewidth=2)

    # moving points (markers)
    o_pt, = ax.plot([], [], [], marker="o", markersize=4)
    p_pt, = ax.plot([], [], [], marker="o", markersize=4)

    def set_axis_lines(pos, R, x_line, y_line, z_line, L):
        # axis endpoints in world
        ex = pos + R[:,0] * L
        ey = pos + R[:,1] * L
        ez = pos + R[:,2] * L

        x_line.set_data([pos[0], ex[0]], [pos[1], ex[1]])
        x_line.set_3d_properties([pos[2], ex[2]])

        y_line.set_data([pos[0], ey[0]], [pos[1], ey[1]])
        y_line.set_3d_properties([pos[2], ey[2]])

        z_line.set_data([pos[0], ez[0]], [pos[1], ez[1]])
        z_line.set_3d_properties([pos[2], ez[2]])

    def update(iP):
        # processed index iP
        t = iP / (nP - 1)
        iO = int(round(t * (nO - 1)))

        ppos = p_pos_s[iP]
        opos = o_pos_s[iO]

        pR = quat_to_rotmat_xyzw(p_quat_s[iP])
        oR = quat_to_rotmat_xyzw(o_quat_s[iO])

        set_axis_lines(opos, oR, o_x_line, o_y_line, o_z_line, axis_len)
        set_axis_lines(ppos, pR, p_x_line, p_y_line, p_z_line, axis_len)

        o_pt.set_data([opos[0]], [opos[1]])
        o_pt.set_3d_properties([opos[2]])

        p_pt.set_data([ppos[0]], [ppos[1]])
        p_pt.set_3d_properties([ppos[2]])

        ax.set_title(f"Overlay frame motion   processed step {iP}/{nP-1}   (orig idx {iO}/{nO-1})")
        return (o_x_line, o_y_line, o_z_line, p_x_line, p_y_line, p_z_line, o_pt, p_pt)

    anim = FuncAnimation(fig, update, frames=nP, interval=interval_ms, blit=False)

    plt.tight_layout()

    if save_path is not None:
        # mp4 requires ffmpeg installed; otherwise you can save as .gif with pillow
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".gif":
            anim.save(save_path, writer="pillow", fps=max(1, int(1000/interval_ms)))
        else:
            anim.save(save_path, writer="ffmpeg", fps=max(1, int(1000/interval_ms)))
        print(f"Saved animation to: {save_path}")

    plt.show()

# ----------------------------
# Main: pick file by index
# ----------------------------
if __name__ == "__main__":
    vicon_original_dir  = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_trimmed/"
    vicon_processed_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_resampled_to_slam_3/"

    vicon_original_csv = sorted(glob.glob(os.path.join(vicon_original_dir, "*.csv")), key=natural_key)
    vicon_processed_csv = sorted(glob.glob(os.path.join(vicon_processed_dir, "*.csv")), key=natural_key)  # fixed

    assert len(vicon_original_csv) == len(vicon_processed_csv), \
        f"Count mismatch: original={len(vicon_original_csv)} processed={len(vicon_processed_csv)}"

    for idx in range(len(vicon_original_csv)):
        # choose which pair to view
        orig_path = vicon_original_csv[idx]
        proc_path = vicon_processed_csv[idx]

        print("Original :", orig_path)
        print("Processed:", proc_path)

        orig_df = load_vicon_quat_csv(orig_path)
        proc_df = load_vicon_quat_csv(proc_path)

        # 1) Static 3D overlay
        plot_overlay_3d(orig_df, proc_df, title=f"Overlay: {os.path.basename(orig_path)}")

        # 2) Animation with moving frames
        # tweak stride/axis_len depending on density and scale
        animate_overlay_frames(
            orig_df, proc_df,
            stride=1,
            axis_len=0.03,       # meters (after auto mm->m conversion)
            interval_ms=25,
            save_path=None       # e.g., "/tmp/overlay.mp4" or "/tmp/overlay.gif"
        )
