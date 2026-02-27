#!/usr/bin/env python3
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# Utils
# ----------------------------
def natural_key(path):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", os.path.basename(path))]

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

def guess_pos_scale_to_meters(pos_xyz):
    med = np.nanmedian(np.linalg.norm(pos_xyz, axis=1))
    if med > 10.0:
        return 1.0 / 1000.0
    return 1.0


def debug_axes(path, n=5):
    pos, Rmats, meta = load_pose_sequence_auto(path)
    print("\n====", os.path.basename(path), "====")
    print("format:", meta["format"])
    print("pos_scale:", meta["pos_scale"])
    print("N:", len(pos))

    # check handedness + orthonormality
    dets = np.linalg.det(Rmats)
    ortho_err = np.max(np.abs(np.einsum("nij,nkj->nik", Rmats, Rmats) - np.eye(3)), axis=(1,2))
    print("det(R): min/mean/max =", dets.min(), dets.mean(), dets.max())
    print("orthonormal err: max =", ortho_err.max())

    # how often tool-x points along +world-x vs -world-x
    tool_x = Rmats[:, :, 0]
    dots = tool_x @ np.array([1.0, 0.0, 0.0])  # dot with world +X
    print("tool_x · world_X: min/mean/max =", dots.min(), dots.mean(), dots.max())
    print("fraction pointing +X:", np.mean(dots > 0.0))
    print("fraction pointing -X:", np.mean(dots < 0.0))

    # print a few samples
    for i in np.linspace(0, len(pos)-1, num=min(n, len(pos)), dtype=int):
        print(f"i={i:5d}  tool_x={tool_x[i]}  dot={dots[i]: .3f}")
# ----------------------------
# Rotations
# ----------------------------
def quat_to_rotmat_xyzw(q):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n

    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s

    return np.array([
        [1.0 - (yy + zz),       xy - wz,         xz + wy],
        [xy + wz,               1.0 - (xx + zz), yz - wx],
        [xz - wy,               yz + wx,         1.0 - (xx + yy)]
    ], dtype=float)

def rotvec_to_rotmat(rv):
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    kx, ky, kz = k
    K = np.array([
        [0,   -kz,  ky],
        [kz,   0,  -kx],
        [-ky, kx,   0]
    ], dtype=float)

    I = np.eye(3)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

# ----------------------------
# CSV loaders (auto-detect)
# ----------------------------
def _find_header_idx(lines):
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("frame,"):
            return i
    return None

def load_vicon_quat_csv(path):
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    h = _find_header_idx(lines)
    if h is None:
        df = pd.read_csv(path)
    else:
        from io import StringIO
        df = pd.read_csv(StringIO("".join(lines[h:])), engine="python")

    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df = df.dropna(subset=["Frame"]).copy()

    expected = ["Frame", "Sub Frame", "RX", "RY", "RZ", "RW", "TX", "TY", "TZ"]
    for c in expected:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["TX", "TY", "TZ", "RX", "RY", "RZ", "RW"]).reset_index(drop=True)
    return df

def load_episode_rotvec_csv(path):
    df = pd.read_csv(path)
    cols_pos = ["robot0_eef_pos_0", "robot0_eef_pos_1", "robot0_eef_pos_2"]
    cols_rv  = ["robot0_eef_rot_axis_angle_0", "robot0_eef_rot_axis_angle_1", "robot0_eef_rot_axis_angle_2"]

    for c in cols_pos + cols_rv:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {path}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=cols_pos + cols_rv).reset_index(drop=True)
    return df

def load_pose_sequence_auto(path):
    df = pd.read_csv(path, nrows=5)
    cols = set(df.columns)

    is_vicon = all(c in cols for c in ["TX","TY","TZ","RX","RY","RZ","RW"])
    is_ep    = all(c in cols for c in ["robot0_eef_pos_0","robot0_eef_pos_1","robot0_eef_pos_2",
                                       "robot0_eef_rot_axis_angle_0","robot0_eef_rot_axis_angle_1","robot0_eef_rot_axis_angle_2"])

    if is_vicon:
        full = load_vicon_quat_csv(path)
        pos = full[["TX","TY","TZ"]].to_numpy(float)
        quat = full[["RX","RY","RZ","RW"]].to_numpy(float)  # xyzw
        s = guess_pos_scale_to_meters(pos)
        pos = pos * s
        Rmats = np.stack([quat_to_rotmat_xyzw(q) for q in quat], axis=0)
        return pos, Rmats, {"format":"vicon_quat", "pos_scale":s, "df":full}

    if is_ep:
        full = load_episode_rotvec_csv(path)
        pos = full[["robot0_eef_pos_0","robot0_eef_pos_1","robot0_eef_pos_2"]].to_numpy(float)
        rv  = full[["robot0_eef_rot_axis_angle_0","robot0_eef_rot_axis_angle_1","robot0_eef_rot_axis_angle_2"]].to_numpy(float)
        s = guess_pos_scale_to_meters(pos)
        pos = pos * s
        Rmats = np.stack([rotvec_to_rotmat(r) for r in rv], axis=0)
        return pos, Rmats, {"format":"episode_rotvec", "pos_scale":s, "df":full}

    raise ValueError(f"Could not auto-detect format for {path}\nColumns seen: {sorted(list(cols))}")

# ----------------------------
# Axis lines + labels
# ----------------------------
def _set_axis_lines_and_labels(pos, R, x_line, y_line, z_line, x_txt, y_txt, z_txt, L, label_offset=1.10):
    ex = pos + R[:,0] * L
    ey = pos + R[:,1] * L
    ez = pos + R[:,2] * L

    x_line.set_data([pos[0], ex[0]], [pos[1], ex[1]])
    x_line.set_3d_properties([pos[2], ex[2]])

    y_line.set_data([pos[0], ey[0]], [pos[1], ey[1]])
    y_line.set_3d_properties([pos[2], ey[2]])

    z_line.set_data([pos[0], ez[0]], [pos[1], ez[1]])
    z_line.set_3d_properties([pos[2], ez[2]])

    # put the label slightly beyond the tip
    lx = pos + R[:,0] * (L * label_offset)
    ly = pos + R[:,1] * (L * label_offset)
    lz = pos + R[:,2] * (L * label_offset)

    x_txt.set_position((lx[0], lx[1])); x_txt.set_3d_properties(lx[2])
    y_txt.set_position((ly[0], ly[1])); y_txt.set_3d_properties(ly[2])
    z_txt.set_position((lz[0], lz[1])); z_txt.set_3d_properties(lz[2])

# ----------------------------
# Animations
# ----------------------------
def animate_single_csv(path, stride=1, axis_len=0.03, interval_ms=25, save_path=None):
    pos, Rmats, meta = load_pose_sequence_auto(path)

    pos_s = pos[::stride]
    R_s   = Rmats[::stride]
    n = len(pos_s)
    if n < 2:
        raise ValueError("Not enough points to animate")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pos_s[:,0], pos_s[:,1], pos_s[:,2], linewidth=1.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    set_3d_equal(ax, pos_s[:,0], pos_s[:,1], pos_s[:,2])

    x_line, = ax.plot([], [], [], linewidth=2)
    y_line, = ax.plot([], [], [], linewidth=2)
    z_line, = ax.plot([], [], [], linewidth=2)
    pt, = ax.plot([], [], [], marker="o", markersize=4)

    # moving labels for the frame axes
    x_txt = ax.text(0, 0, 0, "X", fontsize=10)
    y_txt = ax.text(0, 0, 0, "Y", fontsize=10)
    z_txt = ax.text(0, 0, 0, "Z", fontsize=10)

    def update(i):
        p = pos_s[i]
        R = R_s[i]

        # print("tool x in world:", R[:,0])
        # print("tool y in world:", R[:,1])
        # print("tool z in world:", R[:,2])


        _set_axis_lines_and_labels(p, R, x_line, y_line, z_line, x_txt, y_txt, z_txt, axis_len)

        pt.set_data([p[0]], [p[1]])
        pt.set_3d_properties([p[2]])

        ax.set_title(f"Motion: {os.path.basename(path)}   step {i}/{n-1}   ({meta['format']})")
        return (x_line, y_line, z_line, pt, x_txt, y_txt, z_txt)

    anim = FuncAnimation(fig, update, frames=n, interval=interval_ms, blit=False)
    plt.tight_layout()

    if save_path is not None:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".gif":
            anim.save(save_path, writer="pillow", fps=max(1, int(1000/interval_ms)))
        else:
            anim.save(save_path, writer="ffmpeg", fps=max(1, int(1000/interval_ms)))
        print("Saved animation to:", save_path)

    plt.show()


def print_final_displacement(path):
    # Use your existing auto-loader to get the scaled positions
    pos, Rmats, meta = load_pose_sequence_auto(path)
    
    # Extract the final position (last row)
    final_pos = pos[-1]
    
    # Calculate absolute distance in each axis from the origin (0,0,0)
    dx = abs(final_pos[0])
    dy = abs(final_pos[1])
    dz = abs(final_pos[2])
    
    # Calculate total Euclidean (straight-line) distance
    total_dist = np.linalg.norm(final_pos)
    
    print(f"\n==== Final Pose Analysis: {os.path.basename(path)} ====")
    print(f"Format detected: {meta['format']}")
    print(f"Scale applied:   {meta['pos_scale']}")
    print("-" * 40)
    print(f"Final X Distance: {dx:.4f} m")
    print(f"Final Y Distance: {dy:.4f} m")
    print(f"Final Z Distance: {dz:.4f} m")
    print(f"Total Magnitude:  {total_dist:.4f} m")

def animate_overlay_two_csv(path_a, path_b, stride=1, axis_len=0.03, interval_ms=25, save_path=None):
    posA, RA, metaA = load_pose_sequence_auto(path_a)
    posB, RB, metaB = load_pose_sequence_auto(path_b)

    posA = posA[::stride]; RA = RA[::stride]
    posB = posB[::stride]; RB = RB[::stride]
    nA, nB = len(posA), len(posB)
    if nA < 2 or nB < 2:
        raise ValueError("Not enough points to animate")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(posA[:,0], posA[:,1], posA[:,2], label=f"A: {os.path.basename(path_a)}", linewidth=1.0)
    ax.plot(posB[:,0], posB[:,1], posB[:,2], label=f"B: {os.path.basename(path_b)}", linewidth=2.0)
    ax.legend()

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    all_pos = np.vstack([posA, posB])
    set_3d_equal(ax, all_pos[:,0], all_pos[:,1], all_pos[:,2])

    axA_x, = ax.plot([], [], [], linewidth=2)
    axA_y, = ax.plot([], [], [], linewidth=2)
    axA_z, = ax.plot([], [], [], linewidth=2)

    axB_x, = ax.plot([], [], [], linewidth=2)
    axB_y, = ax.plot([], [], [], linewidth=2)
    axB_z, = ax.plot([], [], [], linewidth=2)

    ptA, = ax.plot([], [], [], marker="o", markersize=4)
    ptB, = ax.plot([], [], [], marker="o", markersize=4)

    # moving labels for A and B frames
    A_x_txt = ax.text(0, 0, 0, "A_X", fontsize=9)
    A_y_txt = ax.text(0, 0, 0, "A_Y", fontsize=9)
    A_z_txt = ax.text(0, 0, 0, "A_Z", fontsize=9)

    B_x_txt = ax.text(0, 0, 0, "B_X", fontsize=9)
    B_y_txt = ax.text(0, 0, 0, "B_Y", fontsize=9)
    B_z_txt = ax.text(0, 0, 0, "B_Z", fontsize=9)

    def update(iB):
        t = iB / (nB - 1)
        iA = int(round(t * (nA - 1)))

        pA, rA = posA[iA], RA[iA]
        pB, rB = posB[iB], RB[iB]

        _set_axis_lines_and_labels(pA, rA, axA_x, axA_y, axA_z, A_x_txt, A_y_txt, A_z_txt, axis_len)
        _set_axis_lines_and_labels(pB, rB, axB_x, axB_y, axB_z, B_x_txt, B_y_txt, B_z_txt, axis_len)

        ptA.set_data([pA[0]], [pA[1]]); ptA.set_3d_properties([pA[2]])
        ptB.set_data([pB[0]], [pB[1]]); ptB.set_3d_properties([pB[2]])

        ax.set_title(
            f"Overlay frames\n"
            f"A({metaA['format']}) idx {iA}/{nA-1}  |  B({metaB['format']}) idx {iB}/{nB-1}"
        )
        return (axA_x, axA_y, axA_z, axB_x, axB_y, axB_z,
                ptA, ptB, A_x_txt, A_y_txt, A_z_txt, B_x_txt, B_y_txt, B_z_txt)

    anim = FuncAnimation(fig, update, frames=nB, interval=interval_ms, blit=False)
    plt.tight_layout()

    if save_path is not None:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".gif":
            anim.save(save_path, writer="pillow", fps=max(1, int(1000/interval_ms)))
        else:
            anim.save(save_path, writer="ffmpeg", fps=max(1, int(1000/interval_ms)))
        print("Saved animation to:", save_path)

    plt.show()

# ----------------------------
# Example main
# ----------------------------
if __name__ == "__main__":
    # episode_path = "/home/hisham246/uwaterloo/cable_route_umi/vicon_final/episode_1.csv"
    episode_path = "/home/hisham246/uwaterloo/VIDP_IROS2026/cable_route/vicon_no_blue_station/episode_12.csv"

    # episode_path = "/home/hisham246/uwaterloo/peg_in_hole_delta_umi/vicon_final/episode_1.csv"
    # episode_path = "/home/hisham246/uwaterloo/umi/reaching_ball_multimodal/csv_filtered/episode_1.csv"

    print_final_displacement(episode_path)
    animate_single_csv(
        episode_path,
        stride=1,
        axis_len=0.03,
        interval_ms=25,
        save_path=None
    )
    debug_axes(episode_path)


# # ----------------------------
# # Example main
# # ----------------------------
# if __name__ == "__main__":
#     # Define the directory containing your episodes
#     base_dir = "/home/hisham246/uwaterloo/VIDP_IROS2026/cable_route/vicon_all_stations/"
    
#     # Find all CSV files matching the episode pattern
#     search_pattern = os.path.join(base_dir, "episode_*.csv")
#     episode_files = glob.glob(search_pattern)
    
#     # Sort them naturally (1, 2, ..., 10, 11) instead of alphabetically (1, 10, 11, ..., 2)
#     episode_files.sort(key=natural_key)
    
#     if not episode_files:
#         print(f"No CSV files found in {base_dir}")
#     else:
#         print(f"Found {len(episode_files)} episodes. Playing them sequentially...")

#     # Loop through each file one by one
#     for episode_path in episode_files:
#         print(f"\n========================================================")
#         print(f"Loading: {os.path.basename(episode_path)}")
#         print(f"========================================================")
        
#         # Run your analysis functions
#         print_final_displacement(episode_path)
#         debug_axes(episode_path)
        
#         # Run the animation. 
#         # Note: plt.show() inside this function will pause the loop. 
#         # Close the plot window to advance to the next episode.
#         animate_single_csv(
#             episode_path,
#             stride=1,
#             axis_len=0.03,
#             interval_ms=25,
#             save_path=None
#         )
