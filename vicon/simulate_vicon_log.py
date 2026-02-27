#!/usr/bin/env python3
"""
Animate ONLY the umi_cable_route pose from your vicon_logs_to_csv/vicon_1.csv format.

Expected columns (exactly what you showed):
  Frame, Timestamp,
  umi_cable_route_Pos_X, umi_cable_route_Pos_Y, umi_cable_route_Pos_Z,
  umi_cable_route_Rot_X, umi_cable_route_Rot_Y, umi_cable_route_Rot_Z, umi_cable_route_Rot_W

Notes:
- Positions are often in mm; set POS_SCALE=1e-3 to convert to meters.
- Rot_* are quaternion in xyzw order.
- Draws:
  - Static world frame at origin (0,0,0)
  - Moving tool frame at each timestep
  - Trajectory line
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "/home/hisham246/uwaterloo/cable_route_umi/vicon_logs_to_csv/vicon_1.csv"
OBJ = "umi_cable_route"

POS_SCALE = 1e-3        # mm -> m (set 1.0 if already meters)
STRIDE = 1              # use every Nth sample
TOOL_AXIS_LEN = 0.03    # meters
WORLD_AXIS_LEN = 0.05   # meters
INTERVAL_MS = 25
SAVE_GIF = None         # e.g. "/tmp/umi_cable_route.gif" or None

# ----------------------------
# Helpers
# ----------------------------
def quat_to_rotmat_xyzw(q):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    x, y, z, w = map(float, q)
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

def set_tool_axes(pos, Rm, x_line, y_line, z_line, x_txt, y_txt, z_txt, L, label_offset=1.10):
    ex = pos + Rm[:,0] * L
    ey = pos + Rm[:,1] * L
    ez = pos + Rm[:,2] * L

    x_line.set_data([pos[0], ex[0]], [pos[1], ex[1]])
    x_line.set_3d_properties([pos[2], ex[2]])

    y_line.set_data([pos[0], ey[0]], [pos[1], ey[1]])
    y_line.set_3d_properties([pos[2], ey[2]])

    z_line.set_data([pos[0], ez[0]], [pos[1], ez[1]])
    z_line.set_3d_properties([pos[2], ez[2]])

    lx = pos + Rm[:,0] * (L * label_offset)
    ly = pos + Rm[:,1] * (L * label_offset)
    lz = pos + Rm[:,2] * (L * label_offset)

    x_txt.set_position((lx[0], lx[1])); x_txt.set_3d_properties(lx[2])
    y_txt.set_position((ly[0], ly[1])); y_txt.set_3d_properties(ly[2])
    z_txt.set_position((lz[0], lz[1])); z_txt.set_3d_properties(lz[2])

# ----------------------------
# Load umi_cable_route from CSV
# ----------------------------
df = pd.read_csv(CSV_PATH)

pos_cols = [f"{OBJ}_Pos_X", f"{OBJ}_Pos_Y", f"{OBJ}_Pos_Z"]
quat_cols = [f"{OBJ}_Rot_X", f"{OBJ}_Rot_Y", f"{OBJ}_Rot_Z", f"{OBJ}_Rot_W"]

for c in pos_cols + quat_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column {c} in {CSV_PATH}")

for c in pos_cols + quat_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=pos_cols + quat_cols).reset_index(drop=True)

pos = df[pos_cols].to_numpy(dtype=np.float64) * POS_SCALE                 # (N,3)
quat = df[quat_cols].to_numpy(dtype=np.float64)                           # (N,4) xyzw
Rmats = np.stack([quat_to_rotmat_xyzw(q) for q in quat], axis=0)           # (N,3,3)

# apply stride
pos_s = pos[::STRIDE]
R_s = Rmats[::STRIDE]
n = len(pos_s)
if n < 2:
    raise ValueError("Not enough points after STRIDE")

print(f"Loaded {n} frames from {os.path.basename(CSV_PATH)} (object={OBJ})")

# ----------------------------
# Plot + animate
# ----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# trajectory line
ax.plot(pos_s[:,0], pos_s[:,1], pos_s[:,2], linewidth=1.5, alpha=0.5)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
set_3d_equal(ax, pos_s[:,0], pos_s[:,1], pos_s[:,2])

# static world frame at origin
w0 = np.zeros(3, dtype=float)
wx = w0 + np.array([WORLD_AXIS_LEN, 0, 0], dtype=float)
wy = w0 + np.array([0, WORLD_AXIS_LEN, 0], dtype=float)
wz = w0 + np.array([0, 0, WORLD_AXIS_LEN], dtype=float)

ax.plot([w0[0], wx[0]], [w0[1], wx[1]], [w0[2], wx[2]], 'r', linewidth=2)
ax.plot([w0[0], wy[0]], [w0[1], wy[1]], [w0[2], wy[2]], 'g', linewidth=2)
ax.plot([w0[0], wz[0]], [w0[1], wz[1]], [w0[2], wz[2]], 'b', linewidth=2)

ax.text(wx[0], wx[1], wx[2], "Xw", fontsize=10, color="r")
ax.text(wy[0], wy[1], wy[2], "Yw", fontsize=10, color="g")
ax.text(wz[0], wz[1], wz[2], "Zw", fontsize=10, color="b")

# moving tool frame
x_line, = ax.plot([], [], [], 'r', linewidth=2)
y_line, = ax.plot([], [], [], 'g', linewidth=2)
z_line, = ax.plot([], [], [], 'b', linewidth=2)
pt, = ax.plot([], [], [], 'ko', markersize=4)

x_txt = ax.text(0, 0, 0, "Xt", fontsize=10, color="r")
y_txt = ax.text(0, 0, 0, "Yt", fontsize=10, color="g")
z_txt = ax.text(0, 0, 0, "Zt", fontsize=10, color="b")

def update(i):
    p = pos_s[i]
    Rm = R_s[i]
    set_tool_axes(p, Rm, x_line, y_line, z_line, x_txt, y_txt, z_txt, TOOL_AXIS_LEN)

    pt.set_data([p[0]], [p[1]])
    pt.set_3d_properties([p[2]])

    ax.set_title(f"{OBJ} moving frame | step {i}/{n-1}")
    return (x_line, y_line, z_line, pt, x_txt, y_txt, z_txt)

anim = FuncAnimation(fig, update, frames=n, interval=INTERVAL_MS, blit=False)
plt.tight_layout()

if SAVE_GIF is not None:
    anim.save(SAVE_GIF, writer="pillow", fps=max(1, int(1000 / INTERVAL_MS)))
    print("Saved GIF to:", SAVE_GIF)

plt.show()