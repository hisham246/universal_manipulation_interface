#!/usr/bin/env python3
import pathlib, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.pose_util import pose_to_mat

register_codecs()

src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_with_vicon_segmented.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_with_vicon_final.zarr.zip"
vicon_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_resampled_to_slam_3"

# Your current extrinsic guess (translation only)
pose_cam_tcp = np.array([-0.01938062, -0.19540817, -0.09206965, 0.0, 0.0, 0.0], dtype=np.float64)
T_cam_tcp = pose_to_mat(pose_cam_tcp).astype(np.float64)
T_tcp_cam = np.linalg.inv(T_cam_tcp)

vicon_pos_scale = 1e-3
pattern = "peg_umi_quat {i}.csv"
one_based = True

def read_vicon_csv(vicon_csv_path: pathlib.Path) -> pd.DataFrame:
    lines = vicon_csv_path.read_text(errors="ignore").splitlines()
    header_idx = None
    for i, ln in enumerate(lines):
        low = ln.strip().lower()
        if low.startswith("frame,") and ("tx" in low) and ("rw" in low):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"Could not find Vicon header row in {vicon_csv_path}")
    df = pd.read_csv(vicon_csv_path, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]
    return df

def vicon_df_to_pos_quat(df: pd.DataFrame, pos_scale: float):
    pos = df[["TX", "TY", "TZ"]].to_numpy(dtype=np.float64) * pos_scale
    quat = df[["RX", "RY", "RZ", "RW"]].to_numpy(dtype=np.float64)
    quat = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12)
    return pos, quat

def pos_quat_to_T(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    Rm = Rotation.from_quat(quat).as_matrix()
    T = np.zeros((len(pos), 4, 4), dtype=np.float64)
    T[:, 3, 3] = 1.0
    T[:, :3, :3] = Rm
    T[:, :3, 3] = pos
    return T

# --- load episode bounds + src/dst eef for episode 0
src_path = str(pathlib.Path(src_path).expanduser().absolute())
dst_path = str(pathlib.Path(dst_path).expanduser().absolute())
vicon_dir_path = pathlib.Path(vicon_dir).expanduser().absolute()

with zarr.ZipStore(src_path, mode="r") as src_store:
    rb_src = ReplayBuffer.copy_from_store(src_store=src_store, store=zarr.MemoryStore())
with zarr.ZipStore(dst_path, mode="r") as dst_store:
    rb_dst = ReplayBuffer.copy_from_store(src_store=dst_store, store=zarr.MemoryStore())

episode_ends = np.asarray(rb_src.root["meta"]["episode_ends"][:], dtype=int)
episode_starts = np.concatenate(([0], episode_ends[:-1]))
ep = 0
s, e = int(episode_starts[ep]), int(episode_ends[ep])
L = e - s

pos_src = np.asarray(rb_src.data["robot0_eef_pos"][s:e], dtype=np.float64)
pos_dst = np.asarray(rb_dst.data["robot0_eef_pos"][s:e], dtype=np.float64)

# --- load vicon csv for episode 0
file_i = (ep + 1) if one_based else ep
vicon_csv_path = vicon_dir_path / pattern.format(i=file_i)
if not vicon_csv_path.is_file():
    raise RuntimeError(f"Missing Vicon CSV: {vicon_csv_path}")

vdf = read_vicon_csv(vicon_csv_path)
pos_cam, quat_cam = vicon_df_to_pos_quat(vdf, pos_scale=vicon_pos_scale)

if len(pos_cam) != L:
    raise RuntimeError(f"Length mismatch: vicon={len(pos_cam)} vs ep_len={L} for {vicon_csv_path}")

T_world_cam = pos_quat_to_T(pos_cam, quat_cam)

# two hypotheses for applying extrinsic
T_world_tcp_A = T_world_cam @ T_cam_tcp   # your current code
T_world_tcp_B = T_world_cam @ T_tcp_cam   # if you accidentally used the inverse direction

pos_tcp_A = T_world_tcp_A[:, :3, 3]
pos_tcp_B = T_world_tcp_B[:, :3, 3]

# --- quick numeric sanity: which one matches src better (start/end only)
def start_end_err(p_ref, p_test):
    return np.linalg.norm(p_ref[0]-p_test[0]), np.linalg.norm(p_ref[-1]-p_test[-1])

print("Start/End position error vs SRC (meters):")
print("  DST (your written zarr):", start_end_err(pos_src, pos_dst))
print("  Vicon tcp A (cam@T_cam_tcp):", start_end_err(pos_src, pos_tcp_A))
print("  Vicon tcp B (cam@inv(T_cam_tcp)):", start_end_err(pos_src, pos_tcp_B))

# --- plot overlay (3D)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(pos_src[:,0], pos_src[:,1], pos_src[:,2], label="SRC SLAM tcp (zarr)")
ax.plot(pos_dst[:,0], pos_dst[:,1], pos_dst[:,2], label="DST Vicon tcp (zarr)")
ax.plot(pos_cam[:,0], pos_cam[:,1], pos_cam[:,2], label="Vicon cam (raw)")
ax.plot(pos_tcp_A[:,0], pos_tcp_A[:,1], pos_tcp_A[:,2], label="Vicon tcp A")
ax.plot(pos_tcp_B[:,0], pos_tcp_B[:,1], pos_tcp_B[:,2], label="Vicon tcp B")

ax.set_title("Episode 0: before/after + Vicon hypotheses")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.legend()
plt.show()

# --- optional: show how much “jitter amplification” you’re getting from the lever arm
# (frame-to-frame displacement stats)
def disp_stats(p):
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return float(np.mean(d)), float(np.std(d)), float(np.max(d))

print("Frame-to-frame displacement mean/std/max (m):")
print("  Vicon cam: ", disp_stats(pos_cam))
print("  Vicon tcp A:", disp_stats(pos_tcp_A))
print("  Vicon tcp B:", disp_stats(pos_tcp_B))
print("  SRC SLAM tcp:", disp_stats(pos_src))
