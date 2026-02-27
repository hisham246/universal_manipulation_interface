#!/usr/bin/env python3
import re
import pathlib
import zarr
import numpy as np
import pandas as pd

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.pose_util import pose_to_mat
from scipy.spatial.transform import Rotation as R

register_codecs()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/dataset_camera_only_segmented.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/dataset_with_vicon.zarr.zip"
vicon_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/vicon_final"

ref_zarr = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/dataset.zarr.zip"


# -----------------------------------------------------------------------------
# Inspect original dataset, store chunk/dtype/compressor info + episode bounds
# -----------------------------------------------------------------------------
with zarr.ZipStore(ref_zarr, mode='r') as src_store:
    root = zarr.open(src_store)

    print("Original dataset structure:")
    orig_chunks = {}
    orig_dtypes = {}
    orig_compressors = {}

    for key in root['data'].keys():
        arr = root['data'][key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, "
              f"chunks={arr.chunks}, compressor={arr.compressor}")
        orig_chunks[key] = arr.chunks
        orig_dtypes[key] = arr.dtype
        orig_compressors[key] = arr.compressor

# -----------------------------------------------------------------------------
# Config & Geometry
# -----------------------------------------------------------------------------
use_pattern = True
pattern = "aligned_episode_{i:03d}.csv"
one_based = True
vicon_pos_scale = 1e-3

vicon_world_offset = np.array([-0.02799, -0.206246, -0.093154], dtype=np.float64)  # <-- put your numbers here

# 1. LOCAL TRANSFORMATION (Camera -> Tool/EE)
# This handles the axis swap and the translation from camera center to peg tip.
R_local = np.array([
    [-1,  0,  0],
    [ 0,  0, -1],
    [ 0, -1,  0]
])
rotvec_local = R.from_matrix(R_local).as_rotvec()
t_offset = [-0.02799, -0.206246, -0.093154]

# Create the local transformation objects
pose_cam_tcp = np.concatenate([t_offset, rotvec_local])
T_cam_tcp = pose_to_mat(pose_cam_tcp)
t_local_shift = T_cam_tcp[:3, 3]
R_local_rot = R.from_matrix(T_cam_tcp[:3, :3])

# 2. GLOBAL WORLD TRANSFORMATION (Vicon World -> SLAM World)
# This rotates the entire scene by 180 degrees around the Z-axis.
R_vicon_to_slam = np.array([
    [-1.0,  0.0, 0.0],
    [ 0.0, -1.0, 0.0],
    [ 0.0,  0.0, 1.0],
], dtype=np.float64)
rot_v2s = R.from_matrix(R_vicon_to_slam)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def get_vicon_csv_for_episode(ep: int, vicon_dir_path: pathlib.Path, csv_list_sorted=None) -> pathlib.Path:
    if use_pattern:
        file_i = (ep + 1) if one_based else ep
        return vicon_dir_path / pattern.format(i=file_i)
    return vicon_dir_path / csv_list_sorted[ep]

def read_aligned_vicon_csv(p: pathlib.Path):
    df = pd.read_csv(p)
    required = ["Pos_X","Pos_Y","Pos_Z","Rot_X","Rot_Y","Rot_Z","Rot_W"]
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df[required]).all(axis=1)].reset_index(drop=True)
    
    pos = df[["Pos_X","Pos_Y","Pos_Z"]].to_numpy(dtype=np.float64) * vicon_pos_scale
    quat = df[["Rot_X","Rot_Y","Rot_Z","Rot_W"]].to_numpy(dtype=np.float64)
    
    # Normalize and ensure hemisphere continuity
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    for i in range(1, len(quat)):
        if np.dot(quat[i-1], quat[i]) < 0:
            quat[i] *= -1
    return pos, quat

# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------
vicon_dir_path = pathlib.Path(vicon_dir).expanduser().absolute()
with zarr.ZipStore(src_path, mode="r") as src_store:
    root = zarr.open(src_store)
    episode_ends_idx = np.asarray(root["meta"]["episode_ends"][:], dtype=int)
    episode_start_idxs = np.concatenate(([0], episode_ends_idx[:-1]))
    num_episodes = len(episode_ends_idx)
    ref_arr = root["data"]["timestamp"] if "timestamp" in root["data"] else root["data"][list(root["data"].array_keys())[0]]
    T_total = int(episode_ends_idx[-1])
    replay_buffer = ReplayBuffer.copy_from_store(src_store=src_store, store=zarr.MemoryStore())

data = replay_buffer.data
eef_pos_new = np.empty((T_total, 3), dtype=np.float32)
eef_rot_new = np.empty((T_total, 3), dtype=np.float32)
demo_start_new = np.empty((T_total, 6), dtype=np.float32)
demo_end_new   = np.empty((T_total, 6), dtype=np.float32)

for ep in range(num_episodes):
    s, e = int(episode_start_idxs[ep]), int(episode_ends_idx[ep])
    vicon_csv_path = get_vicon_csv_for_episode(ep, vicon_dir_path)
    pos_vicon, quat_vicon = read_aligned_vicon_csv(vicon_csv_path)

    L = min(e - s, len(pos_vicon))
    pos_vicon, quat_vicon = pos_vicon[:L], quat_vicon[:L]
    curr_e = s + L

    # # --- 1. LOCAL TRANSFORMATION (Vicon Space) ---
    # # Convert camera quat to scipy rotation
    # rot_cam_vicon = R.from_quat(quat_vicon)
    
    # # Position: Apply the local shift relative to the camera's orientation
    # vicon_dynamic_shift = rot_cam_vicon.apply(t_local_shift)
    # pos_tcp_vicon = pos_vicon + vicon_dynamic_shift

    # # Orientation: Apply the tool axis swap (Right-Multiply)
    # rot_tcp_vicon = rot_cam_vicon * R_local_rot

    # # --- 2. GLOBAL WORLD TRANSFORMATION (Rotate Entire World 180 deg) ---
    # # Rotate the finished position vector around the world Z-axis
    # pos_tcp_slam = pos_tcp_vicon @ R_vicon_to_slam.T 
    
    # # Rotate the finished orientation (Left-Multiply by world rotation)
    # rot_tcp_slam = rot_v2s * rot_tcp_vicon
    # rotvec_slam = rot_tcp_slam.as_rotvec()

        # Convert camera quat to scipy rotation (camera pose in Vicon world)
    rot_cam_vicon = R.from_quat(quat_vicon)

    # --- Step 0: apply constant world offset in Vicon world (position only) ---
    pos_cam_vicon_shift = pos_vicon + vicon_world_offset

    # --- Step 1: global transform Vicon -> SLAM (rotate whole world) ---
    # Position in SLAM
    pos_cam_slam = pos_cam_vicon_shift @ R_vicon_to_slam.T  # row-vector form
    # Orientation in SLAM (left-multiply world rotation)
    rot_cam_slam = rot_v2s * rot_cam_vicon

    # --- Step 2: local camera -> TCP in SLAM world ---
    # Lever-arm shift in SLAM world using SLAM camera orientation
    slam_dynamic_shift = rot_cam_slam.apply(t_local_shift)
    pos_tcp_slam = pos_cam_slam + slam_dynamic_shift

    # Axis convention swap (right-multiply)
    rot_tcp_slam = rot_cam_slam * R_local_rot
    rotvec_slam = rot_tcp_slam.as_rotvec()

    # Store results
    eef_pos_new[s:curr_e] = pos_tcp_slam
    eef_rot_new[s:curr_e] = rotvec_slam

    # Start/End Pose logic (based on the final rotated world)
    start_pose = np.concatenate([pos_tcp_slam[0], rotvec_slam[0]])
    end_pose   = np.concatenate([pos_tcp_slam[-1], rotvec_slam[-1]])
    demo_start_new[s:curr_e] = start_pose
    demo_end_new[s:curr_e]   = end_pose

# Save back to Zarr
for key in ["robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_demo_start_pose", "robot0_demo_end_pose"]:
    if key in data: del data[key]

data.create_dataset("robot0_eef_pos", data=eef_pos_new, dtype=orig_dtypes["robot0_eef_pos"], chunks=(ref_arr.chunks[0], 3), compressor=orig_compressors["robot0_eef_pos"])
data.create_dataset("robot0_eef_rot_axis_angle", data=eef_rot_new, dtype=orig_dtypes["robot0_eef_rot_axis_angle"], chunks=(ref_arr.chunks[0], 3), compressor=orig_compressors["robot0_eef_rot_axis_angle"])
data.create_dataset("robot0_demo_start_pose", data=demo_start_new, dtype=orig_dtypes["robot0_demo_start_pose"], chunks=(ref_arr.chunks[0], 6), compressor=orig_compressors["robot0_demo_start_pose"])
data.create_dataset("robot0_demo_end_pose", data=demo_end_new, dtype=orig_dtypes["robot0_demo_end_pose"], chunks=(ref_arr.chunks[0], 6), compressor=orig_compressors["robot0_demo_end_pose"])

with zarr.ZipStore(dst_path, mode="w") as dst_store:
    replay_buffer.save_to_store(dst_store)

print(f"Done. Saved to {dst_path}")