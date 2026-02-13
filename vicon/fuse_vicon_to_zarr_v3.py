# #!/usr/bin/env python3
# """
# Build robot0_eef_pos / robot0_eef_rot_axis_angle from aligned Vicon CSVs (resampled to episode timestamps),
# apply a constant translation shift in Vicon world axes, and compute robot0_demo_start_pose / end_pose.

# No SLAM reference is used.
# """

# import re
# import pathlib
# import zarr
# import numpy as np
# import pandas as pd

# from diffusion_policy.common.replay_buffer import ReplayBuffer
# from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
# from umi.common.pose_util import pose_to_mat

# from scipy.spatial.transform import Rotation as R

# register_codecs()

# # -----------------------------------------------------------------------------
# # Paths
# # -----------------------------------------------------------------------------
# src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/dataset_camera_only.zarr.zip"
# dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/peg_in_hole_vicon.zarr.zip"

# # NOTE: fix your directory name: you wrote peg_in_hole_umi_with_vicon_3 before
# vicon_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/aligned_vicon_files/aligned_vicon_to_episode"

# # -----------------------------------------------------------------------------
# # Aligned Vicon file naming
# # -----------------------------------------------------------------------------
# # your sync script saved: aligned_episode_001.csv, aligned_episode_002.csv, ...
# use_pattern = True
# pattern = "aligned_episode_{i:03d}.csv"
# one_based = True

# # Rotate Vicon world -> SLAM world (180 deg about +Z)
# R_vicon_to_slam = np.array([
#     [-1.0,  0.0, 0.0],
#     [ 0.0, -1.0, 0.0],
#     [ 0.0,  0.0, 1.0],
# ], dtype=np.float64)

# rot_v2s = R.from_matrix(R_vicon_to_slam)  # scipy Rotation

# # -----------------------------------------------------------------------------
# # Transform config (VICON frame for now)
# # -----------------------------------------------------------------------------
# # If your aligned CSV Pos_* are already meters -> keep 1.0
# # If they are mm -> set 1e-3
# vicon_pos_scale = 1e-3

# # For now: treat this as a CONSTANT translation expressed in Vicon world axes.
# pose_cam_tcp = np.array(
#     [-0.02799, -0.206246, -0.093154, 0.0, 0.0, 0.0],
#     dtype=np.float64
# )
# T_cam_tcp = pose_to_mat(pose_cam_tcp)
# t_shift_vicon = T_cam_tcp[:3, 3].copy()
# t_shift_slam  = R_vicon_to_slam @ t_shift_vicon   # column-vector convention

# def natural_key(s: str):
#     return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

# def get_vicon_csv_for_episode(ep: int, vicon_dir_path: pathlib.Path, csv_list_sorted=None) -> pathlib.Path:
#     if use_pattern:
#         file_i = (ep + 1) if one_based else ep
#         return vicon_dir_path / pattern.format(i=file_i)
#     else:
#         assert csv_list_sorted is not None
#         if ep < 0 or ep >= len(csv_list_sorted):
#             raise RuntimeError(f"Episode {ep} out of range for vicon csv list (len={len(csv_list_sorted)})")
#         return vicon_dir_path / csv_list_sorted[ep]

# def read_aligned_vicon_csv(p: pathlib.Path) -> pd.DataFrame:
#     df = pd.read_csv(p)

#     required = ["Pos_X","Pos_Y","Pos_Z","Rot_X","Rot_Y","Rot_Z","Rot_W"]
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise RuntimeError(f"Missing columns {missing} in {p}. Found: {df.columns.tolist()}")

#     # numeric and finite
#     for c in required:
#         df[c] = pd.to_numeric(df[c], errors="coerce")
#     df = df[np.isfinite(df[required]).all(axis=1)].reset_index(drop=True)

#     return df

# def aligned_df_to_pose_arrays(df: pd.DataFrame):
#     pos = df[["Pos_X","Pos_Y","Pos_Z"]].to_numpy(dtype=np.float64) * vicon_pos_scale
#     quat = df[["Rot_X","Rot_Y","Rot_Z","Rot_W"]].to_numpy(dtype=np.float64)

#     # normalize quat + hemisphere continuity (robust)
#     n = np.linalg.norm(quat, axis=1, keepdims=True)
#     good = (n[:,0] > 1e-8) & np.isfinite(n[:,0])
#     if good.sum() < 2:
#         raise RuntimeError("Quaternion stream has <2 valid samples after filtering.")

#     quat = quat[good]
#     pos  = pos[good]

#     quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
#     for i in range(1, len(quat)):
#         if np.dot(quat[i-1], quat[i]) < 0:
#             quat[i] *= -1

#     rotvec = R.from_quat(quat).as_rotvec()  # (N,3)

#     return pos, rotvec

# # -----------------------------------------------------------------------------
# # Main
# # -----------------------------------------------------------------------------
# src_path = str(pathlib.Path(src_path).expanduser().absolute())
# dst_path = str(pathlib.Path(dst_path).expanduser().absolute())
# vicon_dir_path = pathlib.Path(vicon_dir).expanduser().absolute()
# if not vicon_dir_path.is_dir():
#     raise RuntimeError(f"vicon_dir does not exist: {vicon_dir_path}")

# vicon_csv_names_sorted = None
# if not use_pattern:
#     vicon_csv_names_sorted = sorted([p.name for p in vicon_dir_path.glob("*.csv")], key=natural_key)
#     if len(vicon_csv_names_sorted) == 0:
#         raise RuntimeError(f"No CSV files found in {vicon_dir_path}")

# # Read meta + original dtype/chunks (from camera-only dataset)
# with zarr.ZipStore(src_path, mode="r") as src_store:
#     root = zarr.open(src_store)
#     episode_ends_idx = np.asarray(root["meta"]["episode_ends"][:], dtype=int)
#     episode_start_idxs = np.concatenate(([0], episode_ends_idx[:-1]))
#     num_episodes = len(episode_ends_idx)

#     # choose reference array for chunking if robot arrays don't exist yet
#     # camera0_rgb is usually huge; better to use "timestamp" if it exists
#     if "timestamp" in root["data"]:
#         ref_arr = root["data"]["timestamp"]
#     else:
#         # fallback: any array under data
#         ref_key = list(root["data"].array_keys())[0]
#         ref_arr = root["data"][ref_key]

#     T_data = int(episode_ends_idx[-1])
#     print(f"Loaded meta: num_episodes={num_episodes}, T={T_data}")
#     print(f"t_shift_vicon (meters): {t_shift_vicon}")

# # Load replay buffer into memory
# with zarr.ZipStore(src_path, mode="r") as src_store:
#     replay_buffer = ReplayBuffer.copy_from_store(src_store=src_store, store=zarr.MemoryStore())

# data = replay_buffer.data
# T = T_data
# Dpose = 6

# # Decide dtypes (since they may not exist yet in camera-only dataset)
# pos_dtype = np.float32
# rot_dtype = np.float32
# pose_dtype = np.float32

# eef_pos_new = np.empty((T, 3), dtype=pos_dtype)
# eef_rot_new = np.empty((T, 3), dtype=rot_dtype)
# demo_start_new = np.empty((T, Dpose), dtype=pose_dtype)
# demo_end_new   = np.empty((T, Dpose), dtype=pose_dtype)

# bad_eps = []

# for ep in range(num_episodes):
#     s = int(episode_start_idxs[ep])
#     e = int(episode_ends_idx[ep])
#     ep_len = e - s
#     if ep_len <= 0:
#         raise RuntimeError(f"Episode {ep} has non-positive length: {ep_len}")

#     vicon_csv_path = get_vicon_csv_for_episode(ep, vicon_dir_path, csv_list_sorted=vicon_csv_names_sorted)
#     if not vicon_csv_path.is_file():
#         raise RuntimeError(f"Missing aligned Vicon CSV for episode {ep}: {vicon_csv_path}")

#     df = read_aligned_vicon_csv(vicon_csv_path)
#     pos_cam_vicon, rotvec_cam = aligned_df_to_pose_arrays(df)

#     if pos_cam_vicon.shape[0] != ep_len:
#         # keep going but record, because you said you will delete later
#         bad_eps.append((ep, ep_len, pos_cam_vicon.shape[0], str(vicon_csv_path)))
#         # safest: trim to min length to avoid crash
#         L = min(ep_len, pos_cam_vicon.shape[0])
#         pos_cam_vicon = pos_cam_vicon[:L]
#         rotvec_cam = rotvec_cam[:L]
#         e = s + L
#         ep_len = L

#     # 1) Rotate trajectory into SLAM world axes (position + orientation)
#     pos_cam_slam = pos_cam_vicon @ R_vicon_to_slam.T     # row-vector form
#     rot_slam = rot_v2s * R.from_rotvec(rotvec_cam)       # left-multiply: slam_R = Rv2s * vicon_R
#     rotvec_slam = rot_slam.as_rotvec()

#     # 2) Apply constant shift in SLAM axes
#     pos_tcp_slam = pos_cam_slam + t_shift_slam

#     # Write per-step
#     eef_pos_new[s:e, :] = pos_tcp_slam.astype(pos_dtype, copy=False)
#     eef_rot_new[s:e, :] = rotvec_slam.astype(rot_dtype, copy=False)

#     # Start/end pose per step (repeat)
#     start_pose = np.concatenate([pos_tcp_slam[0], rotvec_slam[0]], axis=0).astype(pose_dtype, copy=False)
#     end_pose   = np.concatenate([pos_tcp_slam[-1], rotvec_slam[-1]], axis=0).astype(pose_dtype, copy=False)

#     demo_start_new[s:e, :] = start_pose[None, :].repeat(ep_len, axis=0)
#     demo_end_new[s:e, :]   = end_pose[None, :].repeat(ep_len, axis=0)


# if bad_eps:
#     print("\nWARNING: length mismatches (ep, zarr_len, vicon_len, file):")
#     for row in bad_eps[:20]:
#         print("  ", row)
#     print(f"  ... total mismatches: {len(bad_eps)}\n")

# # Overwrite / create datasets
# for key in ["robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_demo_start_pose", "robot0_demo_end_pose"]:
#     if key in data:
#         del data[key]

# data.create_dataset("robot0_eef_pos", data=eef_pos_new, chunks=(ref_arr.chunks[0], 3))
# data.create_dataset("robot0_eef_rot_axis_angle", data=eef_rot_new, chunks=(ref_arr.chunks[0], 3))
# data.create_dataset("robot0_demo_start_pose", data=demo_start_new, chunks=(ref_arr.chunks[0], 6))
# data.create_dataset("robot0_demo_end_pose", data=demo_end_new, chunks=(ref_arr.chunks[0], 6))

# # Save
# with zarr.ZipStore(dst_path, mode="w") as dst_store:
#     replay_buffer.save_to_store(dst_store)

# print(f"Saved updated dataset to: {dst_path}")

# # Quick verify
# with zarr.ZipStore(dst_path, mode="r") as verify_store:
#     root_out = zarr.open(verify_store)
#     print("Verify shapes:")
#     print("  robot0_eef_pos:           ", root_out["data"]["robot0_eef_pos"].shape)
#     print("  robot0_eef_rot_axis_angle:", root_out["data"]["robot0_eef_rot_axis_angle"].shape)
#     print("  robot0_demo_start_pose:   ", root_out["data"]["robot0_demo_start_pose"].shape)
#     print("  robot0_demo_end_pose:     ", root_out["data"]["robot0_demo_end_pose"].shape)
#     print("  meta/episode_ends last:   ", int(root_out["meta"]["episode_ends"][-1]))



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
src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/dataset_camera_only.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/peg_in_hole_vicon.zarr.zip"
vicon_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/aligned_vicon_files/aligned_vicon_to_episode"

# -----------------------------------------------------------------------------
# Config & Geometry
# -----------------------------------------------------------------------------
use_pattern = True
pattern = "aligned_episode_{i:03d}.csv"
one_based = True
vicon_pos_scale = 1e-3

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

    # --- 1. LOCAL TRANSFORMATION (Vicon Space) ---
    # Convert camera quat to scipy rotation
    rot_cam_vicon = R.from_quat(quat_vicon)
    
    # Position: Apply the local shift relative to the camera's orientation
    vicon_dynamic_shift = rot_cam_vicon.apply(t_local_shift)
    pos_tcp_vicon = pos_vicon + vicon_dynamic_shift

    # Orientation: Apply the tool axis swap (Right-Multiply)
    rot_tcp_vicon = rot_cam_vicon * R_local_rot

    # --- 2. GLOBAL WORLD TRANSFORMATION (Rotate Entire World 180 deg) ---
    # Rotate the finished position vector around the world Z-axis
    pos_tcp_slam = pos_tcp_vicon @ R_vicon_to_slam.T 
    
    # Rotate the finished orientation (Left-Multiply by world rotation)
    rot_tcp_slam = rot_v2s * rot_tcp_vicon
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

data.create_dataset("robot0_eef_pos", data=eef_pos_new, chunks=(ref_arr.chunks[0], 3))
data.create_dataset("robot0_eef_rot_axis_angle", data=eef_rot_new, chunks=(ref_arr.chunks[0], 3))
data.create_dataset("robot0_demo_start_pose", data=demo_start_new, chunks=(ref_arr.chunks[0], 6))
data.create_dataset("robot0_demo_end_pose", data=demo_end_new, chunks=(ref_arr.chunks[0], 6))

with zarr.ZipStore(dst_path, mode="w") as dst_store:
    replay_buffer.save_to_store(dst_store)

print(f"Done. Saved to {dst_path}")