#!/usr/bin/env python3
"""
Overwrite robot0_eef_pos using Vicon camera positions mapped into SLAM axes,
then add a constant translation shift t_shift that is ALREADY expressed in SLAM axes.

Keep robot0_eef_rot_axis_angle exactly as-is from the original SLAM dataset.
Recompute robot0_demo_start_pose / robot0_demo_end_pose using meta/episode_ends.

Key behavior (per your clarified intent):
- Vicon positions are in Vicon world axes; convert them to SLAM world axes first:
    pos_cam_slam = pos_cam_vicon @ R_vicon_to_slam.T
- Then apply constant translation shift in SLAM axes:
    pos_tcp_slam = pos_cam_slam + t_shift_slam
- Orientation stays exactly the original SLAM tcp orientation:
    rotvec_tcp[t] = original_robot0_eef_rot_axis_angle[t]
"""

import re
import pathlib
import zarr
import numpy as np
import pandas as pd

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.pose_util import pose_to_mat

register_codecs()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_slam.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_vicon_3.zarr.zip"
vicon_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_resampled_to_slam_3"

# -----------------------------------------------------------------------------
# Vicon file mapping options
# -----------------------------------------------------------------------------
use_pattern = True
pattern = "peg_umi_quat {i}.csv"
one_based = True
# use_pattern = False  # use sorted list instead of pattern

# -----------------------------------------------------------------------------
# Vicon / transform config
# -----------------------------------------------------------------------------
vicon_pos_scale = 1e-3  # mm -> m

# This vector is in SLAM axis convention (your clarification)
pose_cam_tcp = np.array(
    [-0.01938062, 0.19540817, -0.09206965, 0.0, 0.0, 0.0],
    dtype=np.float64
)
T_cam_tcp = pose_to_mat(pose_cam_tcp)
t_shift_slam = T_cam_tcp[:3, 3].copy()  # (3,)

# Vicon world axes -> SLAM world axes (your observed 180deg about Z)
R_vicon_to_slam = np.array([
    [-1.0,  0.0, 0.0],
    [ 0.0, -1.0, 0.0],
    [ 0.0,  0.0, 1.0],
], dtype=np.float64)

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

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

    required = ["Frame", "Sub Frame", "TX", "TY", "TZ"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing} in {vicon_csv_path}")

    return df

def vicon_df_to_pos(df: pd.DataFrame, pos_scale: float) -> np.ndarray:
    # (N,3) in Vicon world axes
    return df[["TX", "TY", "TZ"]].to_numpy(dtype=np.float64) * pos_scale

def get_vicon_csv_for_episode(ep: int, vicon_dir_path: pathlib.Path, csv_list_sorted=None) -> pathlib.Path:
    if use_pattern:
        file_i = (ep + 1) if one_based else ep
        return vicon_dir_path / pattern.format(i=file_i)
    else:
        assert csv_list_sorted is not None
        if ep < 0 or ep >= len(csv_list_sorted):
            raise RuntimeError(f"Episode {ep} out of range for vicon csv list (len={len(csv_list_sorted)})")
        return vicon_dir_path / csv_list_sorted[ep]

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
src_path = str(pathlib.Path(src_path).expanduser().absolute())
dst_path = str(pathlib.Path(dst_path).expanduser().absolute())
vicon_dir_path = pathlib.Path(vicon_dir).expanduser().absolute()
if not vicon_dir_path.is_dir():
    raise RuntimeError(f"vicon_dir does not exist: {vicon_dir_path}")

vicon_csv_names_sorted = None
if not use_pattern:
    vicon_csv_names_sorted = sorted([p.name for p in vicon_dir_path.glob("*.csv")], key=natural_key)
    if len(vicon_csv_names_sorted) == 0:
        raise RuntimeError(f"No CSV files found in {vicon_dir_path}")

# Read meta + original dtypes/chunks
with zarr.ZipStore(src_path, mode="r") as src_store:
    root = zarr.open(src_store)
    episode_ends_idx = np.asarray(root["meta"]["episode_ends"][:], dtype=int)
    episode_start_idxs = np.concatenate(([0], episode_ends_idx[:-1]))
    num_episodes = len(episode_ends_idx)

    keys_to_overwrite = [
        "robot0_eef_pos",
        "robot0_demo_start_pose",
        "robot0_demo_end_pose",
    ]
    keys_to_read_keep = ["robot0_eef_rot_axis_angle"]

    orig = {}
    for key in keys_to_overwrite + keys_to_read_keep:
        arr = root["data"][key]
        orig[key] = dict(dtype=arr.dtype, chunks=arr.chunks, compressor=arr.compressor)

    T_data = root["data"]["robot0_eef_pos"].shape[0]
    assert episode_ends_idx[-1] == T_data, f"episode_ends last={episode_ends_idx[-1]} != T={T_data}"

print(f"Loaded meta: num_episodes={num_episodes}, T={episode_ends_idx[-1]}")
print(f"t_shift_slam (meters): {t_shift_slam}")
print(f"R_vicon_to_slam:\n{R_vicon_to_slam}")

# Load replay buffer into memory
with zarr.ZipStore(src_path, mode="r") as src_store:
    replay_buffer = ReplayBuffer.copy_from_store(src_store=src_store, store=zarr.MemoryStore())

data = replay_buffer.data
T = data["robot0_eef_pos"].shape[0]
Dpose = 6

# Keep original SLAM tcp orientation exactly
eef_rot_orig = data["robot0_eef_rot_axis_angle"][:]  # (T,3)

eef_pos_new = np.empty((T, 3), dtype=orig["robot0_eef_pos"]["dtype"])
eef_rot_new = np.empty((T, 3), dtype=orig["robot0_eef_rot_axis_angle"]["dtype"])
demo_start_new = np.empty((T, Dpose), dtype=orig["robot0_demo_start_pose"]["dtype"])
demo_end_new   = np.empty((T, Dpose), dtype=orig["robot0_demo_end_pose"]["dtype"])

for ep in range(num_episodes):
    s = int(episode_start_idxs[ep])
    e = int(episode_ends_idx[ep])
    ep_len = e - s
    if ep_len <= 0:
        raise RuntimeError(f"Episode {ep} has non-positive length: {ep_len}")

    vicon_csv_path = get_vicon_csv_for_episode(ep, vicon_dir_path, csv_list_sorted=vicon_csv_names_sorted)
    if not vicon_csv_path.is_file():
        raise RuntimeError(f"Missing Vicon CSV for episode {ep}: {vicon_csv_path}")

    vdf = read_vicon_csv(vicon_csv_path)
    pos_cam_vicon = vicon_df_to_pos(vdf, pos_scale=vicon_pos_scale)

    if pos_cam_vicon.shape[0] != ep_len:
        raise RuntimeError(
            f"Episode {ep} length mismatch: vicon={pos_cam_vicon.shape[0]} vs ep_len={ep_len} ({vicon_csv_path})"
        )

    # 1) Convert Vicon world axes -> SLAM world axes
    # pos_slam = pos_vicon @ R^T  (row-vector form)
    pos_cam_slam = pos_cam_vicon @ R_vicon_to_slam.T

    # 2) Add shift already defined in SLAM axes
    pos_tcp_slam = pos_cam_slam + t_shift_slam  # (N,3)

    # 3) Preserve original SLAM orientation per step (tcp orientation)
    rotvec_tcp = eef_rot_orig[s:e, :].astype(np.float64, copy=False)

    # Write per-step eef
    eef_pos_new[s:e, :] = pos_tcp_slam.astype(orig["robot0_eef_pos"]["dtype"], copy=False)
    eef_rot_new[s:e, :] = rotvec_tcp.astype(orig["robot0_eef_rot_axis_angle"]["dtype"], copy=False)

    # Start/end pose per step (repeat across episode)
    start_pose = np.concatenate([pos_tcp_slam[0], rotvec_tcp[0]], axis=0).astype(
        orig["robot0_demo_start_pose"]["dtype"], copy=False
    )
    end_pose = np.concatenate([pos_tcp_slam[-1], rotvec_tcp[-1]], axis=0).astype(
        orig["robot0_demo_end_pose"]["dtype"], copy=False
    )

    demo_start_new[s:e, :] = start_pose[None, :].repeat(ep_len, axis=0)
    demo_end_new[s:e, :]   = end_pose[None, :].repeat(ep_len, axis=0)

print("Computed new eef pos from Vicon (axes-aligned to SLAM) + constant SLAM shift; kept SLAM eef rot; recomputed demo start/end.")

# Overwrite datasets
for key in ["robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_demo_start_pose", "robot0_demo_end_pose"]:
    if key in data:
        del data[key]

data.create_dataset(
    "robot0_eef_pos",
    data=eef_pos_new,
    chunks=orig["robot0_eef_pos"]["chunks"],
    compressor=orig["robot0_eef_pos"]["compressor"],
)
data.create_dataset(
    "robot0_eef_rot_axis_angle",
    data=eef_rot_new,
    chunks=orig["robot0_eef_rot_axis_angle"]["chunks"],
    compressor=orig["robot0_eef_rot_axis_angle"]["compressor"],
)
data.create_dataset(
    "robot0_demo_start_pose",
    data=demo_start_new,
    chunks=orig["robot0_demo_start_pose"]["chunks"],
    compressor=orig["robot0_demo_start_pose"]["compressor"],
)
data.create_dataset(
    "robot0_demo_end_pose",
    data=demo_end_new,
    chunks=orig["robot0_demo_end_pose"]["chunks"],
    compressor=orig["robot0_demo_end_pose"]["compressor"],
)

# Save
with zarr.ZipStore(dst_path, mode="w") as dst_store:
    replay_buffer.save_to_store(dst_store)

print(f"Saved updated dataset to: {dst_path}")

# Quick verify
with zarr.ZipStore(dst_path, mode="r") as verify_store:
    root_out = zarr.open(verify_store)
    print("Verify shapes:")
    print("  robot0_eef_pos:           ", root_out["data"]["robot0_eef_pos"].shape)
    print("  robot0_eef_rot_axis_angle:", root_out["data"]["robot0_eef_rot_axis_angle"].shape)
    print("  robot0_demo_start_pose:   ", root_out["data"]["robot0_demo_start_pose"].shape)
    print("  robot0_demo_end_pose:     ", root_out["data"]["robot0_demo_end_pose"].shape)
    print("  meta/episode_ends last:   ", int(root_out["meta"]["episode_ends"][-1]))
