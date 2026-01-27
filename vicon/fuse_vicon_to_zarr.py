#!/usr/bin/env python3
"""
Replace (overwrite) robot0_eef_pos / robot0_eef_rot_axis_angle using Vicon poses,
then recompute robot0_demo_start_pose / robot0_demo_end_pose using meta/episode_ends.

Vicon CSV is expected to contain: TX,TY,TZ,RX,RY,RZ,RW (quat xyzw).
- TX/TY/TZ are scaled by vicon_pos_scale (1e-3 if mm -> m).
- We apply a fixed cam->tcp transform (UMI constants + tcp_offset).
- Output orientation is stored as rotvec (axis-angle vector), i.e. Rotation.as_rotvec().

Writes to a new Zarr zip store (same style as your script).
"""

import os
import re
import pathlib
import zarr
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.pose_util import pose_to_mat  # for cam->tcp

register_codecs()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_with_vicon_segmented.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_with_vicon_final.zarr.zip"

vicon_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_resampled_to_slam_3"

# -----------------------------------------------------------------------------
# Vicon file mapping options
# -----------------------------------------------------------------------------
# Option A (pattern): episode 0 -> file index 1 (one_based=True) => peg_umi_quat 1.csv
use_pattern = True
pattern = "peg_umi_quat {i}.csv"
one_based = True

# Option B (sorted list): ignore pattern and just take all CSVs in vicon_dir sorted naturally
# use_pattern = False

# -----------------------------------------------------------------------------
# Vicon / transform config
# -----------------------------------------------------------------------------
vicon_pos_scale = 1e-3  # mm -> m (use 1.0 if already meters)

# tcp_offset = 0.205            # meters, tip -> mounting screw
# cam_to_center_height = 0.086  # meters (UMI constant)
# cam_to_mount_offset = 0.01465 # meters (GoPro constant)

# If vicon length != episode length:
allow_resample = False  # if False -> raise error
# -----------------------------------------------------------------------------

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
    required = ["Frame", "Sub Frame", "TX", "TY", "TZ", "RX", "RY", "RZ", "RW"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing} in {vicon_csv_path}")
    return df

def vicon_df_to_pos_quat(df: pd.DataFrame, pos_scale: float):
    pos = df[["TX", "TY", "TZ"]].to_numpy(dtype=np.float64) * pos_scale
    quat = df[["RX", "RY", "RZ", "RW"]].to_numpy(dtype=np.float64)

    n = np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12
    quat = quat / n
    return pos, quat

def pos_quat_to_T(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    Rm = Rotation.from_quat(quat).as_matrix()
    T = np.zeros((len(pos), 4, 4), dtype=np.float64)
    T[:, 3, 3] = 1.0
    T[:, :3, :3] = Rm
    T[:, :3, 3] = pos
    return T

# def resample_pos_quat(pos: np.ndarray, quat: np.ndarray, out_len: int):
#     N = pos.shape[0]
#     if N == out_len:
#         return pos, quat
#     if N < 2:
#         raise RuntimeError(f"Cannot resample Vicon with N={N}")

#     t_in = np.linspace(0.0, 1.0, N)
#     t_out = np.linspace(0.0, 1.0, out_len)

#     pos_out = np.stack([np.interp(t_out, t_in, pos[:, d]) for d in range(3)], axis=1)

#     r_in = Rotation.from_quat(quat)
#     slerp = Slerp(t_in, r_in)
#     r_out = slerp(t_out)
#     quat_out = r_out.as_quat()

#     return pos_out.astype(np.float64), quat_out.astype(np.float64)

def get_vicon_csv_for_episode(ep: int, vicon_dir_path: pathlib.Path, csv_list_sorted=None) -> pathlib.Path:
    if use_pattern:
        file_i = (ep + 1) if one_based else ep
        p = vicon_dir_path / pattern.format(i=file_i)
        return p
    else:
        assert csv_list_sorted is not None
        if ep < 0 or ep >= len(csv_list_sorted):
            raise RuntimeError(f"Episode {ep} out of range for vicon csv list (len={len(csv_list_sorted)})")
        return vicon_dir_path / csv_list_sorted[ep]

# -----------------------------------------------------------------------------
# 1) Read episode boundaries + original array metadata (dtype/chunks/compressor)
# -----------------------------------------------------------------------------
src_path = str(pathlib.Path(src_path).expanduser().absolute())
dst_path = str(pathlib.Path(dst_path).expanduser().absolute())
vicon_dir_path = pathlib.Path(vicon_dir).expanduser().absolute()

if not vicon_dir_path.is_dir():
    raise RuntimeError(f"vicon_dir does not exist: {vicon_dir_path}")

vicon_csv_names_sorted = None
if not use_pattern:
    vicon_csv_names_sorted = sorted(
        [p.name for p in vicon_dir_path.glob("*.csv")],
        key=natural_key
    )
    if len(vicon_csv_names_sorted) == 0:
        raise RuntimeError(f"No CSV files found in {vicon_dir_path}")

with zarr.ZipStore(src_path, mode="r") as src_store:
    root = zarr.open(src_store)

    episode_ends_idx = np.asarray(root["meta"]["episode_ends"][:], dtype=int)  # [E], exclusive
    episode_start_idxs = np.concatenate(([0], episode_ends_idx[:-1]))          # [E], inclusive
    num_episodes = len(episode_ends_idx)

    # Original metadata for arrays we will overwrite
    keys_to_overwrite = [
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_demo_start_pose",
        "robot0_demo_end_pose",
    ]
    orig = {}
    for key in keys_to_overwrite:
        arr = root["data"][key]
        orig[key] = dict(dtype=arr.dtype, chunks=arr.chunks, compressor=arr.compressor)

    # Sanity check T
    T_data = root["data"]["robot0_eef_pos"].shape[0]
    assert episode_ends_idx[-1] == T_data, f"episode_ends last={episode_ends_idx[-1]} != T={T_data}"

print(f"Loaded meta: num_episodes={num_episodes}, T={episode_ends_idx[-1]}")

# -----------------------------------------------------------------------------
# 2) Load replay buffer into memory
# -----------------------------------------------------------------------------
with zarr.ZipStore(src_path, mode="r") as src_store:
    replay_buffer = ReplayBuffer.copy_from_store(
        src_store=src_store,
        store=zarr.MemoryStore()
    )

data = replay_buffer.data

T = data["robot0_eef_pos"].shape[0]
Dpos = data["robot0_eef_pos"].shape[1]
Drot = data["robot0_eef_rot_axis_angle"].shape[1]
assert Dpos == 3, f"Expected robot0_eef_pos dim=3, got {Dpos}"
assert Drot == 3, f"Expected robot0_eef_rot_axis_angle dim=3 (rotvec), got {Drot}"
Dpose = Dpos + Drot

# -----------------------------------------------------------------------------
# 3) Build constant cam->tcp transform
# -----------------------------------------------------------------------------
# cam_to_tip_offset = cam_to_mount_offset + tcp_offset
# pose_cam_tcp = np.array([0.0, cam_to_center_height, cam_to_tip_offset, 0.0, 0.0, 0.0], dtype=np.float64)
# T_cam_tcp = pose_to_mat(pose_cam_tcp).astype(np.float64)  # (4,4)

pose_cam_tcp = np.array([-0.01938062, -0.19540817, -0.09206965, 0.0, 0.0, 0.0], dtype=np.float64)
T_cam_tcp = pose_to_mat(pose_cam_tcp)

# -----------------------------------------------------------------------------
# 4) Compute new eef pos/rot (from Vicon) and demo start/end pose arrays
# -----------------------------------------------------------------------------
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
    pos_cam, quat_cam = vicon_df_to_pos_quat(vdf, pos_scale=vicon_pos_scale)

    # if pos_cam.shape[0] != ep_len:
    #     if not allow_resample:
    #         raise RuntimeError(
    #             f"Episode {ep} length mismatch: vicon={pos_cam.shape[0]} vs ep_len={ep_len} ({vicon_csv_path})"
    #         )
    #     pos_cam, quat_cam = resample_pos_quat(pos_cam, quat_cam, out_len=ep_len)

    if pos_cam.shape[0] != ep_len:
        raise RuntimeError(
            f"Episode {ep} length mismatch: vicon={pos_cam.shape[0]} vs ep_len={ep_len} ({vicon_csv_path})"
        )

    # cam->tcp
    T_world_cam = pos_quat_to_T(pos_cam, quat_cam)        # (L,4,4)
    T_world_tcp = T_world_cam @ T_cam_tcp                 # (L,4,4)

    pos_tcp = T_world_tcp[:, :3, 3]                       # (L,3)
    rot_tcp = Rotation.from_matrix(T_world_tcp[:, :3, :3])
    rotvec_tcp = rot_tcp.as_rotvec()                      # (L,3)

    # write per-step eef
    eef_pos_new[s:e, :] = pos_tcp.astype(orig["robot0_eef_pos"]["dtype"], copy=False)
    eef_rot_new[s:e, :] = rotvec_tcp.astype(orig["robot0_eef_rot_axis_angle"]["dtype"], copy=False)

    # start/end pose per step (repeat across episode)
    start_pose = np.concatenate([pos_tcp[0], rotvec_tcp[0]], axis=0).astype(orig["robot0_demo_start_pose"]["dtype"], copy=False)
    end_pose   = np.concatenate([pos_tcp[-1], rotvec_tcp[-1]], axis=0).astype(orig["robot0_demo_end_pose"]["dtype"],   copy=False)

    demo_start_new[s:e, :] = start_pose[None, :].repeat(ep_len, axis=0)
    demo_end_new[s:e, :]   = end_pose[None, :].repeat(ep_len, axis=0)

print("Computed new Vicon-based eef pos/rot and demo start/end poses.")

# -----------------------------------------------------------------------------
# 5) Overwrite datasets in replay buffer (same style as your script)
# -----------------------------------------------------------------------------
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

# meta stays the same (episode_ends unchanged)

# -----------------------------------------------------------------------------
# 6) Save to destination and verify quickly
# -----------------------------------------------------------------------------
with zarr.ZipStore(dst_path, mode="w") as dst_store:
    replay_buffer.save_to_store(dst_store)

print(f"Saved updated dataset to: {dst_path}")

with zarr.ZipStore(dst_path, mode="r") as verify_store:
    root_out = zarr.open(verify_store)
    print("Verify shapes:")
    print("  robot0_eef_pos:           ", root_out["data"]["robot0_eef_pos"].shape)
    print("  robot0_eef_rot_axis_angle:", root_out["data"]["robot0_eef_rot_axis_angle"].shape)
    print("  robot0_demo_start_pose:   ", root_out["data"]["robot0_demo_start_pose"].shape)
    print("  robot0_demo_end_pose:     ", root_out["data"]["robot0_demo_end_pose"].shape)
    print("  meta/episode_ends last:   ", int(root_out["meta"]["episode_ends"][-1]))
