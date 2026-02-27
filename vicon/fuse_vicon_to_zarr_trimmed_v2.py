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
SRC_ZARR = "/home/hisham246/uwaterloo/peg_in_hole_delta_umi/dataset_camera_only.zarr.zip"
DST_ZARR = "/home/hisham246/uwaterloo/peg_in_hole_delta_umi/dataset_with_vicon_trimmed_2.zarr.zip"
VICON_TRIMMED_DIR = "/home/hisham246/uwaterloo/peg_in_hole_delta_umi/vicon_trimmed_3"

# -----------------------------------------------------------------------------
# Vicon CSV naming
# -----------------------------------------------------------------------------
USE_PATTERN = True
PATTERN = "aligned_episode_{i:03d}.csv"
ONE_BASED = True
vicon_pos_scale = 1e-3

# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
# R_local defines the mapping from your DESIRED local axes to the CURRENT local axes:
R_local_mat = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0],
    [0.0, -1.0,  0.0]
])
rot_local = R.from_matrix(R_local_mat)

# 180-degree rotation around Z-axis
# This matrix represents a 180-deg rotation about Z
R_v2s = R.from_matrix(np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0,  1.0],
]))

# rot_v2s = R.from_matrix(R_vicon_to_slam_mat)

# The offset to be applied in the NEW frame
vicon_world_offset = np.array([0.02799, 0.206246, -0.093154], dtype=np.float64)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def ep_csv_path(ep: int, base_dir: pathlib.Path) -> pathlib.Path:
    if USE_PATTERN:
        file_i = (ep + 1) if ONE_BASED else ep
        return base_dir / PATTERN.format(i=file_i)
    files = sorted([p.name for p in base_dir.glob("*.csv")], key=natural_key)
    return base_dir / files[ep]

# def read_trimmed_vicon_csv(p: pathlib.Path):
#     df = pd.read_csv(p)
#     required = ["Pos_X","Pos_Y","Pos_Z","Rot_X","Rot_Y","Rot_Z","Rot_W"]
#     for c in required:
#         df[c] = pd.to_numeric(df[c], errors="coerce")
#     df = df[np.isfinite(df[required]).all(axis=1)].reset_index(drop=True)

#     pos = df[["Pos_X","Pos_Y","Pos_Z"]].to_numpy(dtype=np.float64) * vicon_pos_scale
#     quat = df[["Rot_X","Rot_Y","Rot_Z","Rot_W"]].to_numpy(dtype=np.float64)

#     # normalize + hemisphere continuity
#     quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
#     for i in range(1, len(quat)):
#         if np.dot(quat[i-1], quat[i]) < 0:
#             quat[i] *= -1
#     return pos, quat

def read_trimmed_vicon_csv(p: pathlib.Path):
    df = pd.read_csv(p)
    required = ["Pos_X","Pos_Y","Pos_Z","Rot_X","Rot_Y","Rot_Z","Rot_W"]
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df[required]).all(axis=1)].reset_index(drop=True)

    # 1. Extract raw position (scaled to meters) and quaternions
    pos = df[["Pos_X","Pos_Y","Pos_Z"]].to_numpy(dtype=np.float64) * vicon_pos_scale
    quat = df[["Rot_X","Rot_Y","Rot_Z","Rot_W"]].to_numpy(dtype=np.float64)

    # TRANSFORM POSITION
    # 1. Flip world 180 around Z, 2. Add translation
    pos_final = R_v2s.apply(pos) + vicon_world_offset

    # TRANSFORM ORIENTATION (The Gripper/TCP)
    raw_rotations = R.from_quat(quat)
    
    # We apply the world-flip FIRST (to orient the sensor) 
    # and the local-TCP transformation LAST (to orient the tool)
    final_rotations = R_v2s * raw_rotations * rot_local 

    # Convert to axis-angle (rotation vector) as expected by your script later
    rotvec_final = final_rotations.as_rotvec()

    # Normalize quaternions for continuity (if you still need quats elsewhere)
    quat_transformed = final_rotations.as_quat()
    for i in range(1, len(quat_transformed)):
        if np.dot(quat_transformed[i-1], quat_transformed[i]) < 0:
            quat_transformed[i] *= -1
            
    return pos_final, rotvec_final

def first_non_empty_ep_len(episode_ends: np.ndarray) -> int:
    starts = np.concatenate(([0], episode_ends[:-1]))
    lens = episode_ends - starts
    nz = lens[lens > 0]
    return int(nz[0]) if len(nz) else 1

def is_camera_key(name: str) -> bool:
    return name.startswith("camera") and ("rgb" in name or "depth" in name)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
vicon_dir = pathlib.Path(VICON_TRIMMED_DIR).expanduser().absolute()

with zarr.ZipStore(SRC_ZARR, mode="r") as src_store:
    root = zarr.open(src_store, mode="r")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=int)
    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    num_eps = len(episode_ends)

    # capture encoding from camera-only dataset
    orig_info = {}
    for k in root["data"].keys():
        arr = root["data"][k]
        orig_info[k] = {"dtype": arr.dtype, "chunks": arr.chunks, "compressor": arr.compressor}

    rb = ReplayBuffer.copy_from_store(src_store=src_store, store=zarr.MemoryStore())

data = rb.data
keys = list(data.keys())

# ---- pass 1: compute slices using trimmed vicon length
ep_slices = []
total_new = 0
per_ep_L = []  # store L for later

for ep in range(num_eps):
    s = int(episode_starts[ep])
    e = int(episode_ends[ep])
    ep_len = e - s

    vicon_csv = ep_csv_path(ep, vicon_dir)
    if not vicon_csv.exists():
        raise FileNotFoundError(f"Missing trimmed vicon CSV for ep {ep}: {vicon_csv}")

    pos_v, quat_v = read_trimmed_vicon_csv(vicon_csv)
    n_trim = len(pos_v)

    keep_start = 0
    keep_end = int(np.clip(n_trim, 0, ep_len))

    ep_slices.append((s + keep_start, s + keep_end))
    per_ep_L.append(keep_end - keep_start)
    total_new += (keep_end - keep_start)

    print(f"EP {ep:03d}: orig_len={ep_len:5d} keep=[{keep_start:5d},{keep_end:5d}) new_len={keep_end-keep_start:5d}")

# ---- allocate new arrays for existing keys (this trims camera/video too)
new_arrays = {}
for k in keys:
    arr = data[k]
    new_shape = (total_new,) if arr.ndim == 1 else (total_new,) + arr.shape[1:]
    new_arrays[k] = np.empty(new_shape, dtype=arr.dtype)

# ---- allocate new vicon-derived arrays
eef_pos_new = np.empty((total_new, 3), dtype=np.float32)
eef_rot_new = np.empty((total_new, 3), dtype=np.float32)
demo_start_new = np.empty((total_new, 6), dtype=np.float32)
demo_end_new   = np.empty((total_new, 6), dtype=np.float32)

# ---- pass 2: copy trimmed camera arrays + compute/write trimmed vicon arrays
write_ptr = 0
new_episode_ends = []

for ep, (gs, ge) in enumerate(ep_slices):
    L = ge - gs
    if L <= 0:
        new_episode_ends.append(write_ptr)
        continue

    # copy camera-only arrays trimmed
    for k in keys:
        new_arrays[k][write_ptr:write_ptr + L] = data[k][gs:ge]

    # Get the already-transformed data from the helper
    vicon_csv = ep_csv_path(ep, vicon_dir)
    pos_v, rotvec_v = read_trimmed_vicon_csv(vicon_csv)
    
    pos_v = pos_v[:L]
    rotvec_v = rotvec_v[:L]

    # Assign to new arrays
    eef_pos_new[write_ptr:write_ptr + L] = pos_v.astype(np.float32)
    eef_rot_new[write_ptr:write_ptr + L] = rotvec_v.astype(np.float32)

    sp = np.concatenate([eef_pos_new[write_ptr], eef_rot_new[write_ptr]])
    ep_pose = np.concatenate([eef_pos_new[write_ptr + L - 1], eef_rot_new[write_ptr + L - 1]])
    demo_start_new[write_ptr:write_ptr + L] = sp
    demo_end_new[write_ptr:write_ptr + L] = ep_pose

    write_ptr += L
    new_episode_ends.append(write_ptr)

assert write_ptr == total_new

# ---- rebuild datasets in rb.data
for k in list(data.keys()):
    del data[k]

# chunks: use first episode length (post-trim) for non-camera keys
first_ep_len = first_non_empty_ep_len(np.asarray(new_episode_ends, dtype=int))
tchunk_dyn = int(np.clip(first_ep_len, 32, 2048))
tchunk_dyn = min(tchunk_dyn, total_new) if total_new > 0 else 1

# recreate original keys with same encoding as camera-only
for k in keys:
    info = orig_info[k]
    chunks = info["chunks"]
    compressor = info["compressor"]
    dtype = info["dtype"]

    if chunks is None:
        out_chunks = None
    else:
        if is_camera_key(k):
            tchunk = min(chunks[0], total_new) if total_new > 0 else 1
        else:
            tchunk = tchunk_dyn
        out_chunks = (tchunk,) + tuple(chunks[1:])

    data.create_dataset(k, data=new_arrays[k], dtype=dtype, chunks=out_chunks, compressor=compressor)

# add vicon-derived keys: compressor=None, chunks=(first_ep_len, D)
data.create_dataset("robot0_eef_pos", data=eef_pos_new, dtype=np.float32, chunks=(tchunk_dyn, 3), compressor=None)
data.create_dataset("robot0_eef_rot_axis_angle", data=eef_rot_new, dtype=np.float32, chunks=(tchunk_dyn, 3), compressor=None)
data.create_dataset("robot0_demo_start_pose", data=demo_start_new, dtype=np.float32, chunks=(tchunk_dyn, 6), compressor=None)
data.create_dataset("robot0_demo_end_pose", data=demo_end_new, dtype=np.float32, chunks=(tchunk_dyn, 6), compressor=None)

# update episode_ends
rb.meta["episode_ends"][:] = np.asarray(new_episode_ends, dtype=np.int64)

with zarr.ZipStore(DST_ZARR, mode="w") as dst_store:
    rb.save_to_store(dst_store)

print(f"\nDone. Saved trimmed+fused Zarr to: {DST_ZARR}")
print(f"Total steps: {total_new} (was {int(episode_ends[-1])})")
print(f"Episodes: {num_eps}")