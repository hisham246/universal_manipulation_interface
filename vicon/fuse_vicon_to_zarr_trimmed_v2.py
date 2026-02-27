#!/usr/bin/env python3
import re
import pathlib
import zarr
import numpy as np
import pandas as pd

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.pose_util import pose_to_mat, mat_to_pose
from scipy.spatial.transform import Rotation as R

register_codecs()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# SRC_ZARR = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/dataset_camera_only_segmented.zarr.zip"
# DST_ZARR = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/dataset_with_vicon_segmented.zarr.zip"
# VICON_TRIMMED_DIR = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/aligned_vicon_files_segmented/aligned_vicon_to_episode"


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

MAX_EPISODES = 80

# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
# R_local defines the mapping from your DESIRED local axes to the CURRENT local axes:
R_local_mat = np.array([
    [-1.0,  0.0,  0.0],
    [0.0,  0.0,  -1.0],
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
# vicon_world_offset = np.array([0.02799, 0.206246, -0.093154], dtype=np.float64)
# vicon_world_offset = np.array([0.02799, -0.093154, -0.206246], dtype=np.float64)


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

# def read_trimmed_vicon_csv(p: pathlib.Path):
#     df = pd.read_csv(p)
#     required = ["Pos_X","Pos_Y","Pos_Z","Rot_X","Rot_Y","Rot_Z","Rot_W"]
#     for c in required:
#         df[c] = pd.to_numeric(df[c], errors="coerce")
#     df = df[np.isfinite(df[required]).all(axis=1)].reset_index(drop=True)

#     # 1. Extract raw position (scaled to meters) and quaternions
#     pos = df[["Pos_X","Pos_Y","Pos_Z"]].to_numpy(dtype=np.float64) * vicon_pos_scale
#     quat = df[["Rot_X","Rot_Y","Rot_Z","Rot_W"]].to_numpy(dtype=np.float64)

#     # rot_y_adj = R.from_euler('y', 20, degrees=True)

#     # TRANSFORM POSITION
#     # 1. Flip world 180 around Z, 2. Add translation
#     pos_final = R_v2s.apply(pos) + vicon_world_offset

#     # TRANSFORM ORIENTATION (The Gripper/TCP)
#     raw_rotations = R.from_quat(quat)
    
#     # We apply the world-flip FIRST (to orient the sensor) 
#     # and the local-TCP transformation LAST (to orient the tool)
#     final_rotations = R_v2s * raw_rotations * rot_local

#     # Convert to axis-angle (rotation vector) as expected by your script later
#     rotvec_final = final_rotations.as_rotvec()

#     # Normalize quaternions for continuity (if you still need quats elsewhere)
#     quat_transformed = final_rotations.as_quat()
#     for i in range(1, len(quat_transformed)):
#         if np.dot(quat_transformed[i-1], quat_transformed[i]) < 0:
#             quat_transformed[i] *= -1
            
#     return pos_final, rotvec_final

# -----------------------------------------------------------------------------
# Optional per-episode Y correction from summary_final_angles.csv
# -----------------------------------------------------------------------------
SUMMARY_FINAL_ANGLES = "/home/hisham246/uwaterloo/peg_in_hole_delta_umi/VIDP_data/summary_final_angles.csv"  # <-- set me (or None)

def build_ep_y_correction_map(summary_csv: str, one_based: bool):
    """
    Returns dict ep_index(0-based) -> y_correction_deg (float)
    Uses: y_correction_deg = -int(final_euler_y_deg_xyz)
    """
    df = pd.read_csv(summary_csv)
    if "final_euler_y_deg_xyz" not in df.columns:
        raise ValueError("summary_final_angles.csv missing column: final_euler_y_deg_xyz")

    ep_to_y = {}
    for _, row in df.iterrows():
        fname = str(row["file"]) if "file" in df.columns else ""
        # try to parse episode index from filename, e.g. aligned_episode_001.csv
        m = re.search(r"(\d+)", fname)
        if m is None:
            continue
        i = int(m.group(1))
        ep0 = (i - 1) if one_based else i

        y_final = float(row["final_euler_y_deg_xyz"])
        y_int = int(y_final)  # "integer digit" (trunc toward 0)
        ep_to_y[ep0] = float(y_int)  # correction so final y ~ 0

    return ep_to_y


EP_Y_CORR = None
if SUMMARY_FINAL_ANGLES is not None and len(str(SUMMARY_FINAL_ANGLES)) > 0:
    EP_Y_CORR = build_ep_y_correction_map(SUMMARY_FINAL_ANGLES, one_based=ONE_BASED)
    print(f"Loaded per-episode Y corrections for {len(EP_Y_CORR)} episodes from {SUMMARY_FINAL_ANGLES}")

def read_trimmed_vicon_csv(p: pathlib.Path,  y_corr_deg: float = 0.0):
    df = pd.read_csv(p)
    required = ["Pos_X","Pos_Y","Pos_Z","Rot_X","Rot_Y","Rot_Z","Rot_W"]
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df[required]).all(axis=1)].reset_index(drop=True)

    # camera pose in Vicon world
    pos_V_cam = df[["Pos_X","Pos_Y","Pos_Z"]].to_numpy(dtype=np.float64) * vicon_pos_scale
    quat_V_cam = df[["Rot_X","Rot_Y","Rot_Z","Rot_W"]].to_numpy(dtype=np.float64)
    rot_V_cam = R.from_quat(quat_V_cam)

    # ------------------------------------------------------------
    # Fixed camera -> TCP transform (treat offset as CAMERA-FRAME)
    # (i.e., assumes it was defined at identity alignment)
    # ------------------------------------------------------------
    t_cam = np.array([-0.02799, -0.206246, -0.093154], dtype=np.float64)

    # your desired fixed TCP-axis relabel/rotation (optional)
    # R_cam_tcp = R.from_euler("x", +90, degrees=True)
    R_cam_tcp = R.from_euler("x", +90, degrees=True) * R.from_euler("z", 180, degrees=True)

    tx_cam_tcp = np.eye(4, dtype=np.float64)
    tx_cam_tcp[:3, :3] = R_cam_tcp.as_matrix()
    tx_cam_tcp[:3,  3] = t_cam

    # ------------------------------------------------------------
    # Build T_V_cam(t) and compute T_V_tcp(t) = T_V_cam(t) @ T_cam_tcp
    # ------------------------------------------------------------
    n = len(pos_V_cam)
    tx_V_cam = np.zeros((n, 4, 4), dtype=np.float64)
    tx_V_cam[:, 3, 3] = 1.0
    tx_V_cam[:, :3, 3] = pos_V_cam
    tx_V_cam[:, :3, :3] = rot_V_cam.as_matrix()

    tx_V_tcp = tx_V_cam @ tx_cam_tcp

    # Compare expected delta in Vicon world: delta(t) = p_V_tcp - p_V_cam = R_V_cam(t) @ t_cam
    delta_V = tx_V_tcp[:, :3, 3] - tx_V_cam[:, :3, 3]
    delta_V_expected = rot_V_cam.apply(t_cam)

    print("delta_V mean:", delta_V.mean(axis=0))
    print("delta_V_expected mean:", delta_V_expected.mean(axis=0))
    print("max abs diff:", np.max(np.abs(delta_V - delta_V_expected)))

    # ------------------------------------------------------------
    # World flip (Vicon world -> your desired world): 180 deg about Z
    # ------------------------------------------------------------
    tx_V2S = np.eye(4, dtype=np.float64)
    tx_V2S[:3, :3] = R_v2s.as_matrix()

    tx_S_tcp = tx_V2S @ tx_V_tcp

    # output as (pos, rotvec)
    # pos_final = tx_S_tcp[:, :3, 3]
    # rot_final = R.from_matrix(tx_S_tcp[:, :3, :3])
    # rotvec_final = rot_final.as_rotvec()


    # ------------------------------------------------------------
    # Final adjustment: 20-degree local rotation around Y-axis
    # ------------------------------------------------------------
    # R_y_adj = R.from_euler("y", 20, degrees=True)
    # tx_y_adj = np.eye(4, dtype=np.float64)
    # tx_y_adj[:3, :3] = R_y_adj.as_matrix()


    # Right-multiply to rotate the TCP in place around its local Y-axis
    # tx_S_tcp_final = tx_S_tcp @ tx_y_adj

    #     # output as (pos, rotvec)
    # pos_final = tx_S_tcp_final[:, :3, 3]
    # rot_final = R.from_matrix(tx_S_tcp_final[:, :3, :3])
    # rotvec_final = rot_final.as_rotvec()

        # ------------------------------------------------------------
    # Final adjustment: per-episode local rotation around Y-axis
    # (right-multiply = rotate TCP in its local frame)
    # ------------------------------------------------------------
    if abs(y_corr_deg) > 1e-9:
        R_y_adj = R.from_euler("y", y_corr_deg, degrees=True)
        tx_y_adj = np.eye(4, dtype=np.float64)
        tx_y_adj[:3, :3] = R_y_adj.as_matrix()
        tx_S_tcp = tx_S_tcp @ tx_y_adj

    # output as (pos, rotvec)
    pos_final = tx_S_tcp[:, :3, 3]
    rot_final = R.from_matrix(tx_S_tcp[:, :3, :3])
    rotvec_final = rot_final.as_rotvec()

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

    # ---- limit to first MAX_EPISODES episodes
    if MAX_EPISODES is not None:
        num_eps = min(num_eps, int(MAX_EPISODES))
        episode_ends = episode_ends[:num_eps]
        episode_starts = np.concatenate(([0], episode_ends[:-1]))

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

    # pos_v, quat_v = read_trimmed_vicon_csv(vicon_csv)
    y_corr = float(EP_Y_CORR.get(ep, 0.0)) if EP_Y_CORR is not None else 0.0
    pos_v, quat_v = read_trimmed_vicon_csv(vicon_csv, y_corr_deg=y_corr)
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
    # pos_v, rotvec_v = read_trimmed_vicon_csv(vicon_csv)

    y_corr = float(EP_Y_CORR.get(ep, 0.0)) if EP_Y_CORR is not None else 0.0
    pos_v, rotvec_v = read_trimmed_vicon_csv(vicon_csv, y_corr_deg=y_corr)
    
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