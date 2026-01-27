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
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.pose_util import pose_to_mat
from vicon.gopro_to_tcp_transform import quat_mean_markley  # for cam->tcp

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


def rot_jitter_stats(quat_xyzw: np.ndarray, t_cam_tcp: np.ndarray, name="cam"):
    """
    quat_xyzw: (N,4) in xyzw
    t_cam_tcp: (3,) translation expressed in camera frame
    Returns dict of stats and prints a short summary.
    """
    R = Rotation.from_quat(quat_xyzw)

    # relative rotation between consecutive frames: R_{k}^{-1} R_{k+1}
    dR = R[:-1].inv() * R[1:]
    dang = dR.magnitude()  # radians, (N-1,)

    # equivalent position jitter from lever arm
    lever = float(np.linalg.norm(t_cam_tcp))
    pos_equiv = lever * dang  # meters, small-angle approx is fine

    stats = {
        "lever_m": lever,
        "dang_deg_median": float(np.median(dang) * 180/np.pi),
        "dang_deg_mean": float(np.mean(dang) * 180/np.pi),
        "dang_deg_p95": float(np.percentile(dang, 95) * 180/np.pi),
        "dang_deg_max": float(np.max(dang) * 180/np.pi),
        "pos_equiv_mm_median": float(np.median(pos_equiv) * 1e3),
        "pos_equiv_mm_mean": float(np.mean(pos_equiv) * 1e3),
        "pos_equiv_mm_p95": float(np.percentile(pos_equiv, 95) * 1e3),
        "pos_equiv_mm_max": float(np.max(pos_equiv) * 1e3),
    }

    print(f"\n=== Rotation jitter ({name}) ===")
    print(f"lever arm ||t|| = {stats['lever_m']:.4f} m")
    print("delta angle per frame [deg]: "
          f"median={stats['dang_deg_median']:.4f}, mean={stats['dang_deg_mean']:.4f}, "
          f"p95={stats['dang_deg_p95']:.4f}, max={stats['dang_deg_max']:.4f}")
    print("equiv pos jitter from lever [mm]: "
          f"median={stats['pos_equiv_mm_median']:.3f}, mean={stats['pos_equiv_mm_mean']:.3f}, "
          f"p95={stats['pos_equiv_mm_p95']:.3f}, max={stats['pos_equiv_mm_max']:.3f}")
    return stats

def pos_step_stats(pos: np.ndarray, name="pos"):
    dpos = pos[1:] - pos[:-1]
    step = np.linalg.norm(dpos, axis=1)  # meters
    print(f"\n=== Position step stats ({name}) ===")
    print(f"step [mm]: median={np.median(step)*1e3:.3f}, mean={np.mean(step)*1e3:.3f}, "
          f"p95={np.percentile(step,95)*1e3:.3f}, max={np.max(step)*1e3:.3f}")
    return step


def smooth_quat_slerp(quat_xyzw: np.ndarray, win: int = 5) -> np.ndarray:
    """
    Smooth quaternions with a sliding window by slerping to the window mean.
    win should be odd (e.g., 5, 7, 9). Returns xyzw.
    """
    assert win >= 3 and (win % 2 == 1)
    N = quat_xyzw.shape[0]
    q = quat_xyzw.copy()

    # fix sign continuity first (avoid flips)
    for i in range(1, N):
        if np.dot(q[i-1], q[i]) < 0:
            q[i] *= -1.0

    half = win // 2
    out = q.copy()

    for i in range(N):
        a = max(0, i - half)
        b = min(N, i + half + 1)
        # Markley mean on the window
        qw = quat_mean_markley(q[a:b])
        # slerp current toward mean (alpha=1 means replace; use <1 for gentler)
        # out[i] = Rotation.from_quat(qw).as_quat()
        alpha = 0.5  # 0..1
        out[i] = (Rotation.from_quat(q[i]).slerp(alpha, Rotation.from_quat(qw))).as_quat()

    return out


def smooth_quat_exp(quat_xyzw: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """
    Exponential smoothing on rotations:
      R_s[k] = R_s[k-1] * exp( alpha * log( R_s[k-1]^{-1} R[k] ) )
    alpha in (0,1], smaller -> smoother.
    Returns xyzw.
    """
    R = Rotation.from_quat(quat_xyzw)
    N = len(R)
    Rs = [R[0]]
    for k in range(1, N):
        d = Rs[-1].inv() * R[k]
        rv = d.as_rotvec()
        Rs.append(Rs[-1] * Rotation.from_rotvec(alpha * rv))
    return Rotation.concatenate(Rs).as_quat()

def ang_speed_stats(quat_xyzw: np.ndarray, hz: float, name="cam"):
    dt = 1.0 / hz
    R = Rotation.from_quat(quat_xyzw)
    dR = R[:-1].inv() * R[1:]
    dtheta = dR.magnitude()                 # rad/frame
    omega = dtheta / dt                     # rad/s
    omega_deg = omega * 180/np.pi           # deg/s
    print(f"\n=== Angular speed ({name}) ===")
    print(f"deg/s: median={np.median(omega_deg):.2f}, mean={np.mean(omega_deg):.2f}, "
          f"p95={np.percentile(omega_deg,95):.2f}, max={np.max(omega_deg):.2f}")
    return omega_deg


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

pose_cam_tcp = np.array([-0.01935615, -0.19549504, -0.09199475, 0.0, 0.0, 0.0], dtype=np.float64)
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

    t = T_cam_tcp[:3, 3].copy()

    # quat_cam_smooth = smooth_quat_slerp(quat_cam, win=5)
    # R_smooth = Rotation.from_quat(quat_cam_smooth).as_matrix()
    # p_rigid_smooth = pos_cam + (R_smooth @ t.reshape(3,1)).squeeze(-1)

    R = Rotation.from_quat(quat_cam).as_matrix()  # assumes xyzw
    p_rigid = pos_cam + (R @ t.reshape(3,1)).squeeze(-1)  # correct rigid offset application
    p_world_translate_only = pos_cam + t  # WRONG unless t is in world

    delta = p_rigid - p_world_translate_only
    print("delta stats (m):",
        "mean", np.mean(np.linalg.norm(delta, axis=1)),
        "max",  np.max(np.linalg.norm(delta, axis=1)))

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

    # # cam->tcp
    # T_world_cam = pos_quat_to_T(pos_cam, quat_cam)        # (L,4,4)
    # T_world_tcp = T_world_cam @ T_cam_tcp                 # (L,4,4)

    # pos_tcp = T_world_tcp[:, :3, 3]                       # (L,3)
    # rot_tcp = Rotation.from_matrix(T_world_tcp[:, :3, :3])
    # rotvec_tcp = rot_tcp.as_rotvec()                      # (L,3)

    # smooth quat
    quat_cam_smooth = smooth_quat_exp(quat_cam, alpha=0.3)

    rot_jitter_stats(quat_cam,        t, name=f"ep{ep} cam RAW")
    rot_jitter_stats(quat_cam_smooth, t, name=f"ep{ep} cam SMOOTH")

    R_smooth = Rotation.from_quat(quat_cam_smooth)
    pos_tcp = pos_cam + R_smooth.apply(t)
    rotvec_tcp = R_smooth.as_rotvec()

    dq = np.linalg.norm(quat_cam_smooth - quat_cam, axis=1)
    print("quat diff max:", dq.max())


    # after computing pos_tcp:
    pos_step_stats(pos_cam, name=f"ep{ep} cam")
    pos_step_stats(pos_tcp, name=f"ep{ep} tcp (derived)")

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
