#!/usr/bin/env python3
"""
Replace (overwrite) robot0_eef_pos / robot0_eef_rot_axis_angle using Vicon poses,
then recompute robot0_demo_start_pose / robot0_demo_end_pose using meta/episode_ends.

Key changes vs your current version:
1) No Rotation.slerp() (doesn't exist) and no window-mean slerp attempt.
2) Rotation smoothing uses an error-state exponential filter with cutoff frequency (Hz).
3) The main jitter amplifier is the lever arm: we low-pass filter u(t)=R(t)*t directly (very effective, low lag).
4) Optional: light alpha-beta filtering on pos_tcp to reduce residual jitter with minimal lag.

This avoids “significant lag” while killing the jitter that comes from R(t)*t.
"""

import re
import pathlib
import zarr
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from umi.common.pose_util import pose_to_mat

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
use_pattern = True
pattern = "peg_umi_quat {i}.csv"
one_based = True
# use_pattern = False  # use sorted list instead of pattern

# -----------------------------------------------------------------------------
# Vicon / transform config
# -----------------------------------------------------------------------------
vicon_pos_scale = 1e-3  # mm -> m

# Your cam->tcp translation (in camera frame), no rotation
pose_cam_tcp = np.array([-0.01935615, -0.19549504, -0.09199475, 0.0, 0.0, 0.0], dtype=np.float64)
T_cam_tcp = pose_to_mat(pose_cam_tcp).astype(np.float64)
t_cam_tcp = T_cam_tcp[:3, 3].copy()

# -----------------------------------------------------------------------------
# Filtering config (60 Hz target)
# -----------------------------------------------------------------------------
HZ = 60.0

# Rotation smoothing cutoff (Hz): higher = less lag, less smoothing
ROT_CUTOFF_HZ = 6.0      # try 6.0, if still jittery -> 4.0; if too laggy -> 8-10

# Lever smoothing gain (0..1): higher = less lag, less smoothing
LEVER_K = 0.55           # try 0.55; if still jittery -> 0.35-0.45; if laggy -> 0.7

# Optional position alpha-beta filter on pos_tcp after lever filtering
USE_ALPHA_BETA_POS = True
POS_KP = 0.25            # higher -> less lag
POS_KV = 0.05            # higher -> smoother velocity / less jitter
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
    quat = df[["RX", "RY", "RZ", "RW"]].to_numpy(dtype=np.float64)  # xyzw

    # normalize
    n = np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12
    quat = quat / n

    # sign continuity (prevents sudden flips that hurt any smoothing)
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0:
            quat[i] *= -1.0

    return pos, quat

def get_vicon_csv_for_episode(ep: int, vicon_dir_path: pathlib.Path, csv_list_sorted=None) -> pathlib.Path:
    if use_pattern:
        file_i = (ep + 1) if one_based else ep
        return vicon_dir_path / pattern.format(i=file_i)
    else:
        assert csv_list_sorted is not None
        if ep < 0 or ep >= len(csv_list_sorted):
            raise RuntimeError(f"Episode {ep} out of range for vicon csv list (len={len(csv_list_sorted)})")
        return vicon_dir_path / csv_list_sorted[ep]

def rot_jitter_stats(quat_xyzw: np.ndarray, t_cam_tcp: np.ndarray, name="cam"):
    R = Rotation.from_quat(quat_xyzw)
    dR = R[:-1].inv() * R[1:]
    dang = dR.magnitude()  # rad/frame

    lever = float(np.linalg.norm(t_cam_tcp))
    pos_equiv = lever * dang

    print(f"\n=== Rotation jitter ({name}) ===")
    print(f"lever arm ||t|| = {lever:.4f} m")
    print("delta angle per frame [deg]: "
          f"median={np.median(dang)*180/np.pi:.4f}, mean={np.mean(dang)*180/np.pi:.4f}, "
          f"p95={np.percentile(dang,95)*180/np.pi:.4f}, max={np.max(dang)*180/np.pi:.4f}")
    print("equiv pos jitter from lever [mm]: "
          f"median={np.median(pos_equiv)*1e3:.3f}, mean={np.mean(pos_equiv)*1e3:.3f}, "
          f"p95={np.percentile(pos_equiv,95)*1e3:.3f}, max={np.max(pos_equiv)*1e3:.3f}")

def pos_step_stats(pos: np.ndarray, name="pos"):
    step = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
    print(f"\n=== Position step stats ({name}) ===")
    print(f"step [mm]: median={np.median(step)*1e3:.3f}, mean={np.mean(step)*1e3:.3f}, "
          f"p95={np.percentile(step,95)*1e3:.3f}, max={np.max(step)*1e3:.3f}")

def alpha_from_cutoff(fc_hz: float, hz: float):
    # alpha = 1 - exp(-2*pi*fc*dt)
    dt = 1.0 / hz
    return float(1.0 - np.exp(-2.0 * np.pi * fc_hz * dt))

def smooth_quat_exp_cutoff(quat_xyzw: np.ndarray, hz: float, fc_hz: float) -> np.ndarray:
    """
    Error-state exponential smoothing on SO(3):
      Rf[k] = Rf[k-1] * Exp( alpha * Log( Rf[k-1]^{-1} R[k] ) )
    alpha derived from cutoff frequency to control lag vs smoothing.
    """
    alpha = alpha_from_cutoff(fc_hz, hz)
    Rm = Rotation.from_quat(quat_xyzw)
    Rs = [Rm[0]]
    for k in range(1, len(Rm)):
        d = Rs[-1].inv() * Rm[k]
        rv = d.as_rotvec()
        Rs.append(Rs[-1] * Rotation.from_rotvec(alpha * rv))
    return Rotation.concatenate(Rs).as_quat()

def lever_lowpass(u: np.ndarray, k: float) -> np.ndarray:
    """
    Low-pass filter u[k] = u[k-1] + k*(u_raw[k]-u[k-1]) on 3D vectors.
    k in (0,1]. Larger k => less lag, less smoothing.
    """
    out = u.copy()
    for i in range(1, len(u)):
        out[i] = out[i - 1] + k * (u[i] - out[i - 1])
    return out

def alpha_beta_filter_pos(pos_meas: np.ndarray, hz: float, kp: float, kv: float) -> np.ndarray:
    """
    Alpha-beta filter on position:
      pred = x + v*dt
      err = meas - pred
      x = pred + kp*err
      v = v + (kv/dt)*err
    """
    dt = 1.0 / hz
    x = pos_meas[0].copy()
    v = np.zeros(3, dtype=np.float64)
    out = np.empty_like(pos_meas)
    out[0] = x
    for k in range(1, len(pos_meas)):
        pred = x + v * dt
        err = pos_meas[k] - pred
        x = pred + kp * err
        v = v + (kv / dt) * err
        out[k] = x
    return out

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
        "robot0_eef_rot_axis_angle",
        "robot0_demo_start_pose",
        "robot0_demo_end_pose",
    ]
    orig = {}
    for key in keys_to_overwrite:
        arr = root["data"][key]
        orig[key] = dict(dtype=arr.dtype, chunks=arr.chunks, compressor=arr.compressor)

    T_data = root["data"]["robot0_eef_pos"].shape[0]
    assert episode_ends_idx[-1] == T_data, f"episode_ends last={episode_ends_idx[-1]} != T={T_data}"

print(f"Loaded meta: num_episodes={num_episodes}, T={episode_ends_idx[-1]}")

# Load replay buffer into memory
with zarr.ZipStore(src_path, mode="r") as src_store:
    replay_buffer = ReplayBuffer.copy_from_store(src_store=src_store, store=zarr.MemoryStore())

data = replay_buffer.data
T = data["robot0_eef_pos"].shape[0]
Dpose = 6

eef_pos_new = np.empty((T, 3), dtype=orig["robot0_eef_pos"]["dtype"])
eef_rot_new = np.empty((T, 3), dtype=orig["robot0_eef_rot_axis_angle"]["dtype"])
demo_start_new = np.empty((T, Dpose), dtype=orig["robot0_demo_start_pose"]["dtype"])
demo_end_new   = np.empty((T, Dpose), dtype=orig["robot0_demo_end_pose"]["dtype"])

print(f"Filter settings: HZ={HZ}, ROT_CUTOFF_HZ={ROT_CUTOFF_HZ}, LEVER_K={LEVER_K}, "
      f"USE_ALPHA_BETA_POS={USE_ALPHA_BETA_POS}, POS_KP={POS_KP}, POS_KV={POS_KV}")

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

    if pos_cam.shape[0] != ep_len:
        raise RuntimeError(
            f"Episode {ep} length mismatch: vicon={pos_cam.shape[0]} vs ep_len={ep_len} ({vicon_csv_path})"
        )

    # Diagnostics: what global-translation-only would miss (purely for understanding)
    R_raw = Rotation.from_quat(quat_cam).as_matrix()
    p_rigid = pos_cam + (R_raw @ t_cam_tcp.reshape(3, 1)).squeeze(-1)
    p_translate_only = pos_cam + t_cam_tcp
    delta = p_rigid - p_translate_only
    print("\ndelta stats (m): mean", float(np.mean(np.linalg.norm(delta, axis=1))),
          "max", float(np.max(np.linalg.norm(delta, axis=1))))

    # 1) Smooth rotation (low-lag, cutoff-controlled)
    quat_cam_smooth = smooth_quat_exp_cutoff(quat_cam, hz=HZ, fc_hz=ROT_CUTOFF_HZ)

    # 2) Compute lever term and filter it directly (big jitter reduction, low lag)
    R_smooth = Rotation.from_quat(quat_cam_smooth)
    u_raw = R_smooth.apply(t_cam_tcp)               # (N,3)
    u_f   = lever_lowpass(u_raw, k=LEVER_K)         # (N,3)

    # 3) Derived TCP pose
    pos_tcp = pos_cam + u_f
    rotvec_tcp = R_smooth.as_rotvec()

    # 4) Optional: alpha-beta on position to kill any remaining high-freq jitter with tiny lag
    if USE_ALPHA_BETA_POS:
        pos_tcp = alpha_beta_filter_pos(pos_tcp, hz=HZ, kp=POS_KP, kv=POS_KV)

    # Diagnostics (first few eps only to avoid huge spam)
    if ep < 10:
        rot_jitter_stats(quat_cam,        t_cam_tcp, name=f"ep{ep} cam RAW")
        rot_jitter_stats(quat_cam_smooth, t_cam_tcp, name=f"ep{ep} cam SMOOTH")
        pos_step_stats(pos_cam, name=f"ep{ep} cam")
        pos_step_stats(pos_tcp, name=f"ep{ep} tcp (derived)")
        dq = np.linalg.norm(quat_cam_smooth - quat_cam, axis=1)
        print("quat diff max:", float(dq.max()))

    # write per-step eef
    eef_pos_new[s:e, :] = pos_tcp.astype(orig["robot0_eef_pos"]["dtype"], copy=False)
    eef_rot_new[s:e, :] = rotvec_tcp.astype(orig["robot0_eef_rot_axis_angle"]["dtype"], copy=False)

    # start/end pose per step (repeat across episode)
    start_pose = np.concatenate([pos_tcp[0], rotvec_tcp[0]], axis=0).astype(orig["robot0_demo_start_pose"]["dtype"], copy=False)
    end_pose   = np.concatenate([pos_tcp[-1], rotvec_tcp[-1]], axis=0).astype(orig["robot0_demo_end_pose"]["dtype"],   copy=False)

    demo_start_new[s:e, :] = start_pose[None, :].repeat(ep_len, axis=0)
    demo_end_new[s:e, :]   = end_pose[None, :].repeat(ep_len, axis=0)

print("Computed new Vicon-based eef pos/rot and demo start/end poses.")

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
