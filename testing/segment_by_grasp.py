#!/usr/bin/env python3
"""
Memory-safe trim + recompute demo start/end poses.

Trimming rule:
- For each episode, keep a prefix until the first time the gripper "reaches" THRESHOLD.
- Uses a crossing event by default to avoid trimming immediately if the episode starts already <= THRESHOLD:
    prev > THRESHOLD and curr <= THRESHOLD
- Keep everything BEFORE the reach step (INCLUDE_REACH_STEP=False) or include it (True).

Then stream-copy all timestep arrays from src->dst episode-by-episode (no MemoryStore),
and recompute robot0_demo_start_pose / robot0_demo_end_pose on the trimmed data.
"""

import os
import zarr
import numpy as np
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_with_vicon.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_with_vicon_segmented.zarr.zip"

if os.path.exists(dst_path):
    os.remove(dst_path)

# -----------------------------------------------------------------------------
# Trimming config
# -----------------------------------------------------------------------------
GRIPPER_KEY = "robot0_gripper_width"
THRESHOLD = 0.005

# If False: keep strictly before the step where it first becomes <= THRESHOLD
INCLUDE_REACH_STEP = False

# If True: trigger trimming only on a crossing (prev > thr and curr <= thr).
# This avoids trimming immediately in episodes that start already below threshold.
USE_CROSSING = True

# Prevent empty episodes (set to 1 if you want to allow super short)
MIN_EP_LEN = 1

PRINT_FIRST_N_EPISODES = 5

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def compute_keep_len(ep_g: np.ndarray) -> int:
    """Return number of steps to keep for a single episode gripper signal."""
    L = ep_g.shape[0]
    if L == 0:
        return 0

    if USE_CROSSING:
        if L < 2:
            keep_len = L
        else:
            prev = ep_g[:-1]
            curr = ep_g[1:]
            hit = np.where((prev > THRESHOLD) & (curr <= THRESHOLD))[0]
            if hit.size == 0:
                keep_len = L
            else:
                reach_idx = int(hit[0] + 1)  # first index where curr <= thr
                keep_len = reach_idx + (1 if INCLUDE_REACH_STEP else 0)
    else:
        # First index where ep_g <= THRESHOLD (can trim immediately if episode starts <= thr)
        hit = np.where(ep_g <= THRESHOLD)[0]
        if hit.size == 0:
            keep_len = L
        else:
            reach_idx = int(hit[0])
            keep_len = reach_idx + (1 if INCLUDE_REACH_STEP else 0)

    keep_len = max(int(MIN_EP_LEN), int(keep_len))
    keep_len = min(int(L), int(keep_len))
    return keep_len

# -----------------------------------------------------------------------------
# 1) Open source, read meta episode boundaries
# -----------------------------------------------------------------------------
with zarr.ZipStore(src_path, mode="r") as src_store:
    src_root = zarr.open(src_store, mode="r")
    src_data = src_root["data"]
    src_meta = src_root["meta"]

    episode_ends = np.asarray(src_meta["episode_ends"][:], dtype=int)  # [E], exclusive
    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    E = len(episode_ends)

    T = int(src_data["robot0_eef_pos"].shape[0])
    assert episode_ends[-1] == T, f"episode_ends last={episode_ends[-1]} != T={T}"

print(f"Loaded meta: num_episodes={E}, T={T}")

# -----------------------------------------------------------------------------
# 2) Compute per-episode keep lengths using the gripper signal (read only gripper)
# -----------------------------------------------------------------------------
with zarr.ZipStore(src_path, mode="r") as src_store:
    src_root = zarr.open(src_store, mode="r")
    src_data = src_root["data"]

    if GRIPPER_KEY not in src_data:
        candidates = [k for k in src_data.keys() if "gripper" in k.lower()]
        raise RuntimeError(
            f"GRIPPER_KEY='{GRIPPER_KEY}' not found. Gripper-like keys: {candidates}"
        )

    g_raw = np.asarray(src_data[GRIPPER_KEY][:])  # [T,1] or [T]
    g = g_raw[:, 0] if g_raw.ndim == 2 else g_raw

print(f"Using GRIPPER_KEY='{GRIPPER_KEY}', shape={g_raw.shape}")
print(f"Global gripper stats: min={float(g.min()):.6f}, max={float(g.max()):.6f}")

kept_lengths = np.zeros(E, dtype=int)
for ep in range(E):
    s = int(episode_starts[ep])
    e = int(episode_ends[ep])
    ep_g = g[s:e]
    kept_lengths[ep] = compute_keep_len(ep_g)

    if ep < PRINT_FIRST_N_EPISODES:
        print(
            f"ep {ep:04d}: len={e-s:4d}, kept={kept_lengths[ep]:4d}, "
            f"gripper[min,max]=[{float(ep_g.min()):.6f},{float(ep_g.max()):.6f}]"
        )

new_episode_ends = np.cumsum(kept_lengths).astype(int)
T_new = int(new_episode_ends[-1])
print(f"Trimming complete: T {T} -> {T_new}")

# -----------------------------------------------------------------------------
# 3) Create destination store and stream-copy datasets
# -----------------------------------------------------------------------------
with zarr.ZipStore(src_path, mode="r") as src_store, zarr.ZipStore(dst_path, mode="w") as dst_store:
    src_root = zarr.open(src_store, mode="r")
    dst_root = zarr.group(store=dst_store)

    src_data = src_root["data"]
    src_meta = src_root["meta"]

    # Create groups
    dst_data = dst_root.create_group("data")
    dst_meta = dst_root.create_group("meta")

    # Copy meta (except episode_ends, which we replace)
    for k in src_meta.keys():
        if k == "episode_ends":
            continue
        arr = src_meta[k]
        dst_meta.create_dataset(
            k,
            data=arr[:],
            dtype=arr.dtype,
            chunks=arr.chunks,
            compressor=arr.compressor,
        )

    # Write new episode_ends
    dst_meta.create_dataset(
        "episode_ends",
        data=new_episode_ends,
        dtype=new_episode_ends.dtype,
        chunks=src_meta["episode_ends"].chunks,
        compressor=src_meta["episode_ends"].compressor,
    )

    # Identify timestep-aligned arrays (shape[0] == T) and copy them trimmed
    keys = list(src_data.keys())

    # We will recompute these two after copying everything else
    recompute_keys = {"robot0_demo_start_pose", "robot0_demo_end_pose"}

    # Pre-create all datasets in dst with correct shape (trimmed for timestep-aligned)
    for k in keys:
        arr = src_data[k]
        shape = arr.shape

        if len(shape) == 0:
            # scalar
            dst_data.create_dataset(k, data=arr[()], dtype=arr.dtype)
            continue

        if shape[0] == T:
            # timestep array -> trimmed first dimension
            new_shape = (T_new,) + shape[1:]
        else:
            # leave as-is (episode-level arrays etc.)
            new_shape = shape

        dst_data.create_dataset(
            k,
            shape=new_shape,
            dtype=arr.dtype,
            chunks=arr.chunks,
            compressor=arr.compressor,
        )

    # Stream-copy timestep arrays episode-by-episode (excluding demo poses for now)
    print("Copying datasets (streaming)...")
    out_cursor = 0
    for ep in range(E):
        s = int(episode_starts[ep])
        keep = int(kept_lengths[ep])
        if keep <= 0:
            raise RuntimeError(f"Episode {ep} has keep_len={keep}.")

        src_slice = slice(s, s + keep)
        dst_slice = slice(out_cursor, out_cursor + keep)

        for k in keys:
            arr = src_data[k]
            if len(arr.shape) == 0:
                continue

            if arr.shape[0] == T:
                if k in recompute_keys:
                    continue  # will rewrite later
                dst_data[k][dst_slice] = arr[src_slice]
            else:
                # non-timestep arrays: copy once outside loop (do it only on first episode)
                pass

        out_cursor += keep

    # Copy non-timestep arrays once
    for k in keys:
        arr = src_data[k]
        if len(arr.shape) == 0:
            continue
        if arr.shape[0] != T:
            dst_data[k][:] = arr[:]

    assert out_cursor == T_new, f"out_cursor={out_cursor} != T_new={T_new}"

    # -----------------------------------------------------------------------------
    # 4) Recompute demo start/end poses on trimmed data (episode-by-episode)
    # -----------------------------------------------------------------------------
    print("Recomputing demo start/end poses (streaming)...")
    pos = src_data["robot0_eef_pos"]
    rot = src_data["robot0_eef_rot_axis_angle"]

    out_cursor = 0
    for ep in range(E):
        s = int(episode_starts[ep])
        keep = int(kept_lengths[ep])
        src_slice = slice(s, s + keep)
        dst_slice = slice(out_cursor, out_cursor + keep)

        # start pose uses first of kept segment
        p0 = np.asarray(pos[s])
        r0 = np.asarray(rot[s])
        start_pose = np.concatenate([p0, r0], axis=0)

        # end pose uses last of kept segment
        pend = np.asarray(pos[s + keep - 1])
        rend = np.asarray(rot[s + keep - 1])
        end_pose = np.concatenate([pend, rend], axis=0)

        # Tile across the kept segment
        dst_data["robot0_demo_start_pose"][dst_slice] = start_pose[None, :].repeat(keep, axis=0)
        dst_data["robot0_demo_end_pose"][dst_slice] = end_pose[None, :].repeat(keep, axis=0)

        out_cursor += keep

    print(f"Saved trimmed+recomputed dataset to: {dst_path}")

# -----------------------------------------------------------------------------
# 5) Verify
# -----------------------------------------------------------------------------
with zarr.ZipStore(dst_path, mode="r") as verify_store:
    root_out = zarr.open(verify_store, mode="r")
    print("Verify:")
    print("  T_new:", root_out["data"]["robot0_eef_pos"].shape[0])
    print("  robot0_demo_start_pose:", root_out["data"]["robot0_demo_start_pose"].shape)
    print("  robot0_demo_end_pose:  ", root_out["data"]["robot0_demo_end_pose"].shape)
    print("  num_episodes:", len(root_out["meta"]["episode_ends"][:]))
    print("  meta/episode_ends last:", int(root_out["meta"]["episode_ends"][-1]))
