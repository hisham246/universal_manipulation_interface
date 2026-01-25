#!/usr/bin/env python3
"""
Recompute robot0_demo_start_pose and robot0_demo_end_pose from:
- robot0_eef_pos (position)
- robot0_eef_rot_axis_angle (orientation)
using episode boundaries from meta/episode_ends.

Writes to a new Zarr zip store.
"""

import zarr
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_with_vicon_segmented.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_with_vicon_segmented_2.zarr.zip"

# -----------------------------------------------------------------------------
# 1) Read episode boundaries + original array metadata (dtype/chunks/compressor)
# -----------------------------------------------------------------------------
with zarr.ZipStore(src_path, mode="r") as src_store:
    root = zarr.open(src_store)

    episode_ends_idx = np.asarray(root["meta"]["episode_ends"][:], dtype=int)  # [E], exclusive
    episode_start_idxs = np.concatenate(([0], episode_ends_idx[:-1]))          # [E], inclusive
    num_episodes = len(episode_ends_idx)

    # Grab original metadata for the two datasets we will overwrite
    orig = {}
    for key in ["robot0_demo_start_pose", "robot0_demo_end_pose"]:
        arr = root["data"][key]
        orig[key] = dict(dtype=arr.dtype, chunks=arr.chunks, compressor=arr.compressor)

    # Sanity check
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

pos = np.asarray(data["robot0_eef_pos"][:])             # [T, Dpos]
rot = np.asarray(data["robot0_eef_rot_axis_angle"][:])  # [T, Drot]

T, Dpos = pos.shape
_, Drot = rot.shape
Dpose = Dpos + Drot

# -----------------------------------------------------------------------------
# 3) Recompute demo start/end pose arrays
# -----------------------------------------------------------------------------
demo_start_new = np.empty((T, Dpose), dtype=orig["robot0_demo_start_pose"]["dtype"])
demo_end_new   = np.empty((T, Dpose), dtype=orig["robot0_demo_end_pose"]["dtype"])

for ep in range(num_episodes):
    s = int(episode_start_idxs[ep])
    e = int(episode_ends_idx[ep])
    ep_len = e - s
    if ep_len <= 0:
        raise RuntimeError(f"Episode {ep} has non-positive length: {ep_len}")

    first_idx = s
    last_idx = e - 1

    start_pose = np.concatenate([pos[first_idx], rot[first_idx]], axis=0)
    end_pose   = np.concatenate([pos[last_idx],  rot[last_idx]],  axis=0)

    start_pose = start_pose.astype(orig["robot0_demo_start_pose"]["dtype"], copy=False)
    end_pose   = end_pose.astype(orig["robot0_demo_end_pose"]["dtype"],   copy=False)

    demo_start_new[s:e, :] = start_pose[None, :].repeat(ep_len, axis=0)
    demo_end_new[s:e, :]   = end_pose[None, :].repeat(ep_len, axis=0)

print("Recomputed demo start/end poses.")

# -----------------------------------------------------------------------------
# 4) Overwrite only those two datasets in the replay buffer
# -----------------------------------------------------------------------------
for key in ["robot0_demo_start_pose", "robot0_demo_end_pose"]:
    if key in data:
        del data[key]

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
# 5) Save to destination and verify quickly
# -----------------------------------------------------------------------------
with zarr.ZipStore(dst_path, mode="w") as dst_store:
    replay_buffer.save_to_store(dst_store)

print(f"Saved updated dataset to: {dst_path}")

with zarr.ZipStore(dst_path, mode="r") as verify_store:
    root_out = zarr.open(verify_store)
    print("Verify shapes:")
    print("  robot0_demo_start_pose:", root_out["data"]["robot0_demo_start_pose"].shape)
    print("  robot0_demo_end_pose:  ", root_out["data"]["robot0_demo_end_pose"].shape)
    print("  meta/episode_ends last:", int(root_out["meta"]["episode_ends"][-1]))
