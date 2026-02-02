#!/usr/bin/env python3
"""
Drop selected demos/episodes from a diffusion_policy Zarr replay buffer.

- Reads src zarr.zip
- Removes full episodes given by indices (or keeps first N, etc.)
- Writes dst zarr.zip
"""

import numpy as np
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

# -----------------------------
# CONFIG
# -----------------------------
src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_vicon_3.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_vicon_4.zarr.zip"

# Option A: drop explicit list (0-based episode indices)
episodes_to_drop_0based = [100, 165, 194, 239]

# Option B: drop last K episodes
# K = 10

# Option C: keep only first N episodes (drop the rest)
# N = 100


def main():
    # ---- read meta + inspect shapes/compressors
    with zarr.ZipStore(src_path, mode="r") as src_store:
        root = zarr.open(src_store)

        # episode ends (exclusive)
        episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=int)
        num_episodes = len(episode_ends)
        episode_starts = np.concatenate(([0], episode_ends[:-1]))
        T = int(episode_ends[-1])

        # gather per-array storage info so we preserve it
        orig_chunks = {}
        orig_dtypes = {}
        orig_compressors = {}
        for key in root["data"].keys():
            arr = root["data"][key]
            orig_chunks[key] = arr.chunks
            orig_dtypes[key] = arr.dtype
            orig_compressors[key] = arr.compressor

    # ---- choose which episodes to drop
    drop_set = set(episodes_to_drop_0based)

    # Option B example:
    # drop_set = set(range(num_episodes - K, num_episodes))

    # Option C example:
    # drop_set = set(range(N, num_episodes))

    if any((ep < 0 or ep >= num_episodes) for ep in drop_set):
        raise ValueError(f"Invalid episode index in drop_set. num_episodes={num_episodes}")

    # ---- load replay buffer in memory
    with zarr.ZipStore(src_path, mode="r") as src_store:
        rb = ReplayBuffer.copy_from_store(src_store=src_store, store=zarr.MemoryStore())

    data = rb.data

    # ---- build timestep keep mask (length T)
    keep_mask = np.ones(T, dtype=bool)
    ep_lengths = episode_ends - episode_starts

    for ep in range(num_episodes):
        if ep in drop_set:
            s = int(episode_starts[ep])
            e = int(episode_ends[ep])
            keep_mask[s:e] = False

    T_new = int(keep_mask.sum())
    keep_episode_mask = np.array([ep not in drop_set for ep in range(num_episodes)], dtype=bool)
    num_episodes_new = int(keep_episode_mask.sum())

    print(f"Dropping {len(drop_set)} episodes.")
    print(f"Timesteps: {T} -> {T_new}")
    print(f"Episodes : {num_episodes} -> {num_episodes_new}")

    # ---- recompute new episode_ends (exclusive) in the kept timeline
    new_episode_ends = []
    cum = 0
    for ep in range(num_episodes):
        if ep in drop_set:
            continue
        cum += int(ep_lengths[ep])
        new_episode_ends.append(cum)
    new_episode_ends = np.asarray(new_episode_ends, dtype=int)

    if cum != T_new:
        raise RuntimeError(f"Kept length mismatch: cum={cum}, T_new={T_new}")

    # ---- apply masks to all datasets
    all_keys = list(data.keys())
    for key in all_keys:
        arr = data[key]
        shape = arr.shape
        if len(shape) == 0 or shape[0] == 0:
            continue

        if shape[0] == T:
            new_arr = np.asarray(arr)[keep_mask]
            del data[key]
            data.create_dataset(
                key,
                data=new_arr.astype(orig_dtypes.get(key, new_arr.dtype)),
                chunks=orig_chunks.get(key, arr.chunks),
                compressor=orig_compressors.get(key, arr.compressor),
            )
        elif shape[0] == num_episodes:
            new_arr = np.asarray(arr)[keep_episode_mask]
            del data[key]
            data.create_dataset(
                key,
                data=new_arr.astype(orig_dtypes.get(key, new_arr.dtype)),
                chunks=orig_chunks.get(key, arr.chunks),
                compressor=orig_compressors.get(key, arr.compressor),
            )
        else:
            # leave unchanged
            pass

    # ---- update meta and save
    rb.meta["episode_ends"] = new_episode_ends

    with zarr.ZipStore(dst_path, mode="w") as dst_store:
        rb.save_to_store(dst_store)

    print(f"Saved: {dst_path}")
    print("New meta/episode_ends tail:", new_episode_ends[-5:] if len(new_episode_ends) >= 5 else new_episode_ends)


if __name__ == "__main__":
    main()
