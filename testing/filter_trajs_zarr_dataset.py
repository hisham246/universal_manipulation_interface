#!/usr/bin/env python3
"""
1) Replace robot0_eef_pos for episodes that have filtered CSVs (x,y,z)
   using csv_segmented_filtered_2.
2) Recompute robot0_demo_start_pose and robot0_demo_end_pose based on the
   updated positions and existing orientations.
3) Drop a given list of corrupted episodes from all observations/actions,
   then save to a new Zarr store.
"""

import os
import re
import glob

import zarr
import numpy as np
import pandas as pd

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl

register_codecs()

# -----------------------------------------------------------------------------
# Paths (adjust if needed)
# -----------------------------------------------------------------------------
src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi/peg_in_hole_segmented_2.zarr.zip"

dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi/peg_in_hole_insertion.zarr.zip"

# Directory containing ALL original demos (one CSV per episode, in order)
all_csv_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi/csv_segmented_2/"

# Directory containing only some demos, each containing BOTH original and
# filtered positions; filtered columns are x, y, z.
filtered_csv_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi/csv_segmented_filtered_2/"

# Episodes to drop (1-based indices as given)
episodes_to_drop_1based = [
    22, 33, 62, 76, 84, 88, 94, 101, 106, 112, 119, 127, 131, 141, 147,
    153, 154, 155, 157, 159, 161, 163, 164, 165, 166, 169, 172, 173, 176,
    177, 184, 186, 190, 191, 194, 198, 201, 207, 209, 210, 211, 212, 213,
    214, 216, 223, 226, 228, 232
]
episodes_to_drop_0based = [i - 1 for i in episodes_to_drop_1based]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def natural_key(s: str):
    """Sort key that handles numbers in filenames naturally."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


# -----------------------------------------------------------------------------
# 1) Inspect original dataset, store chunk/dtype/compressor info + episode bounds
# -----------------------------------------------------------------------------
with zarr.ZipStore(src_path, mode='r') as src_store:
    root = zarr.open(src_store)

    print("Original dataset structure:")
    orig_chunks = {}
    orig_dtypes = {}
    orig_compressors = {}

    for key in root['data'].keys():
        arr = root['data'][key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, "
              f"chunks={arr.chunks}, compressor={arr.compressor}")
        orig_chunks[key] = arr.chunks
        orig_dtypes[key] = arr.dtype
        orig_compressors[key] = arr.compressor

    # Episode boundaries from meta (as in your exporter script)
    meta_episode_ends = np.asarray(root['meta']['episode_ends'][:], dtype=int)  # [E], exclusive end indices
    episode_ends_idx = meta_episode_ends.copy()
    episode_start_idxs = np.concatenate(([0], episode_ends_idx[:-1]))          # [E], inclusive starts

    # Sanity check with one data array
    T_data = root['data']['robot0_eef_pos'].shape[0]
    assert episode_ends_idx[-1] == T_data, \
        f"meta episode_ends last={episode_ends_idx[-1]} != T={T_data}"

num_episodes = len(episode_ends_idx)
print(f"\nLoaded original chunks/dtypes/compressors.")
print(f"Number of episodes (from meta): {num_episodes}\n")


# # -----------------------------------------------------------------------------
# # 2) Map CSV filenames to episode indices using ALL demos
# # -----------------------------------------------------------------------------
# all_csv_files = sorted(
#     glob.glob(os.path.join(all_csv_dir, "*.csv")),
#     key=natural_key
# )
# if len(all_csv_files) == 0:
#     raise RuntimeError(f"No CSV files found in all_csv_dir={all_csv_dir}")

# csv_name_to_episode_idx = {
#     os.path.basename(path): idx
#     for idx, path in enumerate(all_csv_files)
# }

# print(f"Found {len(all_csv_files)} CSVs in all_csv_dir.")
# print("Example mapping:", list(csv_name_to_episode_idx.items())[:3])

# # Filtered CSVs subset
# filtered_csv_files = sorted(
#     glob.glob(os.path.join(filtered_csv_dir, "*.csv")),
#     key=natural_key
# )
# print(f"Found {len(filtered_csv_files)} filtered CSVs in filtered_csv_dir.\n")


# # -----------------------------------------------------------------------------
# # 3) Load replay buffer into memory
# # -----------------------------------------------------------------------------
# with zarr.ZipStore(src_path, mode='r') as src_store:
#     replay_buffer = ReplayBuffer.copy_from_store(
#         src_store=src_store,
#         store=zarr.MemoryStore()
#     )

# data = replay_buffer.data

# # Step-level arrays we care about
# pos = np.asarray(data['robot0_eef_pos'][:])              # [T, Dpos]
# rot = np.asarray(data['robot0_eef_rot_axis_angle'][:])   # [T, Drot]

# T = pos.shape[0]
# Dpos = pos.shape[1]
# Drot = rot.shape[1]

# assert T == episode_ends_idx[-1], \
#     f"Replay buffer T={T} != meta episode_ends last={episode_ends_idx[-1]}"

# print(f"Replay buffer size T={T}, num_episodes={num_episodes}")
# print(f"Position dim={Dpos}, rotation dim={Drot}\n")


# # -----------------------------------------------------------------------------
# # 4) Replace robot0_eef_pos with filtered x,y,z for the episodes that have them
# # -----------------------------------------------------------------------------
# pos_new = pos.copy()

# for csv_path in filtered_csv_files:
#     csv_name = os.path.basename(csv_path)
#     if csv_name not in csv_name_to_episode_idx:
#         raise RuntimeError(
#             f"Filtered CSV {csv_name} not found in all_csv_dir mapping."
#         )

#     ep_idx = csv_name_to_episode_idx[csv_name]
#     if ep_idx >= num_episodes:
#         raise RuntimeError(
#             f"Episode index {ep_idx} from CSV {csv_name} "
#             f"exceeds num_episodes={num_episodes}"
#         )

#     s = episode_start_idxs[ep_idx]   # inclusive
#     e = episode_ends_idx[ep_idx]     # exclusive
#     ep_len = e - s

#     df = pd.read_csv(csv_path)

#     # Expect filtered position columns named x, y, z
#     for col in ['x', 'y', 'z']:
#         if col not in df.columns:
#             raise RuntimeError(
#                 f"Expected column '{col}' not found in {csv_name}."
#             )

#     filtered_pos = df[['x', 'y', 'z']].to_numpy()
#     if filtered_pos.shape[0] != ep_len:
#         raise RuntimeError(
#             f"Length mismatch for episode {ep_idx} from {csv_name}: "
#             f"CSV len={filtered_pos.shape[0]}, episode len={ep_len}"
#         )

#     # If robot0_eef_pos has more than 3 dims, overwrite only first 3
#     if Dpos >= 3:
#         pos_new[s:e, :3] = filtered_pos.astype(orig_dtypes['robot0_eef_pos'])
#     else:
#         # If for some reason Dpos != 3, enforce full replacement
#         if Dpos != filtered_pos.shape[1]:
#             raise RuntimeError(
#                 f"Dpos={Dpos} but filtered_pos has dim={filtered_pos.shape[1]}"
#             )
#         pos_new[s:e, :] = filtered_pos.astype(orig_dtypes['robot0_eef_pos'])

#     print(f"Replaced robot0_eef_pos for episode {ep_idx} "
#           f"using filtered CSV {csv_name}")

# print("\nFinished applying filtered positions.\n")


# # -----------------------------------------------------------------------------
# # 5) Recompute robot0_demo_start_pose and robot0_demo_end_pose
# # -----------------------------------------------------------------------------
# demo_start_orig = np.asarray(data['robot0_demo_start_pose'][:])
# demo_end_orig = np.asarray(data['robot0_demo_end_pose'][:])
# T_demo, Dpose = demo_start_orig.shape

# if T_demo != T:
#     raise RuntimeError(
#         f"demo pose length T_demo={T_demo} does not match T={T}"
#     )

# if Dpose != (Dpos + Drot):
#     raise RuntimeError(
#         f"demo pose dim Dpose={Dpose} != Dpos+Drot={Dpos + Drot}"
#     )

# demo_start_new = np.empty((T, Dpose), dtype=orig_dtypes['robot0_demo_start_pose'])
# demo_end_new = np.empty((T, Dpose), dtype=orig_dtypes['robot0_demo_end_pose'])

# for ep in range(num_episodes):
#     s = episode_start_idxs[ep]   # inclusive
#     e = episode_ends_idx[ep]     # exclusive
#     ep_len = e - s

#     first_idx = s
#     last_idx = e - 1

#     start_pose = np.concatenate([pos_new[first_idx], rot[first_idx]])  # [Dpose]
#     end_pose = np.concatenate([pos_new[last_idx], rot[last_idx]])      # [Dpose]

#     start_pose = start_pose.astype(orig_dtypes['robot0_demo_start_pose'])
#     end_pose = end_pose.astype(orig_dtypes['robot0_demo_end_pose'])

#     demo_start_new[s:e, :] = np.tile(start_pose, (ep_len, 1))
#     demo_end_new[s:e, :] = np.tile(end_pose, (ep_len, 1))

# print("Recomputed robot0_demo_start_pose and robot0_demo_end_pose.\n")


# # -----------------------------------------------------------------------------
# # 6) Write updated pos and demo poses back into the in-memory replay buffer
# # -----------------------------------------------------------------------------
# for key in ['robot0_eef_pos', 'robot0_demo_start_pose', 'robot0_demo_end_pose']:
#     if key in data:
#         del data[key]

# data.create_dataset(
#     'robot0_eef_pos',
#     data=pos_new.astype(orig_dtypes['robot0_eef_pos']),
#     chunks=orig_chunks['robot0_eef_pos'],
#     compressor=orig_compressors['robot0_eef_pos']
# )

# data.create_dataset(
#     'robot0_demo_start_pose',
#     data=demo_start_new.astype(orig_dtypes['robot0_demo_start_pose']),
#     chunks=orig_chunks['robot0_demo_start_pose'],
#     compressor=orig_compressors['robot0_demo_start_pose']
# )

# data.create_dataset(
#     'robot0_demo_end_pose',
#     data=demo_end_new.astype(orig_dtypes['robot0_demo_end_pose']),
#     chunks=orig_chunks['robot0_demo_end_pose'],
#     compressor=orig_compressors['robot0_demo_end_pose']
# )

# print("Updated datasets written in-memory.\n")


# # -----------------------------------------------------------------------------
# # 7) Drop corrupted episodes from all relevant arrays
# # -----------------------------------------------------------------------------
# drop_set = set(episodes_to_drop_0based)

# if any(ep >= num_episodes for ep in drop_set):
#     raise RuntimeError(
#         f"Some episodes_to_drop indices exceed num_episodes={num_episodes}"
#     )

# # Precompute episode lengths
# ep_lengths = episode_ends_idx - episode_start_idxs  # [E]

# keep_mask = np.ones(T, dtype=bool)

# for ep in range(num_episodes):
#     s = episode_start_idxs[ep]
#     e = episode_ends_idx[ep]
#     if ep in drop_set:
#         keep_mask[s:e] = False

# keep_indices = np.nonzero(keep_mask)[0]
# T_new = len(keep_indices)

# keep_episode_mask = np.array(
#     [ep not in drop_set for ep in range(num_episodes)],
#     dtype=bool
# )
# num_episodes_new = int(keep_episode_mask.sum())

# print(f"Dropping {len(drop_set)} episodes; "
#       f"time steps go from {T} -> {T_new}, "
#       f"episodes {num_episodes} -> {num_episodes_new}.\n")

# # Recompute new episode_ends meta (exclusive indices) for kept episodes
# new_episode_ends = []
# cum = 0
# for ep in range(num_episodes):
#     if ep in drop_set:
#         continue
#     cum += int(ep_lengths[ep])
#     new_episode_ends.append(cum)

# new_episode_ends = np.asarray(new_episode_ends, dtype=int)
# assert cum == T_new, (
#     f"Sum of kept episode lengths {cum} != T_new {T_new} "
#     "(something inconsistent in dropping logic)"
# )

# # Apply masks to all datasets in data group:
# # - If first dimension is T: use keep_mask
# # - If first dimension is num_episodes: use keep_episode_mask
# all_keys = list(data.keys())

# for key in all_keys:
#     arr = data[key]
#     shape = arr.shape

#     # Skip scalars and zero-length arrays
#     if len(shape) == 0 or shape[0] == 0:
#         continue

#     # Step-level arrays
#     if shape[0] == T:
#         print(f"  Applying time-step mask to {key}")
#         new_arr = np.asarray(arr)[keep_mask]
#         del data[key]
#         data.create_dataset(
#             key,
#             data=new_arr.astype(orig_dtypes.get(key, new_arr.dtype)),
#             chunks=orig_chunks.get(key, arr.chunks),
#             compressor=orig_compressors.get(key, arr.compressor)
#         )

#     # Episode-level arrays (if any)
#     elif shape[0] == num_episodes:
#         print(f"  Applying episode-level mask to {key}")
#         new_arr = np.asarray(arr)[keep_episode_mask]
#         del data[key]
#         data.create_dataset(
#             key,
#             data=new_arr.astype(orig_dtypes.get(key, new_arr.dtype)),
#             chunks=orig_chunks.get(key, arr.chunks),
#             compressor=orig_compressors.get(key, arr.compressor)
#         )

#     else:
#         # Leave as-is
#         pass

# print("\nFinished applying masks.\n")


# # -----------------------------------------------------------------------------
# # 8) Update replay_buffer meta to be consistent with the new size
# # -----------------------------------------------------------------------------
# T_final = data['robot0_eef_pos'].shape[0]
# meta = replay_buffer.meta

# meta['episode_ends'] = new_episode_ends  # exclusive endpoints in new indexing

# # -----------------------------------------------------------------------------
# # 9) Save to destination Zarr zip and verify
# # -----------------------------------------------------------------------------
# with zarr.ZipStore(dst_path, mode='w') as dst_store:
#     replay_buffer.save_to_store(dst_store)

# print(f"Modified dataset saved to: {dst_path}\n")

# print("Verifying output dataset:")
# with zarr.ZipStore(dst_path, mode='r') as verify_store:
#     root_out = zarr.open(verify_store)
#     print(root_out.tree())
#     for key in root_out['data'].keys():
#         arr = root_out['data'][key]
#         print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, "
#               f"chunks={arr.chunks}, compressor={arr.compressor}")
#     print("meta/episode_ends:", root_out['meta']['episode_ends'][:])
