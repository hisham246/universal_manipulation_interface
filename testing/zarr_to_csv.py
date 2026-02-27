import zarr
import pandas as pd
import numpy as np
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
import os

register_codecs()

# Open the zarr datasetS
zarr_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/VIDP_data/dataset_with_vicon_combined.zarr.zip"
csv_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/VIDP_data/dataset_with_vicon_combined/"
os.makedirs(csv_path, exist_ok=True)
root = zarr.open(zarr_path)
print(root.tree())

# # -----------------------------------------------------------------------------
# # Inspect original dataset, store chunk/dtype/compressor info + episode bounds
# # -----------------------------------------------------------------------------
# with zarr.ZipStore(zarr_path, mode='r') as src_store:
#     root = zarr.open(src_store)

#     print("Original dataset structure:")
#     orig_chunks = {}
#     orig_dtypes = {}
#     orig_compressors = {}

#     for key in root['data'].keys():
#         arr = root['data'][key]
#         print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, "
#               f"chunks={arr.chunks}, compressor={arr.compressor}")
#         orig_chunks[key] = arr.chunks
#         orig_dtypes[key] = arr.dtype
#         orig_compressors[key] = arr.compressor

#     # Episode boundaries from meta (as in your exporter script)
#     meta_episode_ends = np.asarray(root['meta']['episode_ends'][:], dtype=int)  # [E], exclusive end indices
#     episode_ends_idx = meta_episode_ends.copy()
#     episode_start_idxs = np.concatenate(([0], episode_ends_idx[:-1]))          # [E], inclusive starts

#     # Sanity check with one data array
#     T_data = root['data']['robot0_eef_pos'].shape[0]
#     assert episode_ends_idx[-1] == T_data, \
#         f"meta episode_ends last={episode_ends_idx[-1]} != T={T_data}"

# num_episodes = len(episode_ends_idx)
# print(f"\nLoaded original chunks/dtypes/compressors.")
# print(f"Number of episodes (from meta): {num_episodes}\n")
# # --------------------------------------------------------------

# Extract episode boundaries
episode_ends = root['meta']['episode_ends'][:]
episode_starts = np.concatenate(([0], episode_ends[:-1]))

# List of data arrays to extract
# data_keys = ["timestamp", "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_stiffness"]
# data_keys = ["timestamp", "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_demo_start_pose", "robot0_demo_end_pose", "robot0_gripper_width"]
data_keys = ["timestamp", "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_demo_start_pose", "robot0_demo_end_pose"]
# data_keys = ["robot0_eef_pos", "robot0_eef_rot_axis_angle"]
# data_keys = ["timestamp"]
# data_keys = ["robot0_gripper_width"]



# Iterate over episodes and extract data
for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
    episode_data = {}

    # Extract data for this episode
    for key in data_keys:
        episode_data[key] = root["data"][key][start:end]

    # timestamps = episode_data["timestamp"]
    # positions = episode_data["robot0_eef_pos"]

    # # Compute Cartesian velocity
    # vel = np.gradient(positions, timestamps, axis=0)
    # episode_data["robot0_eef_vel"] = vel.astype(np.float32)

    df = pd.DataFrame()
    for key, values in episode_data.items():
        if values.ndim == 2:
            columns = [f"{key}_{j}" for j in range(values.shape[1])]
        else:
            columns = [key]
        df = pd.concat([df, pd.DataFrame(values, columns=columns)], axis=1)

    csv_filename = f"episode_{i+1}.csv"
    df.to_csv(csv_path + csv_filename, index=False)
    print(f"Saved {csv_filename}")