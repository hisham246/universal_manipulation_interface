import zarr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl

Xi_ref = np.load("/home/hisham246/uwaterloo/vic-umi/stiffness_estimation/pc_gmm/gmm_output/surface_wiping/run_20250618_092508/Xi_ref.npy")

register_codecs()

# Open the zarr dataset
zarr_path = "/home/hisham246/uwaterloo/umi/surface_wiping_trial_1/dataset.zarr.zip"
csv_path = "/home/hisham246/uwaterloo/umi/surface_wiping_trial_1/csv/"
os.makedirs(csv_path, exist_ok=True)
root = zarr.open(zarr_path)

# Extract episode boundaries
episode_ends = root['meta']['episode_ends'][:]
episode_starts = np.concatenate(([0], episode_ends[:-1]))

# List of data arrays to extract
data_keys = ["timestamp", "robot0_eef_pos"]

# Iterate over episodes and extract data
for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
    episode_data = {}

    # Extract data for this episode
    for key in data_keys:
        episode_data[key] = root["data"][key][start:end]

Xi_ref_T = Xi_ref.T

# Sanity check
assert Xi_ref_T.shape[0] == episode_ends[-1], "Xi_ref length doesn't match Zarr episode boundaries"

# Segment Xi_ref into episodes
Xi_ref_episodes = [Xi_ref_T[start:end] for start, end in zip(episode_starts, episode_ends)]

for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
    # Zarr dataset extraction
    eef_pos = root["data"]["robot0_eef_pos"][start:end]  # shape (T_i, 3)
    xi_ref_ep = Xi_ref_episodes[i]                      # shape (T_i, 3)

    # Plot side-by-side
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(eef_pos[:, 0], eef_pos[:, 1], eef_pos[:, 2], label='robot0_eef_pos (zarr)', color='blue')
    ax.plot(xi_ref_ep[:, 0], xi_ref_ep[:, 1], xi_ref_ep[:, 2], label='Xi_ref (PC-GMM)', color='red')
    ax.set_title(f'Episode {i}')
    ax.legend()
    plt.show()