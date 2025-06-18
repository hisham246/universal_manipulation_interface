import zarr
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/hisham246/uwaterloo/universal_manipulation_interface')
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
import os

# Register required codecs
register_codecs()

# Open the Zarr dataset
zarr_path = "/home/hisham246/uwaterloo/umi/pickplace_trial_2/pickplace_trial_2.zarr"
csv_dir = "/home/hisham246/uwaterloo/umi/pickplace_trial_2/csv"

os.makedirs(csv_dir, exist_ok=True)
root = zarr.open(zarr_path)

# Extract episode boundaries
episode_ends = root['meta']['episode_ends'][:]
episode_starts = np.concatenate(([0], episode_ends[:-1]))

# List of data arrays to extract
# data_keys = ["timestamp", "action", "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_joint_pos", "robot0_joint_vel"]
data_keys = ["timestamp", "action", "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_joint_pos", "robot0_joint_vel"]


# Iterate over episodes and extract data
for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
    episode_data = {}

    # Extract data for this episode
    for key in data_keys:
        episode_data[key] = root["data"][key][start:end]

    # Convert to Dataframe
    df = pd.DataFrame()
    for key, values in episode_data.items():
        if values.ndim == 2:
            columns = [f"{key}_{j}" for j in range(values.shape[1])]
        else:
            columns = [key]
        df = pd.concat([df, pd.DataFrame(values, columns=columns)], axis=1)

    # Save to CSV
    csv_filename = f"episode_{i}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_filename}")