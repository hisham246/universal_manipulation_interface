import zarr
import pandas as pd
import numpy as np
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl

# Register required codecs
register_codecs()

# Open the Zarr dataset
zarr_path = "/home/hisham246/uwaterloo/umi/pickplace/pickplace.zarr.zip"
root = zarr.open(zarr_path)

# Extract episode boundaries
episode_ends = root['meta']['episode_ends'][:]
episode_starts = np.concatenate(([0], episode_ends[:-1]))  # Start indices

# List of data arrays to extract
data_keys = ["robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width"]

# Iterate over episodes and extract data
for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
    episode_data = {}

    # Extract data for this episode
    for key in data_keys:
        episode_data[key] = root["data"][key][start:end]

    # Convert to DataFrame
    df = pd.DataFrame()
    for key, values in episode_data.items():
        if values.ndim == 2:
            columns = [f"{key}_{j}" for j in range(values.shape[1])]
        else:
            columns = [key]
        df = pd.concat([df, pd.DataFrame(values, columns=columns)], axis=1)

    # Save to CSV
    csv_filename = f"episode_{i+1}.csv"
    df.to_csv("/home/hisham246/uwaterloo/umi/pickplace/csv/" + csv_filename, index=False)
    print(f"Saved {csv_filename}")