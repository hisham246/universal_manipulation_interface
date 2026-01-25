import zarr
import pandas as pd
import numpy as np
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
import os

register_codecs()

# Open the zarr dataset
zarr_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_with_vicon.zarr.zip"
csv_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/slam/"
os.makedirs(csv_path, exist_ok=True)
root = zarr.open(zarr_path)
print(root.tree()) 

# Extract episode boundaries
episode_ends = root['meta']['episode_ends'][:]
episode_starts = np.concatenate(([0], episode_ends[:-1]))

# List of data arrays to extract
# data_keys = ["timestamp", "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_stiffness"]
data_keys = ["timestamp", "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_demo_start_pose", "robot0_demo_end_pose", "robot0_gripper_width"]
# data_keys = ["robot0_eef_pos", "robot0_eef_rot_axis_angle"]


# Iterate over episodes and extract data
for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
    episode_data = {}

    # Extract data for this episode
    for key in data_keys:
        episode_data[key] = root["data"][key][start:end]

    # timestamps = episode_data["timestamp"]
    positions = episode_data["robot0_eef_pos"]

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