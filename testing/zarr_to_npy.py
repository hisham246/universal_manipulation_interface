import zarr
import numpy as np
import os
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

# Configuration
zarr_path = "/home/hisham246/uwaterloo/umi/surface_wiping_tp/surface_wiping_tp.zarr.zip"
output_dir = "/home/hisham246/uwaterloo/umi/surface_wiping_tp/dataset/"
os.makedirs(output_dir, exist_ok=True)

# Open zarr dataset
root = zarr.open(zarr_path)

print(root.tree())

# # Extract episode boundaries
# episode_ends = root['meta']['episode_ends'][:]
# episode_starts = np.concatenate(([0], episode_ends[:-1]))

# # All data keys to extract
# data_keys = [
#     "timestamp",
#     "robot0_eef_pos", 
#     "robot0_eef_rot_axis_angle",
#     "robot0_gripper_width",
#     "robot0_demo_start_pose",
#     "robot0_demo_end_pose",
#     "camera0_rgb"
# ]

# print(f"Extracting {len(episode_ends)} episodes...")

# # Extract each episode
# for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
#     episode_dir = os.path.join(output_dir, f"episode_{i+1}")
#     os.makedirs(episode_dir, exist_ok=True)
    
#     # Extract each data type as separate numpy arrays
#     for key in data_keys:
#         if key in root["data"]:
#             data = root["data"][key][start:end]
#             np.save(os.path.join(episode_dir, f"{key}.npy"), data)
    
#     print(f"Saved episode_{i+1} - {end-start} timesteps")

# print(f"Extraction complete! Files saved to: {output_dir}")
# print("Each episode folder contains:")
# for key in data_keys:
#     print(f"  - {key}.npy")