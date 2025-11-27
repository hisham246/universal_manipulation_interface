"""
This script adds a filtered action trajectory to to replace an original trajectory in a zarr dataset.
"""

import zarr
import pandas as pd
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
import os, re, glob

register_codecs()

# Paths
src_path = "/home/hisham246/uwaterloo/umi/reaching_ball_multimodal/reaching_ball_multimodal.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/umi/reaching_ball_multimodal/reaching_ball_multimodal_filtered.zarr.zip"
filtered_data_path = "/home/hisham246/uwaterloo/umi/reaching_ball_multimodal/csv_filtered"

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

# Open the original Zarr dataset
with zarr.ZipStore(src_path, mode='r') as src_store:
    root = zarr.open(src_store)

    # List of data keys to replace
    data_keys = ["robot0_eef_pos", "robot0_demo_end_pose", "robot0_demo_start_pose"]

    # Load the filtered CSV files and organize them by episode
    filtered_csv_files = sorted(glob.glob(os.path.join(filtered_data_path, "*.csv")), key=natural_key)

    pos_data_list = []
    rot_data_list = []  # To hold rotation data (robot0_eef_rot_axis_angle)
    demo_start_poses_list = []
    demo_end_poses_list = []
    for i, csv_file in enumerate(filtered_csv_files):
        # Load the filtered data for this episode
        csv_path = os.path.join(filtered_data_path, csv_file)
        df = pd.read_csv(csv_path)
        pos_data = df.filter(regex="robot0_eef_pos").values
        rot_data = df.filter(regex="robot0_eef_rot_axis_angle").values  # Extract rotation data
        start_pos = np.array([pos_data[0] for _ in range(pos_data.shape[0])])
        start_rot = np.array([rot_data[0] for _ in range(rot_data.shape[0])])
        end_pos = np.array([pos_data[-1] for _ in range(pos_data.shape[0])])
        end_rot = np.array([rot_data[-1] for _ in range(rot_data.shape[0])])
        pos_data_list.append(pos_data)
        rot_data_list.append(rot_data)
        demo_start_poses_list.append(np.concatenate([start_pos, start_rot], axis=1))  # Concatenate position and rotation
        demo_end_poses_list.append(np.concatenate([end_pos, end_rot], axis=1))  # Concatenate position and rotation

    # Concatenate all position and rotation data
    all_pos_data = np.concatenate(pos_data_list, axis=0)
    all_rot_data = np.concatenate(rot_data_list, axis=0)  # All rotations
    print(f"Total concatenated position data shape: {all_pos_data.shape}")
    print(f"Total concatenated rotation data shape: {all_rot_data.shape}")

    # Create the replay buffer from the Zarr store
    replay_buffer = ReplayBuffer.copy_from_store(
        src_store=src_store,
        store=zarr.MemoryStore()
    )

    # Get the data object to modify
    data = replay_buffer.data

    # 1. Drop "robot0_eef_pos", "robot0_demo_end_pose", and "robot0_demo_start_pose" from the dataset
    for key in data_keys:
        if key in data:
            del data[key]

    # 2. Construct "robot0_demo_start_pose" and "robot0_demo_end_pose"
    demo_start_poses = np.concatenate(demo_start_poses_list, axis=0)
    demo_end_poses = np.concatenate(demo_end_poses_list, axis=0)
    print(f"Shape of demo_start_poses: {demo_start_poses.shape}")
    print(f"Shape of demo_end_poses: {demo_end_poses.shape}")

    with zarr.ZipStore(dst_path, mode='w') as dst_store:
        # Verify the shape of the data
        print(f"Shape of robot0_eef_pos: {all_pos_data.shape}")
        # You can now determine the correct chunking based on the data shapes
        # Assuming the data is 2D and we want to chunk by the first axis (number of episodes)
        chunk_shape = (1, demo_start_poses.shape[1])  # Adjust the chunk size if needed
        
        data.create_dataset(
            'robot0_eef_pos',
            data=all_pos_data,
            shape=all_pos_data.shape,
            dtype='f4',
            compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=1),
            chunks=chunk_shape  # Use the calculated chunk shape
        )

        data.create_dataset(
            'robot0_demo_start_pose',
            data=demo_start_poses,
            shape=demo_start_poses.shape,
            dtype='f4',
            compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=1),
            chunks=chunk_shape  # Use the calculated chunk shape
        )

        data.create_dataset(
            'robot0_demo_end_pose',
            data=demo_end_poses,
            shape=demo_end_poses.shape,
            dtype='f4',
            compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=1),
            chunks=chunk_shape  # Use the calculated chunk shape
        )

        # Save the modified replay buffer to the new Zarr store
        replay_buffer.save_to_store(dst_store)

    print(f"Modified dataset saved to: {dst_path}")