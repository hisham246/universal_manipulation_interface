"""
This script adds a filtered action trajectory to replace an original trajectory in a zarr dataset.
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

# Open the original Zarr dataset to inspect structure and get chunk sizes
with zarr.ZipStore(src_path, mode='r') as src_store:
    root = zarr.open(src_store)
    
    print("Original dataset structure:")
    for key in root['data'].keys():
        arr = root['data'][key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, chunks={arr.chunks}, compressor={arr.compressor}")
    
    # Store original chunk sizes and dtypes
    orig_chunks = {key: root['data'][key].chunks for key in root['data'].keys()}
    orig_dtypes = {key: root['data'][key].dtype for key in root['data'].keys()}

# Load the filtered CSV files and organize them by episode
filtered_csv_files = sorted(glob.glob(os.path.join(filtered_data_path, "*.csv")), key=natural_key)

pos_data_list = []
rot_data_list = []
demo_start_poses_list = []
demo_end_poses_list = []

for csv_file in filtered_csv_files:
    df = pd.read_csv(csv_file)
    pos_data = df.filter(regex="robot0_eef_pos").values
    rot_data = df.filter(regex="robot0_eef_rot_axis_angle").values
    
    start_pos = np.tile(pos_data[0], (pos_data.shape[0], 1))
    start_rot = np.tile(rot_data[0], (rot_data.shape[0], 1))
    end_pos = np.tile(pos_data[-1], (pos_data.shape[0], 1))
    end_rot = np.tile(rot_data[-1], (rot_data.shape[0], 1))
    
    pos_data_list.append(pos_data)
    rot_data_list.append(rot_data)
    demo_start_poses_list.append(np.concatenate([start_pos, start_rot], axis=1))
    demo_end_poses_list.append(np.concatenate([end_pos, end_rot], axis=1))

# Concatenate with correct dtypes from original
all_pos_data = np.concatenate(pos_data_list, axis=0).astype(orig_dtypes['robot0_eef_pos'])
all_rot_data = np.concatenate(rot_data_list, axis=0).astype(orig_dtypes['robot0_eef_rot_axis_angle'])
demo_start_poses = np.concatenate(demo_start_poses_list, axis=0).astype(orig_dtypes['robot0_demo_start_pose'])
demo_end_poses = np.concatenate(demo_end_poses_list, axis=0).astype(orig_dtypes['robot0_demo_end_pose'])

print(f"\nFiltered data shapes and dtypes:")
print(f"  robot0_eef_pos: {all_pos_data.shape}, dtype={all_pos_data.dtype}")
print(f"  robot0_eef_rot_axis_angle: {all_rot_data.shape}, dtype={all_rot_data.dtype}")
print(f"  robot0_demo_start_pose: {demo_start_poses.shape}, dtype={demo_start_poses.dtype}")
print(f"  robot0_demo_end_pose: {demo_end_poses.shape}, dtype={demo_end_poses.dtype}")

# Re-open and modify
with zarr.ZipStore(src_path, mode='r') as src_store:
    replay_buffer = ReplayBuffer.copy_from_store(
        src_store=src_store,
        store=zarr.MemoryStore()
    )
    
    data = replay_buffer.data
    
    data_keys = ["robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_demo_start_pose", "robot0_demo_end_pose"]
    
    for key in data_keys:
        if key in data:
            del data[key]
    
    # Recreate with original chunks, dtypes, and no compression
    data.create_dataset(
        'robot0_eef_pos',
        data=all_pos_data,
        chunks=orig_chunks['robot0_eef_pos'],
        compressor=None
    )
    
    data.create_dataset(
        'robot0_eef_rot_axis_angle',
        data=all_rot_data,
        chunks=orig_chunks['robot0_eef_rot_axis_angle'],
        compressor=None
    )
    
    data.create_dataset(
        'robot0_demo_start_pose',
        data=demo_start_poses,
        chunks=orig_chunks['robot0_demo_start_pose'],
        compressor=None
    )
    
    data.create_dataset(
        'robot0_demo_end_pose',
        data=demo_end_poses,
        chunks=orig_chunks['robot0_demo_end_pose'],
        compressor=None
    )
    
    with zarr.ZipStore(dst_path, mode='w') as dst_store:
        replay_buffer.save_to_store(dst_store)

print(f"\nModified dataset saved to: {dst_path}")

# Verify
print("\nVerifying output dataset:")
with zarr.ZipStore(dst_path, mode='r') as verify_store:
    root = zarr.open(verify_store)
    for key in root['data'].keys():
        arr = root['data'][key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, chunks={arr.chunks}, compressor={arr.compressor}")