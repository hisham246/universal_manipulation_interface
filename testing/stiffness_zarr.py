"""
This script adds a stiffness dataset to an existing zarr dataset.
"""

import zarr
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

src_path = "/home/hisham246/uwaterloo/umi/surface_wiping_trial_1/dataset.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/umi/surface_wiping_trial_1/dataset_stiffness.zarr.zip"

stiffness_path = "/home/hisham246/uwaterloo/vic-umi/stiffness_estimation/pc_gmm/gmm_output/surface_wiping/run_20250618_092508/stiffness.npy"
stiffness = np.load(stiffness_path)


with zarr.ZipStore(src_path, mode='r') as src_store:
    replay_buffer = ReplayBuffer.copy_from_store(
        src_store=src_store,
        store=zarr.MemoryStore()
    )

data = replay_buffer.data
n_timesteps = data['robot0_eef_pos'].shape[0]

# Save the stiffness matrices
data.create_dataset(
    'robot0_stiffness',
    data=stiffness,
    shape=stiffness.shape,
    dtype='f4',
    compressor=None,
    chunks=(1, 6)
)

with zarr.ZipStore(dst_path, mode='w') as dst_store:
    replay_buffer.save_to_store(dst_store)