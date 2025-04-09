"""
This script adds a stiffness dataset to an existing zarr dataset.
"""

import zarr
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

zarr_path = "/home/hisham246/uwaterloo/vic-umi_gopro/dataset.zarr.zip"

src_path = "/home/hisham246/uwaterloo/vic-umi_gopro/dataset.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/vic-umi_gopro/dataset_stiffness.zarr.zip"

with zarr.ZipStore(src_path, mode='r') as src_store:
    replay_buffer = ReplayBuffer.copy_from_store(
        src_store=src_store,
        store=zarr.MemoryStore()
    )

data = replay_buffer.data
n_timesteps = data['robot0_eef_pos'].shape[0]
D = 6

stiffness = np.random.uniform(0.0, 1.0, size=(n_timesteps, D)).astype(np.float32)

data.create_dataset(
    'robot0_stiffness',
    data=stiffness,
    shape=stiffness.shape,
    dtype='f4',
    compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=1),
    chunks=(1, D)
)

with zarr.ZipStore(dst_path, mode='w') as dst_store:
    replay_buffer.save_to_store(dst_store)


