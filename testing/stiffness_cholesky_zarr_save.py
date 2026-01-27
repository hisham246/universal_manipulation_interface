import os, re, glob
import zarr
import numpy as np
import pandas as pd
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

def natural_key(path):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', path)]

src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_umi_with_vicon_segmented.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/peg_in_hole_insertion.zarr.zip"

data_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/stiffness_profiles"

stiffness_files = sorted([f for f in os.listdir(data_path) if f.startswith("stiffness_episode_") and f.endswith(".csv")], key=natural_key)

stiffness_vec_list = []

for file in stiffness_files:
    file_path = os.path.join(data_path, file)
    stiffness_df = pd.read_csv(file_path)

    N = stiffness_df.shape[0]

    for i in range(N):
        L = np.linalg.cholesky(np.diag(stiffness_df.iloc[i]))
        U = L.T
        chol_vec_upper = np.array([U[0, 0], U[0, 1], U[1, 1], U[0, 2], U[1, 2], U[2, 2]])
        stiffness_vec_list.append(chol_vec_upper)

stiffness_vec = np.array(stiffness_vec_list)

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
    data=stiffness_vec,
    shape=stiffness_vec.shape,
    dtype='float32',
    compressor=None,
    chunks=(319, 6)
)

with zarr.ZipStore(dst_path, mode='w') as dst_store:
    replay_buffer.save_to_store(dst_store)