import os
import numpy as np
import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl

register_codecs()

# Paths
input_path = "/home/hisham246/uwaterloo/umi/surface_wiping_trial_1/dataset.zarr.zip"
output_path = "/home/hisham246/uwaterloo/umi/surface_wiping_trial_1/filtered_dataset.zarr.zip"

# Open input Zarr
root_in = zarr.open(input_path, mode='r')
data_in = root_in["data"]
meta_in = root_in["meta"]
episode_ends = meta_in["episode_ends"][:]
episode_starts = np.concatenate([[0], episode_ends[:-1]])

# Episodes to drop (1-indexed for clarity)
drop_episodes = {2, 3, 21}  # change as needed
keep_indices = [i for i in range(len(episode_starts)) if (i + 1) not in drop_episodes]

# Compute new episode boundaries
new_episode_ends = []
current_idx = 0
for i in keep_indices:
    start, end = episode_starts[i], episode_ends[i]
    new_len = end - start
    current_idx += new_len
    new_episode_ends.append(current_idx)

# Open new Zarr
store_out = zarr.ZipStore(output_path, mode='w')
root_out = zarr.group(store=store_out)
data_out = root_out.require_group("data")
meta_out = root_out.require_group("meta")

# Get keys and shapes
keys = list(data_in.array_keys())

# Create datasets in output Zarr
for key in keys:
    dtype = data_in[key].dtype
    shape = (sum(episode_ends[i] - episode_starts[i] for i in keep_indices),) + data_in[key].shape[1:]
    chunks = (1,) + data_in[key].chunks[1:]
    compressor = data_in[key].compressor
    data_out.create_dataset(
        name=key, shape=shape, dtype=dtype, chunks=chunks, compressor=compressor
    )

# Copy kept data
out_idx = 0
for i in keep_indices:
    start, end = episode_starts[i], episode_ends[i]
    for key in keys:
        data_out[key][out_idx:out_idx + (end - start)] = data_in[key][start:end]
    out_idx += (end - start)

# Save new episode ends
meta_out.create_dataset("episode_ends", data=np.array(new_episode_ends, dtype=np.int64))

store_out.close()
print(f"Filtered dataset saved to {output_path}")
