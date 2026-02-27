#!/usr/bin/env python3
import numpy as np
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ZARR_1 = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/VIDP_data/dataset_with_vicon_segmented_filtered.zarr.zip"
ZARR_2 = "/home/hisham246/uwaterloo/peg_in_hole_delta_umi/VIDP_data/dataset_with_vicon_trimmed_2.zarr.zip"
OUT_ZARR = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/VIDP_data/dataset_with_vicon_combined.zarr.zip"


# Block size for streaming (number of frames per read/write)
# 512 is a good balance for speed vs RAM
COPY_BLOCK_T = 512

def _copy_attrs(src_obj, dst_obj):
    try:
        dst_obj.attrs.update(dict(src_obj.attrs))
    except Exception:
        pass

def main():
    # 1. Open both source Zarrs
    store1 = zarr.ZipStore(ZARR_1, mode="r")
    store2 = zarr.ZipStore(ZARR_2, mode="r")
    src1 = zarr.open(store1, mode="r")
    src2 = zarr.open(store2, mode="r")

    # 2. Recompute Metadata
    ends1 = np.asarray(src1["meta"]["episode_ends"][:], dtype=int)
    ends2 = np.asarray(src2["meta"]["episode_ends"][:], dtype=int)
    
    len1 = int(ends1[-1])
    len2 = int(ends2[-1])
    total_len = len1 + len2
    total_episodes = len(ends1) + len(ends2)

    # Offset episode ends for the second dataset
    new_episode_ends = np.concatenate([ends1, ends2 + len1], axis=0)

    print(f"Merging: {len(ends1)} eps ({len1} steps) + {len(ends2)} eps ({len2} steps)")
    print(f"Target:  {total_episodes} eps ({total_len} steps)")

    # 3. Create destination Zarr
    with zarr.ZipStore(OUT_ZARR, mode="w") as dst_store:
        dst_root = zarr.open(dst_store, mode="w")
        dst_meta = dst_root.create_group("meta")
        dst_data = dst_root.create_group("data")

        # Copy attributes
        _copy_attrs(src1, dst_root)
        _copy_attrs(src1["meta"], dst_meta)
        _copy_attrs(src1["data"], dst_data)

        # Write the combined episode_ends
        dst_meta.create_dataset(
            "episode_ends",
            data=new_episode_ends,
            dtype=new_episode_ends.dtype,
            chunks=src1["meta"]["episode_ends"].chunks,
            compressor=src1["meta"]["episode_ends"].compressor
        )

        # 4. Stream-copy arrays from data/
        for key in src1["data"].keys():
            arr1 = src1["data"][key]
            arr2 = src2["data"][key]
            
            # Create the dataset in destination with the combined shape
            dst_shape = (total_len,) + arr1.shape[1:]
            out = dst_data.create_dataset(
                key,
                shape=dst_shape,
                dtype=arr1.dtype,
                chunks=arr1.chunks,
                compressor=arr1.compressor
            )
            _copy_attrs(arr1, out)

            print(f"Streaming dataset: {key}")
            
            # Copy from first source
            for s in range(0, len1, COPY_BLOCK_T):
                e = min(len1, s + COPY_BLOCK_T)
                out[s:e] = arr1[s:e]

            # Copy from second source (offset the write position)
            for s in range(0, len2, COPY_BLOCK_T):
                e = min(len2, s + COPY_BLOCK_T)
                out[len1 + s : len1 + e] = arr2[s:e]

    store1.close()
    store2.close()
    print(f"Done! Combined Zarr saved to: {OUT_ZARR}")

if __name__ == "__main__":
    main()