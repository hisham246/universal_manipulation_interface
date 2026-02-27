# #!/usr/bin/env python3
# """
# Drop selected demos/episodes from a diffusion_policy Zarr replay buffer.

# - Reads src zarr.zip
# - Removes full episodes given by indices (or keeps first N, etc.)
# - Writes dst zarr.zip
# """

# import numpy as np
# import zarr

# from diffusion_policy.common.replay_buffer import ReplayBuffer
# from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

# register_codecs()

# # -----------------------------
# # CONFIG
# # -----------------------------
# src_path = "/home/hisham246/uwaterloo/cable_route_umi/dataset_camera_only.zarr.zip"
# dst_path = "/home/hisham246/uwaterloo/cable_route_umi/dataset_camera_only_filtered.zarr.zip"

# # Option A: drop explicit list (0-based episode indices)
# episodes_to_drop_0based = [300, 301, 302, 303]

# # Option B: drop last K episodes
# # K = 10

# # Option C: keep only first N episodes (drop the rest)
# # N = 100


# def main():
#     # ---- read meta + inspect shapes/compressors
#     with zarr.ZipStore(src_path, mode="r") as src_store:
#         root = zarr.open(src_store)

#         # episode ends (exclusive)
#         episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=int)
#         num_episodes = len(episode_ends)
#         episode_starts = np.concatenate(([0], episode_ends[:-1]))
#         T = int(episode_ends[-1])

#         # gather per-array storage info so we preserve it
#         orig_chunks = {}
#         orig_dtypes = {}
#         orig_compressors = {}
#         for key in root["data"].keys():
#             arr = root["data"][key]
#             orig_chunks[key] = arr.chunks
#             orig_dtypes[key] = arr.dtype
#             orig_compressors[key] = arr.compressor

#     # ---- choose which episodes to drop
#     drop_set = set(episodes_to_drop_0based)

#     # Option B example:
#     # drop_set = set(range(num_episodes - K, num_episodes))

#     # Option C example:
#     # drop_set = set(range(N, num_episodes))

#     if any((ep < 0 or ep >= num_episodes) for ep in drop_set):
#         raise ValueError(f"Invalid episode index in drop_set. num_episodes={num_episodes}")

#     # ---- load replay buffer in memory
#     with zarr.ZipStore(src_path, mode="r") as src_store:
#         rb = ReplayBuffer.copy_from_store(src_store=src_store, store=zarr.MemoryStore())

#     data = rb.data

#     # ---- build timestep keep mask (length T)
#     keep_mask = np.ones(T, dtype=bool)
#     ep_lengths = episode_ends - episode_starts

#     for ep in range(num_episodes):
#         if ep in drop_set:
#             s = int(episode_starts[ep])
#             e = int(episode_ends[ep])
#             keep_mask[s:e] = False

#     T_new = int(keep_mask.sum())
#     keep_episode_mask = np.array([ep not in drop_set for ep in range(num_episodes)], dtype=bool)
#     num_episodes_new = int(keep_episode_mask.sum())

#     print(f"Dropping {len(drop_set)} episodes.")
#     print(f"Timesteps: {T} -> {T_new}")
#     print(f"Episodes : {num_episodes} -> {num_episodes_new}")

#     # ---- recompute new episode_ends (exclusive) in the kept timeline
#     new_episode_ends = []
#     cum = 0
#     for ep in range(num_episodes):
#         if ep in drop_set:
#             continue
#         cum += int(ep_lengths[ep])
#         new_episode_ends.append(cum)
#     new_episode_ends = np.asarray(new_episode_ends, dtype=int)

#     if cum != T_new:
#         raise RuntimeError(f"Kept length mismatch: cum={cum}, T_new={T_new}")

#     # ---- apply masks to all datasets
#     all_keys = list(data.keys())
#     for key in all_keys:
#         arr = data[key]
#         shape = arr.shape
#         if len(shape) == 0 or shape[0] == 0:
#             continue

#         if shape[0] == T:
#             new_arr = np.asarray(arr)[keep_mask]
#             del data[key]
#             data.create_dataset(
#                 key,
#                 data=new_arr.astype(orig_dtypes.get(key, new_arr.dtype)),
#                 chunks=orig_chunks.get(key, arr.chunks),
#                 compressor=orig_compressors.get(key, arr.compressor),
#             )
#         elif shape[0] == num_episodes:
#             new_arr = np.asarray(arr)[keep_episode_mask]
#             del data[key]
#             data.create_dataset(
#                 key,
#                 data=new_arr.astype(orig_dtypes.get(key, new_arr.dtype)),
#                 chunks=orig_chunks.get(key, arr.chunks),
#                 compressor=orig_compressors.get(key, arr.compressor),
#             )
#         else:
#             # leave unchanged
#             pass

#     # ---- update meta and save
#     rb.meta["episode_ends"] = new_episode_ends

#     with zarr.ZipStore(dst_path, mode="w") as dst_store:
#         rb.save_to_store(dst_store)

#     print(f"Saved: {dst_path}")
#     print("New meta/episode_ends tail:", new_episode_ends[-5:] if len(new_episode_ends) >= 5 else new_episode_ends)


# if __name__ == "__main__":
#     main()


# More efficient
#!/usr/bin/env python3
"""
Drop selected demos/episodes from a diffusion_policy Zarr replay buffer (zarr.zip),
without loading everything into RAM.

- Reads src zarr.zip
- Drops full episodes by index
- Writes dst zarr.zip
- Copies data in chunks by episode ranges (streaming)
"""

import numpy as np
import zarr

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()

# -----------------------------
# CONFIG
# -----------------------------
src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/dataset_with_vicon_segmented.zarr.zip"
dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/dataset_with_vicon_segmented_filtered.zarr.zip"

# Option A: drop explicit list (0-based episode indices)
episodes_to_drop_0based = [163, 164]

# Option B: drop last K episodes
# K = 10

# Option C: keep only first N episodes (drop the rest)
# N = 100

# How many timesteps to copy per write (tune if needed)
COPY_BLOCK_T = 512


def _copy_attrs(src_obj, dst_obj):
    # copy group/array attributes if present
    try:
        dst_obj.attrs.update(dict(src_obj.attrs))
    except Exception:
        pass


def main():
    # ---- open src and read episode structure
    with zarr.ZipStore(src_path, mode="r") as src_store:
        src_root = zarr.open(src_store, mode="r")

        episode_ends = np.asarray(src_root["meta"]["episode_ends"][:], dtype=int)
        num_episodes = len(episode_ends)
        episode_starts = np.concatenate(([0], episode_ends[:-1]))
        ep_lengths = episode_ends - episode_starts
        T = int(episode_ends[-1])

        # choose which episodes to drop
        drop_set = set(episodes_to_drop_0based)

        # Option B:
        # drop_set = set(range(num_episodes - K, num_episodes))

        # Option C:
        # drop_set = set(range(N, num_episodes))

        if any((ep < 0 or ep >= num_episodes) for ep in drop_set):
            raise ValueError(f"Invalid episode index in drop_set. num_episodes={num_episodes}")

        keep_episode_mask = np.array([ep not in drop_set for ep in range(num_episodes)], dtype=bool)
        kept_eps = np.nonzero(keep_episode_mask)[0].tolist()

        kept_ranges = []
        for ep in kept_eps:
            s = int(episode_starts[ep])
            e = int(episode_ends[ep])
            kept_ranges.append((s, e))

        T_new = int(sum(e - s for (s, e) in kept_ranges))
        num_episodes_new = len(kept_ranges)

        print(f"Dropping {len(drop_set)} episodes.")
        print(f"Timesteps: {T} -> {T_new}")
        print(f"Episodes : {num_episodes} -> {num_episodes_new}")

        # recompute new episode_ends (exclusive) in the kept timeline
        new_episode_ends = []
        cum = 0
        for (s, e) in kept_ranges:
            cum += (e - s)
            new_episode_ends.append(cum)
        new_episode_ends = np.asarray(new_episode_ends, dtype=int)

        if cum != T_new:
            raise RuntimeError(f"Kept length mismatch: cum={cum}, T_new={T_new}")

        # ---- create dst and stream-copy
        with zarr.ZipStore(dst_path, mode="w") as dst_store:
            dst_root = zarr.open(dst_store, mode="w")

            # create groups
            dst_meta = dst_root.create_group("meta")
            dst_data = dst_root.create_group("data")

            # copy group attrs
            _copy_attrs(src_root, dst_root)
            _copy_attrs(src_root["meta"], dst_meta)
            _copy_attrs(src_root["data"], dst_data)

            # copy meta arrays (but replace episode_ends)
            for k in src_root["meta"].keys():
                if k == "episode_ends":
                    continue
                src_arr = src_root["meta"][k]
                # meta arrays are usually small; ok to read into memory
                out = dst_meta.create_dataset(
                    k,
                    data=np.asarray(src_arr),
                    dtype=src_arr.dtype,
                    chunks=src_arr.chunks,
                    compressor=src_arr.compressor,
                )
                _copy_attrs(src_arr, out)

            out_ep = dst_meta.create_dataset(
                "episode_ends",
                data=new_episode_ends,
                dtype=new_episode_ends.dtype,
                chunks=src_root["meta"]["episode_ends"].chunks,
                compressor=src_root["meta"]["episode_ends"].compressor,
            )
            _copy_attrs(src_root["meta"]["episode_ends"], out_ep)

            # copy each dataset under data/
            for key in src_root["data"].keys():
                src_arr = src_root["data"][key]
                shape = src_arr.shape

                if len(shape) == 0 or shape[0] == 0:
                    # empty/scalar, just copy
                    out = dst_data.create_dataset(
                        key,
                        data=np.asarray(src_arr),
                        dtype=src_arr.dtype,
                        chunks=src_arr.chunks,
                        compressor=src_arr.compressor,
                    )
                    _copy_attrs(src_arr, out)
                    continue

                if shape[0] == T:
                    # timestep-aligned array: stream-copy kept_ranges
                    dst_shape = (T_new,) + shape[1:]
                    out = dst_data.create_dataset(
                        key,
                        shape=dst_shape,
                        dtype=src_arr.dtype,
                        chunks=src_arr.chunks,
                        compressor=src_arr.compressor,
                    )
                    _copy_attrs(src_arr, out)

                    write_pos = 0
                    for (s, e) in kept_ranges:
                        n = e - s
                        # copy in blocks along time dimension
                        for ss in range(s, e, COPY_BLOCK_T):
                            ee = min(e, ss + COPY_BLOCK_T)
                            block = src_arr[ss:ee]
                            out[write_pos : write_pos + (ee - ss)] = block
                            write_pos += (ee - ss)

                    if write_pos != T_new:
                        raise RuntimeError(f"[{key}] wrote {write_pos} steps, expected {T_new}")

                elif shape[0] == num_episodes:
                    # episode-aligned array: copy only kept episodes
                    dst_shape = (num_episodes_new,) + shape[1:]
                    out = dst_data.create_dataset(
                        key,
                        shape=dst_shape,
                        dtype=src_arr.dtype,
                        chunks=src_arr.chunks,
                        compressor=src_arr.compressor,
                    )
                    _copy_attrs(src_arr, out)

                    # typically small; still avoid one giant np.asarray if it might be big
                    for i, ep in enumerate(kept_eps):
                        out[i] = src_arr[ep]

                else:
                    # other arrays: copy as-is
                    out = dst_data.create_dataset(
                        key,
                        data=np.asarray(src_arr),
                        dtype=src_arr.dtype,
                        chunks=src_arr.chunks,
                        compressor=src_arr.compressor,
                    )
                    _copy_attrs(src_arr, out)

            print(f"Saved: {dst_path}")
            print("New meta/episode_ends tail:", new_episode_ends[-5:] if len(new_episode_ends) >= 5 else new_episode_ends)


if __name__ == "__main__":
    main()