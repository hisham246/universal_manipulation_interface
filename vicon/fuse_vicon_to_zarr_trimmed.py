#!/usr/bin/env python3
import os
import re
import pathlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

SRC_ZARR = "/home/hisham246/uwaterloo/cable_route_umi/dataset_with_vicon.zarr.zip"
DST_ZARR = "/home/hisham246/uwaterloo/cable_route_umi/dataset_with_vicon_trimmed.zarr.zip"

VICON_ALIGNED_DIR = "/home/hisham246/uwaterloo/cable_route_umi/aligned_vicon_files/aligned_vicon_to_episode/"
VICON_TRIMMED_DIR = "/home/hisham246/uwaterloo/cable_route_umi/vicon_trimmed/"

USE_PATTERN = True
PATTERN = "aligned_episode_{i:03d}.csv"
ONE_BASED = True

FORCE_PREFIX_MODE = False
TS_CANDIDATES = ["timestamp", "Timestamp", "time", "Time", "t", "T"]

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def ep_csv_path(ep: int, base_dir: pathlib.Path) -> pathlib.Path:
    if USE_PATTERN:
        file_i = (ep + 1) if ONE_BASED else ep
        return base_dir / PATTERN.format(i=file_i)
    files = sorted([p.name for p in base_dir.glob("*.csv")], key=natural_key)
    return base_dir / files[ep]

def find_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in TS_CANDIDATES:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def load_len_and_first_last_ts(csv_path: pathlib.Path) -> Tuple[int, Optional[float], Optional[float], Optional[str]]:
    df = pd.read_csv(csv_path)
    n = len(df)
    if n == 0:
        return 0, None, None, None

    ts_col = find_timestamp_col(df)
    if ts_col is None:
        return n, None, None, None

    ts = pd.to_numeric(df[ts_col], errors="coerce").to_numpy()
    ts = ts[np.isfinite(ts)]
    if len(ts) == 0:
        return n, None, None, ts_col

    return n, float(ts[0]), float(ts[-1]), ts_col

def infer_kept_range_from_trim(aligned_csv: pathlib.Path, trimmed_csv: pathlib.Path, force_prefix: bool = False) -> Tuple[int, int]:
    n_trim, trim_first_ts, _, _ = load_len_and_first_last_ts(trimmed_csv)
    if n_trim <= 0:
        return 0, 0

    if force_prefix or (trim_first_ts is None):
        return 0, n_trim

    dfA = pd.read_csv(aligned_csv)
    ts_colA = find_timestamp_col(dfA)
    if ts_colA is None:
        return 0, n_trim

    tsA = pd.to_numeric(dfA[ts_colA], errors="coerce").to_numpy()
    diffs = np.abs(tsA - trim_first_ts)
    if not np.isfinite(diffs).any():
        return 0, n_trim

    start = int(np.nanargmin(diffs))
    end = min(start + n_trim, len(dfA))
    start = max(0, min(start, end))
    return start, end

def main():
    vicon_aligned_dir = pathlib.Path(VICON_ALIGNED_DIR).expanduser().absolute()
    vicon_trimmed_dir = pathlib.Path(VICON_TRIMMED_DIR).expanduser().absolute()

    with zarr.ZipStore(SRC_ZARR, mode="r") as src_store:
        root = zarr.open(src_store, mode="r")
        episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=int)
        episode_starts = np.concatenate(([0], episode_ends[:-1]))
        num_eps = len(episode_ends)

        orig_info: Dict[str, Dict] = {}
        for k in root["data"].keys():
            arr = root["data"][k]
            orig_info[k] = {"dtype": arr.dtype, "chunks": arr.chunks, "compressor": arr.compressor}

        rb = ReplayBuffer.copy_from_store(src_store=src_store, store=zarr.MemoryStore())

    data = rb.data
    keys = list(data.keys())

    ep_slices: List[Tuple[int, int]] = []
    total_new = 0
    for ep in range(num_eps):
        s = int(episode_starts[ep])
        e = int(episode_ends[ep])
        ep_len = e - s

        aligned_csv = ep_csv_path(ep, vicon_aligned_dir)
        trimmed_csv = ep_csv_path(ep, vicon_trimmed_dir)

        if not trimmed_csv.exists():
            raise FileNotFoundError(f"Missing trimmed CSV for ep {ep}: {trimmed_csv}")
        if not aligned_csv.exists():
            raise FileNotFoundError(f"Missing aligned CSV for ep {ep}: {aligned_csv}")

        keep_start, keep_end = infer_kept_range_from_trim(aligned_csv, trimmed_csv, FORCE_PREFIX_MODE)
        keep_start = int(np.clip(keep_start, 0, ep_len))
        keep_end = int(np.clip(keep_end, keep_start, ep_len))

        ep_slices.append((s + keep_start, s + keep_end))
        total_new += (keep_end - keep_start)

        print(f"EP {ep:03d}: orig_len={ep_len:5d} keep=[{keep_start:5d},{keep_end:5d}) new_len={keep_end-keep_start:5d}")

    new_arrays: Dict[str, np.ndarray] = {}
    for k in keys:
        arr = data[k]
        new_shape = (total_new,) if arr.ndim == 1 else (total_new,) + arr.shape[1:]
        new_arrays[k] = np.empty(new_shape, dtype=arr.dtype)

    write_ptr = 0
    new_episode_ends = []
    for (gs, ge) in ep_slices:
        L = ge - gs
        for k in keys:
            new_arrays[k][write_ptr:write_ptr + L] = data[k][gs:ge]
        write_ptr += L
        new_episode_ends.append(write_ptr)

    assert write_ptr == total_new

    # ---- recompute demo start/end poses BEFORE writing datasets
    need_keys = ["robot0_eef_pos", "robot0_eef_rot_axis_angle"]
    if all(k in new_arrays for k in need_keys):
        T = total_new
        demo_start = np.empty((T, 6), dtype=np.float32)
        demo_end   = np.empty((T, 6), dtype=np.float32)

        starts = np.concatenate(([0], np.asarray(new_episode_ends[:-1], dtype=int)))
        ends   = np.asarray(new_episode_ends, dtype=int)

        eef_pos = new_arrays["robot0_eef_pos"].astype(np.float32, copy=False)
        eef_rot = new_arrays["robot0_eef_rot_axis_angle"].astype(np.float32, copy=False)

        for s, e in zip(starts, ends):
            if e <= s:
                continue
            sp = np.concatenate([eef_pos[s], eef_rot[s]])
            ep = np.concatenate([eef_pos[e-1], eef_rot[e-1]])
            demo_start[s:e] = sp
            demo_end[s:e] = ep

        new_arrays["robot0_demo_start_pose"] = demo_start
        new_arrays["robot0_demo_end_pose"] = demo_end
    else:
        print("Skipping recompute of demo start/end poses: missing eef pose keys.")

    # ---- rebuild datasets
    for k in list(data.keys()):
        del data[k]

    # choose tchunk from first NON-empty episode
    lens = np.diff(np.concatenate(([0], np.asarray(new_episode_ends, dtype=int))))
    first_non_empty = int(lens[lens > 0][0]) if np.any(lens > 0) else total_new

    min_tchunk, max_tchunk = 32, 2048
    tchunk_dyn = int(np.clip(first_non_empty, min_tchunk, max_tchunk))
    tchunk_dyn = min(tchunk_dyn, total_new) if total_new > 0 else 1

    def is_camera_key(name: str) -> bool:
        return name.startswith("camera") and ("rgb" in name or "depth" in name)

    for k in keys:
        info = orig_info[k]
        chunks = info["chunks"]
        compressor = info["compressor"]
        dtype = info["dtype"]

        if chunks is None:
            out_chunks = None
        else:
            if is_camera_key(k):
                tchunk = min(chunks[0], total_new) if total_new > 0 else 1
            else:
                tchunk = tchunk_dyn
            out_chunks = (tchunk,) + tuple(chunks[1:])

        data.create_dataset(k, data=new_arrays[k], dtype=dtype, chunks=out_chunks, compressor=compressor)

    rb.meta["episode_ends"][:] = np.asarray(new_episode_ends, dtype=np.int64)

    with zarr.ZipStore(DST_ZARR, mode="w") as dst_store:
        rb.save_to_store(dst_store)

    print(f"\nDone.\nSaved trimmed Zarr to: {DST_ZARR}")
    print(f"Total steps: {total_new} (was {int(episode_ends[-1])})")
    print(f"Episodes: {num_eps}")

if __name__ == "__main__":
    main()