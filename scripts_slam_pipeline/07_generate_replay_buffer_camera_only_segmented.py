#!/usr/bin/env python3
"""
Generate a Zarr ReplayBuffer containing:
  - timestamp (float64, absolute unix seconds)
  - camera{K}_rgb (uint8 images)

Episodes come from dataset_plan_camera_only.pkl, but are TRIMMED using external
episode CSV files (e.g., episode_1.csv, episode_2.csv, ...) containing gripper width(s).

Trimming logic (same as your low-dim script):
  - For each episode, find the earliest index where ANY gripper width >= threshold.
  - If trigger index == 0 -> skip episode (default).
  - Otherwise, truncate all episode data to [:trigger_idx].
  - Video frame_end is recomputed as frame_start + truncated_length
"""

import sys
import os
import json
import pathlib
import pickle
import re
import multiprocessing
import concurrent.futures
from collections import defaultdict
from typing import Optional

import click
import zarr
import numpy as np
import cv2
import av
import pandas as pd
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform,
    draw_predefined_mask,
    inpaint_tag
)

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()


# -----------------------------
# Helpers
# -----------------------------
def resolve_episode_csv_dir(ipath: pathlib.Path, episode_csv_dir: Optional[str]) -> pathlib.Path:
    """
    If episode_csv_dir is None:
      - use ipath / "camera_timestamps" (common convention)
    Else:
      - if absolute -> use it directly
      - if relative -> interpret relative to ipath
    """
    if episode_csv_dir is None:
        return ipath.joinpath("camera_timestamps")
    p = pathlib.Path(os.path.expanduser(episode_csv_dir))
    if p.is_absolute():
        return p
    return ipath.joinpath(p)


def find_gripper_width_columns(df: pd.DataFrame) -> list[str]:
    """
    Find columns that look like gripper widths.
    Your sample file uses: robot0_gripper_width_0

    This matcher is permissive; tighten if needed.
    """
    cols = []
    for c in df.columns:
        cl = c.lower()
        if ("gripper" in cl and "width" in cl) or re.search(r"gripper.*width", cl):
            cols.append(c)
    return cols


def compute_cutoff_from_episode_csv(csv_path: str, threshold: float) -> tuple[bool, int, int]:
    """
    Returns:
      found_trigger: bool
      min_cutoff: earliest index where ANY gripper width >= threshold (valid only if found_trigger=True)
      csv_len: number of rows
    """
    df = pd.read_csv(csv_path)
    csv_len = len(df)

    width_cols = find_gripper_width_columns(df)
    if len(width_cols) == 0:
        raise ValueError(
            f"No gripper-width-like columns found in {csv_path}. "
            f"Columns: {list(df.columns)}"
        )

    min_cutoff = None
    for c in width_cols:
        w = pd.to_numeric(df[c], errors="coerce").to_numpy()
        idx = np.where(np.isfinite(w) & (w >= threshold))[0]
        if len(idx) > 0:
            first = int(idx[0])
            if (min_cutoff is None) or (first < min_cutoff):
                min_cutoff = first

    if min_cutoff is None:
        return False, csv_len, csv_len
    return True, min_cutoff, csv_len


# -----------------------------
# Main
# -----------------------------
@click.command()
@click.argument("input", nargs=-1)
@click.option("-o", "--output", required=True, help="Zarr path (.zip)")
@click.option("-or", "--out_res", type=str, default="224,224")
@click.option("-of", "--out_fov", type=float, default=None)
@click.option("-cl", "--compression_level", type=int, default=99)
@click.option("-nm", "--no_mirror", is_flag=True, default=False,
              help="Disable mirror observation by masking them out")
@click.option("-ms", "--mirror_swap", is_flag=True, default=False)
@click.option("-n", "--num_workers", type=int, default=None)

# trimming options
@click.option("--episode_csv_dir", type=str, default=None,
              help="Directory with episode_1.csv, episode_2.csv, ... used for trimming. "
                   "If omitted, defaults to <session>/camera_timestamps/. "
                   "If relative, interpreted relative to each session directory.")
@click.option("--gripper_threshold", type=float, default=0.0075,
              help="Threshold used to truncate episodes (same as your lowdim script).")
@click.option("--skip_if_trigger_at_0/--keep_if_trigger_at_0", default=True,
              help="If trigger occurs at index 0, skip the episode (default: skip).")
def main(
    input, output, out_res, out_fov, compression_level,
    no_mirror, mirror_swap, num_workers,
    episode_csv_dir, gripper_threshold, skip_if_trigger_at_0
):
    if os.path.exists(output):
        click.confirm(f"Output file {output} exists! Overwrite?", abort=True)

    out_res = tuple(int(x) for x in out_res.split(","))
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    cv2.setNumThreads(1)

    # --- Fisheye Rectification Logic ---
    fisheye_converter = None
    if out_fov is not None:
        if len(input) == 0:
            raise ValueError("No input session directories provided.")
        ipath_first = pathlib.Path(os.path.expanduser(input[0])).absolute()
        intr_path = ipath_first.joinpath("calibration", "gopro_intrinsics_2_7k.json")
        if not intr_path.is_file():
            raise FileNotFoundError(f"Intrinsics not found at {intr_path}. Required for out_fov.")

        opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open("r")))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=out_res,
            out_fov=out_fov
        )

    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())

    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = []

    for ipath_str in input:
        ipath = pathlib.Path(os.path.expanduser(ipath_str)).absolute()
        demos_path = ipath.joinpath("demos")
        plan_path = ipath.joinpath("dataset_plan_camera_only.pkl")
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan_camera_only.pkl")
            continue

        plan = pickle.load(plan_path.open("rb"))
        videos_dict = defaultdict(list)

        # per-session episode numbering: episode_1.csv corresponds to first plan episode in this session
        csv_dir = resolve_episode_csv_dir(ipath, episode_csv_dir)
        episode_counter = 0

        for plan_episode in plan:
            episode_counter += 1

            cameras = plan_episode["cameras"]
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)

            ts_full = plan_episode["episode_timestamps"].astype(np.float64)
            full_len = len(ts_full)

            # --- Trimming using episode CSV ---
            trim_len = full_len
            csv_path = csv_dir.joinpath(f"episode_{episode_counter}.csv")
            if csv_path.is_file():
                found_trigger, min_cutoff, csv_len = compute_cutoff_from_episode_csv(
                    str(csv_path),
                    threshold=gripper_threshold
                )
                effective_full = min(full_len, csv_len)

                if found_trigger:
                    if min_cutoff == 0 and skip_if_trigger_at_0:
                        # skip episode entirely
                        continue
                    trim_len = min(min_cutoff, effective_full)
                else:
                    trim_len = effective_full
            else:
                # If you want to hard-require CSVs, replace this with raise FileNotFoundError(...)
                print(f"Warning: missing {csv_path} -> keeping full episode length ({full_len})")

            if trim_len <= 0:
                continue

            # Save timestamps (trimmed)
            episode_data = {
                "timestamp": ts_full[:trim_len]
            }
            out_replay_buffer.add_episode(data=episode_data, compressors=None)

            # Build video tasks (trimmed length defines frame_end)
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera["video_path"]
                video_path = demos_path.joinpath(video_path_rel).absolute()

                v_start, _ = camera["video_start_end"]  # ignore original end
                v_end = v_start + trim_len

                videos_dict[str(video_path)].append({
                    "camera_idx": cam_id,
                    "frame_start": int(v_start),
                    "frame_end": int(v_end),
                    "buffer_start": int(buffer_start)
                })

            buffer_start += trim_len

        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())

    if len(vid_args) == 0:
        raise RuntimeError("No videos/tasks collected. Check inputs and plans.")

    # Determine input image resolution from first video
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width

    # Create camera datasets sized by total frames in the buffer
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    total_frames = out_replay_buffer["timestamp"].shape[0]
    for cam_id in range(n_cameras):
        name = f"camera{cam_id}_rgb"
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(total_frames,) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    def video_to_zarr(replay_buffer, mp4_path, tasks):
        # Optional: tag inpainting if tag_detection.pkl exists next to the mp4
        pkl_path = os.path.join(os.path.dirname(mp4_path), "tag_detection.pkl")
        tag_data = None
        if os.path.exists(pkl_path):
            tag_data = pickle.load(open(pkl_path, "rb"))

        resize_tf = get_image_transform(in_res=(iw, ih), out_res=out_res)
        tasks = sorted(tasks, key=lambda x: x["frame_start"])
        camera_idx = tasks[0]["camera_idx"]
        name = f"camera{camera_idx}_rgb"
        img_array = replay_buffer.data[name]

        # Mirror swap geometry (optional)
        is_mirror = None
        if mirror_swap:
            ow, oh = out_res
            mirror_mask = np.ones((oh, ow, 3), dtype=np.uint8)
            mirror_mask = draw_predefined_mask(
                mirror_mask, color=(0, 0, 0), mirror=True, gripper=False, finger=False
            )
            is_mirror = (mirror_mask[..., 0] == 0)

        curr_task_idx = 0
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            in_stream.thread_count = 1
            buffer_idx = 0

            for frame_idx, frame in tqdm(
                enumerate(container.decode(in_stream)),
                total=in_stream.frames,
                leave=False
            ):
                if curr_task_idx >= len(tasks):
                    break

                task = tasks[curr_task_idx]

                if frame_idx < task["frame_start"]:
                    continue
                elif frame_idx < task["frame_end"]:
                    if frame_idx == task["frame_start"]:
                        buffer_idx = task["buffer_start"]

                    img = frame.to_ndarray(format="rgb24")

                    # inpaint tags
                    if tag_data is not None:
                        this_det = tag_data[frame_idx]
                        for corners in [x["corners"] for x in this_det["tag_dict"].values()]:
                            img = inpaint_tag(img, corners)

                    # mask out mirror/gripper
                    img = draw_predefined_mask(
                        img, color=(0, 0, 0),
                        mirror=no_mirror, gripper=True, finger=False
                    )

                    # resize or fisheye rectify
                    if fisheye_converter is None:
                        img = resize_tf(img)
                    else:
                        img = fisheye_converter.forward(img)

                    # mirror swap
                    if mirror_swap and is_mirror is not None:
                        img[is_mirror] = img[:, ::-1, :][is_mirror]

                    img_array[buffer_idx] = img
                    buffer_idx += 1

                    if (frame_idx + 1) == task["frame_end"]:
                        curr_task_idx += 1

    with tqdm(total=len(vid_args)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    pbar.update(len(completed))
                futures.add(executor.submit(video_to_zarr, out_replay_buffer, mp4_path, tasks))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print(f"{len(all_videos)} videos used in total!")
    print(f"Saving ReplayBuffer with trimmed timestamps to {output}")

    with zarr.ZipStore(output, mode="w") as zip_store:
        out_replay_buffer.save_to_store(store=zip_store)

    print("Done!")


if __name__ == "__main__":
    main()
