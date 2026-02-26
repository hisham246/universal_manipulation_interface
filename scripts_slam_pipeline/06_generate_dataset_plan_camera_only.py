#!/usr/bin/env python3
"""
Generate dataset_plan.pkl using ONLY camera videos:
- No SLAM required (no camera_trajectory.csv)
- No ArUco required (no tag_detection.pkl)
Outputs:
  all_plans = [{
    "episode_timestamps": np.ndarray,  # seconds since epoch
    "grippers": [],                    # empty (fill later from Vicon)
    "cameras": [{
        "video_path": str,             # relative to demos/
        "video_start_end": (int,int)   # [start_frame, end_frame)
    }]
  }]
"""

import os, sys, math, pathlib, pickle, collections
import click
import numpy as np
import pandas as pd
import av
from exiftool import ExifToolHelper
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from umi.common.timecode_util import mp4_get_start_datetime

DAY = 86400.0

def normalize_epoch_day_buckets(video_meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Your desired behavior:
    - Find dominant epoch-day bucket (most frequent).
    - Subtract 86400 seconds from ALL rows in that dominant bucket.
    """
    df = video_meta_df.copy()
    ts = df["start_timestamp"].to_numpy(np.float64)
    day_bucket = np.floor(ts / DAY).astype(np.int64)

    vc = pd.Series(day_bucket).value_counts()
    if len(vc) <= 1:
        return df

    dominant_day = int(vc.index[0])  # most frequent bucket

    mask = (day_bucket == dominant_day)
    print("Applying dominant-day shift (-86400s):")
    print(f"  dominant_day={dominant_day}")
    print(f"  shifting {mask.sum()} / {len(df)} videos by -86400 sec")

    # optional: print a few
    for vd in df.loc[mask, "video_dir"].head(10):
        print(f"  - {Path(vd).name}")
    if mask.sum() > 10:
        print("  ...")

    df.loc[mask, "start_timestamp"] = df.loc[mask, "start_timestamp"].astype(np.float64) - DAY
    df.loc[mask, "end_timestamp"]   = df.loc[mask, "end_timestamp"].astype(np.float64) - DAY
    return df

def shift_after_demo_index(video_meta_df: pd.DataFrame,
                           cutoff_demo_num: int,
                           shift_sec: float = -DAY) -> pd.DataFrame:
    """
    Apply a constant shift (default -86400) to videos whose demo number >= cutoff.
    Assumes video_dir looks like .../demos/demo_###/ (as in your dataset).
    """
    df = video_meta_df.copy()

    # extract demo number from path name demo_14, demo_015, etc.
    demo_nums = df["video_dir"].apply(lambda p: int(Path(p).name.split("_")[-1])).to_numpy(np.int64)

    mask = demo_nums >= cutoff_demo_num
    if mask.any():
        print(f"Shifting {mask.sum()} videos with demo_num >= {cutoff_demo_num} by {shift_sec:+.0f} sec")
        # optional: print a few
        for vd in df.loc[mask, "video_dir"].head(10):
            print(f"  - {Path(vd).name}")
        if mask.sum() > 10:
            print("  ...")

    df.loc[mask, "start_timestamp"] = df.loc[mask, "start_timestamp"].astype(np.float64) + shift_sec
    df.loc[mask, "end_timestamp"]   = df.loc[mask, "end_timestamp"].astype(np.float64) + shift_sec
    return df

@click.command()
@click.option('-i', '--input', 'input_dir', required=True, help='Session directory (contains demos/)')
@click.option('-o', '--output', default=None, help='Output dataset_plan.pkl path')
@click.option('-ml', '--min_episode_length', type=int, default=24, help='Minimum frames per episode')
@click.option('--ignore_cameras', type=str, default=None, help="comma separated camera serials to ignore")
def main(input_dir, output, min_episode_length, ignore_cameras):
    input_path = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    demos_dir = input_path.joinpath('demos')
    if output is None:
        output = input_path.joinpath('dataset_plan_camera_only.pkl')
    else:
        output = pathlib.Path(os.path.expanduser(output)).absolute()

    # --------------------------
    # Stage 1: Collect video metadata
    # --------------------------
    video_dirs = sorted([x.parent for x in demos_dir.glob('demo_*/raw_video.mp4')])

    ignore_cam_serials = set()
    if ignore_cameras:
        ignore_cam_serials = set([s.strip() for s in ignore_cameras.split(',') if s.strip()])

    rows = []
    fps_ref = None

    with ExifToolHelper() as et:
        for vd in video_dirs:
            mp4_path = vd.joinpath('raw_video.mp4')
            meta = list(et.get_metadata(str(mp4_path)))[0]
            cam_serial = meta['QuickTime:CameraSerialNumber']

            if cam_serial in ignore_cam_serials:
                print(f"Ignored {vd.name} (camera serial {cam_serial})")
                continue

            start_date = mp4_get_start_datetime(str(mp4_path))
            start_ts = start_date.timestamp()

            with av.open(str(mp4_path), 'r') as container:
                stream = container.streams.video[0]
                n_frames = stream.frames
                fps = stream.average_rate

            if fps_ref is None:
                fps_ref = fps
            else:
                if fps_ref != fps:
                    raise RuntimeError(f"Inconsistent fps: {float(fps_ref)} vs {float(fps)} in {vd.name}")

            duration_sec = float(n_frames / fps)
            end_ts = start_ts + duration_sec

            rows.append({
                'video_dir': vd,
                'camera_serial': cam_serial,
                'start_date': start_date,
                'n_frames': int(n_frames),
                'fps': fps,
                'start_timestamp': float(start_ts),
                'end_timestamp': float(end_ts)
            })

    if not rows:
        raise RuntimeError("No valid demo videos found under demos/demo_*/raw_video.mp4")

    video_meta_df = pd.DataFrame(rows)
    video_meta_df = normalize_epoch_day_buckets(video_meta_df)
    serial_count = video_meta_df['camera_serial'].value_counts()
    n_cameras = len(serial_count)

    print("Found cameras:")
    print(serial_count)

    # --------------------------
    # Stage 2: Bundle into demo intervals where all cameras are recording
    # (same logic as your current script, but without SLAM/tag requirements)
    # --------------------------
    events = []
    for vid_idx, row in video_meta_df.iterrows():
        events.append({'vid_idx': vid_idx, 'camera_serial': row['camera_serial'],
                       't': row['start_timestamp'], 'is_start': True})
        events.append({'vid_idx': vid_idx, 'camera_serial': row['camera_serial'],
                       't': row['end_timestamp'], 'is_start': False})
    events = sorted(events, key=lambda x: x['t'])

    demo_data_list = []
    on_videos = set()
    on_cameras = set()
    used_videos = set()
    t_demo_start = None

    for ev in events:
        if ev['is_start']:
            on_videos.add(ev['vid_idx'])
            on_cameras.add(ev['camera_serial'])
        else:
            on_videos.remove(ev['vid_idx'])
            on_cameras.remove(ev['camera_serial'])

        # Start interval when all cameras are on
        if len(on_cameras) == n_cameras:
            t_demo_start = ev['t']

        # End interval when a camera stops
        elif t_demo_start is not None:
            assert not ev['is_start']
            t_start = t_demo_start
            t_end = ev['t']

            demo_vid_idxs = set(on_videos)
            demo_vid_idxs.add(ev['vid_idx'])
            used_videos.update(demo_vid_idxs)

            demo_data_list.append({
                "video_idxs": sorted(demo_vid_idxs),
                "start_timestamp": float(t_start),
                "end_timestamp": float(t_end)
            })
            t_demo_start = None

    unused = set(video_meta_df.index) - used_videos
    for vid_idx in unused:
        print(f"Warning: video {video_meta_df.loc[vid_idx]['video_dir'].name} unused in any demo interval")

    if not demo_data_list:
        raise RuntimeError("No overlapping intervals found where all cameras are recording.")

    # --------------------------
    # Stage 3: Generate frames-only plan
    # --------------------------
    all_plans = []
    dt = 1.0 / float(video_meta_df.iloc[0]['fps'])

    for demo_idx, demo in enumerate(demo_data_list):
        video_idxs = demo['video_idxs']
        start_ts = demo['start_timestamp']
        end_ts = demo['end_timestamp']

        demo_df = video_meta_df.loc[video_idxs].copy()

        # Choose an alignment reference video that minimizes modulo mismatch
        alignment_costs = []
        for _, row in demo_df.iterrows():
            this_cost = []
            for _, other in demo_df.iterrows():
                diff = other['start_timestamp'] - row['start_timestamp']
                this_cost.append(diff % dt)
            alignment_costs.append(sum(this_cost))
        align_row = demo_df.iloc[int(np.argmin(alignment_costs))]
        align_video_start = float(align_row['start_timestamp'])

        # Snap demo start to frame grid
        start_ts = start_ts + dt - ((start_ts - align_video_start) % dt)

        # Compute per-video start frame, and shared n_frames
        cam_start_frame = {}
        n_frames = int((end_ts - start_ts) / dt)

        for _, row in demo_df.iterrows():
            vs = float(row['start_timestamp'])
            ve = float(row['end_timestamp'])

            start_frame = math.ceil((start_ts - vs) / dt)
            video_n_frames = math.floor((ve - start_ts) / dt) - 1

            if start_frame < 0:
                video_n_frames += start_frame
                start_frame = 0

            cam_start_frame[row['camera_serial']] = int(start_frame)
            n_frames = min(n_frames, int(video_n_frames))

        if n_frames < min_episode_length:
            print(f"Skipping demo {demo_idx}: only {n_frames} frames after alignment.")
            continue

        episode_timestamps = np.arange(n_frames, dtype=np.float64) * dt + start_ts

        cameras = []
        # Keep deterministic order: sort by camera serial
        for _, row in demo_df.sort_values('camera_serial').iterrows():
            vd = row['video_dir']
            # Make video_path relative to demos/ like the original script
            rel = str(vd.joinpath('raw_video.mp4').relative_to(demos_dir))
            start_frame = cam_start_frame[row['camera_serial']]
            cameras.append({
                "video_path": rel,
                "video_start_end": (start_frame, start_frame + n_frames)
            })

        all_plans.append({
            "episode_timestamps": episode_timestamps,
            "grippers": [],
            "cameras": cameras
        })

    if not all_plans:
        raise RuntimeError("No episodes produced. Try lowering --min_episode_length or check video overlaps.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as f:
        pickle.dump(all_plans, f)

    print(f"Wrote {len(all_plans)} episodes to {output}")

if __name__ == "__main__":
    main()
