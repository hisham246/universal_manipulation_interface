#!/usr/bin/env python3
"""
Gripper-only dataset plan generator.

Example:
python scripts_slam_pipeline/06_generate_dataset_plan.py \
  -i data_workspace/cup_in_the_wild/20240105_zhenjia_packard_2nd_conference_room \
  --gripper_only

This version:
- does NOT require demos/mapping/tx_slam_tag.json
- does NOT require demo_*/camera_trajectory.csv
- generates dataset_plan.pkl containing only:
    episode_timestamps
    grippers: [{gripper_width, camera_serial, gripper_hardware_id, source_video}]
Optionally includes cameras list if --include_cameras is set.
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import pickle
import json
import math
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import av
from exiftool import ExifToolHelper

from umi.common.timecode_util import mp4_get_start_datetime
from umi.common.cv_util import get_gripper_width
from umi.common.interpolation_util import (
    get_gripper_calibration_interpolator,
    get_interp1d,
)


def get_bool_segments(bool_seq):
    bool_seq = np.array(bool_seq, dtype=bool)
    segment_ends = (np.nonzero(np.diff(bool_seq))[0] + 1).tolist()
    segment_bounds = [0] + segment_ends + [len(bool_seq)]
    segments = []
    segment_type = []
    for i in range(len(segment_bounds) - 1):
        start = segment_bounds[i]
        end = segment_bounds[i + 1]
        this_type = bool_seq[start]
        segments.append(slice(start, end))
        segment_type.append(this_type)
    segment_type = np.array(segment_type, dtype=bool)
    return segments, segment_type


@click.command()
@click.option("-i", "--input", required=True, help="Project/session directory")
@click.option("-o", "--output", default=None, help="Output dataset_plan.pkl (default: <session>/dataset_plan.pkl)")
@click.option("-nz", "--nominal_z", type=float, default=0.072, help="Nominal Z value for gripper finger tag")
@click.option("-ml", "--min_episode_length", type=int, default=24, help="Minimum episode length in frames")
@click.option("--ignore_cameras", type=str, default=None, help="Comma-separated camera serials to ignore")
@click.option(
    "--gripper_only",
    is_flag=True,
    default=True,
    help="Generate plan using only gripper widths (no SLAM poses, no camera trajectories).",
)
@click.option(
    "--include_cameras",
    is_flag=True,
    default=False,
    help="If set, include cameras video_path/video_start_end in the plan (still no SLAM).",
)
def main(input, output, nominal_z, min_episode_length, ignore_cameras, gripper_only, include_cameras):
    input_path = pathlib.Path(os.path.expanduser(input)).absolute()
    demos_dir = input_path.joinpath("demos")
    if output is None:
        output = input_path.joinpath("dataset_plan_gripper_only.pkl")
    else:
        output = pathlib.Path(os.path.expanduser(output)).absolute()

    if not demos_dir.is_dir():
        raise FileNotFoundError(f"Missing demos dir: {demos_dir}")

    # load gripper calibration(s)
    gripper_id_gripper_cal_map = dict()
    cam_serial_gripper_cal_map = dict()

    with ExifToolHelper() as et:
        for gripper_cal_path in demos_dir.glob("gripper*/gripper_range.json"):
            mp4_path = gripper_cal_path.parent.joinpath("raw_video.mp4")
            if not mp4_path.is_file():
                print(f"Warning: missing {mp4_path}, skipping calibration folder {gripper_cal_path.parent}")
                continue

            meta = list(et.get_metadata(str(mp4_path)))[0]
            cam_serial = meta.get("QuickTime:CameraSerialNumber", None)
            if cam_serial is None:
                print(f"Warning: no CameraSerialNumber in {mp4_path}, skipping")
                continue

            gripper_range_data = json.load(gripper_cal_path.open("r"))
            gripper_id = int(gripper_range_data["gripper_id"])
            max_width = float(gripper_range_data["max_width"])
            min_width = float(gripper_range_data["min_width"])

            gripper_cal_data = {
                "aruco_measured_width": [min_width, max_width],
                "aruco_actual_width": [min_width, max_width],
            }
            gripper_cal_interp = get_gripper_calibration_interpolator(**gripper_cal_data)
            gripper_id_gripper_cal_map[gripper_id] = gripper_cal_interp
            cam_serial_gripper_cal_map[cam_serial] = gripper_cal_interp

    if len(gripper_id_gripper_cal_map) == 0 and len(cam_serial_gripper_cal_map) == 0:
        raise RuntimeError("No gripper calibration found under demos/gripper*/gripper_range.json")

    # stage 1: gather video metadata (only requires raw_video.mp4 + tag_detection.pkl in gripper_only mode)
    video_dirs = sorted([x.parent for x in demos_dir.glob("demo_*/raw_video.mp4")])

    ignore_cam_serials = set()
    if ignore_cameras is not None:
        ignore_cam_serials = set([s.strip() for s in ignore_cameras.split(",") if len(s.strip()) > 0])

    fps = None
    rows = []
    with ExifToolHelper() as et:
        for video_dir in video_dirs:
            mp4_path = video_dir.joinpath("raw_video.mp4")
            meta = list(et.get_metadata(str(mp4_path)))[0]
            cam_serial = meta.get("QuickTime:CameraSerialNumber", None)
            if cam_serial is None:
                print(f"Ignored {video_dir.name}, missing CameraSerialNumber in metadata")
                continue

            if cam_serial in ignore_cam_serials:
                print(f"Ignored {video_dir.name} (camera serial ignored)")
                continue

            # require tag detections for gripper-only plan
            pkl_path = video_dir.joinpath("tag_detection.pkl")
            if not pkl_path.is_file():
                print(f"Ignored {video_dir.name}, no tag_detection.pkl")
                continue

            # in non-gripper-only mode (not recommended here), camera trajectory would be required
            if not gripper_only:
                csv_path = video_dir.joinpath("camera_trajectory.csv")
                if not csv_path.is_file():
                    print(f"Ignored {video_dir.name}, no camera_trajectory.csv")
                    continue

            start_date = mp4_get_start_datetime(str(mp4_path))
            start_timestamp = start_date.timestamp()

            with av.open(str(mp4_path), "r") as container:
                stream = container.streams.video[0]
                n_frames = stream.frames
                if fps is None:
                    fps = stream.average_rate
                else:
                    if fps != stream.average_rate:
                        print(
                            f"Inconsistent fps: {float(fps)} vs {float(stream.average_rate)} in {video_dir.name}"
                        )
                        raise RuntimeError("Inconsistent fps across videos")

            duration_sec = float(n_frames / fps)
            end_timestamp = start_timestamp + duration_sec

            rows.append(
                {
                    "video_dir": video_dir,
                    "camera_serial": cam_serial,
                    "start_date": start_date,
                    "n_frames": n_frames,
                    "fps": fps,
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                }
            )

    if len(rows) == 0:
        raise RuntimeError("No valid videos found (need demo_*/raw_video.mp4 + tag_detection.pkl).")

    video_meta_df = pd.DataFrame(data=rows)

    # stage 2: match videos into demos (same logic: segments where all cameras are recording)
    serial_count = video_meta_df["camera_serial"].value_counts()
    print("Found following cameras:")
    print(serial_count)
    n_cameras = len(serial_count)

    events = []
    for vid_idx, row in video_meta_df.iterrows():
        events.append({"vid_idx": vid_idx, "camera_serial": row["camera_serial"], "t": row["start_timestamp"], "is_start": True})
        events.append({"vid_idx": vid_idx, "camera_serial": row["camera_serial"], "t": row["end_timestamp"], "is_start": False})
    events = sorted(events, key=lambda x: x["t"])

    demo_data_list = []
    on_videos = set()
    on_cameras = set()
    used_videos = set()
    t_demo_start = None

    for event in events:
        if event["is_start"]:
            on_videos.add(event["vid_idx"])
            on_cameras.add(event["camera_serial"])
        else:
            on_videos.remove(event["vid_idx"])
            on_cameras.remove(event["camera_serial"])

        if len(on_videos) != len(on_cameras):
            raise RuntimeError("Mismatch between active videos and cameras (unexpected)")

        if len(on_cameras) == n_cameras:
            t_demo_start = event["t"]
        elif t_demo_start is not None:
            # demo ends when one camera stops
            if event["is_start"]:
                raise RuntimeError("Unexpected start event while demo active")
            t_start = t_demo_start
            t_end = event["t"]

            demo_vid_idxs = set(on_videos)
            demo_vid_idxs.add(event["vid_idx"])  # undo removal
            used_videos.update(demo_vid_idxs)

            demo_data_list.append(
                {"video_idxs": sorted(demo_vid_idxs), "start_timestamp": t_start, "end_timestamp": t_end}
            )
            t_demo_start = None

    unused_videos = set(video_meta_df.index) - used_videos
    for vid_idx in sorted(list(unused_videos)):
        print(f"Warning: video {video_meta_df.loc[vid_idx]['video_dir'].name} unused in any demo")

    # stage 3: identify gripper id (hardware) using aruco tags inside tag_detection.pkl
    finger_tag_det_th = 0.8
    vid_idx_gripper_hardware_id_map = dict()
    cam_serial_gripper_ids_map = collections.defaultdict(list)

    for vid_idx, row in video_meta_df.iterrows():
        video_dir = row["video_dir"]
        pkl_path = video_dir.joinpath("tag_detection.pkl")
        if not pkl_path.is_file():
            vid_idx_gripper_hardware_id_map[vid_idx] = -1
            continue

        tag_data = pickle.load(pkl_path.open("rb"))
        n_frames = len(tag_data)
        if n_frames == 0:
            vid_idx_gripper_hardware_id_map[vid_idx] = -1
            cam_serial_gripper_ids_map[row["camera_serial"]].append(-1)
            continue

        tag_counts = collections.defaultdict(lambda: 0)
        for frame in tag_data:
            for key in frame.get("tag_dict", {}).keys():
                tag_counts[key] += 1

        tag_stats = collections.defaultdict(lambda: 0.0)
        for k, v in tag_counts.items():
            tag_stats[k] = v / n_frames

        if len(tag_stats) == 0:
            gripper_id_by_tag = -1
        else:
            max_tag_id = int(np.max(list(tag_stats.keys())))
            tag_per_gripper = 6
            max_gripper_id = max_tag_id // tag_per_gripper

            gripper_prob_map = dict()
            for gripper_id in range(max_gripper_id + 1):
                left_id = gripper_id * tag_per_gripper
                right_id = left_id + 1
                left_prob = tag_stats[left_id]
                right_prob = tag_stats[right_id]
                gripper_prob = min(left_prob, right_prob)
                if gripper_prob > 0:
                    gripper_prob_map[gripper_id] = gripper_prob

            gripper_id_by_tag = -1
            if len(gripper_prob_map) > 0:
                gripper_probs = sorted(gripper_prob_map.items(), key=lambda x: x[-1])
                gripper_id = gripper_probs[-1][0]
                gripper_prob = gripper_probs[-1][1]
                if gripper_prob >= finger_tag_det_th:
                    gripper_id_by_tag = int(gripper_id)

        cam_serial_gripper_ids_map[row["camera_serial"]].append(gripper_id_by_tag)
        vid_idx_gripper_hardware_id_map[vid_idx] = gripper_id_by_tag

    series = pd.Series(
        data=list(vid_idx_gripper_hardware_id_map.values()),
        index=list(vid_idx_gripper_hardware_id_map.keys()),
    )
    video_meta_df["gripper_hardware_id"] = series

    cam_serial_gripper_hardware_id_map = dict()
    for cam_serial, gripper_ids in cam_serial_gripper_ids_map.items():
        counter = collections.Counter(gripper_ids)
        gripper_id = counter.most_common(1)[0][0]
        cam_serial_gripper_hardware_id_map[cam_serial] = gripper_id

    # stage 4: camera indexing
    # In gripper-only mode we do not disambiguate left/right; we just sort camera serials
    unique_serials = sorted(video_meta_df["camera_serial"].unique().tolist())
    cam_serial_cam_idx_map = {cs: i for i, cs in enumerate(unique_serials)}
    video_meta_df["camera_idx"] = video_meta_df["camera_serial"].map(cam_serial_cam_idx_map)

    rows = []
    for cs, ci in cam_serial_cam_idx_map.items():
        example_vid = video_meta_df.loc[video_meta_df["camera_serial"] == cs].iloc[0]["video_dir"].name
        rows.append(
            {
                "camera_idx": ci,
                "camera_serial": cs,
                "gripper_hw_idx": cam_serial_gripper_hardware_id_map.get(cs, -1),
                "example_vid": example_vid,
            }
        )
    camera_serial_df = pd.DataFrame(data=rows).set_index("camera_idx").sort_index()
    print("Assigned camera_idx by serial order (gripper-only mode):")
    print(camera_serial_df)

    # stage 6: generate dataset plan (gripper widths only)
    total_avaliable_time = 0.0
    total_used_time = 0.0
    n_dropped_demos = 0
    all_plans = []

    for demo_idx, demo_data in enumerate(demo_data_list):
        video_idxs = demo_data["video_idxs"]
        start_timestamp = float(demo_data["start_timestamp"])
        end_timestamp = float(demo_data["end_timestamp"])
        total_avaliable_time += (end_timestamp - start_timestamp)

        demo_video_meta_df = video_meta_df.loc[video_idxs].copy()
        demo_video_meta_df.set_index("camera_idx", inplace=True)
        demo_video_meta_df.sort_index(inplace=True)

        # determine dt from fps (assumes consistent fps)
        dt = None
        for _, row in demo_video_meta_df.iterrows():
            dt = 1.0 / float(row["fps"])
            break
        if dt is None:
            print(f"Skipped demo {demo_idx} (no rows)")
            n_dropped_demos += 1
            continue

        # align start_timestamp to frame grid to be consistent across videos
        # pick the first camera as alignment reference
        align_cam_idx = demo_video_meta_df.index[0]
        align_video_start = float(demo_video_meta_df.loc[align_cam_idx]["start_timestamp"])
        start_timestamp = start_timestamp + dt - ((start_timestamp - align_video_start) % dt)

        # compute frame window length using time overlap
        n_frames = int((end_timestamp - start_timestamp) / dt)
        cam_start_frame_idxs = {}

        for cam_idx, row in demo_video_meta_df.iterrows():
            video_start_frame = math.ceil((start_timestamp - float(row["start_timestamp"])) / dt)
            video_n_frames = math.floor((float(row["end_timestamp"]) - start_timestamp) / dt) - 1
            if video_start_frame < 0:
                video_n_frames += video_start_frame
                video_start_frame = 0
            cam_start_frame_idxs[int(cam_idx)] = int(video_start_frame)
            n_frames = min(n_frames, int(video_n_frames))

        if n_frames <= 0:
            print(f"Skipped demo {demo_idx} (no overlapping frames after alignment)")
            n_dropped_demos += 1
            continue

        demo_timestamps = np.arange(n_frames, dtype=np.float64) * float(dt) + float(start_timestamp)

        # load gripper widths per gripper-video (any video with a detected gripper_hardware_id >= 0)
        gripper_video_entries = []
        all_gripper_widths = []
        all_is_valid = []

        for cam_idx, row in demo_video_meta_df.iterrows():
            ghi = int(row["gripper_hardware_id"])
            if ghi < 0:
                continue

            video_dir = row["video_dir"]
            start_frame_idx = cam_start_frame_idxs[int(cam_idx)]

            # optional manual check
            check_path = video_dir.joinpath("check_result.txt")
            if check_path.is_file():
                if not check_path.open("r").read().startswith("true"):
                    print(f"Skipping {video_dir.name}, manually filtered with check_result.txt!=true")
                    continue

            pkl_path = video_dir.joinpath("tag_detection.pkl")
            if not pkl_path.is_file():
                print(f"Skipping {video_dir.name}, no tag_detection.pkl.")
                continue

            tag_detection_results = pickle.load(open(pkl_path, "rb"))
            if len(tag_detection_results) < start_frame_idx + n_frames:
                print(f"Skipping {video_dir.name}, tag_detection length too short for alignment window.")
                continue

            tag_detection_results = tag_detection_results[start_frame_idx : start_frame_idx + n_frames]
            video_timestamps = np.array([x["time"] for x in tag_detection_results], dtype=np.float64)

            # choose calibration interpolator
            if ghi in gripper_id_gripper_cal_map:
                gripper_cal_interp = gripper_id_gripper_cal_map[ghi]
            elif row["camera_serial"] in cam_serial_gripper_cal_map:
                gripper_cal_interp = cam_serial_gripper_cal_map[row["camera_serial"]]
                print(
                    f"Gripper id {ghi} not found in calibrations {list(gripper_id_gripper_cal_map.keys())}. "
                    f"Falling back to camera serial map."
                )
            else:
                raise RuntimeError("Gripper calibration not found.")

            left_id = 6 * ghi
            right_id = left_id + 1

            gripper_timestamps = []
            gripper_widths = []
            is_valid = np.zeros(len(tag_detection_results), dtype=bool)

            for k, td in enumerate(tag_detection_results):
                width = get_gripper_width(
                    td.get("tag_dict", {}),
                    left_id=left_id,
                    right_id=right_id,
                    nominal_z=nominal_z,
                )
                if width is not None:
                    is_valid[k] = True
                    gripper_timestamps.append(td["time"])
                    gripper_widths.append(float(gripper_cal_interp(width)))

            if is_valid.sum() < 60:
                print(f"Skipping {video_dir.name}, only {is_valid.sum()} valid gripper frames.")
                continue

            gripper_det_ratio = float(len(gripper_widths) / len(tag_detection_results))
            if gripper_det_ratio < 0.9:
                print(f"Warning: {video_dir.name} only {gripper_det_ratio:.3f} of gripper tags detected.")

            gripper_interp = get_interp1d(gripper_timestamps, gripper_widths)
            gripper_width_full = gripper_interp(video_timestamps)

            all_gripper_widths.append(gripper_width_full)
            all_is_valid.append(is_valid)

            gripper_video_entries.append(
                {
                    "camera_serial": row["camera_serial"],
                    "gripper_hardware_id": ghi,
                    "source_video": str(video_dir.joinpath("raw_video.mp4").relative_to(video_dir.parent)),
                    "video_timestamps": video_timestamps,
                }
            )

        if len(all_gripper_widths) == 0:
            print(f"Skipped demo {demo_idx} (no usable gripper videos).")
            n_dropped_demos += 1
            continue

        # aggregate validity across all gripper videos used
        all_is_valid_arr = np.array(all_is_valid)
        is_step_valid = np.all(all_is_valid_arr, axis=0)

        # remove valid segments that are too short
        seg_slices, seg_type = get_bool_segments(is_step_valid)
        for s, ok in zip(seg_slices, seg_type):
            if ok and (s.stop - s.start) < min_episode_length:
                is_step_valid[s.start : s.stop] = False

        # re-segment and emit episodes
        seg_slices, seg_type = get_bool_segments(is_step_valid)
        for s, ok in zip(seg_slices, seg_type):
            if not ok:
                continue

            start = int(s.start)
            end = int(s.stop)
            total_used_time += float((end - start) * dt)

            # choose timestamps from the first gripper video in the bundle
            episode_timestamps = gripper_video_entries[0]["video_timestamps"][start:end]

            grippers = []
            for gi, entry in enumerate(gripper_video_entries):
                grippers.append(
                    {
                        "gripper_width": all_gripper_widths[gi][start:end],
                        "camera_serial": entry["camera_serial"],
                        "gripper_hardware_id": entry["gripper_hardware_id"],
                        "source_video": entry["source_video"],
                    }
                )

            plan = {
                "episode_timestamps": episode_timestamps,
                "grippers": grippers,
            }

            if include_cameras:
                cameras = []
                for cam_idx, row in demo_video_meta_df.iterrows():
                    video_dir = row["video_dir"]
                    vid_start_frame = cam_start_frame_idxs[int(cam_idx)]
                    cameras.append(
                        {
                            "video_path": str(video_dir.joinpath("raw_video.mp4").relative_to(video_dir.parent)),
                            "video_start_end": (start + vid_start_frame, end + vid_start_frame),
                            "camera_serial": row["camera_serial"],
                            "camera_idx": int(cam_idx),
                        }
                    )
                plan["cameras"] = cameras

            all_plans.append(plan)

    used_ratio = (total_used_time / total_avaliable_time) if total_avaliable_time > 0 else 0.0
    print(f"{int(used_ratio * 100)}% of raw data are used.")
    print("n_dropped_demos", n_dropped_demos)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(all_plans, f)

    print(f"Wrote {len(all_plans)} episodes to {output}")


if __name__ == "__main__":
    main()