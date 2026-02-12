#!/usr/bin/env python3
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
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

@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-of', '--out_fov', type=float, default=None)
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-nm', '--no_mirror', is_flag=True, default=False, help="Disable mirror observation by masking them out")
@click.option('-ms', '--mirror_swap', is_flag=True, default=False)
@click.option('-n', '--num_workers', type=int, default=None)
def main(input, output, out_res, out_fov, compression_level, 
         no_mirror, mirror_swap, num_workers):
    
    if os.path.exists(output):
        click.confirm(f'Output file {output} exists! Overwrite?', abort=True)
        
    out_res = tuple(int(x) for x in out_res.split(','))
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)
            
    # --- RESTORED: Fisheye Rectification Logic ---
    fisheye_converter = None
    if out_fov is not None:
        # Assuming the first input path contains the calibration directory
        ipath_first = pathlib.Path(os.path.expanduser(input[0])).absolute()
        intr_path = ipath_first.joinpath('calibration', 'gopro_intrinsics_2_7k.json')
        if not intr_path.is_file():
            raise FileNotFoundError(f"Intrinsics not found at {intr_path}. Required for out_fov.")
            
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=out_res,
            out_fov=out_fov
        )
        
    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = list()
    
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan_camera_only.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan_camera_only.pkl")
            continue
        
        plan = pickle.load(plan_path.open('rb'))
        videos_dict = defaultdict(list)
        
        for plan_episode in plan:
            cameras = plan_episode['cameras']
            if n_cameras is None:
                n_cameras = len(cameras)
            
            episode_data = dict()
            # Absolute Unix timestamps are saved here
            episode_data['timestamp'] = plan_episode['episode_timestamps'].astype(np.float64)
            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            
            n_frames = len(plan_episode['episode_timestamps'])
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera['video_path']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                
                v_start, v_end = camera['video_start_end']
                videos_dict[str(video_path)].append({
                    'camera_idx': cam_id,
                    'frame_start': v_start,
                    'frame_end': v_end,
                    'buffer_start': buffer_start
                })
            buffer_start += n_frames
        
        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
    
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    for cam_id in range(n_cameras):
        name = f'camera{cam_id}_rgb'
        total_frames = out_replay_buffer['timestamp'].shape[0]
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(total_frames,) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    def video_to_zarr(replay_buffer, mp4_path, tasks):
        # Optional: Load tags if they exist for inpainting
        pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        tag_data = None
        if os.path.exists(pkl_path):
            tag_data = pickle.load(open(pkl_path, 'rb'))

        resize_tf = get_image_transform(in_res=(iw, ih), out_res=out_res)
        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = tasks[0]['camera_idx']
        name = f'camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name]
        
        # --- RESTORED: Mirror Swap Geometry ---
        is_mirror = None
        if mirror_swap:
            ow, oh = out_res
            mirror_mask = np.ones((oh,ow,3),dtype=np.uint8)
            mirror_mask = draw_predefined_mask(
                mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
            is_mirror = (mirror_mask[...,0] == 0)
        
        curr_task_idx = 0
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            in_stream.thread_count = 1
            buffer_idx = 0
            
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), 
                                        total=in_stream.frames, leave=False):
                if curr_task_idx >= len(tasks): break
                task = tasks[curr_task_idx]
                
                if frame_idx < task['frame_start']: continue
                elif frame_idx < task['frame_end']:
                    if frame_idx == task['frame_start']:
                        buffer_idx = task['buffer_start']
                    
                    img = frame.to_ndarray(format='rgb24')

                    if tag_data is not None:
                        this_det = tag_data[frame_idx]
                        for corners in [x['corners'] for x in this_det['tag_dict'].values()]:
                            img = inpaint_tag(img, corners)
                        
                    # --- RESTORED: Original Masking and Fisheye Pipeline ---
                    img = draw_predefined_mask(img, color=(0,0,0), 
                        mirror=no_mirror, gripper=True, finger=False)
                    
                    if fisheye_converter is None:
                        img = resize_tf(img)
                    else:
                        img = fisheye_converter.forward(img)
                        
                    if mirror_swap and is_mirror is not None:
                        img[is_mirror] = img[:,::-1,:][is_mirror]
                        
                    img_array[buffer_idx] = img
                    buffer_idx += 1
                    
                    if (frame_idx + 1) == task['frame_end']:
                        curr_task_idx += 1

    with tqdm(total=len(vid_args)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))
                futures.add(executor.submit(video_to_zarr, out_replay_buffer, mp4_path, tasks))
            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print(f"Saving ReplayBuffer with high-precision timestamps to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(store=zip_store)
    print("Done!")

if __name__ == "__main__":
    main()