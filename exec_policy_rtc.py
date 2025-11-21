"""
Usage:
(umi): python exec_policy.py -o output

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from collections import deque
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.vic_umi_env import VicUmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
import pandas as pd

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', default='129.97.71.27')
@click.option('--gripper_ip', default='129.97.71.27')
@click.option('--gripper_port', type=int, default=4242)
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--steps_per_inference', '-si', default=5, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=120, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_crop', is_flag=True, default=False)
@click.option('--mirror_swap', is_flag=True, default=False)
@click.option('--use_rtc', is_flag=True, default=True)
# @click.option('--rtc_delay', type=int, default=2)

def main(output, robot_ip, gripper_ip, gripper_port,
    match_dataset, match_camera,
    vis_camera_idx, steps_per_inference, max_duration,
    frequency, no_mirror, sim_fov, camera_intrinsics, 
    mirror_crop, mirror_swap, use_rtc):

    # Diffusion Transformer
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/diffusion_transformer_pickplace.ckpt'

    # Diffusion UNet
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/diffusion_unet_pickplace_2.ckpt'
    ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/reaching_ball_multimodal_16.ckpt'
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/reaching_ball_unet.ckpt'
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/surface_wiping_unet_position_control.ckpt'
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/surface_wiping_unet_position_control_16_actions.ckpt'

    # Compliance policy unet
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/diffusion_unet_compliance_trial_2.ckpt'

    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        # with Spacemouse(shm_manager=shm_manager) as sm, \
        with KeystrokeCounter() as key_counter, \
            VicUmiEnv(
                output_dir=output, 
                robot_ip=robot_ip,
                gripper_ip=gripper_ip,
                gripper_port=gripper_port,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=None,
                # init_joints=init_joints,
                # enable_multi_cam_vis=True,
                # latency
                # camera_obs_latency=0.145,
                # robot_obs_latency=0.0001,
                # gripper_obs_latency=0.01,
                # robot_action_latency=0.0,
                # gripper_action_latency=0.0,
                camera_obs_latency=0.0,
                robot_obs_latency=0.0,
                gripper_obs_latency=0.0,
                robot_action_latency=0.0,
                gripper_action_latency=0.0,
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_crop=mirror_crop,
                mirror_swap=mirror_swap,
                # dev_video_path='/dev/video13',
                # action
                max_pos_speed=3.5,
                max_rot_speed=3.5,
                # robot_type=robot_type,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break
                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # RTC configuration
            rtc_schedule = "exp"
            rtc_max_guidance = 5.0
            DELAY_BUF_LEN = 8          # small rolling buffer for conservative delay forecast
            S_MIN = steps_per_inference  # minimum execution horizon per loop
            delay_buf = deque([S_MIN], maxlen=DELAY_BUF_LEN)

            # Track previous chunk for RTC
            prev_action_chunk = None
            chunk_generation_count = 0

            action_log = []
            actions_executed_from_current_chunk = 0

            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr

            device = torch.device('cuda')
            policy.eval().to(device)

            policy.debug_rtc = True

            print("Warming up policy inference")

            obs = env.get_obs()            
            episode_start_pose = np.concatenate([
                    obs[f'robot0_eef_pos'],
                    obs[f'robot0_eef_rot_axis_angle']
                ], axis=-1)[-1]
            with torch.no_grad():
                s = time.time()
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                if use_rtc and prev_action_chunk is not None and chunk_generation_count > 0:
                    # H, D = policy.action_horizon, policy.action_dim

                    # # How many actions from the current chunk have already been executed?
                    # s_exec = max(actions_executed_from_current_chunk, 0)
                    # # s_exec = min(s_exec, H)  # clamp
                    # s_exec = min(s_exec, H - 1)

                    # Conservative delay forecast d = max(buffer) and clamp to feasible region
                    # d_forecast = max(delay_buf) if len(delay_buf) > 0 else S_MIN
                    # d_forecast = int(max(0, min(d_forecast, s_exec, H - s_exec)))

                    # # Enforce H - s > d so we get a non-empty decay region
                    # d_forecast = max(delay_buf) if len(delay_buf) > 0 else S_MIN
                    # d_forecast = min(d_forecast, s_exec, H - s_exec - 2)  # <-- note the -1
                    # d_forecast = int(max(0, d_forecast))

                    # MIN_DECAY   = 3   # how many steps you want in the smooth region
                    # MAX_PREFIX  = 2   # how many steps can be “hard” prefix at most
                    # d_forecast = max(delay_buf) if len(delay_buf) > 0 else S_MIN
                    # # ensure enough blend: d <= H - s_exec - MIN_DECAY
                    # prefix_cap_decay = H - s_exec - MIN_DECAY
                    # # ensure prefix isn’t too long either
                    # prefix_cap = min(prefix_cap_decay, MAX_PREFIX)
                    # prefix_cap = max(0, prefix_cap)
                    # d_forecast = min(d_forecast, s_exec, prefix_cap)
                    # d_forecast = int(max(0, d_forecast))

                    # d_forecast = 4

                    # # Prefix attention horizon = H - s (ignore the soon-to-be-executed suffix)
                    # prefix_attn_h = max(0, H - s_exec)

                    # # Slice remaining prefix from previous chunk and right-pad to H
                    # remaining = prev_action_chunk[s_exec:]                      # [H - s_exec, D]
                    # if remaining.shape[0] < H:
                    #     pad = np.zeros((H - remaining.shape[0], remaining.shape[1]), dtype=remaining.dtype)
                    #     remaining = np.concatenate([remaining, pad], axis=0)    # [H, D]

                    # prev_chunk_tensor = torch.from_numpy(remaining).unsqueeze(0).to(device)

                    # H, D = policy.action_horizon, policy.action_dim
                    # s = steps_per_inference  # fixed execution horizon

                    # # Choose a fixed or measured delay d, with 0 <= d <= s <= H - d
                    # d_forecast = min(4, s, H - s)   # simple clamp

                    # Prefix attention horizon is H - s (as in the paper)
                    # prefix_attn_h = H - s

                    # Use the full previous chunk, no index shift
                    # prev_chunk_tensor = torch.from_numpy(prev_action_chunk).unsqueeze(0).to(device)


                    H, D = policy.action_horizon, policy.action_dim

                    # 1. How many steps from the *current* chunk will be (or were) executed
                    s_exec = max(actions_executed_from_current_chunk, 0)
                    s_exec = int(min(s_exec, H))  # safety

                    # 2. Conservative delay estimate from buffer (Algorithm 1: use max)
                    if len(delay_buf) > 0:
                        d_raw = max(delay_buf)
                    else:
                        d_raw = S_MIN  # fall back to min execution horizon

                    # 3. Clamp to feasible region: 0 ≤ d ≤ s_exec and 0 ≤ d ≤ H - s_exec - MIN_DECAY
                    MIN_DECAY = 3   # want at least 3 steps in the smooth region
                    d_max_by_s = s_exec
                    d_max_by_h = max(0, H - s_exec - MIN_DECAY)
                    d_max = max(0, min(d_max_by_s, d_max_by_h))

                    d_forecast = int(np.clip(d_raw, 0, d_max))

                    # 4. Prefix attention horizon: H - s_exec
                    prefix_attn_h = max(0, H - s_exec)

                    # 5. Remaining prefix from previous chunk, padded back to length H
                    remaining = prev_action_chunk[s_exec:]      # shape [H - s_exec, D]
                    if remaining.shape[0] < H:
                        pad = np.zeros((H - remaining.shape[0], remaining.shape[1]), dtype=remaining.dtype)
                        remaining = np.concatenate([remaining, pad], axis=0)
                    prev_chunk_tensor = torch.from_numpy(remaining).unsqueeze(0).to(device)


                    result = policy.realtime_action(
                        obs_dict,
                        prev_action_chunk=prev_chunk_tensor,
                        inference_delay=d_forecast,
                        prefix_attention_horizon=prefix_attn_h,
                        prefix_attention_schedule=rtc_schedule,
                        max_guidance_weight=rtc_max_guidance,
                        n_steps=policy.num_inference_steps
                    )
                    print(f"[RTC] Chunk {chunk_generation_count}: d={d_forecast}, s={s_exec}, H={H}, prefix_h={prefix_attn_h}")
                else:
                    d_forecast = 0
                    # First chunk or standard inference
                    result = policy.predict_action_flow(obs_dict)
                    print(f"[Standard] Initial chunk")

                raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                action = get_real_umi_action(raw_action, obs, action_pose_repr)

                # # Store for next RTC iteration
                # delay_buf.append(max(actions_executed_from_current_chunk, 0))
                # prev_action_chunk = raw_action.copy()
                # actions_executed_from_current_chunk = 0
                # chunk_generation_count += 1
                
                # inference_time = time.time() - s
                # print(f"Inference took {inference_time*1000:.1f}ms")

                inference_time = time.time() - s
                delay_steps = int(np.ceil(inference_time / dt))
                delay_steps = max(delay_steps, 0)

                delay_buf.append(delay_steps)
                print(f"Inference took {inference_time*1000:.1f}ms -> delay ~ {delay_steps} steps")

                prev_action_chunk = raw_action.copy()
                chunk_generation_count += 1
                actions_executed_from_current_chunk = 0

            print('Ready!')

            while True:
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    prev_action_chunk = None
                    chunk_generation_count = 0
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    # last_action_end_time = time.time()
                    action_log = []

                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        # print("Observations:", obs)
                        episode_start_pose = np.concatenate([
                            obs[f'robot0_eef_pos'],
                            obs[f'robot0_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                episode_start_pose=episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

                            if use_rtc and prev_action_chunk is not None and chunk_generation_count > 0:
                                # H, D = policy.action_horizon, policy.action_dim

                                # # Actions executed so far from current chunk => execution horizon s
                                # s_exec = max(actions_executed_from_current_chunk, 0)
                                # s_exec = min(s_exec, H)

                                # Conservative delay forecast d
                                # d_forecast = max(delay_buf) if len(delay_buf) > 0 else S_MIN
                                # d_forecast = int(max(0, min(d_forecast, s_exec, H - s_exec)))

                                # d_forecast = max(delay_buf) if len(delay_buf) > 0 else S_MIN
                                # d_forecast = min(d_forecast, s_exec, H - s_exec - 2)  # <-- note the -1
                                # d_forecast = int(max(0, d_forecast))

                                # MIN_DECAY   = 3   # how many steps you want in the smooth region
                                # MAX_PREFIX  = 2   # how many steps can be “hard” prefix at most
                                # d_forecast = max(delay_buf) if len(delay_buf) > 0 else S_MIN
                                # # ensure enough blend: d <= H - s_exec - MIN_DECAY
                                # prefix_cap_decay = H - s_exec - MIN_DECAY
                                # # ensure prefix isn’t too long either
                                # prefix_cap = min(prefix_cap_decay, MAX_PREFIX)
                                # prefix_cap = max(0, prefix_cap)
                                # d_forecast = min(d_forecast, s_exec, prefix_cap)
                                # d_forecast = int(max(0, d_forecast))

                                # d_forecast = 4

                                # # Prefix attention horizon = H - s
                                # prefix_attn_h = max(0, H - s_exec)

                                # # Remaining prefix from the *previous* chunk, right-padded to H
                                # remaining = prev_action_chunk[s_exec:]                   # [H - s_exec, D]
                                # if remaining.shape[0] < H:
                                #     pad = np.zeros((H - remaining.shape[0], remaining.shape[1]), dtype=remaining.dtype)
                                #     remaining = np.concatenate([remaining, pad], axis=0)
                                # prev_chunk_tensor = torch.from_numpy(remaining).unsqueeze(0).to(device)

                                H, D = policy.action_horizon, policy.action_dim

                                # 1. How many steps from the *current* chunk will be (or were) executed
                                s_exec = max(actions_executed_from_current_chunk, 0)
                                s_exec = int(min(s_exec, H))  # safety

                                # 2. Conservative delay estimate from buffer (Algorithm 1: use max)
                                if len(delay_buf) > 0:
                                    d_raw = max(delay_buf)
                                else:
                                    d_raw = S_MIN  # fall back to min execution horizon

                                # 3. Clamp to feasible region: 0 ≤ d ≤ s_exec and 0 ≤ d ≤ H - s_exec - MIN_DECAY
                                MIN_DECAY = 3   # want at least 3 steps in the smooth region
                                d_max_by_s = s_exec
                                d_max_by_h = max(0, H - s_exec - MIN_DECAY)
                                d_max = max(0, min(d_max_by_s, d_max_by_h))

                                d_forecast = int(np.clip(d_raw, 0, d_max))

                                # 4. Prefix attention horizon: H - s_exec
                                prefix_attn_h = max(0, H - s_exec)

                                # 5. Remaining prefix from previous chunk, padded back to length H
                                remaining = prev_action_chunk[s_exec:]      # shape [H - s_exec, D]
                                if remaining.shape[0] < H:
                                    pad = np.zeros((H - remaining.shape[0], remaining.shape[1]), dtype=remaining.dtype)
                                    remaining = np.concatenate([remaining, pad], axis=0)
                                prev_chunk_tensor = torch.from_numpy(remaining).unsqueeze(0).to(device)

                                result = policy.realtime_action(
                                    obs_dict,
                                    prev_action_chunk=prev_chunk_tensor,
                                    inference_delay=d_forecast,
                                    prefix_attention_horizon=prefix_attn_h,
                                    prefix_attention_schedule=rtc_schedule,
                                    max_guidance_weight=rtc_max_guidance,
                                    n_steps=policy.num_inference_steps
                                )
                                print(f"[RTC] Generated chunk {chunk_generation_count} with d={d_forecast}, s={s_exec}, prefix_h={prefix_attn_h}")
                            else:
                                d_forecast = 0
                                result = policy.predict_action_flow(obs_dict)
                                print(f"[Standard] Generated initial chunk")

                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)

                            # # After a new chunk is produced, log observed delay (s_exec), then reset
                            # delay_buf.append(max(actions_executed_from_current_chunk, 0))
                            # prev_action_chunk = raw_action.copy()
                            # chunk_generation_count += 1
                            # actions_executed_from_current_chunk = 0

                            # inference_time = time.time() - s
                            # print(f"Inference took {inference_time*1000:.1f}ms")

                            inference_time = time.time() - s
                            delay_steps = int(np.ceil(inference_time / dt))
                            delay_steps = max(delay_steps, 0)

                            delay_buf.append(delay_steps)
                            print(f"Inference took {inference_time*1000:.1f}ms -> delay ~ {delay_steps} steps")

                            prev_action_chunk = raw_action.copy()
                            chunk_generation_count += 1
                            actions_executed_from_current_chunk = 0

                            # action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                            # this_target_poses = action
                            # actions_to_execute = len(this_target_poses)

                        # # action_exec_latency = 0.01
                        # curr_time = time.time()
                        # is_new = action_timestamps > (curr_time)
                        # # print("Is new:", is_new)
                        # if np.sum(is_new) == 0:
                        #     # exceeded time budget, still do something
                        #     this_target_poses = this_target_poses[[-1]]
                        #     # schedule on next available step
                        #     next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                        #     action_timestamp = eval_t_start + (next_step_idx) * dt
                        #     print('Over budget', action_timestamp - curr_time)
                        #     action_timestamps = np.array([action_timestamp])
                        #     actions_to_execute = 1
                        # else:
                        #     this_target_poses = this_target_poses[is_new]
                        #     action_timestamps = action_timestamps[is_new]
                        #     actions_to_execute = len(this_target_poses)

                            # if actions_to_execute > steps_per_inference:
                            #     this_target_poses = this_target_poses[:steps_per_inference]
                            #     action_timestamps = action_timestamps[:steps_per_inference]
                            #     actions_to_execute = steps_per_inference

                            start_idx = d_forecast
                            end_idx   = min(start_idx + steps_per_inference, action.shape[0])

                            this_target_poses = action[start_idx:end_idx]
                            actions_to_execute = len(this_target_poses)

                            # update timestamps to align with the *next* s steps
                            action_timestamps = (
                                np.arange(actions_to_execute, dtype=np.float64) * dt + obs_timestamps[-1])

                        actions_executed_from_current_chunk += actions_to_execute

                        for a, t in zip(this_target_poses, action_timestamps):
                            a = a.tolist()
                            # print("Actions", a)
                            action_log.append({
                                'timestamp': t,
                                'ee_pos_0': a[0],
                                'ee_pos_1': a[1],
                                'ee_pos_2': a[2],
                                'ee_rot_0': a[3],
                                'ee_rot_1': a[4],
                                'ee_rot_2': a[5]
                            })

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        if mirror_crop:
                            vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                            crop_img = obs['camera0_rgb_mirror_crop'][-1]
                            vis_img = np.concatenate([vis_img, crop_img], axis=1)
                        else:
                            vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print('Stopped.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)

                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")

                    if hasattr(policy, "_rtc_W_log"):
                        np.save(os.path.join(output, f"rtc_W_log_episode_{episode_id}.npy"), np.array(policy._rtc_W_log, dtype=object))
                        print(f"Saved W logs with {len(policy._rtc_W_log)} entries.")

                    if hasattr(policy, "_rtc_guidance_log"):
                        np.save(os.path.join(output, f"rtc_guidance_log_episode_{episode_id}.npy"),
                                np.array(policy._rtc_guidance_log))
                        print(f"Saved guidance log with {len(policy._rtc_guidance_log)} entries.")

                    # stop robot.
                    env.end_episode()
                    if len(action_log) > 0:
                        df = pd.DataFrame(action_log)
                        csv_path = os.path.join(output, f"policy_actions_episode_{episode_id}.csv")
                        df.to_csv(csv_path, index=False)
                        print(f"Saved actions to {csv_path}")
                
                print("Stopped.")

# %%
if __name__ == '__main__':
    main()