"""
Usage:
(umi): python exec_policy_rtc.py -o output

Real-Time Chunking (RTC) Implementation for UMI.
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

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', default='129.97.71.27')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--steps_per_inference', '-si', default=4, type=int, help="Minimum action horizon (s_min).")
@click.option('--max_duration', '-md', default=360, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_crop', is_flag=True, default=False)
@click.option('--mirror_swap', is_flag=True, default=False)

def main(output, robot_ip, match_dataset, match_camera, vis_camera_idx, steps_per_inference, 
    max_duration, frequency, no_mirror, sim_fov, camera_intrinsics, 
    mirror_crop, mirror_swap):

    # Load Checkpoint
    ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/peg_in_hole_position_control.ckpt'
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']

    # Setup Timing
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    
    # Fisheye setup
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

    print(f"RTC Mode: s_min={steps_per_inference}, frequency={frequency}Hz")

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            VicUmiEnv(
                output_dir=output, 
                robot_ip=robot_ip,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=None,
                camera_obs_latency=0.0,
                robot_obs_latency=0.0,
                robot_action_latency=0.0,
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_crop=mirror_crop,
                mirror_swap=mirror_swap,
                max_pos_speed=1.5,
                max_rot_speed=1.5,
                shm_manager=shm_manager) as env:
            
            cv2.setNumThreads(2)
            print("Waiting for camera...")
            time.sleep(1.0)

            # --- RTC State Variables ---
            DELAY_BUF_LEN = 10
            delay_buf = deque([steps_per_inference], maxlen=DELAY_BUF_LEN)
            prev_raw_chunk = None
            chunk_generation_count = 0
            
            # --- Load Model ---
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16
            
            # RTC specific parameters
            policy.debug_rtc = True 
            rtc_max_guidance = 5.0
            rtc_schedule = "exp"
            
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr

            device = torch.device('cuda')
            policy.eval().to(device)

            # --- Warmup ---
            print("Warming up policy...")
            obs = env.get_obs()
            episode_start_pose = np.concatenate([
                    obs[f'robot0_eef_pos'],
                    obs[f'robot0_eef_rot_axis_angle']
                ], axis=-1)[-1]

            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs,
                    shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=obs_pose_rep,
                    episode_start_pose=episode_start_pose
                )
                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
                )
                _ = policy.predict_action_flow(obs_dict)
            print('Ready!')

            while True:
                # ========== Episode Loop ==============
                try:
                    policy.reset()
                    
                    # Reset RTC State
                    prev_raw_chunk = None # A_cur in paper
                    delay_buf.clear()
                    delay_buf.append(steps_per_inference)
                    
                    # Start Logic Clock
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    
                    # This is our logical clock for the robot queue
                    # We start scheduling actions starting exactly at eval_t_start
                    next_execution_time = eval_t_start 
                    
                    env.start_episode(eval_t_start)
                    precise_wait(eval_t_start - 1/60, time_func=time.time)
                    print("Episode Started.")
                    
                    step_idx = 0
                    
                    while True:
                        # 1. Get Observation
                        # Note: In RTC, we grab obs asynchronously. 
                        # The robot continues moving based on previously sent queue.
                        obs = env.get_obs()
                        episode_start_pose = np.concatenate([
                            obs[f'robot0_eef_pos'],
                            obs[f'robot0_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        
                        # 2. Estimate Latency (d) & Execution Horizon (s)
                        # d = max observed latency
                        d_steps = int(max(delay_buf))
                        # s = execution horizon. Must be at least d to avoid buffer underrun.
                        # Also respect user's steps_per_inference (s_min)
                        s_steps = max(d_steps, steps_per_inference)
                        
                        # 3. Prepare RTC Inference
                        with torch.no_grad():
                            infer_start = time.time()
                            
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta,
                                obs_pose_repr=obs_pose_rep,
                                episode_start_pose=episode_start_pose
                            )
                            obs_dict = dict_apply(
                                obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
                            )
                            
                            H = policy.action_horizon
                            D = policy.action_dim
                            
                            # Initialize prev_chunk if first step
                            if prev_raw_chunk is None:
                                # Naive prediction for first step
                                base = policy.predict_action_flow(obs_dict)
                                prev_raw_chunk = base['action_pred'][0].detach().cpu().numpy()
                                # Fill buffer with zeros if naive pred is short? usually H match
                            
                            # RTC Constraints
                            # d: freeze horizon (actions already in queue)
                            # s: execution horizon (actions we commit to this iter)
                            # Clamping
                            d = min(d_steps, H-1)
                            s = min(s_steps, H)
                            
                            # Prepare previous chunk tensor
                            prev_chunk_tensor = torch.from_numpy(prev_raw_chunk).unsqueeze(0).to(device)
                            
                            # Inpaint / Guided Inference
                            prefix_attn_h = max(0, H - s) # How much to overlap
                            
                            result = policy.realtime_action(
                                obs_dict,
                                prev_action_chunk=prev_chunk_tensor,
                                inference_delay=d,
                                prefix_attention_horizon=prefix_attn_h,
                                prefix_attention_schedule=rtc_schedule,
                                max_guidance_weight=rtc_max_guidance,
                                n_steps=policy.num_inference_steps
                            )
                            
                            curr_raw_chunk = result['action_pred'][0].detach().cpu().numpy()
                            
                            # Update Latency Stats
                            infer_dur = time.time() - infer_start
                            measured_d = int(np.ceil(infer_dur / dt))
                            delay_buf.append(measured_d)
                            
                            # 4. Filter & Send Actions
                            # We must convert RAW model actions to WORLD frame
                            curr_action_world = get_real_umi_action(curr_raw_chunk, obs, action_pose_repr)
                            
                            # Algorithm 1 Logic:
                            # 0 to d: Frozen (already sent in previous iters). Do NOT resend.
                            # d to s: The new extension we commit to now.
                            # s to H: Future plan (will be inpainted next time).
                            
                            start_idx = d
                            end_idx = s
                            
                            if end_idx > start_idx:
                                # Extract new actions
                                actions_to_send = curr_action_world[start_idx : end_idx]
                                
                                # Generate Timestamps using Logical Clock
                                # This guarantees they are in the future relative to the *queue end*
                                timestamps_to_send = (
                                    np.arange(len(actions_to_send), dtype=np.float64) + 1
                                ) * dt + (next_execution_time - dt) # (start at next_execution_time)
                                
                                # Execute
                                env.exec_actions(
                                    actions=actions_to_send,
                                    timestamps=timestamps_to_send,
                                    compensate_latency=True
                                )
                                
                                # Advance Logical Clock
                                next_execution_time += (len(actions_to_send) * dt)
                                
                                # Logging
                                print(f"[RTC] d_est={d_steps} s={s} | Infer={infer_dur*1000:.1f}ms | Sent {len(actions_to_send)} steps")
                            else:
                                print(f"[RTC Warning] s ({s}) <= d ({d}). No new actions sent! System lagging.")

                            # 5. Shift Chunk for Next Iteration (Algorithm 1 Line 14)
                            # We shift by 's' (steps executed/committed)
                            if s < H:
                                shifted = curr_raw_chunk[s:] 
                                pad = np.zeros((s, D), dtype=curr_raw_chunk.dtype)
                                prev_raw_chunk = np.concatenate([shifted, pad], axis=0)
                            else:
                                prev_raw_chunk = np.zeros_like(curr_raw_chunk)

                        # Visualization & Keys
                        if mirror_crop:
                            vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                            crop_img = obs['camera0_rgb_mirror_crop'][-1]
                            vis_img = np.concatenate([vis_img, crop_img], axis=1)
                        else:
                            vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                        
                        text = f'RTC | Delay: {d_steps} | Steps: {s_steps}'
                        cv2.putText(vis_img, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                        cv2.imshow('default', vis_img[...,::-1])
                        
                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        if any(k == KeyCode(char='s') for k in press_events):
                            env.end_episode()
                            break
                        
                        if (time.time() - eval_t_start) > max_duration:
                            env.end_episode()
                            break
                        
                        step_idx += 1

                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.end_episode()
                    break
            
            print("Stopped.")

if __name__ == '__main__':
    main()