"""
Usage:
(umi): python exec_policy_rtc.py -o output

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
@click.option('--gripper_ip', default='129.97.71.27')
@click.option('--gripper_port', type=int, default=4242)
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--steps_per_inference', '-si', default=5, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=360, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_crop', is_flag=True, default=False)
@click.option('--mirror_swap', is_flag=True, default=False)

def main(output, robot_ip, gripper_ip, gripper_port,
         match_dataset, match_camera, vis_camera_idx, steps_per_inference, 
    max_duration, frequency, no_mirror, sim_fov, camera_intrinsics, 
    mirror_crop, mirror_swap):

    # Diffusion UNet
    ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/reaching_ball_multimodal_16.ckpt'
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/peg_in_hole_position_control.ckpt'

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
                # action
                max_pos_speed=1.5,
                max_rot_speed=1.5,
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
            delay_buf = deque([steps_per_inference], maxlen=DELAY_BUF_LEN)

            # Track previous chunk in *model* action space (like action_chunk in eval_flow.py)
            prev_raw_chunk = None      # shape [H, D] in policy action space
            chunk_generation_count = 0

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

            # Grab one observation and build obs_dict just like in the main loop
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

                t0 = time.time()
                _ = policy.predict_action_flow(obs_dict)   # just to warm up
                infer_time = time.time() - t0

                delay_steps = int(np.ceil(infer_time / dt))
                delay_steps = max(delay_steps, 0)
                delay_buf.append(delay_steps)

                print(f"[Warmup] Inference took {infer_time*1000:.1f}ms -> delay ~ {delay_steps} steps")

            print('Ready!')


            while True:
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    prev_raw_chunk = None
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

                    # Logical end-of-queue time for the robot’s action schedule
                    next_action_time = eval_t_start

                    is_first_chunk = True
                    # iter_idx = 0

                    while True:
                        # t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

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
                        # ====== Algorithm 1 / eval_flow-style RTC ======
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

                            H, D = policy.action_horizon, policy.action_dim
                            s_min = min(steps_per_inference, H)  # minimum execution horizon

                            if prev_raw_chunk is None:
                                base_result = policy.predict_action_flow(obs_dict)
                                prev_raw_chunk = base_result['action_pred'][0].detach().cpu().numpy()
                                is_first_chunk = True
                            else:
                                is_first_chunk = False

                            # === 1) Forecast delay (for RTC inpainting) ===
                            if len(delay_buf) > 0:
                                d_forecast_raw = max(delay_buf)
                            else:
                                d_forecast_raw = s_min

                            # s_horizon: how many steps of this chunk we intend to execute
                            s_horizon = int(min(max(s_min, d_forecast_raw), H))

                            # Clamp forecast delay into [0, s_horizon]
                            d_forecast = int(min(max(0, d_forecast_raw), s_horizon))

                            prefix_attn_h = max(0, H - s_horizon)

                            prev_chunk_tensor = torch.from_numpy(prev_raw_chunk).unsqueeze(0).to(device)
                            result = policy.realtime_action(
                                obs_dict,
                                prev_action_chunk=prev_chunk_tensor,
                                inference_delay=d_forecast,          # like inference_delay in LeRobot
                                prefix_attention_horizon=prefix_attn_h,
                                prefix_attention_schedule=rtc_schedule,
                                max_guidance_weight=rtc_max_guidance,
                                n_steps=policy.num_inference_steps
                            )
                            curr_raw_chunk = result['action_pred'][0].detach().cpu().numpy()

                            # === 2) Measure REAL delay (how many steps executed while thinking) ===
                            # real delay
                            inference_time = time.time() - infer_start
                            real_delay = int(np.ceil(inference_time / dt))
                            real_delay = max(0, min(real_delay, s_horizon))
                            if is_first_chunk:
                                # no old actions were being executed yet
                                real_delay = 0
                            delay_buf.append(real_delay)
                            print(f"[RTC] Inference took {inference_time*1000:.1f}ms -> real_delay ~ {real_delay} steps (H={H}, s={s_horizon})")

                            # === 3) Convert current RTC-refined chunk to world-space actions ===
                            curr_action_world = get_real_umi_action(curr_raw_chunk, obs, action_pose_repr)

                            H_world = min(curr_action_world.shape[0], H)
                            s_horizon = min(s_horizon, H_world)
                            real_delay = min(real_delay, s_horizon)

                            # === 4) Like ActionQueue._replace_actions_queue: drop first real_delay actions ===
                            this_target_poses = curr_action_world[real_delay:s_horizon]
                            actions_to_execute = this_target_poses.shape[0]


                        # ====== END RTC BLOCK ======

                        if actions_to_execute > 0:
                            # Schedule these actions after the current logical queue end
                            action_timestamps = next_action_time + (
                                (np.arange(actions_to_execute, dtype=np.float64) + 1) * dt
                            )
                            # 5) Timestamps for the next s steps
                            # action_timestamps = (
                            #     np.arange(actions_to_execute, dtype=np.float64) * dt + obs_timestamps[-1]
                            # )
                            # execute actions
                            env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps,
                                compensate_latency=True
                            )
                            # print(f"Submitted {len(this_target_poses)} steps of actions.")

                            # Advance the logical queue end to the last scheduled time
                            next_action_time = action_timestamps[-1]
                        
                        else:
                            # Nothing to send this iteration (very high delay or other issue)
                            print(f"[RTC] real_delay={real_delay}, s={s_horizon}, no new actions scheduled this loop.")

                        # 6) Shift by s_horizon (the committed window), NOT by actions_to_execute.
                        if s_horizon < H:
                            shifted = curr_raw_chunk[s_horizon:]            # true future tail
                            pad = np.zeros((s_horizon, D), dtype=curr_raw_chunk.dtype)
                            prev_raw_chunk = np.concatenate([shifted, pad], axis=0)
                        else:
                            prev_raw_chunk = np.zeros_like(curr_raw_chunk)

                        chunk_generation_count += 1

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
                        # cv2.imshow('default', vis_img[...,::-1])

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

                        # precise_wait(t_cycle_end - frame_latency)
                        # iter_idx += steps_per_inference


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
                
                print("Stopped.")

# %%
if __name__ == '__main__':
    main()