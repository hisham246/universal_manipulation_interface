"""
Usage:
(umi): python exec_policy_rtc_lerobot.py -o output

LeRobot-style RTC architecture:
- Actor thread that sends 1 action every dt to the robot
- Get-actions thread that runs RTC (realtime_action) and overwrites the future queue
- Main thread handles episodes, keyboard, visualization
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from collections import deque
import pathlib
import time
import threading
from typing import Optional

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
from umi.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_umi_obs_dict,
    get_real_umi_action
)

OmegaConf.register_new_resolver("eval", eval, replace=True)

class RTCSharedState:
    def __init__(self, H: int, D_raw: int, D_world: int, d_init: int = 0, delay_buf_size: int = 100):
        self.H = int(H)
        self.D_raw = int(D_raw)
        self.D_world = int(D_world)

        self.M = threading.Lock()
        self.C = threading.Condition(self.M)

        self.t = 0
        self.ocur = None

        self.Acur_raw   = np.zeros((self.H, self.D_raw), dtype=np.float32)
        self.Acur_world = np.zeros((self.H, self.D_world), dtype=np.float32)

        self.Q = deque([int(d_init)], maxlen=delay_buf_size)

        self.episode_start_time = None
        self.step_idx_global = 0

        self.acur_ready = False


# =========================
# Robot wrapper around VicUmiEnv
# =========================

class UmiRobotWrapper:
    def __init__(self, env: VicUmiEnv):
        self.env = env

    def get_obs(self):
        return self.env.get_obs()

    def send_action_at(self, action_world_1d: np.ndarray, target_time: float, compensate_latency: bool = True):
        actions = action_world_1d[None, :]
        timestamps = np.array([target_time], dtype=np.float64)
        self.env.exec_actions(actions=actions, timestamps=timestamps, compensate_latency=compensate_latency)


# =========================
# Observation & action processors
# =========================

def umi_build_obs_dict(env_obs, shape_meta, obs_pose_repr):
    """
    Wraps get_real_umi_obs_dict exactly like your current code:
    - computes episode_start_pose from last eef pose
    - returns numpy obs_dict
    """
    episode_start_pose = np.concatenate(
        [env_obs['robot0_eef_pos'], env_obs['robot0_eef_rot_axis_angle']],
        axis=-1
    )[-1]

    obs_dict_np = get_real_umi_obs_dict(
        env_obs=env_obs,
        shape_meta=shape_meta,
        obs_pose_repr=obs_pose_repr,
        episode_start_pose=episode_start_pose
    )
    return obs_dict_np


def umi_action_processor(action_world_1d: np.ndarray) -> np.ndarray:
    """
    Hook to clamp / scale actions before sending to robot.

    For now, this is identity.
    Adjust here if you need safety clamps (e.g., on position deltas or gripper).
    """
    return action_world_1d


# =========================
# RTC get-actions thread (LeRobot-style)
# =========================

def get_actions_umi_thread(
    policy,
    robot_wrapper: UmiRobotWrapper,
    shared: RTCSharedState,
    shutdown_event: threading.Event,
    shape_meta,
    obs_pose_repr,
    action_pose_repr,
    smin: int,
    rtc_schedule: str = "exp",
    rtc_max_guidance: float = 5.0,
    debug: bool = True,
):
    device = next(policy.parameters()).device
    H = shared.H
    D_raw = shared.D_raw
    D_world = shared.D_world

    # ---- Bootstrap: wait for first obs, then compute Ainit ----
    with shared.M:
        obs0 = shared.ocur

    while (obs0 is None) and (not shutdown_event.is_set()):
        time.sleep(0.001)
        with shared.M:
            obs0 = shared.ocur

    if shutdown_event.is_set():
        return

    obs_dict_np = umi_build_obs_dict(obs0, shape_meta, obs_pose_repr)
    obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

    with torch.no_grad():
        base = policy.predict_action_flow(obs_dict)
        Araw = base["action_pred"][0].detach().cpu().numpy()           # (H, D_raw)
        assert Araw.shape[0] == H, f"Araw H mismatch: {Araw.shape} vs H={H}"
        assert Araw.shape[1] == D_raw, f"Araw D_raw mismatch: {Araw.shape} vs D_raw={D_raw}"

        Aworld = get_real_umi_action(Araw, obs0, action_pose_repr)     # (H, D_world)
        assert Aworld.shape[0] == H, f"Aworld H mismatch: {Aworld.shape} vs H={H}"
        assert Aworld.shape[1] == D_world, f"Aworld D_world mismatch: {Aworld.shape} vs D_world={D_world}"

    with shared.C:
        shared.Acur_raw[...] = Araw
        shared.Acur_world[...] = Aworld
        shared.acur_ready = True
        shared.C.notify_all()

    if debug:
        print(f"[RTC] initialized Acur (bootstrap) raw={Araw.shape} world={Aworld.shape}")

    # ---- Main loop ----
    while not shutdown_event.is_set():
        # Wait until at least smin steps have been executed into the current chunk
        with shared.C:
            while (shared.t < smin) and (not shutdown_event.is_set()):
                shared.C.wait(timeout=0.01)
            if shutdown_event.is_set():
                break

            # Snapshot execution index into current chunk
            s_exec = int(min(shared.t, H))  # allow H, never > H

            # Previous chunk tail (Aprev = Acur[s:])
            Aprev_raw = shared.Acur_raw[s_exec:].copy()  # (H-s_exec, D_raw)

            # Latest observation
            obs = shared.ocur
            if obs is None:
                continue

            # Delay forecast: max(Q)
            d_hat = int(max(shared.Q)) if len(shared.Q) else 0
            d_hat = int(min(max(d_hat, 0), H - 1))

        # Build obs_dict outside lock
        obs_dict_np = umi_build_obs_dict(obs, shape_meta, obs_pose_repr)
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

        # Pad Aprev_raw back to length H for policy API
        if Aprev_raw.shape[0] < H:
            pad = np.zeros((H - Aprev_raw.shape[0], D_raw), dtype=Aprev_raw.dtype)
            Aprev_raw_full = np.concatenate([Aprev_raw, pad], axis=0)
        else:
            Aprev_raw_full = Aprev_raw[:H]

        prev_chunk_tensor = torch.from_numpy(Aprev_raw_full).unsqueeze(0).to(device)

        overlap_len = max(0, H - s_exec)

        infer_start = time.time()
        with torch.no_grad():
            result = policy.realtime_action(
                obs_dict,
                prev_action_chunk=prev_chunk_tensor,
                inference_delay=d_hat,
                prefix_attention_horizon=overlap_len,
                prefix_attention_schedule=rtc_schedule,
                max_guidance_weight=rtc_max_guidance,
                n_steps=policy.num_inference_steps,
            )
        infer_end = time.time()

        Anew_raw = result["action_pred"][0].detach().cpu().numpy()      # (H, D_raw)
        assert Anew_raw.shape == (H, D_raw), f"Anew_raw shape {Anew_raw.shape} != {(H, D_raw)}"

        Anew_world = get_real_umi_action(Anew_raw, obs, action_pose_repr)  # (H, D_world)
        assert Anew_world.shape == (H, D_world), f"Anew_world shape {Anew_world.shape} != {(H, D_world)}"

        # Swap Acur and compute observed delay
        with shared.C:
            shared.Acur_raw[...] = Anew_raw
            shared.Acur_world[...] = Anew_world

            observed_delay_steps = int(shared.t - s_exec)
            observed_delay_steps = int(min(max(observed_delay_steps, 0), H - 1))

            shared.t = observed_delay_steps
            shared.Q.append(shared.t)
            shared.C.notify_all()

        if debug:
            qmax = int(max(shared.Q)) if len(shared.Q) else 0
            print(
                f"[RTC] s_exec={s_exec} overlap={overlap_len} "
                f"d_hat={d_hat} d_obs={observed_delay_steps} "
                f"infer={(infer_end-infer_start)*1000:.1f}ms Qmax={qmax} reset_t={shared.t}"
            )


# =========================
# Actor thread (LeRobot-style)
# =========================

def actor_control_umi_thread(
    robot_wrapper: UmiRobotWrapper,
    shared: RTCSharedState,
    shutdown_event: threading.Event,
    frequency: float,
    debug: bool = False,
):
    dt = 1.0 / frequency

    # Wait until Acur is ready
    with shared.C:
        while (not shared.acur_ready) and (not shutdown_event.is_set()):
            shared.C.wait(timeout=0.01)
    if shutdown_event.is_set():
        return

    # Wait until episode_start_time (wall clock)
    with shared.M:
        t0_wall = shared.episode_start_time
    if t0_wall is None:
        t0_wall = time.time()
        with shared.M:
            shared.episode_start_time = t0_wall
            shared.step_idx_global = 0

    while (time.time() < t0_wall) and (not shutdown_event.is_set()):
        precise_wait(t0_wall, time_func=time.time)
    if shutdown_event.is_set():
        return

    next_tick_mono = time.monotonic()

    while not shutdown_event.is_set():
        # 1) get observation (onext)
        obs = robot_wrapper.get_obs()

        # 2) GETACTION critical section
        with shared.C:
            shared.ocur = obs

            # If we consumed the whole chunk, block until RTC swaps a new chunk and resets t
            while (shared.t >= shared.H) and (not shutdown_event.is_set()):
                shared.C.wait(timeout=0.01)
            if shutdown_event.is_set():
                break

            # Advance into the chunk and wake inference
            shared.t += 1
            shared.C.notify()

            idx = shared.t - 1  # guaranteed < H
            action = shared.Acur_world[idx].copy()

            # Deterministic timestamp grid, but "future-safe" so env.exec_actions won't drop it
            now = time.time()
            if shared.episode_start_time is None:
                shared.episode_start_time = now
                shared.step_idx_global = 0

            eps = 0.002  # 2ms safety
            while shared.episode_start_time + shared.step_idx_global * dt <= now + eps:
                shared.step_idx_global += 1

            target_time = shared.episode_start_time + shared.step_idx_global * dt
            shared.step_idx_global += 1

        # 3) execute action
        robot_wrapper.send_action_at(action, target_time)

        # 4) sleep to maintain loop timing
        next_tick_mono += dt
        precise_wait(next_tick_mono, time_func=time.monotonic)

        if debug and (shared.step_idx_global % 20 == 0):
            with shared.M:
                qmax = int(max(shared.Q)) if len(shared.Q) else 0
                print(f"[ACTOR] t={shared.t} step_global={shared.step_idx_global} Qmax={qmax}")

# =========================
# Main CLI
# =========================

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

def main(
    output, robot_ip, gripper_ip, gripper_port,
    match_dataset, match_camera, vis_camera_idx, steps_per_inference,
    max_duration, frequency, no_mirror, sim_fov, camera_intrinsics,
    mirror_crop, mirror_swap
):

    # Diffusion UNet ckpt
    ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/reaching_ball_multimodal_16.ckpt'
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/peg_in_hole_position_control.ckpt'

    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']

    dt = 1.0 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)

    # Fisheye / camera conversion
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r'))
        )
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)

    from multiprocessing.managers import SharedMemoryManager
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
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_crop=mirror_crop,
                mirror_swap=mirror_swap,
                max_pos_speed=1.5,
                max_rot_speed=1.5,
                shm_manager=shm_manager,
                enable_gripper=False,
            ) as env:

            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # Optional: load match_dataset (unchanged from your script)
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(
                    str(match_zarr_path), mode='r')
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
            rtc_max_guidance = 10.0

            # Create workspace & policy AFTER fork
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model

            policy.num_inference_steps = 16
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr

            device = torch.device('cuda')
            policy.eval().to(device)

            policy.debug_rtc = True

            print("Warming up policy inference")
            # Warmup before any episodes (safe to call env.get_obs here; recording not started yet)
            obs = env.get_obs()
            episode_start_pose = np.concatenate([
                obs['robot0_eef_pos'],
                obs['robot0_eef_rot_axis_angle']
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
                _ = policy.predict_action_flow(obs_dict)
                infer_time = time.time() - t0
                delay_steps = int(np.ceil(infer_time / dt))
                delay_steps = max(delay_steps, 0)
                print(f"[Warmup] Inference took {infer_time*1000:.1f}ms -> ~{delay_steps} steps")

            print('Ready!')

            # Wrap env in LeRobot-style robot wrapper
            robot_wrapper = UmiRobotWrapper(env)

            try:
                while True:
                    # New episode

                    policy.reset()

                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start_mono = time.monotonic() + start_delay

                    shutdown_event = threading.Event()

                    # Prime first obs so we can infer D_world correctly
                    obs0 = env.get_obs()

                    H = int(policy.action_horizon)
                    D_raw = int(policy.action_dim)

                    # Infer D_world from your adapter (pose6 + gripper1 => 7)
                    Araw_dummy = np.zeros((H, D_raw), dtype=np.float32)
                    Aworld_dummy = get_real_umi_action(Araw_dummy, obs0, action_pose_repr)
                    D_world = int(Aworld_dummy.shape[1])

                    shared = RTCSharedState(H=H, D_raw=D_raw, D_world=D_world, d_init=0, delay_buf_size=100)

                    # Anchor shared episode start time to the same start time used by env.start_episode
                    shared.episode_start_time = eval_t_start
                    shared.step_idx_global = 0

                    # Start recording
                    env.start_episode(eval_t_start)

                    # Store initial observation for RTC bootstrap
                    with shared.M:
                        shared.ocur = obs0


                    frame_latency = 1 / 60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")

                    rtc_thread = threading.Thread(
                        target=get_actions_umi_thread,
                        args=(
                            policy,
                            robot_wrapper,
                            shared,
                            shutdown_event,
                            cfg.task.shape_meta,
                            obs_pose_rep,
                            action_pose_repr,
                            steps_per_inference,   # smin
                        ),
                        kwargs={
                            "rtc_schedule": rtc_schedule,
                            "rtc_max_guidance": rtc_max_guidance,
                            "debug": True,
                        },
                        daemon=True,
                    )

                    actor_thread = threading.Thread(
                        target=actor_control_umi_thread,
                        args=(robot_wrapper, shared, shutdown_event, frequency),
                        kwargs={"debug": False},
                        daemon=True,
                    )

                    # start inference first (optional), actor will wait on shared.acur_ready anyway
                    rtc_thread.start()
                    actor_thread.start()

                    t_start = t_start_mono
                    episode_id = env.replay_buffer.n_episodes
                    print(f"Episode {episode_id} running...")

                    while True:
                        # Visualization based on latest obs from actor thread
                        with shared.M:
                            obs_vis = shared.ocur
                        if obs_vis is not None:
                            if mirror_crop:
                                vis_img = obs_vis[f'camera{vis_camera_idx}_rgb'][-1]
                                crop_img = obs_vis['camera0_rgb_mirror_crop'][-1]
                                vis_img = np.concatenate([vis_img, crop_img], axis=1)
                            else:
                                vis_img = obs_vis[f'camera{vis_camera_idx}_rgb'][-1]

                            text = 'Episode: {}, Time: {:.1f}'.format(
                                episode_id, time.monotonic() - t_start
                            )
                            cv2.putText(
                                vis_img,
                                text,
                                (10, 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                thickness=1,
                                color=(255, 255, 255)
                            )
                            # cv2.imshow('default', vis_img[..., ::-1])

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                print('Stopped by user.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max duration reached.")
                            stop_episode = True

                        if stop_episode:
                            shutdown_event.set()
                            actor_thread.join()
                            rtc_thread.join()
                            env.end_episode()
                            break

                        time.sleep(0.01)

            except KeyboardInterrupt:
                print("Interrupted!")
                shutdown_event.set()
                try:
                    actor_thread.join(timeout=2.0)
                    rtc_thread.join(timeout=2.0)
                except Exception:
                    pass

                # Save RTC logs if present
                try:
                    if hasattr(policy, "_rtc_W_log"):
                        np.save(os.path.join(output, f"rtc_W_log_episode_{episode_id}.npy"),
                                np.array(policy._rtc_W_log, dtype=object))
                        print(f"Saved W logs with {len(policy._rtc_W_log)} entries.")

                    if hasattr(policy, "_rtc_guidance_log"):
                        np.save(os.path.join(output, f"rtc_guidance_log_episode_{episode_id}.npy"),
                                np.array(policy._rtc_guidance_log))
                        print(f"Saved guidance log with {len(policy._rtc_guidance_log)} entries.")
                except Exception as e:
                    print(f"Error saving RTC logs: {e}")

                env.end_episode()
                print("Stopped.")


if __name__ == '__main__':
    main()
