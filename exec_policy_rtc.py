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


# =========================
# Latency tracking utilities for Real-Time Chunking (RTC).
# =========================
    
class LatencyTracker:
    """Tracks recent latencies (seconds) and provides max/percentiles."""

    def __init__(self, maxlen: int = 100):
        self._values = deque(maxlen=maxlen)
        self.reset()

    def reset(self) -> None:
        self._values.clear()
        self.max_latency = 0.0

    def add(self, latency: float) -> None:
        val = float(latency)
        if val < 0:
            return
        self._values.append(val)
        self.max_latency = max(self.max_latency, val)

    def __len__(self) -> int:
        return len(self._values)

    def max(self) -> float:
        return self.max_latency

    def percentile(self, q: float) -> float:
        if not self._values:
            return 0.0
        q = float(q)
        if q <= 0.0:
            return min(self._values)
        if q >= 1.0:
            return self.max_latency
        vals = np.array(list(self._values), dtype=np.float32)
        return float(np.quantile(vals, q))

    def p95(self) -> float:
        return self.percentile(0.95)


# =========================
# Robot wrapper around VicUmiEnv
# =========================

class UmiRobotWrapper:
    """
    LeRobot-like wrapper around VicUmiEnv.

    - update_observation(): called by actor thread, fetches env.get_obs()
    - get_observation(): used by RTC thread to read latest obs
    - send_action(): takes one world-space action (1D np array) and calls env.exec_actions
    """

    def __init__(self, env: VicUmiEnv, dt: float):
        self.env = env
        self.dt = dt

        self._latest_obs = None
        self._obs_lock = threading.Lock()

        self._next_action_time = None
        self._action_time_lock = threading.Lock()

    def update_observation(self):
        """Actor thread: fetch new obs from env and store it."""
        obs = self.env.get_obs()
        with self._obs_lock:
            self._latest_obs = obs
        return obs

    def get_observation(self):
        """RTC thread: read latest obs set by actor thread."""
        with self._obs_lock:
            return self._latest_obs

    def send_action(self, action_world_1d: np.ndarray, compensate_latency: bool = True):
        """
        Actor thread: send a single step to env.exec_actions as (1, D)
        with a single timestamp. No explicit chunk/timestamp logic exposed.
        """
        assert action_world_1d.ndim == 1
        now = time.time()

        with self._action_time_lock:
            if self._next_action_time is None:
                self._next_action_time = now
            # ensure we don't drift backwards in time
            self._next_action_time = max(self._next_action_time + self.dt, now + 0.1 * self.dt)
            t = self._next_action_time

        actions = action_world_1d[None, :]                  # (1, D)
        timestamps = np.array([t], dtype=np.float64)        # (1,)

        self.env.exec_actions(
            actions=actions,
            timestamps=timestamps,
            compensate_latency=compensate_latency
        )

    def reset_action_time(self, start_time: float):
        with self._action_time_lock:
            self._next_action_time = start_time


# =========================
# Simple world-space ActionQueue
# =========================

class UmiActionQueue:
    """
    Minimal LeRobot-like ActionQueue:

    - Stores world-space actions (each entry is a 1D np.array)
    - Actor thread calls get() to pop next action
    - RTC thread calls replace_future_with() to overwrite the future plan
    """

    def __init__(self, max_size: Optional[int] = None):
        self._queue = deque()
        self._lock = threading.Lock()
        self._max_size = max_size
        self._last_index = 0  # for debugging / stats

    def get(self):
        with self._lock:
            if not self._queue:
                return None
            self._last_index += 1
            return self._queue.popleft()

    def qsize(self):
        with self._lock:
            return len(self._queue)

    def clear(self):
        with self._lock:
            self._queue.clear()
            self._last_index = 0

    def replace_future_with(self, new_actions: np.ndarray):
        """
        Replace queue with new_actions (np.array of shape [T, D]).
        Equivalent to RTC's behavior where we drop old future and insert the
        new RTC-refined chunk (after real_delay trimming).
        """
        if not isinstance(new_actions, np.ndarray):
            new_actions = np.asarray(new_actions)

        with self._lock:
            self._queue.clear()
            if self._max_size is not None and new_actions.shape[0] > self._max_size:
                new_actions = new_actions[:self._max_size]
            for i in range(new_actions.shape[0]):
                self._queue.append(new_actions[i].copy())
            self._last_index = 0

    def get_action_index(self):
        with self._lock:
            return self._last_index


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
    action_queue: UmiActionQueue,
    latency_tracker: LatencyTracker,
    shutdown_event: threading.Event,
    shape_meta,
    obs_pose_repr,
    action_pose_repr,
    dt: float,
    steps_per_inference: int,
    rtc_schedule: str = "exp",
    rtc_max_guidance: float = 5.0,
    debug: bool = True,
):
    """
    UMI version of LeRobot's get_actions() thread.

    - Uses LatencyTracker to forecast inference_delay in steps
    - Uses predict_action_flow + realtime_action (RTC) to generate new chunks
    - Converts model-space chunk to world-space with get_real_umi_action
    - Drops first real_delay steps (already executed) and overwrites the future queue
    """
    device = next(policy.parameters()).device
    H, D = policy.action_horizon, policy.action_dim
    time_per_step = dt

    # When queue size goes below this, we ask for a new chunk.
    # You can tune this; starting with steps_per_inference is reasonable.
    action_queue_size_to_get_new_actions = steps_per_inference

    prev_raw_chunk = None
    chunk_idx = 0

    if debug:
        print("[GET_ACTIONS_UMI] started")

    # BEGIN: RTC get-actions loop (LeRobot-style)
    try:
        while not shutdown_event.is_set():
            # Throttle chunk generation based on queue size
            if action_queue.qsize() > action_queue_size_to_get_new_actions:
                time.sleep(0.001)
                continue

            env_obs = robot_wrapper.get_observation()
            if env_obs is None:
                # Wait until actor thread has produced at least one observation
                time.sleep(0.001)
                continue

            # Build obs_dict (numpy -> torch)
            obs_dict_np = umi_build_obs_dict(
                env_obs=env_obs,
                shape_meta=shape_meta,
                obs_pose_repr=obs_pose_repr,
            )
            obs_dict = dict_apply(
                obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
            )

            s_min = min(steps_per_inference, H)

            with torch.no_grad():
                is_first_chunk = prev_raw_chunk is None

                # Forecast inference_delay in steps from latency history
                max_latency = latency_tracker.max() or 0.0
                if max_latency > 0.0:
                    d_forecast_raw = int(np.ceil(max_latency / time_per_step))
                else:
                    d_forecast_raw = s_min

                s_horizon = int(min(max(s_min, d_forecast_raw), H))
                d_forecast = int(min(max(0, d_forecast_raw), s_horizon))
                prefix_attn_h = max(0, H - s_horizon)

                infer_start = time.time()

                # First chunk: bootstrap prev_raw_chunk with predict_action_flow
                if is_first_chunk:
                    base_result = policy.predict_action_flow(obs_dict)
                    prev_raw_chunk = base_result['action_pred'][0].detach().cpu().numpy()

                prev_chunk_tensor = torch.from_numpy(prev_raw_chunk).unsqueeze(0).to(device)

                result = policy.realtime_action(
                    obs_dict,
                    prev_action_chunk=prev_chunk_tensor,
                    inference_delay=d_forecast,
                    prefix_attention_horizon=prefix_attn_h,
                    prefix_attention_schedule=rtc_schedule,
                    max_guidance_weight=rtc_max_guidance,
                    n_steps=policy.num_inference_steps
                )

                curr_raw_chunk = result['action_pred'][0].detach().cpu().numpy()

                # Measure real latency and convert to step-based delay
                inference_time = time.time() - infer_start  # seconds
                latency_tracker.add(inference_time)

                real_delay_steps = int(np.ceil(inference_time / time_per_step))
                real_delay_steps = max(0, min(real_delay_steps, s_horizon))
                if is_first_chunk:
                    # First chunk: nothing old is being executed yet
                    real_delay_steps = 0

                # Convert model-space chunk to world-space actions
                curr_action_world = get_real_umi_action(
                    curr_raw_chunk,
                    env_obs,
                    action_pose_repr
                )

                H_world = min(curr_action_world.shape[0], H)
                s_horizon = min(s_horizon, H_world)
                real_delay_steps = min(real_delay_steps, s_horizon)

                # Drop first real_delay_steps actions (executed while policy was thinking)
                this_chunk_world = curr_action_world[real_delay_steps:s_horizon]
                actions_to_queue = this_chunk_world.shape[0]

                if actions_to_queue > 0:
                    action_queue.replace_future_with(this_chunk_world)

                if debug:
                    print(
                        f"[GET_ACTIONS_UMI] chunk {chunk_idx} | "
                        f"lat={inference_time*1000:.1f}ms | "
                        f"d_forecast={d_forecast} | real_delay={real_delay_steps} | "
                        f"s_horizon={s_horizon} | qsize={action_queue.qsize()}"
                    )

                # Shift prev_raw_chunk by s_horizon (committed window)
                if s_horizon < H:
                    shifted = curr_raw_chunk[s_horizon:]
                    pad = np.zeros((s_horizon, D), dtype=curr_raw_chunk.dtype)
                    prev_raw_chunk = np.concatenate([shifted, pad], axis=0)
                else:
                    prev_raw_chunk = np.zeros_like(curr_raw_chunk)

                chunk_idx += 1

            time.sleep(0.0005)

    except Exception as e:
        print(f"[GET_ACTIONS_UMI] Exception: {e}")
    finally:
        if debug:
            print("[GET_ACTIONS_UMI] exiting")
    # END: RTC get-actions loop (LeRobot-style)


# =========================
# Actor thread (LeRobot-style)
# =========================

def actor_control_umi_thread(
    robot_wrapper: UmiRobotWrapper,
    action_queue: UmiActionQueue,
    shutdown_event: threading.Event,
    frequency: float,
    debug: bool = False,
):
    """
    LeRobot-style actor:
    - runs at fixed frequency
    - refreshes observation
    - pops next action from queue
    - sends action to robot
    """
    dt = 1.0 / frequency
    t_start = time.monotonic()
    i = 0
    executed = 0

    last_tick = time.monotonic()

    try:
        while not shutdown_event.is_set():
            loop_start = time.monotonic()
            obs = robot_wrapper.update_observation()
            after_obs = time.monotonic()

            action = action_queue.get()
            if action is not None:
                action = umi_action_processor(action)
                robot_wrapper.send_action(action)
            after_action = time.monotonic()

            if i % 10 == 0:
                print(
                    f"[ACTOR] dt_real={loop_start-last_tick:.3f}s, "
                    f"get_obs={after_obs-loop_start:.3f}s, "
                    f"send_action={after_action-after_obs:.3f}s"
                )
            last_tick = loop_start

            t_next = t_start + (i + 1) * dt
            precise_wait(t_next, time_func=time.monotonic)
            i += 1

    except Exception as e:
        print(f"[ACTOR_UMI] Exception: {e}")
    finally:
        if debug:
            print(f"[ACTOR_UMI] exiting, executed {executed} actions")
    # END: Actor loop (LeRobot-style)


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
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=360, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=20, type=float, help="Control frequency in Hz.")
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
            rtc_max_guidance = 20.0

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
            robot_wrapper = UmiRobotWrapper(env, dt=dt)

            try:
                while True:
                    # New episode
                    policy.reset()

                    # Per-episode queue & latency tracker
                    action_queue = UmiActionQueue(
                        max_size=policy.action_horizon * 2
                    )
                    latency_tracker = LatencyTracker(maxlen=100)
                    shutdown_event = threading.Event()

                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay

                    t_start_mono = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    robot_wrapper.reset_action_time(eval_t_start)

                    # Wait a bit before start to align camera frames
                    frame_latency = 1 / 60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")

                    # Threads
                    actor_thread = threading.Thread(
                        target=actor_control_umi_thread,
                        args=(robot_wrapper, action_queue, shutdown_event, frequency),
                        kwargs={'debug': False},
                        daemon=True
                    )
                    rtc_thread = threading.Thread(
                        target=get_actions_umi_thread,
                        args=(
                            policy,
                            robot_wrapper,
                            action_queue,
                            latency_tracker,
                            shutdown_event,
                            cfg.task.shape_meta,
                            obs_pose_rep,
                            action_pose_repr,
                            dt,
                            steps_per_inference,
                        ),
                        kwargs={
                            'rtc_schedule': rtc_schedule,
                            'rtc_max_guidance': rtc_max_guidance,
                            'debug': True
                        },
                        daemon=True
                    )

                    actor_thread.start()
                    rtc_thread.start()

                    # Main loop: keyboard, visualization, duration
                    t_start = t_start_mono
                    episode_id = env.replay_buffer.n_episodes
                    print(f"Episode {episode_id} running...")

                    while True:
                        # Visualization based on latest obs from actor thread
                        obs_vis = robot_wrapper.get_observation()
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
