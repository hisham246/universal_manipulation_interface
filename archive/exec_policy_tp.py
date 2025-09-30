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

import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

from scipy.spatial.transform import Rotation as R
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

# quat utilities
def q_norm(q): return q / (np.linalg.norm(q) + 1e-12)
def q_conj(q): return np.array([-q[0], -q[1], -q[2], q[3]])
def q_mul(a,b):
    x1,y1,z1,w1 = a; x2,y2,z2,w2 = b
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])
def slerp(q0,q1,t):
    q0=q_norm(q0); q1=q_norm(q1)
    if np.dot(q0,q1) < 0: q1 = -q1
    d = np.clip(np.dot(q0,q1), -1.0, 1.0)
    if d > 0.9995:
        out = q_norm(q0 + t*(q1-q0))
    else:
        th = np.arccos(d)
        out = (np.sin((1-t)*th)*q0 + np.sin(t*th)*q1)/np.sin(th)
    return q_norm(out)
def geo_angle(q0,q1):
    d = np.clip(abs(np.dot(q_norm(q0), q_norm(q1))), -1.0, 1.0)
    return 2.0*np.arccos(d)  # [0, pi]

# geodesic mean around a reference quaternion
def weighted_mean_quats_around(q_ref, quats, weights):
    # map each quat to tangent at q_ref, average, exp back
    vecs = []
    for q in quats:
        if np.dot(q, q_ref) < 0: q = -q
        q_err = q_mul(q_conj(q_ref), q)
        rotvec = R.from_quat(q_err).as_rotvec()
        vecs.append(rotvec)
    vbar = np.average(np.stack(vecs, 0), axis=0, weights=weights)
    q_delta = R.from_rotvec(vbar).as_quat()
    return q_mul(q_ref, q_delta)

# SE(3) step limiter
def limit_se3_step(p_prev, q_prev, p_cmd, q_cmd, v_max, w_max, dt):
    # translation
    dp = p_cmd - p_prev
    n = np.linalg.norm(dp)
    max_dp = v_max * dt
    if n > max_dp:
        dp *= (max_dp / (n + 1e-12))
    p_new = p_prev + dp
    # rotation
    ang = geo_angle(q_prev, q_cmd)
    max_dang = w_max * dt
    if ang > max_dang + 1e-9:
        t = max_dang / ang
        q_new = slerp(q_prev, q_cmd, t)
    else:
        # keep hemisphere continuity
        q_new = q_cmd if np.dot(q_prev, q_cmd) >= 0 else -q_cmd
    return p_new, q_norm(q_new)


OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', default='129.97.71.27')
@click.option('--gripper_ip', default='129.97.71.27')
@click.option('--gripper_port', type=int, default=4242)
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--steps_per_inference', '-si', default= 1, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_crop', is_flag=True, default=False)
@click.option('--mirror_swap', is_flag=True, default=False)
@click.option('--temporal_ensembling', is_flag=True, default=True, help='Enable temporal ensembling for inference.')


def main(output, robot_ip, gripper_ip, gripper_port,
    match_dataset, match_camera,
    vis_camera_idx, 
    steps_per_inference, max_duration,
    frequency, 
    no_mirror, sim_fov, camera_intrinsics, 
    mirror_crop, mirror_swap, temporal_ensembling):

    ENSEMBLE_MAX_CANDS = 6

    # Diffusion Transformer
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/diffusion_transformer_pickplace.ckpt'

    # Diffusion UNet
    # ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/diffusion_unet_pickplace_2.ckpt'
    ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/surface_wiping_unet_position_control.ckpt'

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
                # latency
                # camera_obs_latency=0.145,
                # robot_obs_latency=0.0001,
                # gripper_obs_latency=0.01,
                # robot_action_latency=0.2,
                # gripper_action_latency=0.1,
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
                max_rot_speed=2.0,
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

            print("Warming up policy inference")
            obs = env.get_obs()            
            episode_start_pose = np.concatenate([
                    obs[f'robot0_eef_pos'],
                    obs[f'robot0_eef_rot_axis_angle']
                ], axis=-1)[-1]
            
            # SE(3) state & limits
            v_max = 0.75   # m/s
            w_max = 0.9    # rad/s

            p_last = obs['robot0_eef_pos'][-1].copy()
            q_last = R.from_rotvec(obs['robot0_eef_rot_axis_angle'][-1].copy()).as_quat()
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                # assert action.shape[-1] == 16
                assert action.shape[-1] == 10
                action = get_real_umi_action(action, obs, action_pose_repr)
                # assert action.shape[-1] == 10
                assert action.shape[-1] == 7
                del result

            print('Ready!')

            while True:
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    if temporal_ensembling:
                        max_steps = int(max_duration * frequency) + steps_per_inference
                        temporal_action_buffer = [None] * max_steps

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    action_log = []
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        episode_start_pose = np.concatenate([
                            obs[f'robot0_eef_pos'],
                            obs[f'robot0_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        obs_timestamps = obs['timestamp']
                        # print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                episode_start_pose=episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)

                            g_now = float(action[-1, 6]) if action.ndim == 2 else float(action[6])

                            if temporal_ensembling:
                                # Scatter predictions into buffers
                                made_idx = iter_idx
                                for j, a in enumerate(action):
                                    t_abs = iter_idx + j
                                    if 0 <= t_abs < len(temporal_action_buffer):
                                        if temporal_action_buffer[t_abs] is None:
                                            temporal_action_buffer[t_abs] = []
                                        # store (action, made_idx)
                                        temporal_action_buffer[t_abs].append((a, made_idx))

                            # Execute sequence of smoothed actions (one step per loop when ensembling)
                            this_target_poses = []
                            # we'll fill action_timestamps after we know how many poses we kept

                            # Estimate inference latency to align schedule to the sensor clock
                            inference_latency = time.time() - s
                            execution_buffer = 0.10
                            schedule_offset = inference_latency + execution_buffer

                            # We normally execute only 1 step per loop when ensembling
                            for i in range(steps_per_inference):      # usually 1
                                t_target = iter_idx + i
                                if 0 <= t_target < len(temporal_action_buffer) and temporal_action_buffer[t_target]:
                                    cached_pairs = temporal_action_buffer[t_target]  # list of (action, made_idx)

                                    # Keep only the newest ENSEMBLE_MAX_CANDS by made_idx (recency)
                                    # cached_pairs may already be roughly ordered, but we sort to be safe.
                                    cached_pairs = sorted(cached_pairs, key=lambda x: x[1])[-ENSEMBLE_MAX_CANDS:]

                                    acts = [p[0] for p in cached_pairs]          # actions as arrays [x y z rx ry rz g]
                                    made = np.array([p[1] for p in cached_pairs])  # their generation iter_idx

                                    # Ages in ticks relative to the newest one: newest age = 0
                                    ages = (made.max() - made).astype(np.float64)

                                    # Exponential weights by age (newest gets the largest weight)
                                    m = 0.23
                                    w = np.exp(-m * ages)
                                    w = w / w.sum()

                                    # Positions: weighted mean
                                    Ps = np.stack([a[:3] for a in acts], axis=0)
                                    p_cmd = (Ps * w[:, None]).sum(axis=0)

                                    # Rotations: geodesic mean around q_last
                                    quats = [R.from_rotvec(a[3:6]).as_quat() for a in acts]
                                    q_cmd = weighted_mean_quats_around(q_last, quats, w)

                                elif i < len(action):
                                    # Fallback if no cache yet for this t_target
                                    p_cmd = action[i][:3]
                                    q_cmd = R.from_rotvec(action[i][3:6]).as_quat()
                                else:
                                    break

                                # SE(3) rate limit around last command
                                p_safe, q_safe = limit_se3_step(p_last, q_last, p_cmd, q_cmd, v_max, w_max, dt)
                                p_last, q_last = p_safe, q_safe

                                a_exec = np.zeros_like(action[0])
                                a_exec[:3]  = p_safe
                                a_exec[3:6] = R.from_quat(q_safe).as_rotvec()
                                a_exec[6]   = g_now   # keep gripper smooth across blends

                                this_target_poses.append(a_exec)

                            # Schedule from sensor clock and drop late actions
                            if len(this_target_poses) > 0:
                                obs_ts = float(obs_timestamps[-1])
                                action_timestamps = obs_ts + schedule_offset + dt * np.arange(len(this_target_poses), dtype=np.float64)

                                # late-action filter (keep only actions sufficiently in the future)
                                action_exec_latency = 0.01
                                curr_time = time.time()
                                is_new = action_timestamps > (curr_time + action_exec_latency)
                                print("Is new:", is_new)

                                if not np.any(is_new):
                                    # exceeded time budget, still execute *something* (last pose) at next grid time
                                    this_target_poses = np.asarray(this_target_poses)[[-1]]
                                    next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                    action_timestamps = np.array([eval_t_start + next_step_idx * dt], dtype=np.float64)
                                else:
                                    this_target_poses = np.asarray(this_target_poses)[is_new]
                                    action_timestamps = action_timestamps[is_new]
                            else:
                                # empty safety fallback (should rarely happen)
                                this_target_poses = np.asarray([])
                                action_timestamps = np.asarray([])

                            # print('Inference latency:', time.time() - s)
                            for a, t in zip(this_target_poses, action_timestamps):
                                a = a.tolist()
                                action_log.append({
                                    'timestamp': t,
                                    'ee_pos_0': a[0],
                                    'ee_pos_1': a[1],
                                    'ee_pos_2': a[2],
                                    'ee_rot_0': a[3],
                                    'ee_rot_1': a[4],
                                    'ee_rot_2': a[5]
                                })

                            # print("Action:", this_target_poses)

                            # execute one step
                            env.exec_actions(
                                actions=np.stack(this_target_poses),
                                timestamps=np.array(action_timestamps),
                                compensate_latency=True
                            )

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