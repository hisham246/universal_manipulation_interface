"""
Usage:
(umi): python exec_policy_lipo.py -o output
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import os
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
import cvxpy as cp
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

# LiPo
class ActionLiPo:
    def __init__(self, solver="CLARABEL", 
                 chunk_size=100, 
                 blending_horizon=10, 
                 action_dim=7, 
                 len_time_delay=0,
                 dt=0.0333,
                 epsilon_blending=0.02,
                 epsilon_path=0.003):
        """
        ActionLiPo (Action Lightweight Post-Optimizer) for action optimization.      
        Parameters:
        - solver: The solver to use for the optimization problem.
        - chunk_size: The size of the action chunk to optimize.
        - blending_horizon: The number of actions to blend with past actions.
        - action_dim: The dimension of the action space.
        - len_time_delay: The length of the time delay for the optimization.
        - dt: Time step for the optimization.
        - epsilon_blending: Epsilon value for blending actions.
        - epsilon_path: Epsilon value for path actions.
        """

        self.solver = solver
        self.N = chunk_size
        self.B = blending_horizon
        self.D = action_dim
        self.TD = len_time_delay

        self.dt = dt
        self.epsilon_blending = epsilon_blending
        self.epsilon_path = epsilon_path
        
        JM = 3  # margin for jerk calculation
        self.JM = JM
        self.epsilon = cp.Variable((self.N+JM, self.D)) # previous + 3 to consider previous vel/acc/jrk
        self.ref = cp.Parameter((self.N+JM, self.D),value=np.zeros((self.N+JM, self.D))) # previous + 3
        
        D_j = np.zeros((self.N+JM, self.N+JM))
        for i in range(self.N - 2):
            D_j[i, i]     = -1
            D_j[i, i+1]   = 3
            D_j[i, i+2]   = -3
            D_j[i, i+3]   = 1
        D_j = D_j / self.dt**3

        q_total = self.epsilon + self.ref  # (N, D)
        cost = cp.sum([cp.sum_squares(D_j @ q_total[:, d]) for d in range(self.D)])

        constraints = []

        constraints += [self.epsilon[self.B+JM:] <= self.epsilon_path]
        constraints += [self.epsilon[self.B+JM:] >= - self.epsilon_path]
        constraints += [self.epsilon[JM+1+self.TD:self.B+JM] <= self.epsilon_blending]
        constraints += [self.epsilon[JM+1+self.TD:self.B+JM] >= - self.epsilon_blending]
        constraints += [self.epsilon[0:JM+1+self.TD] == 0.0]

        np.set_printoptions(precision=3, suppress=True, linewidth=100)

        self.p = cp.Problem(cp.Minimize(cost), constraints)

        # Initialize the problem & warm up
        self.p.solve(warm_start=True, verbose=False, solver=self.solver, time_limit=0.05)
        
        self.log = []

    def solve(self, actions: np.ndarray, past_actions: np.ndarray, len_past_actions: int):
        """
        Solve the optimization problem with the given actions and past actions.
        Parameters:
        - actions: The current actions to optimize.
        - past_actions: The past actions to blend with.
        - len_past_actions: The number of past actions to consider for blending.
        Returns:
        - solved: The optimized actions after solving the problem.
        - ref: The reference actions used in the optimization.
        """

        blend_len = len_past_actions
        JM = self.JM
        self.ref.value[JM:] = actions.copy()
        
        if blend_len > 0:
            # update last actions
            self.ref.value[:JM+self.TD] = past_actions[-blend_len-JM:-blend_len + self.TD].copy()
            ratio_space = np.linspace(0, 1, blend_len-self.TD) # (B,1)    
            self.ref.value[JM+self.TD:blend_len+JM] = ratio_space[:, None] * actions[self.TD:blend_len] + (1 - ratio_space[:, None]) * past_actions[-blend_len+self.TD:]
        else: # blend_len == 0
            # update last actions
            self.ref.value[:JM] = actions[0]
            
        t0 = time.time()
        try:
            self.p.solve(warm_start=True, verbose=False, solver=self.solver, time_limit=0.05)
        except Exception as e:
            return None, e
        t1 = time.time()

        solved_time = t1 - t0
        self.solved = self.epsilon.value.copy() + self.ref.value.copy()

        self.log.append({
            "time": solved_time,
            "epsilon": self.epsilon.value.copy(),
            "ref": self.ref.value.copy(),
            "solved": self.solved.copy()
        })

        return self.solved[JM:].copy(), self.ref.value[JM:].copy()

    def get_log(self):
        return self.log

    def reset_log(self):
        self.log = []

    def print_solved_times(self):
        if self.log:
            avg_time = np.mean([entry["time"] for entry in self.log])
            std_time = np.std([entry["time"] for entry in self.log])
            num_logs = len(self.log)
            print(f"Number of logs: {num_logs}")
            print(f"Average solved time: {avg_time:.4f} seconds, Std: {std_time:.4f} seconds")
        else:
            print("No logs available.")

@click.command()
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', default='129.97.71.27')
@click.option('--gripper_ip', default='129.97.71.27')
@click.option('--gripper_port', type=int, default=4242)
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--steps_per_inference', '-si', default= 8, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_crop', is_flag=True, default=False)
@click.option('--mirror_swap', is_flag=True, default=False)

def main(output, robot_ip, gripper_ip, gripper_port,
    match_dataset, match_camera, vis_camera_idx, 
    steps_per_inference, max_duration,
    frequency, no_mirror, sim_fov, 
    camera_intrinsics, mirror_crop, mirror_swap):

    # LiPo Configuration
    # Adjust these parameters based on your needs
    lipo_chunk_size = 16  # Larger than steps_per_inference to allow overlap
    lipo_blending_horizon = 8  # How many steps to blend between chunks
    lipo_time_delay = 0  # Account for inference latency (in timesteps)
    
    ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/reaching_ball_multimodal.ckpt'

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
    
    # Initialize LiPo
    lipo = ActionLiPo(
        solver="CLARABEL",
        chunk_size=lipo_chunk_size,
        blending_horizon=lipo_blending_horizon,
        action_dim=7,  # Action dimension
        len_time_delay=lipo_time_delay,
        dt=dt,
        epsilon_blending=0.02,  # Allow more deviation in blending zone
        epsilon_path=0.003     # Tighter constraint for main trajectory
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
                # camera_obs_latency=0.0,
                # robot_obs_latency=0.0,
                # gripper_obs_latency=0.0,
                # robot_action_latency=0.0,
                # gripper_action_latency=0.0,
                camera_obs_latency=0.145,
                robot_obs_latency=0.0001,
                gripper_obs_latency=0.01,
                robot_action_latency=0.2,
                gripper_action_latency=0.1,
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_crop=mirror_crop,
                mirror_swap=mirror_swap,
                max_pos_speed=2.5,
                max_rot_speed=1.5,
                shm_manager=shm_manager) as env:
            
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # [Match dataset loading code remains the same]
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

            obs = env.get_obs()
            print("Got observations")
            episode_start_pose = np.concatenate([
                    obs[f'robot0_eef_pos'],
                    obs[f'robot0_eef_rot_axis_angle']
                ], axis=-1)[-1]
            

            # Initialize with first action chunk for LiPo
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
                assert action.shape[-1] == 10
                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 7
                del result

            print("Waiting to get to the stop button...")
            time.sleep(3.0)
            print('Ready!')

            while True:                
                try:
                    # start episode
                    policy.reset()
                    lipo.reset_log()  # Reset LiPo logs
                    
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")

                    iter_idx = 0
                    action_log = []
                    
                    # LiPo state tracking
                    prev_optimized_chunk = None
                    action_chunk_counter = 0
                    
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
                            action_chunk = get_real_umi_action(raw_action, obs, action_pose_repr)
                            
                            # Extend action chunk to match LiPo chunk size
                            if len(action_chunk) < lipo_chunk_size:
                                # Repeat last action to fill the chunk
                                last_action = action_chunk[-1:].repeat(lipo_chunk_size - len(action_chunk), axis=0)
                                extended_action_chunk = np.concatenate([action_chunk, last_action], axis=0)
                            else:
                                extended_action_chunk = action_chunk[:lipo_chunk_size]
                            
                            # Apply LiPo smoothing
                            if prev_optimized_chunk is not None:
                                smoothed_actions, reference_actions = lipo.solve(
                                    extended_action_chunk, 
                                    prev_optimized_chunk, 
                                    len_past_actions=lipo_blending_horizon
                                )
                            else:
                                smoothed_actions, reference_actions = lipo.solve(
                                    extended_action_chunk, 
                                    None, 
                                    len_past_actions=0
                                )
                            
                            if smoothed_actions is not None:
                                # Store the optimized chunk for next iteration
                                prev_optimized_chunk = smoothed_actions.copy()
                                # Use the first steps_per_inference actions from smoothed chunk
                                action = smoothed_actions[:steps_per_inference]
                            else:
                                # Fallback to original actions if LiPo fails
                                action = action_chunk[:steps_per_inference]
                            
                            action_chunk_counter += 1
                            
                            action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                            this_target_poses = action

                            action_exec_latency = 0.01
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)

                            if np.sum(is_new) == 0:
                                this_target_poses = this_target_poses[[-1]]
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + (next_step_idx) * dt
                                print('Over budget', action_timestamp - curr_time)
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]

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

                            # execute actions
                            env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps,
                                compensate_latency=True
                            )
                            print(f"Submitted {len(this_target_poses)} steps of smoothed actions.")

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
                    # Print LiPo performance stats
                    lipo.print_solved_times()
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