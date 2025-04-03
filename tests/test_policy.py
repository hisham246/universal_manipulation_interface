import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

from moviepy.editor import VideoFileClip
import av
import click
import cv2
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.umi_env import UmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)

from umi.real_world.uvc_camera import UvcCamera
from umi.common.usb_util import create_usb_list
from umi.common.precise_sleep import precise_wait

def main():
    # Paths
    ckpt_path = '/home/hisham246/uwaterloo/test_policy.ckpt'
    dev_video_path = '/dev/video0'

    # Load model checkpoint
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.eval().to('cuda')
    policy.num_inference_steps = 16

    # Pose representations
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_rep = cfg.task.pose_repr.action_pose_repr

    # Inference horizon
    obs_horizon = cfg.task.shape_meta.obs.camera0_rgb.horizon
    obs_shape = cfg.task.shape_meta.obs.camera0_rgb.shape[-2:]

    cv2.setNumThreads(1)
    fps = 10
    dt = 1 / fps


    with SharedMemoryManager() as shm_manager:
        with UvcCamera(shm_manager=shm_manager, dev_video_path='/dev/video0') as camera:
            print("Ready to capture frames from camera!")
            buffer = []
            t_start = time.monotonic()
            iter_idx = 0

            while True:
                t_cycle_end = t_start + (iter_idx + 1) * dt

                data = camera.get()
                img = data['color']
                img = cv2.resize(img, tuple(obs_shape[::-1]))
                buffer.append(img)

                if len(buffer) == obs_horizon:
                    frames = np.stack(buffer)
                    buffer.pop(0)

                    obs = {
                        'camera0_rgb': frames.astype(np.uint8),
                        'robot0_eef_pos': np.zeros((obs_horizon, 3), dtype=np.float32),
                        'robot0_eef_rot_axis_angle': np.zeros((obs_horizon, 3), dtype=np.float32),
                        'robot0_eef_pos_wrt_start': np.zeros((obs_horizon, 3), dtype=np.float32),
                        'robot0_eef_rot_axis_angle_wrt_start': np.zeros((obs_horizon, 3), dtype=np.float32),
                        'robot0_gripper_width': np.zeros((obs_horizon, 1), dtype=np.float32),
                        'timestamp': np.ones((obs_horizon,), dtype=np.float64) * time.time()
                    }

                    obs_dict_np = get_real_umi_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta, obs_pose_repr=obs_pose_rep)
                    obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to('cuda'))

                    with torch.no_grad():
                        torch.cuda.reset_peak_memory_stats()
                        policy.reset()
                        result = policy.predict_action(obs_dict)
                        raw_action = result['action_pred'][0].detach().cpu().numpy()
                        final_action = get_real_umi_action(raw_action, obs, action_pose_rep)
                        print("Predicted action:", final_action)
                        print("Peak memory usage (MB):", torch.cuda.max_memory_allocated() / 1e6)

                cv2.imshow('Live Camera Feed', img)
                if cv2.pollKey() & 0xFF == ord('q'):
                    break

                iter_idx += 1
                time.sleep(max(0, t_cycle_end - time.monotonic()))

if __name__ == '__main__':
    main()