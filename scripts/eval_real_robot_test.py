# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import numpy as np
import cv2
import hydra
import torch
import dill
import time
import json
import pathlib
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.real_world.uvc_camera import UvcCamera
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.franka_hand_controller import FrankaHandController
from umi.real_world.keystroke_counter import KeystrokeCounter, KeyCode, Key
from umi.common.usb_util import create_usb_list
from umi.common.precise_sleep import precise_wait
from umi.common.cv_util import get_image_transform
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from umi.common.interpolation_util import get_interp1d, PoseInterpolator
from umi.common.pose_util import mat_to_pose, pose10d_to_mat, mat_to_pose, mat_to_rot6d

# %%
def get_obs_dict(camera, controller, gripper, n_obs_steps, img_tf):
    camera_data = camera.get(k=n_obs_steps)
    robot_data = controller.get_all_state()
    gripper_data = gripper.get_state()

    raw_imgs = camera_data['color']
    img_timestamps = camera_data['camera_capture_timestamp']
    current_time = time.time()
    
    # Estimate latency for the most recent frame
    cam_latency = current_time - img_timestamps[-1]
    print(f"[Camera Latency] Latest frame delay: {cam_latency:.4f} sec")
    imgs = list()
    for x in raw_imgs:
        # bgr to rgb
        imgs.append(img_tf(x)[...,::-1])
    imgs = np.array(imgs)
    # T,H,W,C to T,C,H,W
    imgs = np.moveaxis(imgs,-1,1).astype(np.float32) / 255.

    # gripper_interp = get_interp1d(
    #     t=gripper_data['gripper_receive_timestamp'],
    #     x=gripper_data['gripper_position'] / 1000) # mm to meters

    # gripper_interp = get_interp1d(
    #     t=gripper_data['timestamp']['seconds'],
    #     x=gripper_data['width']) # meters
    robot_interp = PoseInterpolator(
        t=robot_data['robot_receive_timestamp'],
        x=robot_data['ActualTCPPose'])

    robot_eef_pose = robot_interp(img_timestamps).astype(np.float32)
    # gripper_width = gripper_interp(img_timestamps).astype(np.float32)
    gripper_width = np.float32(gripper_data['width'])

    obs_dict = {
        'img': imgs,
        'robot_eef_pose': robot_eef_pose,
        'gripper_width': gripper_width
    }
    return obs_dict, img_timestamps

    
# %%
@click.command()
# @click.option('-i', '--input', required=True)
@click.option('-rh', '--robot_hostname', default='129.97.71.27')
@click.option('-gh', '--gripper_hostname', default='129.97.71.27')
@click.option('-gp', '--gripper_port', type=int, default=4242)
@click.option('-gs', '--gripper_speed', type=float, default=0.05)
@click.option('-gf', '--gripper_force', type=float, default=20.0)
@click.option('-f', '--frequency', type=float, default=30)
@click.option('-v', '--video_path', default='/dev/video0')
@click.option('-s', '--steps_per_inference', type=int, default=16)

def main(robot_hostname, 
         gripper_hostname, 
         gripper_port, 
         gripper_speed, 
         gripper_force, 
         frequency, 
         video_path, 
         steps_per_inference):
    cv2.setNumThreads(1)

    # load checkpoint
    ckpt_path = '/home/hisham246/uwaterloo/pickplace.ckpt'
    
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    print("Workspace", workspace)


    # diffusion model
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda')
    policy.eval().to(device)

    # set inference params
    policy.num_inference_steps = 16
    policy.n_action_steps = 8
    n_obs_steps = 2

    # setup env
    max_pos_speed = 0.25
    max_rot_speed = 0.6
    cube_diag = np.linalg.norm([1,1,1])

    # # load tcp offset
    # tx_cam_gripper = np.array(
    #     json.load(open('data/calibration/robot_world_hand_eye.json', 'r')
    #               )['tx_gripper2camera'])
    # tx_gripper_cam = np.linalg.inv(tx_cam_gripper)
    # tcp_offset_pose = mat_to_pose(tx_gripper_cam)

    # tcp_offset = 0
    dt = 1/frequency
    command_latency = dt / 2

    dev_video_path = video_path
    # enumerate UBS device to find Elgato Capture Card
    device_list = create_usb_list()
    dev_usb_path = None
    for dev in device_list:
        if 'AVerMedia' in dev['description']:
            dev_usb_path = dev['path']
            print('Found :', dev['description'])
            break
    
    capture_res = (1280, 720)
    out_res = (224, 224)
    img_tf = get_image_transform(in_res=capture_res, 
        out_res=out_res, crop_ratio=0.65)

    with SharedMemoryManager() as shm_manager:
        # with KeystrokeCounter() as key_counter,\
        # with FrankaInterpolationController(
        #     shm_manager=shm_manager,
        #     robot_ip=robot_hostname,
        #     frequency=200,
        #     Kx_scale=1.0,
        #     Kxd_scale=np.array([2.0,1.5,2.0,1.0,1.0,1.0]),
        #     verbose=False) as controller,\
        # Spacemouse(shm_manager=shm_manager) as sm,\
        # UvcCamera(shm_manager=shm_manager,
        #           dev_video_path=dev_video_path,
        #           resolution=capture_res) as camera,\
        # FrankaHandController(
        #     host=gripper_hostname,
        #     port=gripper_port,
        #     speed=gripper_speed,
        #     force=gripper_force,
        #     update_rate=frequency) as gripper:
        gripper = FrankaHandController(
            host=gripper_hostname,
            port=gripper_port,
            speed=gripper_speed,
            force=gripper_force,
            update_rate=frequency
        )
        gripper.start()
        with KeystrokeCounter() as key_counter, \
            FrankaInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_hostname,
                frequency=100,
                Kx_scale=5.0,
                Kxd_scale=2.0,
                verbose=False
            ) as controller, \
            Spacemouse(
                shm_manager=shm_manager
            ) as sm, \
            UvcCamera(shm_manager=shm_manager,
                      dev_video_path=dev_video_path,
                      resolution=capture_res) as camera:
            
            time.sleep(1.0)
            # policy warmup
            obs_dict_np, obs_timestamps = get_obs_dict(
                camera=camera, 
                controller=controller, 
                gripper=gripper, 
                n_obs_steps=n_obs_steps,
                img_tf=img_tf)
            with torch.no_grad():
                # def process_obs(x):
                #     if isinstance(x, np.ndarray):
                #         x = torch.from_numpy(x).to(device)
                #     else:
                #         x = torch.tensor(x, dtype=torch.float32).to(device)

                #     # Handle image: shape [T, C, H, W] → [1, T, C, H, W]
                #     if x.ndim == 4 and x.shape[1] == 3:
                #         x = x.unsqueeze(0)
                #     # Low-dim: [T] → [1, T, 1], [T, D] → [1, T, D]
                #     elif x.ndim == 1:
                #         x = x.unsqueeze(0).unsqueeze(-1)
                #     elif x.ndim == 2:
                #         x = x.unsqueeze(0)
                #     elif x.ndim == 3:
                #         pass  # already has [B, T, D]
                #     else:
                #         raise ValueError(f"Unexpected shape: {x.shape}")
                #     return x

                # obs_dict = dict_apply(obs_dict_np, process_obs)
                obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device) if isinstance(x, np.ndarray) else torch.tensor(x).unsqueeze(0).to(device))
                # obs_dict = dict_apply(obs_dict_np, 
                #     lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                # print('Obs Dict:', obs_dict)
                # for key in obs_dict:
                #     print(key, obs_dict[key].shape)

                # # Rename keys to match normalizer
                # if 'img' in obs_dict:
                #     obs_dict['camera0_rgb'] = obs_dict.pop('img')
                # if 'robot_eef_pose' in obs_dict:
                #     obs_dict['robot0_eef_pos'] = obs_dict['robot_eef_pose'][:, :, :3]
                #     obs_dict['robot0_eef_rot_axis_angle'] = obs_dict['robot_eef_pose'][:, :, 3:]
                #     obs_dict.pop('robot_eef_pose')
                # if 'gripper_width' in obs_dict:
                #     obs_dict['robot0_gripper_width'] = obs_dict.pop('gripper_width')

                # Rename and reshape keys to match the expected input format of the policy
                if 'img' in obs_dict:
                    obs_dict['camera0_rgb'] = obs_dict.pop('img')

                if 'robot_eef_pose' in obs_dict:
                    # Expected shape: (B, T, 3) for position and (B, T, 6) for rotation (after conversion)
                    obs_dict['robot0_eef_pos'] = obs_dict['robot_eef_pose'][:, :, :3]

                    # Convert rotation from axis-angle to 6D
                    rot_axis_angle = obs_dict['robot_eef_pose'][:, :, 3:].cpu().numpy()
                    batch, T, _ = rot_axis_angle.shape
                    rotmat = st.Rotation.from_rotvec(rot_axis_angle.reshape(-1, 3)).as_matrix()
                    rot6d = mat_to_rot6d(rotmat.reshape(batch, T, 3, 3))
                    rot6d_tensor = torch.from_numpy(rot6d).to(device).float()

                    obs_dict['robot0_eef_rot_axis_angle'] = rot6d_tensor
                    obs_dict['robot0_eef_rot_axis_angle_wrt_start'] = rot6d_tensor.clone()
                    obs_dict.pop('robot_eef_pose')
                if 'gripper_width' in obs_dict:
                    gripper_1 = obs_dict.pop('gripper_width')  # shape: [1]
                    if gripper_1.ndim == 1:
                        # reshape to (B, T, D) = (1, 2, 1)
                        gripper_1 = gripper_1.reshape(1, -1, 1).repeat(1, 2, 1)
                    elif gripper_1.ndim == 2:
                        # reshape to (B, T, D)
                        gripper_1 = gripper_1.unsqueeze(-1)
                    obs_dict['robot0_gripper_width'] = gripper_1

                # print("Expected shape:", policy.obs_encoder.key_shape_map['camera0_rgb'])
                # print("Actual shape:", obs_dict['camera0_rgb'].shape)
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")

                gripper_speed = 200.
                state = controller.get_state()
                print("State:", state)
                target_pose = state['ActualTCPPose']
                gripper_pos = gripper.get_state()
                gripper_target_pos = gripper_pos['width']
                t_start = time.monotonic()
                # gripper.restart_put(t_start-time.monotonic() + time.time())

                iter_idx = 0
                while True:
                    # s = time.time()
                    # t_cycle_end = t_start + (iter_idx + 1) * dt
                    # t_sample = t_cycle_end - command_latency
                    # t_command_target = t_cycle_end + dt
                    
                    # data = camera.get()
                    # img = data['color']

                    # # Display the resulting frame
                    # cv2.imshow('frame', img)
                    # if cv2.pollKey() & 0xFF == ord('q'):
                    #     break

                    # precise_wait(t_cycle_end)
                    # iter_idx += 1

                    # precise_wait(t_sample)
                    # sm_state = sm.get_motion_state_transformed()
                    # # print(sm_state)
                    # dpos = sm_state[:3] * (0.25 / frequency)
                    # drot_xyz = sm_state[3:] * (0.6 / frequency)

                    # drot = st.Rotation.from_euler('xyz', drot_xyz)
                    # target_pose[:3] += dpos
                    # target_pose[3:] = (drot * st.Rotation.from_rotvec(
                    #     target_pose[3:])).as_rotvec()

                    # dpos = 0
                    # if sm.is_button_pressed(0):
                    #     # close gripper
                    #     dpos = -gripper_speed / frequency
                    # if sm.is_button_pressed(1):
                    #     dpos = gripper_speed / frequency
                    # gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, 90.)

                    # if t_command_target > time.monotonic():
                    #     # skip outdated command
                    #     controller.schedule_waypoint(target_pose, 
                    #         t_command_target-time.monotonic()+time.time())
                    #     # gripper.schedule_waypoint(gripper_target_pos, 
                    #     #     t_command_target-time.monotonic()+time.time())

                    # precise_wait(t_cycle_end)
                    # iter_idx += 1

                # ========== policy control loop ==============
                    print("Robot in control!")
                    try:
                        policy.reset()
                        start_delay = 1.0
                        eval_t_start = time.time() + start_delay
                        t_start = time.monotonic() + start_delay
                        # wait for 1/30 sec to get the closest frame actually
                        # reduces overall latency
                        frame_latency = 1/59
                        precise_wait(eval_t_start - frame_latency, time_func=time.time)
                        print("Started!")
                        iter_idx = 0
                        while True:
                            # calculate timing
                            t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                            # get obs
                            obs_dict_np, obs_timestamps = get_obs_dict(
                                camera=camera, 
                                controller=controller, 
                                gripper=gripper, 
                                n_obs_steps=n_obs_steps,
                                img_tf=img_tf)
                            with torch.no_grad():
                                s = time.time()
                                # obs_dict = dict_apply(obs_dict_np, 
                                #     lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device) if isinstance(x, np.ndarray) else torch.tensor(x).unsqueeze(0).to(device))
                                
                                                # Rename and reshape keys to match the expected input format of the policy
                                if 'img' in obs_dict:
                                    obs_dict['camera0_rgb'] = obs_dict.pop('img')

                                if 'robot_eef_pose' in obs_dict:
                                    # Expected shape: (B, T, 3) for position and (B, T, 6) for rotation (after conversion)
                                    obs_dict['robot0_eef_pos'] = obs_dict['robot_eef_pose'][:, :, :3]

                                    # Convert rotation from axis-angle to 6D
                                    rot_axis_angle = obs_dict['robot_eef_pose'][:, :, 3:].cpu().numpy()
                                    batch, T, _ = rot_axis_angle.shape
                                    rotmat = st.Rotation.from_rotvec(rot_axis_angle.reshape(-1, 3)).as_matrix()
                                    rot6d = mat_to_rot6d(rotmat.reshape(batch, T, 3, 3))
                                    rot6d_tensor = torch.from_numpy(rot6d).to(device).float()

                                    obs_dict['robot0_eef_rot_axis_angle'] = rot6d_tensor
                                    obs_dict['robot0_eef_rot_axis_angle_wrt_start'] = rot6d_tensor.clone()
                                    obs_dict.pop('robot_eef_pose')
                                if 'gripper_width' in obs_dict:
                                    gripper_1 = obs_dict.pop('gripper_width')  # shape: [1]
                                    if gripper_1.ndim == 1:
                                        # reshape to (B, T, D) = (1, 2, 1)
                                        gripper_1 = gripper_1.reshape(1, -1, 1).repeat(1, 2, 1)
                                    elif gripper_1.ndim == 2:
                                        # reshape to (B, T, D)
                                        gripper_1 = gripper_1.unsqueeze(-1)
                                    obs_dict['robot0_gripper_width'] = gripper_1

                                # print("Expected shape:", policy.obs_encoder.key_shape_map['camera0_rgb'])
                                # print("Actual shape:", obs_dict['camera0_rgb'].shape)
                                result = policy.predict_action(obs_dict)
                                action = result['action'][0].detach().to('cpu').numpy()
                                result = policy.predict_action(obs_dict)
                                action = result['action'][0].detach().to('cpu').numpy()
                                print('Inference latency:', time.time() - s)
                            
                            # action conversion
                            action_pose10d = action[:,:9]
                            action_grip = action[:,9:]
                            action_pose = mat_to_pose(pose10d_to_mat(action_pose10d))
                            action = np.concatenate([action_pose, action_grip], axis=-1)

                            # deal with timing
                            # the same step actions are always the target for
                            action_timestamps = (np.arange(len(action), dtype=np.float64)
                                ) * dt + obs_timestamps[-1]
                            action_exec_latency = 0.01
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            if np.sum(is_new) == 0:
                                # exceeded time budget, still do something
                                action = action[[-1]]
                                # schedule on next available step
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + (next_step_idx) * dt
                                print('Over budget', action_timestamp - curr_time)
                                action_timestamps = np.array([action_timestamp])
                            else:
                                action = action[is_new]
                                action_timestamps = action_timestamps[is_new]

                            # execute actions
                            robot_action = action[:,:6]
                            gripper_action = action[:,-1]

                            print('Action:', robot_action, gripper_action)

                            for i in range(len(action_timestamps)):
                                controller.schedule_waypoint(robot_action[i], action_timestamps[i])
                                gripper.send_target(gripper_action[i])

                            # visualize
                            # vis_img = camera.get()['color']
                            # cv2.imshow('main camera', vis_img)

                            key_stroke = cv2.pollKey()
                            if key_stroke == ord('s'):
                                print('Stopped.')
                                break

                            # wait for execution
                            precise_wait(t_cycle_end - frame_latency)
                            iter_idx += steps_per_inference

                    except KeyboardInterrupt:
                        print("Interrupted!")
                        # stop robot.

                    print("Stopped.")

# %%
if __name__ == "__main__":
    main()
