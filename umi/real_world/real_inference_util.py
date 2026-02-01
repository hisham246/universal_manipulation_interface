from typing import Dict, Callable, Tuple, List
import numpy as np
import collections
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pose_repr_util import (
    compute_relative_pose, 
    convert_pose_mat_rep
)
from umi.common.pose_util import (
    pose_to_mat, mat_to_pose, 
    mat_to_pose10d, pose10d_to_mat)
from diffusion_policy.model.common.rotation_transformer import \
    RotationTransformer


USE_CONST_STIFFNESS = True
CONST_KX_TRANS = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)


def make_T(R: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T

def change_of_basis(T_fix: np.ndarray, T_in: np.ndarray) -> np.ndarray:
    """
    Conjugation: T_out = T_fix @ T_in @ inv(T_fix)
    Works with T_in shaped (...,4,4) thanks to numpy broadcasting rules
    if T_fix is (4,4) and T_in is (...,4,4).
    """
    T_fix_inv = np.linalg.inv(T_fix)
    return T_fix @ T_in @ T_fix_inv

# 180 deg about +z (CCW)
R_Z_180 = np.array([
    [-1.0,  0.0, 0.0],
    [ 0.0, -1.0, 0.0],
    [ 0.0,  0.0, 1.0],
], dtype=np.float64)

T_Z_180 = make_T(R_Z_180)
T_Z_180_INV = np.linalg.inv(T_Z_180)

def real_pose_mat_to_model_pose_mat(T_real: np.ndarray) -> np.ndarray:
    # model frame is rotated version; map real -> model
    return change_of_basis(T_Z_180_INV, T_real)

def model_pose_mat_to_real_pose_mat(T_model: np.ndarray) -> np.ndarray:
    # map model -> real
    return change_of_basis(T_Z_180, T_model)


def chol_to_stiffness(chol_vec):
    chol_vec = np.asarray(chol_vec)
    if chol_vec.ndim == 1:
        chol_vec = chol_vec[None, :]  # shape (1, 6)

    stiffness_list = []
    for vec in chol_vec:
        U = np.zeros((3, 3))
        U[0, 0] = vec[0]
        U[0, 1] = vec[1]
        U[1, 1] = vec[2]
        U[0, 2] = vec[3]
        U[1, 2] = vec[4]
        U[2, 2] = vec[5]

        Kx_matrix = U.T @ U
        stiffness_vector = np.diag(Kx_matrix)
        stiffness_list.append(stiffness_vector)

    return np.array(stiffness_list)

def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def get_real_umi_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        obs_pose_repr: str='abs',
        tx_robot1_robot0: np.ndarray=None,
        episode_start_pose: List[np.ndarray]=None,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    # process non-pose
    obs_shape_meta = shape_meta['obs']
    robot_prefix_map = collections.defaultdict(list)
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        # elif type == 'low_dim' and ('eef' not in key):
        #     this_data_in = env_obs[key]
        #     obs_dict_np[key] = this_data_in
        #     # handle multi-robots
        #     ks = key.split('_')
        #     if ks[0].startswith('robot'):
        #         robot_prefix_map[ks[0]].append(key)
        elif type == 'low_dim':
            # handle multi-robots - always track robot prefixes
            ks = key.split('_')
            if ks[0].startswith('robot'):
                robot_prefix_map[ks[0]].append(key)
            # Only add non-eef keys directly (eef handled below)
            if 'eef' not in key:
                this_data_in = env_obs[key]
                obs_dict_np[key] = this_data_in

    # generate relative pose
    for robot_prefix in robot_prefix_map.keys():
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[robot_prefix + '_eef_pos'],
            env_obs[robot_prefix + '_eef_rot_axis_angle']
        ], axis=-1))

        # solve reltaive obs
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat, 
            base_pose_mat=pose_mat[-1],
            pose_rep=obs_pose_repr,
            backward=False)

        # pose_mat_real = pose_to_mat(np.concatenate([
        #     env_obs[robot_prefix + '_eef_pos'],
        #     env_obs[robot_prefix + '_eef_rot_axis_angle']
        # ], axis=-1))

        # # convert whole history to model frame
        # pose_mat = real_pose_mat_to_model_pose_mat(pose_mat_real)

        # obs_pose_mat = convert_pose_mat_rep(
        #     pose_mat,
        #     base_pose_mat=pose_mat[-1],
        #     pose_rep=obs_pose_repr,
        #     backward=False
        # )

        obs_pose = mat_to_pose10d(obs_pose_mat)
        obs_dict_np[robot_prefix + '_eef_pos'] = obs_pose[...,:3]
        obs_dict_np[robot_prefix + '_eef_rot_axis_angle'] = obs_pose[...,3:]
    
    # generate pose relative to other robot
    n_robots = len(robot_prefix_map)
    for robot_id in range(n_robots):
        # convert pose to mat
        assert f'robot{robot_id}' in robot_prefix_map
        tx_robota_tcpa = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_id}_eef_pos'],
            env_obs[f'robot{robot_id}_eef_rot_axis_angle']
        ], axis=-1))
        for other_robot_id in range(n_robots):
            if robot_id == other_robot_id:
                continue
            tx_robotb_tcpb = pose_to_mat(np.concatenate([
                env_obs[f'robot{other_robot_id}_eef_pos'],
                env_obs[f'robot{other_robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            tx_robota_robotb = tx_robot1_robot0
            if robot_id == 0:
                tx_robota_robotb = np.linalg.inv(tx_robot1_robot0)
            tx_robota_tcpb = tx_robota_robotb @ tx_robotb_tcpb

            rel_obs_pose_mat = convert_pose_mat_rep(
                tx_robota_tcpa,
                base_pose_mat=tx_robota_tcpb[-1],
                pose_rep='relative',
                backward=False)
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            obs_dict_np[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]

            # tx_robota_tcpa_real = pose_to_mat(np.concatenate([
            #     env_obs[f'robot{robot_id}_eef_pos'],
            #     env_obs[f'robot{robot_id}_eef_rot_axis_angle']
            # ], axis=-1))
            # tx_robota_tcpa = real_pose_mat_to_model_pose_mat(tx_robota_tcpa_real)

            # tx_robotb_tcpb_real = pose_to_mat(np.concatenate([
            #     env_obs[f'robot{other_robot_id}_eef_pos'],
            #     env_obs[f'robot{other_robot_id}_eef_rot_axis_angle']
            # ], axis=-1))
            # tx_robotb_tcpb = real_pose_mat_to_model_pose_mat(tx_robotb_tcpb_real)

            # # if you use robot-to-robot transform, it must be in the same frame too
            # tx_robota_robotb = tx_robot1_robot0
            # if tx_robota_robotb is not None:
            #     tx_robota_robotb = real_pose_mat_to_model_pose_mat(tx_robota_robotb)

            # if robot_id == 0:
            #     tx_robota_robotb = np.linalg.inv(tx_robota_robotb)

            # tx_robota_tcpb = tx_robota_robotb @ tx_robotb_tcpb

            # rel_obs_pose_mat = convert_pose_mat_rep(
            #     tx_robota_tcpa,
            #     base_pose_mat=tx_robota_tcpb[-1],
            #     pose_rep='relative',
            #     backward=False
            # )


    # generate relative pose with respect to episode start
    if episode_start_pose is not None:
        for robot_id in range(n_robots):        
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                env_obs[f'robot{robot_id}_eef_pos'],
                env_obs[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            
            # get start pose
            # start_pose = episode_start_pose[robot_id]
            if isinstance(episode_start_pose, list):
                start_pose = episode_start_pose[robot_id]
            else:
                start_pose = episode_start_pose
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)

            # start_pose_mat_real = pose_to_mat(start_pose)
            # start_pose_mat = real_pose_mat_to_model_pose_mat(start_pose_mat_real)

            # pose_mat_real = pose_to_mat(np.concatenate([
            #     env_obs[f'robot{robot_id}_eef_pos'],
            #     env_obs[f'robot{robot_id}_eef_rot_axis_angle']
            # ], axis=-1))
            # pose_mat = real_pose_mat_to_model_pose_mat(pose_mat_real)

            # rel_obs_pose_mat = convert_pose_mat_rep(
            #     pose_mat,
            #     base_pose_mat=start_pose_mat,
            #     pose_rep='relative',
            #     backward=False
            # )

            
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # obs_dict_np[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

    return obs_dict_np

def get_real_umi_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs'
    ):

    # n_robots = int(action.shape[-1] // 16)
    # n_robots = int(action.shape[-1] // 10)
    # n_robots = int(action.shape[-1] // 9)
    n_robots = int(action.shape[-1] // 15)

    env_action = list()
    for robot_idx in range(n_robots):
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_idx}_eef_pos'][-1],
            env_obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]
        ], axis=-1))

        # # current robot pose in real frame (single pose, 4x4)
        # pose_mat_real = pose_to_mat(np.concatenate([
        #     env_obs[f'robot{robot_idx}_eef_pos'][-1],
        #     env_obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]
        # ], axis=-1))

        # # convert base pose to model frame
        # pose_mat_model = real_pose_mat_to_model_pose_mat(pose_mat_real)

        start = robot_idx * 15
        action_pose10d = action[..., start:start+9]
        action_chol = action[..., start+9:start+15]
        # action_grip = action[..., start+15:start+16]

        # start = robot_idx * 10
        # start = robot_idx * 9
        # action_grip = action[..., start+9:start+10]
        # action_pose10d = action[..., start:start+9]

        action_pose_mat = pose10d_to_mat(action_pose10d)

        # solve relative action
        action_mat = convert_pose_mat_rep(
            action_pose_mat, 
            base_pose_mat=pose_mat,
            pose_rep=action_pose_repr,
            backward=True)

        # Convert action to pose
        action_pose = mat_to_pose(action_mat)

        # # policy output pose10d is in model frame representation
        # action_pose_mat_model = pose10d_to_mat(action_pose10d)

        # # convert model-relative -> model-absolute (or abs->abs) using model base
        # action_mat_model = convert_pose_mat_rep(
        #     action_pose_mat_model,
        #     base_pose_mat=pose_mat_model,
        #     pose_rep=action_pose_repr,
        #     backward=True
        # )

        # # now convert model absolute pose to real absolute pose
        # action_mat_real = model_pose_mat_to_real_pose_mat(action_mat_model)

        # action_pose = mat_to_pose(action_mat_real)

        # Convert action to stiffness
        # action_stiffness = chol_to_stiffness(action_chol)

        if USE_CONST_STIFFNESS:
            # action_pose has shape (T, 6), so match stiffness to (T, 3)
            T = action_pose.shape[0] if action_pose.ndim == 2 else 1
            action_stiffness = np.tile(CONST_KX_TRANS[None, :], (T, 1))
        else:
            action_stiffness = chol_to_stiffness(action_chol)


        env_action.append(action_pose)
        env_action.append(action_stiffness)
        # env_action.append(action_grip)

    act = np.concatenate(env_action, axis=-1)
    return act
