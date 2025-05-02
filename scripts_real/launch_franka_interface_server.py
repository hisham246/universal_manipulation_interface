import zerorpc
from polymetis import RobotInterface, GripperInterface
import scipy.spatial.transform as st
import numpy as np
import torch

class FrankaInterface:
    def __init__(self):
        self.robot = RobotInterface('localhost')
        self.gripper = GripperInterface('localhost')

    # Robot methods
    def get_robot_state(self):
        data = self.robot.get_robot_state()
        data_state_dict = {
            "timestamp": float(data.timestamp.nanos),
            "joint_positions": list(data.joint_positions),
            "joint_velocities": list(data.joint_velocities),
            "joint_torques_computed": list(data.joint_torques_computed),
            "prev_joint_torques_computed": list(data.prev_joint_torques_computed),
            "prev_joint_torques_computed_safened": list(data.prev_joint_torques_computed_safened),
            "motor_torques_measured": list(data.motor_torques_measured),
            "motor_torques_external": list(data.motor_torques_external),
            "motor_torques_desired": list(data.motor_torques_desired),
        }
        return data_state_dict
    
    def get_ee_pose(self):
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()
    
    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy().tolist()
    
    def get_joint_velocities(self):
        return self.robot.get_joint_velocities().numpy().tolist()
    
    def move_to_joint_positions(self, positions, time_to_go):
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(positions),
            time_to_go=time_to_go
        )
    
    def start_cartesian_impedance(self, Kx, Kxd):
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx),
            Kxd=torch.Tensor(Kxd)
        )

    def update_desired_ee_pose(self, pose):
        pose = np.asarray(pose)
        self.robot.update_desired_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat())
        )

    def update_current_policy(self, param_dict):
        param_dict_torch = {
            key: torch.Tensor(val)
            for key, val in param_dict.items()
        }
        return self.robot.update_current_policy(param_dict_torch)

    def terminate_current_policy(self):
        self.robot.terminate_current_policy()

    # Gripper methods
    def get_gripper_state(self):
        gripper_state = self.gripper.get_state()
        gripper_state_dict = {
            "timestamp": {"seconds": gripper_state.timestamp.seconds, "nanoseconds": gripper_state.timestamp.nanos},
            "width": gripper_state.width
        }
        return gripper_state_dict
    
    def gripper_goto(self, width, speed, force):
        self.gripper.goto(width, speed, force)
    
    def gripper_grasp(self, speed, force, grasp_width):
        self.gripper.grasp(speed, force, grasp_width)

s = zerorpc.Server(FrankaInterface())
s.bind("tcp://0.0.0.0:4242")
s.run()