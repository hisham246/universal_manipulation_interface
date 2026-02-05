import time
import zerorpc
import csv
import numpy as np
import pathlib
import argparse
import pandas as pd
import os
import signal
import sys

# --- Precise Wait ---
def precise_wait(t_end, slack_time=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_remain = t_end - t_start
    if t_remain > slack_time:
        time.sleep(t_remain - slack_time)
    while time_func() < t_end:
        pass

# --- Interpolator (Minimal implementation if UMI is missing) ---
try:
    from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
except ImportError:
    # Fallback if UMI is not installed, simple linear interpolation
    from scipy.spatial.transform import Rotation as R
    from scipy.interpolate import interp1d

    class PoseTrajectoryInterpolator:
        def __init__(self, times, poses):
            self.times = times
            self.poses = poses
            self.pos_interp = interp1d(times, poses[:, :3], axis=0, fill_value="extrapolate")
            self.rot_obj = R.from_rotvec(poses[:, 3:])
            self.slerp = self.rot_obj.as_quat() # We simplify by just interpolating rotvecs for fallback
            self.rot_interp = interp1d(times, poses[:, 3:], axis=0, fill_value="extrapolate")

        def __call__(self, t):
            pos = self.pos_interp(t)
            rot = self.rot_interp(t)
            return np.concatenate([pos, rot])

# --- Interface ---
class FrankaInterface:
    def __init__(self, ip='129.97.71.27', port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_robot_state(self):
        return self.server.get_robot_state()

    def get_ee_pose(self):
        data = self.server.get_ee_pose()
        pos = np.array(data[:3])
        rot_vec = np.array(data[3:])
        return np.concatenate([pos, rot_vec])
    
    # We skip move_to_joint_positions as it is causing errors
    
    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(Kx.tolist(), Kxd.tolist())
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        self.server.update_desired_ee_pose(pose.tolist())

    def terminate_current_policy(self):
        self.server.terminate_current_policy()
    
    def close(self):
        self.server.close()

# --- Main Replay Logic ---
def replay_trajectory(csv_path, robot_ip, speed_scale=1.0):
    robot = None
    try:
        # 1. Load Data
        print(f"Loading trajectory from {csv_path}...")
        df = pd.read_csv(csv_path)
        ee_cols = [f'ee_pose_{i}' for i in range(6)]
        
        # Verify columns
        if not all(col in df.columns for col in ee_cols):
            print(f"Error: CSV is missing columns {ee_cols}")
            return

        recorded_poses = df[ee_cols].to_numpy()
        recorded_times = df['timestamp'].to_numpy()
        recorded_times = (recorded_times - recorded_times[0]) / speed_scale
        traj_duration = recorded_times[-1]

        # Trajectory Interpolator
        traj_interpolator = PoseTrajectoryInterpolator(times=recorded_times, poses=recorded_poses)

        # 2. Setup Robot & Homing
        print(f"Connecting to robot at {robot_ip}...")
        robot = FrankaInterface(ip=robot_ip)
        
        # Default Gains
        Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) 
        Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0])

        # A. Get Current Pose
        current_pose = robot.get_ee_pose()
        start_pose = traj_interpolator(0.0)

        # B. Generate Soft Homing Trajectory (Current -> Start)
        print("Planning soft homing move (Current -> Start of CSV)...")
        homing_duration = 5.0 # Seconds
        homing_times = np.array([0.0, homing_duration])
        homing_poses = np.stack([current_pose, start_pose])
        homing_interpolator = PoseTrajectoryInterpolator(times=homing_times, poses=homing_poses)

        # 3. Execution Loop
        print("Starting Impedance Control...")
        robot.start_cartesian_impedance(Kx, Kxd)
        
        # Real-time Priority
        try:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
        except:
            pass

        frequency = 1000
        dt = 1.0 / frequency
        
        # --- Phase 1: Homing ---
        print(f"Phase 1: Moving to start pose ({homing_duration}s)...")
        t_start_homing = time.monotonic()
        while True:
            t_now = time.monotonic()
            t_play = t_now - t_start_homing
            
            if t_play > homing_duration:
                break
            
            target = homing_interpolator(t_play)
            robot.update_desired_ee_pose(target)
            precise_wait(t_start_homing + (int(t_play/dt) + 1) * dt)

        # Buffer
        print("Buffered wait (0.5s)...")
        time.sleep(0.5)

        # --- Phase 2: Replay ---
        print(f"Phase 2: Replaying Trajectory ({traj_duration:.2f}s)...")
        t_start_replay = time.monotonic()
        while True:
            t_now = time.monotonic()
            t_play = t_now - t_start_replay
            
            if t_play > traj_duration:
                break
                
            target = traj_interpolator(t_play)
            robot.update_desired_ee_pose(target)
            precise_wait(t_start_replay + (int(t_play/dt) + 1) * dt)

        print("Trajectory finished.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if robot:
            print("Terminating policy...")
            try:
                robot.terminate_current_policy()
                robot.close()
            except:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--ip", type=str, default="129.97.71.27")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    replay_trajectory(args.csv, args.ip, args.speed)