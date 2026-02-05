import time
import zerorpc
import csv
import numpy as np
import pathlib
import argparse
import pandas as pd
import os
import gc

# --- Precise Wait ---
def precise_wait(t_end, slack_time=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_remain = t_end - t_start
    if t_remain > slack_time:
        time.sleep(t_remain - slack_time)
    while time_func() < t_end:
        pass

# --- Interpolator ---
try:
    from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
except ImportError:
    from scipy.spatial.transform import Rotation as R
    from scipy.interpolate import interp1d

    class PoseTrajectoryInterpolator:
        def __init__(self, times, poses):
            self.times = times
            self.poses = poses
            self.pos_interp = interp1d(times, poses[:, :3], axis=0, fill_value="extrapolate")
            self.rot_obj = R.from_rotvec(poses[:, 3:])
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
        # Returns dict with 'joint_positions', 'joint_velocities' etc.
        return self.server.get_robot_state()

    def get_ee_pose(self):
        data = self.server.get_ee_pose()
        pos = np.array(data[:3])
        rot_vec = np.array(data[3:])
        return np.concatenate([pos, rot_vec])
    
    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(Kx.tolist(), Kxd.tolist())
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        self.server.update_desired_ee_pose(pose.tolist())

    def terminate_current_policy(self):
        self.server.terminate_current_policy()
    
    def close(self):
        self.server.close()

# --- Main Replay Logic ---
def replay_trajectory(csv_path, output_log_path, robot_ip, speed_scale=1.0):
    robot = None
    try:
        # 1. Load Data
        print(f"Loading trajectory from {csv_path}...")
        df = pd.read_csv(csv_path)
        ee_cols = [f'ee_pose_{i}' for i in range(6)]
        
        if not all(col in df.columns for col in ee_cols):
            print(f"Error: CSV is missing columns {ee_cols}")
            return

        recorded_poses = df[ee_cols].to_numpy()
        recorded_times = df['timestamp'].to_numpy()
        recorded_times = (recorded_times - recorded_times[0]) / speed_scale
        traj_duration = recorded_times[-1]

        traj_interpolator = PoseTrajectoryInterpolator(times=recorded_times, poses=recorded_poses)

        # 2. Setup Robot
        print(f"Connecting to robot at {robot_ip}...")
        robot = FrankaInterface(ip=robot_ip)
        
        Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) 
        Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0])

        current_pose = robot.get_ee_pose()
        start_pose = traj_interpolator(0.0)

        # Soft Homing Setup
        homing_duration = 5.0 
        homing_times = np.array([0.0, homing_duration])
        homing_poses = np.stack([current_pose, start_pose])
        homing_interpolator = PoseTrajectoryInterpolator(times=homing_times, poses=homing_poses)

        # 3. Pre-allocate Logging Memory
        # Columns: [timestamp, 
        #           desired_ee (6), actual_ee (6), 
        #           actual_q (7), actual_qd (7)]
        # Total columns = 1 + 6 + 6 + 7 + 7 = 27
        
        frequency = 1000
        dt = 1.0 / frequency
        total_duration = homing_duration + 0.5 + traj_duration
        num_steps = int(total_duration * frequency) + 1000 # Buffer
        
        log_buffer = np.zeros((num_steps, 27))
        log_idx = 0
        
        print(f"Allocated memory for {num_steps} steps. Starting...")
        
        # 4. Execution Loop
        robot.start_cartesian_impedance(Kx, Kxd)
        
        try:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
        except:
            pass

        gc.disable() # Important for avoiding hiccups
        
        # --- Helper for Logging ---
        def run_phase(start_time, duration, interpolator):
            nonlocal log_idx
            while True:
                t_now = time.monotonic()
                t_play = t_now - start_time
                
                if t_play > duration:
                    break
                
                # A. Get Target
                target = interpolator(t_play)
                
                # B. Send Command
                robot.update_desired_ee_pose(target)
                
                # C. Measure Actuals (The Expensive Part)
                # Note: 'get_robot_state' usually returns dictionary. 
                # We assume keys based on your previous logs.
                state = robot.get_robot_state() 
                actual_ee = robot.get_ee_pose()
                
                # Extract specific fields (robustly)
                # In your trial_1.csv, keys were 'joint_positions', 'joint_velocities'
                actual_q = state.get('joint_positions', np.zeros(7))
                actual_qd = state.get('joint_velocities', np.zeros(7))

                # D. Log to Buffer
                # 0: Time
                log_buffer[log_idx, 0] = t_play 
                # 1-6: Desired EE
                log_buffer[log_idx, 1:7] = target
                # 7-12: Actual EE
                log_buffer[log_idx, 7:13] = actual_ee
                # 13-19: Actual Q
                log_buffer[log_idx, 13:20] = actual_q
                # 20-26: Actual Qd
                log_buffer[log_idx, 20:27] = actual_qd
                
                log_idx += 1
                
                precise_wait(start_time + (int(t_play/dt) + 1) * dt)

        # --- Phase 1: Homing ---
        print(f"Phase 1: Homing ({homing_duration}s)...")
        run_phase(time.monotonic(), homing_duration, homing_interpolator)

        # Buffer
        time.sleep(0.5)

        # --- Phase 2: Replay ---
        print(f"Phase 2: Replaying ({traj_duration:.2f}s)...")
        run_phase(time.monotonic(), traj_duration, traj_interpolator)

        print("Trajectory finished.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        gc.enable()
        if robot:
            print("Terminating policy...")
            try:
                robot.terminate_current_policy()
                robot.close()
            except:
                pass

        # 5. Save Logs
        if 'log_idx' in locals() and log_idx > 0:
            print(f"Saving {log_idx} log entries to {output_log_path}...")
            
            # Create Headers
            headers = ['time_rel']
            headers += [f'desired_ee_{i}' for i in range(6)]
            headers += [f'actual_ee_{i}' for i in range(6)]
            headers += [f'actual_q_{i}' for i in range(7)]
            headers += [f'actual_qd_{i}' for i in range(7)]
            
            # Slice only filled data
            final_data = log_buffer[:log_idx]
            
            # Save
            pathlib.Path(output_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(final_data)
            print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Input trajectory CSV")
    parser.add_argument("--log", type=str, default="replay_log.csv", help="Output response log CSV")
    parser.add_argument("--ip", type=str, default="129.97.71.27")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    replay_trajectory(args.csv, args.log, args.ip, args.speed)