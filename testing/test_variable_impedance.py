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

    def update_impedance_gains(self, Kx: np.ndarray, Kxd: np.ndarray):
        # The user provided snippet uses a specific dict format for update
        self.server.update_current_policy({
            'Kx_tgt_vec': Kx.tolist(),
            'Kxd_tgt_vec': Kxd.tolist()
        })

    def terminate_current_policy(self):
        self.server.terminate_current_policy()
    
    def close(self):
        self.server.close()

# --- Main Replay Logic ---
def replay_trajectory(csv_path, stiffness_path, output_log_path, robot_ip, 
                      speed_scale=1.0, stiffness_freq=10.0):
    robot = None
    try:
        # 1. Load Trajectory Data
        print(f"Loading trajectory from {csv_path}...")
        df_traj = pd.read_csv(csv_path)
        ee_cols = [f'ee_pose_{i}' for i in range(6)]
        
        if not all(col in df_traj.columns for col in ee_cols):
            print(f"Error: Trajectory CSV is missing columns {ee_cols}")
            return

        recorded_poses = df_traj[ee_cols].to_numpy()
        recorded_times = df_traj['timestamp'].to_numpy()
        recorded_times = (recorded_times - recorded_times[0]) / speed_scale
        traj_duration = recorded_times[-1]

        traj_interpolator = PoseTrajectoryInterpolator(times=recorded_times, poses=recorded_poses)

        # 2. Load Stiffness Data
        print(f"Loading stiffness from {stiffness_path}...")
        df_stiff = pd.read_csv(stiffness_path)
        if not all(col in df_stiff.columns for col in ['Kx', 'Ky', 'Kz']):
            print("Error: Stiffness CSV must have Kx, Ky, Kz columns")
            return
            
        stiffness_data = df_stiff[['Kx', 'Ky', 'Kz']].to_numpy() # (N, 3)
        num_stiff_samples = len(stiffness_data)
        
        # 3. Setup Robot
        print(f"Connecting to robot at {robot_ip}...")
        robot = FrankaInterface(ip=robot_ip)
        
        # Initial Gains (taken from first line of file + default rot)
        init_stiff_trans = stiffness_data[0]
        init_stiff_rot = np.array([30.0, 30.0, 30.0])
        init_Kx = np.concatenate([init_stiff_trans, init_stiff_rot])
        # Damping formula: d = 2 * 0.707 * sqrt(k)
        init_Kxd = 2.0 * 0.707 * np.sqrt(init_Kx)

        current_pose = robot.get_ee_pose()
        start_pose = traj_interpolator(0.0)

        # Soft Homing Setup
        homing_duration = 5.0 
        homing_times = np.array([0.0, homing_duration])
        homing_poses = np.stack([current_pose, start_pose])
        homing_interpolator = PoseTrajectoryInterpolator(times=homing_times, poses=homing_poses)

        # 4. Pre-allocate Logging Memory
        # Columns: [timestamp (1), 
        #           desired_ee (6), actual_ee (6), 
        #           actual_q (7), actual_qd (7),
        #           desired_stiffness (6), desired_damping (6)]
        # Total columns = 1 + 6 + 6 + 7 + 7 + 6 + 6 = 39
        
        frequency = 1000
        dt = 1.0 / frequency
        total_duration = homing_duration + 0.5 + traj_duration
        num_steps = int(total_duration * frequency) + 2000
        
        log_buffer = np.zeros((num_steps, 39))
        log_idx = 0
        
        print(f"Allocated memory for {num_steps} steps. Starting...")
        
        # 5. Execution Loop
        robot.start_cartesian_impedance(init_Kx, init_Kxd)
        
        try:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
        except:
            pass

        gc.disable()
        
        def run_phase(start_time, duration, interpolator, update_stiffness=False):
            nonlocal log_idx
            last_stiff_idx = -1
            
            # Current Gains state
            curr_Kx = init_Kx.copy()
            curr_Kxd = init_Kxd.copy()

            while True:
                t_now = time.monotonic()
                t_play = t_now - start_time
                
                if t_play > duration:
                    break
                
                # A. Trajectory Target
                target_pose = interpolator(t_play)
                robot.update_desired_ee_pose(target_pose)
                
                # B. Variable Stiffness Update
                if update_stiffness:
                    # Calculate index with looping
                    # idx = (time * freq) % count
                    s_idx = int(t_play * stiffness_freq) % num_stiff_samples
                    
                    # Only send command if index changed (Optimization)
                    if s_idx != last_stiff_idx:
                        k_trans = stiffness_data[s_idx]
                        k_rot = np.array([30.0, 30.0, 30.0])
                        
                        curr_Kx = np.concatenate([k_trans, k_rot])
                        curr_Kxd = 2.0 * 0.707 * np.sqrt(curr_Kx)
                        
                        robot.update_impedance_gains(curr_Kx, curr_Kxd)
                        last_stiff_idx = s_idx

                # C. Measure Actuals
                state = robot.get_robot_state() 
                actual_ee = robot.get_ee_pose()
                
                actual_q = state.get('joint_positions', np.zeros(7))
                actual_qd = state.get('joint_velocities', np.zeros(7))

                # D. Log
                row = log_buffer[log_idx]
                row[0] = t_play
                row[1:7]   = target_pose
                row[7:13]  = actual_ee
                row[13:20] = actual_q
                row[20:27] = actual_qd
                row[27:33] = curr_Kx
                row[33:39] = curr_Kxd
                
                log_idx += 1
                
                precise_wait(start_time + (int(t_play/dt) + 1) * dt)

        # --- Phase 1: Homing (Fixed Stiffness) ---
        print(f"Phase 1: Homing ({homing_duration}s)...")
        # During homing, we don't vary stiffness (safer)
        run_phase(time.monotonic(), homing_duration, homing_interpolator, update_stiffness=False)

        # Buffer
        time.sleep(0.5)

        # --- Phase 2: Replay (Variable Stiffness) ---
        print(f"Phase 2: Replaying ({traj_duration:.2f}s) with variable stiffness...")
        run_phase(time.monotonic(), traj_duration, traj_interpolator, update_stiffness=True)

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

        # 6. Save Logs
        if 'log_idx' in locals() and log_idx > 0:
            print(f"Saving {log_idx} log entries to {output_log_path}...")
            
            headers = ['time_rel']
            headers += [f'desired_ee_{i}' for i in range(6)]
            headers += [f'actual_ee_{i}' for i in range(6)]
            headers += [f'actual_q_{i}' for i in range(7)]
            headers += [f'actual_qd_{i}' for i in range(7)]
            headers += [f'stiffness_{i}' for i in range(6)]
            headers += [f'damping_{i}' for i in range(6)]
            
            final_data = log_buffer[:log_idx]
            
            pathlib.Path(output_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(final_data)
            print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Trajectory CSV")
    parser.add_argument("--stiffness", type=str, required=True, help="Stiffness Profile CSV")
    parser.add_argument("--log", type=str, default="impedance_log.csv", help="Output Log CSV")
    parser.add_argument("--ip", type=str, default="129.97.71.27")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--freq", type=float, default=30.0, help="Frequency to play stiffness profile (Hz)")
    args = parser.parse_args()

    replay_trajectory(args.csv, args.stiffness, args.log, args.ip, args.speed, args.freq)