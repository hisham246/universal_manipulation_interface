import time
import multiprocessing as mp
import zerorpc
import csv
import numpy as np
import pathlib
import argparse
import os
import gc

# --- 1. Precise Wait Helper ---
def precise_wait(t_end, slack_time=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_remain = t_end - t_start
    if t_remain > slack_time:
        time.sleep(t_remain - slack_time)
    while time_func() < t_end:
        pass

# --- 2. Interface ---
class FrankaInterface:
    def __init__(self, ip='129.97.71.27', port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_robot_state(self):
        return self.server.get_robot_state()
    
    def get_ee_pose(self):
        # Returns 6D vector: [x, y, z, rot_x, rot_y, rot_z]
        data = self.server.get_ee_pose()
        pos = np.array(data[:3])
        rot_vec = np.array(data[3:])
        return np.concatenate([pos, rot_vec])
    
    def close(self):
        self.server.close()

# --- 3. Optimized Recorder Process ---
class FrankaMemoryRecorder(mp.Process):
    def __init__(self, ip, output_path, duration, frequency=1000):
        super().__init__(name="FrankaMemoryRecorder")
        self.ip = ip
        self.output_path = pathlib.Path(output_path)
        self.duration = duration
        self.frequency = frequency
        self.daemon = True 

    def run(self):
        # A. Setup Real-Time Priority
        try:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            print("[Recorder] Real-time priority enabled (SCHED_RR).")
        except Exception:
            print("[Recorder] Warning: Could not set SCHED_RR priority. Run with sudo for better performance.")

        # B. Connect
        robot = FrankaInterface(self.ip)
        print(f"[Recorder] Connected. Pre-allocating memory for {self.duration}s...")

        # C. Pre-allocate Memory & Determine Headers
        # Fetch one sample to determine structure
        example_state = robot.get_robot_state()
        example_ee = robot.get_ee_pose()
        
        headers = ['timestamp', 'iter_time']
        
        # 1. Add Robot State Headers
        # We need to ensure we iterate in the same order as values()
        # To be safe, we will use a fixed list of keys or just iterate sorted keys
        state_keys = sorted(example_state.keys()) 
        
        flat_example_len = 0
        
        for key in state_keys:
            val = example_state[key]
            if isinstance(val, (list, np.ndarray, tuple)):
                flat_example_len += len(val)
                headers.extend([f"{key}_{i}" for i in range(len(val))])
            else:
                flat_example_len += 1
                headers.append(key)

        # 2. Add EE Pose Headers
        headers.extend([f"ee_pose_{i}" for i in range(6)])
        flat_example_len += 6

        num_features = flat_example_len
        num_samples = int(self.duration * self.frequency)
        
        # The Buffer: (Rows, Columns + 2 for timestamps)
        buffer = np.zeros((num_samples, 2 + num_features), dtype=np.float64)

        # D. The Loop
        dt = 1.0 / self.frequency
        print(f"[Recorder] Starting loop at {self.frequency}Hz...")
        
        gc.disable()
        
        t_start = time.monotonic()
        t_loop_start = t_start
        
        try:
            for i in range(num_samples):
                t_now = time.monotonic()
                
                # --- CRITICAL SECTION START ---
                # 1. Fetch Data
                state = robot.get_robot_state()
                ee_pose = robot.get_ee_pose()
                
                # 2. Parse into Buffer
                buffer[i, 0] = time.time()          
                buffer[i, 1] = t_now - t_loop_start 
                
                row_idx = 2
                
                # Unpack Robot State (using sorted keys for consistency)
                for key in state_keys:
                    val = state[key]
                    if isinstance(val, (list, np.ndarray, tuple)):
                        l = len(val)
                        buffer[i, row_idx : row_idx+l] = val
                        row_idx += l
                    else:
                        buffer[i, row_idx] = val
                        row_idx += 1
                
                # Unpack EE Pose
                buffer[i, row_idx : row_idx+6] = ee_pose
                # --- CRITICAL SECTION END ---

                # 3. Wait
                precise_wait(t_start + (i + 1) * dt)

        except KeyboardInterrupt:
            print("\n[Recorder] Interrupted.")
        except Exception as e:
            print(f"\n[Recorder] Error: {e}")
        finally:
            gc.enable()
            robot.close()

            # E. Save to Disk
            print(f"[Recorder] Saving {i} samples to {self.output_path}...")
            
            final_data = buffer[:i+1]
            
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(final_data)
            
            print(f"[Recorder] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="panda_state_polymetis/trial_3.csv")
    parser.add_argument("--ip", type=str, default="129.97.71.27")
    parser.add_argument("--time", type=float, default=10.0, help="Duration to record (seconds)")
    args = parser.parse_args()

    recorder = FrankaMemoryRecorder(args.ip, args.output, args.time)
    recorder.start()
    recorder.join()