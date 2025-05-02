import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
import torch
from umi.common.pose_util import pose_to_mat, mat_to_pose
import zerorpc
import csv

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class FrankaInterface:
    # IP Address of NUC Labwork 4
    def __init__(self, ip='129.97.71.27', port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_robot_state(self):
        state = self.server.get_robot_state()
        return state
    
    def get_ee_pose(self):
        data = self.server.get_ee_pose()
        pos = np.array(data[:3])
        rot_vec = np.array(data[3:])
        return np.concatenate([pos, rot_vec])
    
    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())
    
    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(
            Kx.tolist(),
            Kxd.tolist()
        )

    def get_joint_pos_desired(self, pose: np.ndarray):
        return np.array(self.server.get_joint_pos_desired(pose.tolist()))
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        self.server.update_desired_ee_pose(pose.tolist())

    def terminate_current_policy(self):
        self.server.terminate_current_policy()

    def close(self):
        self.server.close()


class FrankaInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        robot_ip,
        robot_port=4242,
        frequency=1000,
        Kx_scale=1.0,
        Kxd_scale=1.0,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0
        ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FrankaPositionalController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.frequency = frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd','get_joint_velocities'),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
            
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
            
        # Start Polymetis interface
        robot = FrankaInterface(self.robot_ip, self.robot_port)

        robot_state_path_1 = "/home/hisham246/uwaterloo/robot_state_pickplace_test_1.csv"

        robot_state_path_2 = "/home/hisham246/uwaterloo/robot_state_pickplace_test_2.csv"

        # Get example state to construct headers
        example_state = robot.get_robot_state()
        csv_fieldnames_1 = []
        for key, value in example_state.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                csv_fieldnames_1.extend([f"{key}_{i}" for i in range(len(value))])
            else:
                csv_fieldnames_1.append(key)

        # Write header
        with open(robot_state_path_1, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames_1)
            writer.writeheader()


        # Define the headers
        csv_fieldnames_2 = (
            [f"ee_pose_{i}" for i in range(6)] +
            [f"joint_pos_{i}" for i in range(7)] +
            [f"joint_vel_{i}" for i in range(7)]
        )

        # Write header once
        with open(robot_state_path_2, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames_2)
            writer.writeheader()
        
        try:
            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")
            
            # init pose
            if self.joints_init is not None:
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=self.joints_init_duration
                )

            # main loop
            dt = 1. / self.frequency
            curr_pose = robot.get_ee_pose()

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            # Start Franka cartesian impedance policy
            robot.start_cartesian_impedance(
                Kx=self.Kx,
                Kxd=self.Kxd
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)

                # tip_pose = pose_interp(t_now)
                # flange_pose = mat_to_pose(pose_to_mat(tip_pose) @ tx_tip_flange)


                ee_pose = pose_interp(t_now)
                print("Joint Pos Desired:", robot.get_joint_pos_desired(ee_pose))

                # send command to robot
                robot.update_desired_ee_pose(ee_pose)

                # result = robot.update_desired_ee_pose(ee_pose)

                # if result is not None:
                #     update, q_des = result
                #     print("Desired joint positions:", q_des)

                # update robot state
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(robot, func_name)()

                robot_state = robot.get_robot_state()
                flat_state = {}
                for key, value in robot_state.items():
                    if isinstance(value, (list, np.ndarray)):
                        for i, v in enumerate(value):
                            flat_state[f"{key}_{i}"] = v
                    else:
                        flat_state[key] = value

                with open(robot_state_path_1, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames_1)
                    writer.writerow(flat_state)


                # Collect and flatten low-level state
                ee_pose = state['ActualTCPPose']        # 6D pose
                joint_pos = state['ActualQ']            # 7D positions
                joint_vel = state['ActualQd']           # 7D velocities

                lowlevel_row = {}
                for i in range(6):
                    lowlevel_row[f"ee_pose_{i}"] = ee_pose[i]
                for i in range(7):
                    lowlevel_row[f"joint_pos_{i}"] = joint_pos[i]
                    lowlevel_row[f"joint_vel_{i}"] = joint_vel[i]

                # Write to low-level CSV
                with open(robot_state_path_2, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames_2)
                    writer.writerow(lowlevel_row)

                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    # commands = self.input_queue.get_all()
                    # n_cmd = len(commands['cmd'])
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[FrankaPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[FrankaPositionalController] Actual frequency {1/(time.monotonic() - t_now)}")

        finally:
            # manditory cleanup
            # terminate
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            robot.terminate_current_policy()
            del robot
            self.ready_event.set()

            if self.verbose:
                print(f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")
