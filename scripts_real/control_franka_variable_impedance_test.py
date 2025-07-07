import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.precise_sleep import precise_wait
from umi.real_world.keystroke_counter import KeystrokeCounter, KeyCode
from umi.real_world.franka_variable_impedance_controller import FrankaVariableImpedanceController
from umi.real_world.franka_hand_controller import FrankaHandController

@click.command()
@click.option('-rh', '--robot_hostname', default='129.97.71.27')
@click.option('-f', '--frequency', type=float, default=30)
@click.option('-gs', '--gripper_speed', type=float, default=0.05)
@click.option('-gf', '--gripper_force', type=float, default=20.0)
@click.option('-gp', '--gripper_port', type=int, default=4242)
@click.option('-gh', '--gripper_host', default='129.97.71.27')

def main(robot_hostname, frequency, gripper_speed, gripper_force, gripper_port, gripper_host):
    max_pos_speed = 0.25
    max_rot_speed = 0.6
    max_gripper_width = 0.1  # meters
    dt = 1 / frequency
    command_latency = dt / 2

    # Define impedance profiles (stiffness/damping)
    impedance_profiles = [
        (np.array([100, 100, 100, 10, 10, 10]), np.array([10, 10, 10, 1, 1, 1])),
        (np.array([150, 150, 150, 20, 20, 20]), np.array([20, 20, 20, 2, 2, 2])),
        (np.array([200, 200, 200, 30, 30, 30]), np.array([30, 30, 30, 3, 3, 3])),
        (np.array([300, 300, 300, 30, 30, 30]), np.array([30, 30, 30, 3, 3, 3])),
        (np.array([400, 400, 400, 30, 30, 30]), np.array([30, 30, 30, 3, 3, 3])),
        (np.array([500, 500, 500, 30, 30, 30]), np.array([30, 30, 30, 3, 3, 3]))]
    
    # impedance_profiles = [
    #     (np.array([1000, 1000, 1000, 30, 30, 30]), np.array([30, 30, 30, 3, 3, 3])),
    #     (np.array([1000, 1000, 1000, 30, 30, 30]), np.array([30, 30, 30, 3, 3, 3])),
    #     (np.array([1000, 1000, 1000, 30, 30, 30]), np.array([30, 30, 30, 3, 3, 3])),
    #     (np.array([1000, 1000, 1000, 30, 30, 30]), np.array([30, 30, 30, 3, 3, 3])),
    #     (np.array([1000, 1000, 1000, 30, 30, 30]), np.array([30, 30, 30, 3, 3, 3])),
    #     (np.array([1000, 1000, 1000, 30, 30, 30]), np.array([30, 30, 30, 3, 3, 3])),
    #     (np.array([1000, 1000, 1000, 30, 30, 30]), np.array([30, 30, 30, 3, 3, 3]))]

    imp_idx = 0
    last_impedance_update = time.monotonic()
    impedance_update_period = 5.0  # seconds

    with SharedMemoryManager() as shm_manager:
        gripper = FrankaHandController(
            host=gripper_host,
            port=gripper_port,
            speed=gripper_speed,
            force=gripper_force,
            update_rate=frequency
        )
        gripper.start()

        with KeystrokeCounter() as key_counter, \
             FrankaVariableImpedanceController(
                 shm_manager=shm_manager,
                 robot_ip=robot_hostname,
                 frequency=100,
                 Kx_scale=1.0,
                 Kxd_scale=1.0,
                 verbose=True  # set to True for feedback
             ) as controller, \
             Spacemouse(shm_manager=shm_manager) as sm:

            print("Ready to control robot and gripper.")

            state = controller.get_state()
            target_pose = state['ActualTCPPose']
            gripper_width = 0.08  # starting width

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            while not stop:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # Check if it's time to change impedance
                now = time.monotonic()
                if now - last_impedance_update >= impedance_update_period:
                    imp_idx = (imp_idx + 1) % len(impedance_profiles)
                    Kx, Kxd = impedance_profiles[imp_idx]
                    controller.set_impedance(Kx, Kxd)
                    print(f"[Impedance Switch] New Kx: {Kx}, Kxd: {Kxd}")
                    last_impedance_update = now

                # Handle keystrokes
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True

                precise_wait(t_sample)

                # Spacemouse control
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3] * (max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)
                drot = st.Rotation.from_euler('xyz', drot_xyz)

                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()

                # Gripper control
                dwidth = 0
                if sm.is_button_pressed(0):
                    dwidth = -gripper_speed / frequency
                if sm.is_button_pressed(1):
                    dwidth = gripper_speed / frequency
                gripper_width = np.clip(gripper_width + dwidth, 0.0, max_gripper_width)

                # Send commands
                controller.schedule_waypoint(
                    target_pose,
                    t_command_target - time.monotonic() + time.time()
                )
                gripper.send_target(gripper_width)

                precise_wait(t_cycle_end)
                iter_idx += 1

        gripper.stop()
        controller.terminate_current_policy()

if __name__ == '__main__':
    main()
