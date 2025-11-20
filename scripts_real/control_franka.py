import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.precise_sleep import precise_wait
from umi.real_world.keystroke_counter import KeystrokeCounter, KeyCode
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.franka_hand_controller import FrankaHandController

@click.command()
@click.option('-rh', '--robot_hostname', default='129.97.71.27')
@click.option('-f', '--frequency', type=float, default=10)
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
            FrankaInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_hostname,
                frequency=1000,
                Kx_scale=5.0,
                Kxd_scale=2.0,
                verbose=False,
                use_interpolation=True
            ) as controller, \
            Spacemouse(
                shm_manager=shm_manager
            ) as sm:

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

                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True

                precise_wait(t_sample)

                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3] * (max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)
                drot = st.Rotation.from_euler('xyz', drot_xyz)

                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()

                dwidth = 0
                if sm.is_button_pressed(0):
                    dwidth = -gripper_speed / frequency
                if sm.is_button_pressed(1):
                    dwidth = gripper_speed / frequency

                gripper_width = np.clip(gripper_width + dwidth, 0.0, max_gripper_width)

                controller.schedule_waypoint(target_pose, t_command_target - time.monotonic() + time.time())
                gripper.send_target(gripper_width)

                precise_wait(t_cycle_end)
                iter_idx += 1

        gripper.stop()
        controller.terminate_current_policy()

if __name__ == '__main__':
    main()