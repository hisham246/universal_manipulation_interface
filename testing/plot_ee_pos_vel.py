import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "/home/hisham246/uwaterloo/testing_peg_in_hole_vanilla/robot_state_2_episode_3.csv"
# file_path_time = "/home/hisham246/uwaterloo/test_reaching_rtc/robot_state_1_episode_1.csv"

action = pd.read_csv(file_path)
action = action.iloc[:, :7]
action = action.dropna()


state = pd.read_csv(file_path)
cols = state.columns[:7]
state = state.drop(columns=cols)
state = state.dropna()

# time_data = pd.read_csv(file_path_time)

# Time (shifted to start at 0)
time = action['timestamp'].to_numpy()
time = time[520:]
time = time - time[0]

# Commanded positions
cmd_pos = action[['commanded_ee_pose_0', 'commanded_ee_pose_1', 'commanded_ee_pose_2']].to_numpy()
cmd_pos = cmd_pos[520:]
act_pos = state[['actual_ee_pose_0', 'actual_ee_pose_1', 'actual_ee_pose_2']].to_numpy()
act_pos = act_pos[520:]


# Velocity with fixed dt

freq = 10.0          # your control frequency
dt = 1.0 / freq      # = 0.1 s

# central difference: v[i] = (x[i+1] - x[i-1]) / (2*dt)
cmd_vel = (cmd_pos[2:] - cmd_pos[:-2]) / (2.0 * dt)
act_vel = (act_pos[2:] - act_pos[:-2]) / (2.0 * dt)
time_vel = time[1:-1]

# optional light smoothing (3-sample moving average)
cmd_vel_smooth = pd.DataFrame(cmd_vel).rolling(3, center=True).mean().to_numpy()
act_vel_smooth = pd.DataFrame(act_vel).rolling(3, center=True).mean().to_numpy()
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()
labels = ['x', 'y', 'z']

# Position
for i in range(3):
    axes[i].plot(time, cmd_pos[:, i], label='Commanded Position')
    axes[i].plot(time, act_pos[:, i], label='Actual Position')
    axes[i].set_title(f'Position - {labels[i]}')
    axes[i].set_xlabel('Time [s]')
    axes[i].set_ylabel('Position [m]')
    axes[i].grid(True)
    axes[i].legend()

# Velocity
for i in range(3):
    axes[i+3].plot(time_vel, cmd_vel_smooth[:, i], label='Commanded Velocity')
    axes[i+3].plot(time_vel, act_vel_smooth[:, i], label='Actual Velocity')
    axes[i+3].set_title(f'Velocity - {labels[i]}')
    axes[i+3].set_xlabel('Time [s]')
    axes[i+3].set_ylabel('Velocity [m/s]')
    axes[i+3].grid(True)
    axes[i+3].legend()

plt.tight_layout()
plt.show()
