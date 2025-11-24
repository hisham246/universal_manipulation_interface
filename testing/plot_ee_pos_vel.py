import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path_action = "/home/hisham246/uwaterloo/rtc_test_reaching_final/robot_state_2_episode_6.csv"
file_path_time = "/home/hisham246/uwaterloo/rtc_test_reaching_final/robot_state_1_episode_6.csv"

action = pd.read_csv(file_path_action)
time_data = pd.read_csv(file_path_time)

# Time (shifted to start at 0)
time = time_data['timestamp'].to_numpy()
time = time - time[0]

# Commanded positions
cmd_pos = action[['ee_pose_0', 'ee_pose_1', 'ee_pose_2']].to_numpy()

# Velocity with fixed dt

freq = 10.0          # your control frequency
dt = 1.0 / freq      # = 0.1 s

# central difference: v[i] = (x[i+1] - x[i-1]) / (2*dt)
cmd_vel = (cmd_pos[2:] - cmd_pos[:-2]) / (2.0 * dt)
time_vel = time[1:-1]

# optional light smoothing (3-sample moving average)
cmd_vel_smooth = pd.DataFrame(cmd_vel).rolling(3, center=True).mean().to_numpy()

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()
labels = ['x', 'y', 'z']

# Position
for i in range(3):
    axes[i].plot(time, cmd_pos[:, i], label='Commanded')
    axes[i].set_title(f'Position - {labels[i]}')
    axes[i].set_xlabel('Time [s]')
    axes[i].set_ylabel('Position [m]')
    axes[i].grid(True)
    axes[i].legend()

# Velocity
for i in range(3):
    axes[i+3].plot(time_vel, cmd_vel_smooth[:, i], label='Velocity (10 Hz)')
    axes[i+3].set_title(f'Velocity - {labels[i]}')
    axes[i+3].set_xlabel('Time [s]')
    axes[i+3].set_ylabel('Velocity [m/s]')
    axes[i+3].grid(True)
    axes[i+3].legend()

plt.tight_layout()
plt.show()
