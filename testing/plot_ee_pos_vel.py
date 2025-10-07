import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
# file_path_state = "/home/hisham246/uwaterloo/polymetis_reaching/robot_state_2_episode_4.csv"
file_path_action = "/home/hisham246/uwaterloo/reaching_rtc/policy_actions_episode_0.csv"
# state = pd.read_csv(file_path_state)
action = pd.read_csv(file_path_action)

# Extract time, commanded positions, and actual positions
# time_state = action['timestamp'] - action['timestamp'].iloc[0]
time_action = action['timestamp'] - action['timestamp'].iloc[0]
cmd_pos = action[['ee_pos_0', 'ee_pos_1', 'ee_pos_2']].to_numpy()
# act_pos = state[['ee_pose_0', 'ee_pose_1', 'ee_pose_2']].to_numpy()

# # Discard the first row to avoid zero velocity artifacts
# time = time[1:]
# cmd_pos = cmd_pos[1:]
# act_pos = act_pos[1:]

# Compute velocities
# dt_state = np.gradient(time_state)
# dt_action = np.gradient(time_action)
# cmd_vel = np.gradient(cmd_pos, axis=0) / dt_action[:, None]
# act_vel = np.gradient(act_pos, axis=0) / dt_state[:, None]

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
axes = axes.flatten()
labels = ['x', 'y', 'z']

# Position plots
for i in range(3):
    axes[i].plot(time_action, cmd_pos[:, i], label='Commanded')
    # axes[i].plot(act_pos[:, i], label='Actual')
    axes[i].set_title(f'Position - {labels[i]}')
    axes[i].set_xlabel('Time [s]')
    axes[i].set_ylabel('Position [m]')
    axes[i].legend()
    axes[i].grid(True)

# # Velocity plots
# for i in range(3):
#     axes[i+3].plot(time_action, cmd_vel[:, i], label='Commanded')
#     axes[i+3].plot(time_state, act_vel[:, i], label='Actual')
#     axes[i+3].set_title(f'Velocity - {labels[i]}')
#     axes[i+3].set_xlabel('Time [s]')
#     axes[i+3].set_ylabel('Velocity [m/s]')
#     axes[i+3].legend()
#     axes[i+3].grid(True)

plt.tight_layout()
plt.show()