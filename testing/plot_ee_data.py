# With compensation: episode 3 and 7
# Without compensation: episode 5
# Faster control frequency: epsiode 9

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "/home/hisham246/uwaterloo/pickplace_test_3/csv/episode_15.csv"
df = pd.read_csv(file_path)

# Extract time, commanded positions, and actual positions
time = df['timestamp'] - df['timestamp'].iloc[0]
cmd_pos = df[['action_0', 'action_1', 'action_2']].to_numpy()
act_pos = df[['robot0_eef_pos_0', 'robot0_eef_pos_1', 'robot0_eef_pos_2']].to_numpy()

# Discard the first row to avoid zero velocity artifacts
time = time[1:]
cmd_pos = cmd_pos[1:]
act_pos = act_pos[1:]

# Compute velocities
dt = np.gradient(time)
cmd_vel = np.gradient(cmd_pos, axis=0) / dt[:, None]
act_vel = np.gradient(act_pos, axis=0) / dt[:, None]

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()
labels = ['x', 'y', 'z']

# Position plots
for i in range(3):
    axes[i].plot(time, cmd_pos[:, i], label='Commanded')
    axes[i].plot(time, act_pos[:, i], label='Actual')
    axes[i].set_title(f'Position - {labels[i]}')
    axes[i].set_xlabel('Time [s]')
    axes[i].set_ylabel('Position [m]')
    axes[i].legend()
    axes[i].grid(True)

# Velocity plots
for i in range(3):
    axes[i+3].plot(time, cmd_vel[:, i], label='Commanded')
    axes[i+3].plot(time, act_vel[:, i], label='Actual')
    axes[i+3].set_title(f'Velocity - {labels[i]}')
    axes[i+3].set_xlabel('Time [s]')
    axes[i+3].set_ylabel('Velocity [m/s]')
    axes[i+3].legend()
    axes[i+3].grid(True)

plt.tight_layout()
plt.show()