import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('/home/hisham246/uwaterloo/pickplace/csv/episode_1.csv')

# Normalize time
time = df['timestamp'] - df['timestamp'].iloc[0]
interval = 0.8
max_time = time.max()

# Joint velocities
joint_vel_cols = [f'robot0_joint_vel_{i}' for i in range(7)]
joint_vels = df[joint_vel_cols]

# End-effector velocity (derived from position)
eef_pos_cols = [f'robot0_eef_pos_{i}' for i in range(3)]
eef_pos = df[eef_pos_cols].values
dt = np.gradient(df['timestamp'].values)
eef_vel = np.gradient(eef_pos, axis=0) / dt[:, None]

plt.figure(figsize=(12, 12))

# 1. Joint velocity subplot
plt.subplot(3, 1, 1)
for i in range(7):
    plt.plot(time, joint_vels[f'robot0_joint_vel_{i}'], label=f'Joint {i}')
for t in range(int(max_time // interval) + 1):
    plt.axvline(x=t * interval, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Joint Velocity (rad/s)')
plt.title('Robot0 Joint Velocities')
plt.legend()
plt.tight_layout()

# 2. End-effector velocity subplot
plt.subplot(3, 1, 2)
for i in range(3):
    plt.plot(time, eef_vel[:, i], label=f'EE Vel Axis {i}')
for t in range(int(max_time // interval) + 1):
    plt.axvline(x=t * interval, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Time (s)')
plt.ylabel('End-Effector Velocity (m/s)')
plt.title('End-Effector Linear Velocities')
plt.legend()
plt.tight_layout()

# 3. End-effector position subplot
plt.subplot(3, 1, 3)
for i in range(3):
    plt.plot(time, eef_pos[:, i], label=f'EE Pos Axis {i}')
for t in range(int(max_time // interval) + 1):
    plt.axvline(x=t * interval, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Time (s)')
plt.ylabel('End-Effector Position (m)')
plt.title('End-Effector Linear Positions')
plt.legend()
plt.tight_layout()

plt.show()