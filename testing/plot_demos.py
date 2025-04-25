import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline, Akima1DInterpolator, interp1d

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

# Plotting
plt.figure(figsize=(12, 8))

# Joint velocity subplot
plt.subplot(2, 1, 1)
for i in range(7):
    plt.plot(time, joint_vels[f'robot0_joint_vel_{i}'], label=f'Joint {i}')
for t in range(int(max_time // interval) + 1):
    plt.axvline(x=t * interval, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Joint Velocity (rad/s)')
plt.title('Robot0 Joint Velocities')
plt.legend()
plt.tight_layout()

# End-effector velocity subplot
plt.subplot(2, 1, 2)
for i in range(3):
    plt.plot(time, eef_vel[:, i], label=f'EEF Vel Axis {i}')
for t in range(int(max_time // interval) + 1):
    plt.axvline(x=t * interval, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Time (s)')
plt.ylabel('End-Effector Velocity (m/s)')
plt.title('End-Effector Linear Velocities')
plt.legend()
plt.tight_layout()
plt.show()


# # Prepare for PCHIP interpolation
# eef_pchip_interp = np.zeros_like(eef_pos)

# # Apply PCHIP interpolation for each axis in segments
# timestamps = df['timestamp'].values
# segment_duration = 0.2
# fps = 1 / np.median(np.diff(timestamps))
# samples_per_segment = int(segment_duration * fps)
# print(samples_per_segment)
# num_segments = len(df) // samples_per_segment

# for axis in range(3):
#     for i in range(num_segments):
#         start = i * samples_per_segment
#         end = start + samples_per_segment
#         if end > len(df):
#             end = len(df)

#         t_seg = timestamps[start:end]
#         pchip = CubicSpline(t_seg, eef_pos[start:end, axis])
#         eef_pchip_interp[start:end, axis] = pchip(t_seg)

# # Compute velocities from PCHIP interpolated positions
# dt = np.gradient(timestamps)
# eef_pchip_vel = np.gradient(eef_pchip_interp, axis=0) / dt[:, None]

# plt.figure(figsize=(12, 4))

# colors = ['tab:blue', 'tab:orange', 'tab:green']
# for i in range(3):
#     plt.plot(time, eef_pchip_vel[:, i], label=f'New Vel Axis {i}', linestyle='-', color=colors[i])
# for t in range(int(max_time // interval) + 1):
#     plt.axvline(x=t * interval, color='gray', linestyle='--', linewidth=0.8)

# plt.xlabel('Time (s)')
# plt.ylabel('End-Effector Velocity (m/s)')
# plt.title('End-Effector Linear Velocities Using PCHIP Interpolation')
# plt.legend()
# plt.tight_layout()
# plt.show()


# timestamps = df['timestamp'].values
# eef_pos = df[[f'robot0_eef_pos_{i}' for i in range(3)]].values
# time = timestamps - timestamps[0]
# interval = 0.1
# max_time = time.max()

# # Define keypoint times every 0.2s
# key_times = np.arange(0, max_time, interval)
# key_indices = [np.argmin(np.abs(time - t)) for t in key_times]
# key_timestamps = timestamps[key_indices]
# key_positions = eef_pos[key_indices]

# # Interpolate each axis using CubicSpline (no zero-velocity constraints)
# eef_interp = np.zeros_like(eef_pos)
# for axis in range(3):
#     cs = CubicSpline(key_timestamps, key_positions[:, axis])
#     eef_interp[:, axis] = cs(timestamps)

# # Compute velocities from interpolated trajectory
# dt = np.gradient(timestamps)
# eef_interp_vel = np.gradient(eef_interp, axis=0) / dt[:, None]

# # Plot result
# plt.figure(figsize=(12, 4))
# colors = ['tab:blue', 'tab:orange', 'tab:green']
# for i in range(3):
#     plt.plot(time, eef_interp_vel[:, i], label=f'Interpolated Vel Axis {i}', color=colors[i])
# for t in key_times:
#     plt.axvline(x=t, color='gray', linestyle='--', linewidth=0.8)

# plt.xlabel('Time (s)')
# plt.ylabel('End-Effector Velocity (m/s)')
# plt.title('Velocities Using Cubic Interpolation Between 0.2s-Spaced Positions')
# plt.legend()
# plt.tight_layout()
# # plt.grid(True)
# plt.show()