import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d
import scipy.spatial.transform as st
# import 

# Load the CSV file
file_path = '/home/hisham246/uwaterloo/pickplace/csv/episode_1.csv'
df = pd.read_csv(file_path)

t_all = df['timestamp'].values
pos_all = df[['robot0_eef_pos_0', 'robot0_eef_pos_1', 'robot0_eef_pos_2']].values
rot_all = df[['robot0_eef_rot_axis_angle_0', 'robot0_eef_rot_axis_angle_1', 'robot0_eef_rot_axis_angle_2']].values

# Sample every ~0.8 seconds
sampling_interval = 0.8
sampled_indices = [0]
for i in range(1, len(t_all)):
    if t_all[i] - t_all[sampled_indices[-1]] >= sampling_interval:
        sampled_indices.append(i)

t_sampled = t_all[sampled_indices]
pos_sampled = pos_all[sampled_indices]
rot_sampled = rot_all[sampled_indices]

# Create fine time grid
t_fine = np.linspace(t_sampled[0], t_sampled[-1], 500)

# Interpolators
pchip_interp = [PchipInterpolator(t_sampled, pos_sampled[:, i]) for i in range(3)]
cubic_interp = [CubicSpline(t_sampled, pos_sampled[:, i]) for i in range(3)]
linear_interp = [interp1d(t_sampled, pos_sampled[:, i], kind='linear', bounds_error=False, fill_value=(pos_sampled[0, i], pos_sampled[-1, i])) for i in range(3)]

# Evaluate interpolations
pos_pchip = np.stack([f(t_fine) for f in pchip_interp], axis=1)
pos_cubic = np.stack([f(t_fine) for f in cubic_interp], axis=1)
pos_linear = np.stack([f(t_fine) for f in linear_interp], axis=1)

# Compute velocities
vel_pchip = np.gradient(pos_pchip, t_fine, axis=0)
vel_cubic = np.gradient(pos_cubic, t_fine, axis=0)
vel_linear = np.gradient(pos_linear, t_fine, axis=0)

# Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

labels = ['vx', 'vy', 'vz']
for i in range(3):
    axes[i].plot(t_fine, vel_pchip[:, i], label='PCHIP', linestyle='-')
    axes[i].plot(t_fine, vel_cubic[:, i], label='CubicSpline', linestyle='--')
    axes[i].plot(t_fine, vel_linear[:, i], label='Linear', linestyle=':')
    axes[i].set_ylabel(labels[i])
    axes[i].legend()
    axes[i].set_title(f"Velocity component: {labels[i]}")

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()

# tools.display_dataframe_to_user(name="Sampled Pose Timestamps", dataframe=pd.DataFrame({
#     'timestamp': t_sampled,
#     'x': pos_sampled[:, 0],
#     'y': pos_sampled[:, 1],
#     'z': pos_sampled[:, 2],
# }))
plt.show()