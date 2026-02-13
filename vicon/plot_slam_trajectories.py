import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import re

def natural_key(path):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', path)]

data_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/vicon_final"
# data_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_4"

# data_dir = "/home/hisham246/uwaterloo/umi/reaching_ball_multimodal/csv_filtered"


episode_files = sorted([f for f in os.listdir(data_dir) if f.startswith("episode_") and f.endswith(".csv")], key=natural_key)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Combined plot of all episodes
for file in sorted(episode_files):
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)

    if all(col in df.columns for col in ["robot0_eef_pos_0", "robot0_eef_pos_1", "robot0_eef_pos_2"]):
        ax.plot(df["robot0_eef_pos_0"], df["robot0_eef_pos_1"], df["robot0_eef_pos_2"], alpha=1.0)
    else:
        print(f"Skipping {file}: position columns not found.")

# Draw world coordinate frame (origin at 0,0,0)
origin = np.array([0, 0, 0])
length = 0.05

# X-axis (Red), Y-axis (Green), Z-axis (Blue)
ax.quiver(*origin, 1, 0, 0, length=length, color='r', normalize=True)
ax.quiver(*origin, 0, 1, 0, length=length, color='g', normalize=True)
ax.quiver(*origin, 0, 0, 1, length=length, color='b', normalize=True)

# labels
ax.text(length, 0, 0, 'X', color='r')
ax.text(0, length, 0, 'Y', color='g')
ax.text(0, 0, length, 'Z', color='b')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Position of End-Effector")
plt.tight_layout()
plt.show()

fig_2d = plt.figure(figsize=(8, 6))
ax_2d = fig_2d.add_subplot(111)

for file in sorted(episode_files):
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)

    if all(col in df.columns for col in ["robot0_eef_pos_0", "robot0_eef_pos_1"]):
        ax_2d.plot(df["robot0_eef_pos_0"], df["robot0_eef_pos_1"], label=file)
    else:
        print(f"Skipping {file}: position columns not found.")

ax_2d.set_xlabel("X")
ax_2d.set_ylabel("Y")
ax_2d.set_title("3D Position of End-Effector (X vs Y)")
# ax_2d.legend()
plt.tight_layout()

# Plot each episode's trajectory
for file in episode_files:
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)

    if all(col in df.columns for col in ["robot0_eef_pos_0", "robot0_eef_pos_1", "robot0_eef_pos_2"]):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the 3D trajectory for the current file
        ax.plot(df["robot0_eef_pos_0"], df["robot0_eef_pos_1"], df["robot0_eef_pos_2"], alpha=1.0)

        # Draw world coordinate frame (origin at 0,0,0)
        origin = np.array([0, 0, 0])
        length = 0.05

        # X-axis (Red), Y-axis (Green), Z-axis (Blue)
        ax.quiver(*origin, 1, 0, 0, length=length, color='r', normalize=True)
        ax.quiver(*origin, 0, 1, 0, length=length, color='g', normalize=True)
        ax.quiver(*origin, 0, 0, 1, length=length, color='b', normalize=True)

        # labels
        ax.text(length, 0, 0, 'X', color='r')
        ax.text(0, length, 0, 'Y', color='g')
        ax.text(0, 0, length, 'Z', color='b')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"3D Position of End-Effector - {file}")
        plt.tight_layout()

        # Show the plot for the current file
        plt.show()

    else:
        print(f"Skipping {file}: position columns not found.")
