import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

stiffness_dir = "/home/hisham246/uwaterloo/vic-umi/stiffness_estimation/pc_gmm/gmm_output/surface_wiping/run_20250618_092508/stiffness.npy"

stiffness = np.load(stiffness_dir)

diagonals = np.array([np.diag(Kp) for Kp in stiffness])

min_value = np.min(diagonals)

print(f"Minimum diagonal stiffness value: {min_value:.2f}")

# # Plot stiffness components
# plt.figure(figsize=(12, 6))
# components = ['X', 'Y', 'Z']
# colors = ['r', 'g', 'b']

# for i in range(3):
#     plt.plot(stiffness[:, i], color=colors[i], label=f'Stiffness {components[i]}')

# plt.title("Per-Axis Diagonal Stiffness Profiles Along Trajectory")
# plt.xlabel("Trajectory Point Index")
# plt.ylabel("Stiffness Value")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()