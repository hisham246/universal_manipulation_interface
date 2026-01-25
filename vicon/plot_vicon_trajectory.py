import pandas as pd
import matplotlib.pyplot as plt

# Load the trajectory CSV (skip the metadata rows)
path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat/peg_umi_quat 100.csv"
df = pd.read_csv(path, skiprows=[0, 1, 2, 4])  # keeps header row, skips units row

# Extract XYZ (here: TX, TY, TZ in mm)
x = df["TX"].astype(float).to_numpy() / 1000.0  # convert mm to meters
y = df["TY"].astype(float).to_numpy() / 1000.0
z = df["TZ"].astype(float).to_numpy() / 1000.0

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("3D Trajectory (XYZ positions)")

plt.show()