import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Setup Constants
# ---------------------------------------------------------
# This offset is now applied IN the SLAM frame AFTER rotation
vicon_world_offset = np.array([-0.02799, -0.206246, -0.093154])

rot_v2s = R.from_euler('z', 180, degrees=True)

# Local Axis Convention (e.g., UMI expects Z-forward)
R_local_mat = np.array([
    [ 1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0, -1.0,  0.0]
])
rot_local = R.from_matrix(R_local_mat)

# ---------------------------------------------------------
# 2. Create Sample Data
# ---------------------------------------------------------
steps = 10
pos_v = np.zeros((steps, 3))
pos_v[:, 0] = np.linspace(0, 0.2, steps) # Move 20cm in +X

quat_v = R.from_euler('xyz', [0, 0, 0]).as_quat()
quat_v = np.tile(quat_v, (steps, 1))
rot_v_obj = R.from_quat(quat_v)

# ---------------------------------------------------------
# 3. The Transformation Pipeline (Revised)
# ---------------------------------------------------------
# STEP 1: Rotate the entire trajectory around the origin
pos_rotated = rot_v2s.apply(pos_v)

# STEP 2: Translate in the new world frame
pos_slam = pos_rotated + vicon_world_offset

# STEP 3: Handle Orientations
# Global world rotation * Original Orientation * Local Convention
rot_final = rot_v2s * rot_v_obj * rot_local

# ---------------------------------------------------------
# 4. Plotting (Fixed AttributeError)
# ---------------------------------------------------------
def plot_trajectory(pos_data, rot_obj, title, ax):
    ax.plot(pos_data[:, 0], pos_data[:, 1], pos_data[:, 2], 'k--', alpha=0.3)
    
    colors = ['r', 'g', 'b'] # X, Y, Z
    scale = 0.04 

    # rot_obj is a Scipy Rotation object (array of rotations)
    mats = rot_obj.as_matrix() 

    for i in range(len(pos_data)):
        p = pos_data[i]
        mat = mats[i]
        for j in range(3):
            ax.quiver(p[0], p[1], p[2], mat[0, j], mat[1, j], mat[2, j], 
                      color=colors[j], length=scale, normalize=True)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])

fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection='3d')
plot_trajectory(pos_v, rot_v_obj, "1. Raw Vicon", ax1)

ax2 = fig.add_subplot(122, projection='3d')
plot_trajectory(pos_slam, rot_final, "2. Final (Rotated -> Translated -> Local)", ax2)

plt.tight_layout()
plt.show()