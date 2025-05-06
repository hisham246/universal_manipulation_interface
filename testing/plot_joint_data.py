import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
desired_joint_positions_path = "/home/hisham246/uwaterloo/12_steps/joint_pos_desired.csv"
actual_joint_path = "/home/hisham246/uwaterloo/12_steps/robot_state_pickplace_test_1.csv"

desired_joint_positions_df = pd.read_csv(desired_joint_positions_path)
actual_joint_df = pd.read_csv(actual_joint_path)

# Extract time, commanded positions, and actual positions
time = actual_joint_df['timestamp'] - actual_joint_df['timestamp'].iloc[0]

print(time)

q_des = desired_joint_positions_df[['joint_pos_desired_0', 
                                    'joint_pos_desired_1', 
                                    'joint_pos_desired_2', 
                                    'joint_pos_desired_3',
                                    'joint_pos_desired_4',
                                    'joint_pos_desired_5',
                                    'joint_pos_desired_6']].to_numpy()
q = actual_joint_df[['joint_positions_0', 
                     'joint_positions_1', 
                     'joint_positions_2', 
                     'joint_positions_3',
                     'joint_positions_4',
                     'joint_positions_5',
                     'joint_positions_6']].to_numpy()
dq = actual_joint_df[['joint_velocities_0', 
                     'joint_velocities_1', 
                     'joint_velocities_2', 
                     'joint_velocities_3',
                     'joint_velocities_4',
                     'joint_velocities_5',
                     'joint_velocities_6']].to_numpy()

# Discard the first row to avoid zero velocity artifacts
time = time[1:]
q_des = q_des[2:]
q = q[1:]
dq = dq[1:]

# Compute desired joint velocities
dt = np.gradient(time)
dq_des = np.gradient(q_des, axis=0) / dt[:, None]


# Plot Figure 1: Joint Positions
fig1, axes1 = plt.subplots(7, 1, figsize=(12, 14))
for i in range(7):
    axes1[i].plot(time, q_des[:, i], label='Desired')
    axes1[i].plot(time, q[:, i], label='Actual')
    axes1[i].set_ylabel(f'Joint {i}')
    axes1[i].legend()
    axes1[i].grid(True)
axes1[-1].set_xlabel('Time [s]')
fig1.suptitle('Joint Positions Over Time')
plt.tight_layout()
plt.show()

# Plot Figure 2: Joint Velocities
fig2, axes2 = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
for i in range(7):
    axes2[i].plot(time, dq_des[:, i], label='Desired')
    axes2[i].plot(time, dq[:, i], label='Actual')
    axes2[i].set_ylabel(f'Joint {i}')
    axes2[i].legend()
    axes2[i].grid(True)
axes2[-1].set_xlabel('Time [s]')
fig2.suptitle('Joint Velocities Over Time')
plt.tight_layout()
plt.show()