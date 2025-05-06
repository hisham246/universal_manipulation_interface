import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
joint_path = "/home/hisham246/uwaterloo/12_steps/robot_state_pickplace_test_1.csv"

joint_df = pd.read_csv(joint_path)

# Extract time, commanded positions, and actual positions
time = joint_df['timestamp'] - joint_df['timestamp'].iloc[0]

tau_des = joint_df[[f'motor_torques_desired_{i}' for i in range(7)]].to_numpy()
tau = joint_df[[f'motor_torques_measured_{i}' for i in range(7)]].to_numpy()

# Discard the first row to avoid zero velocity artifacts
time = time[1:]
tau_des = tau_des[1:]
tau = tau[1:]


# Plot Figure 1: Joint Torques
fig1, axes1 = plt.subplots(7, 1, figsize=(12, 14))
for i in range(7):
    axes1[i].plot(time, tau_des[:, i], label='Desired')
    axes1[i].plot(time, tau[:, i], label='Actual')
    axes1[i].set_ylabel(f'Joint {i}')
    axes1[i].legend()
    axes1[i].grid(True)
axes1[-1].set_xlabel('Time [s]')
fig1.suptitle('Joint Torques Over Time')
plt.tight_layout()
plt.show()