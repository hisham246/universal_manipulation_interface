import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/home/hisham246/uwaterloo/pickplace/csv/episode_1.csv')

# Extract time and joint velocities
time = df['timestamp'] - df['timestamp'].iloc[0]  # Normalize time to start at 0
joint_vel_cols = [f'robot0_joint_vel_{i}' for i in range(7)]
joint_vels = df[joint_vel_cols]

# Plot
plt.figure(figsize=(12, 6))
for i in range(7):
    plt.plot(time, joint_vels[f'robot0_joint_vel_{i}'], label=f'Joint {i}')

# Add vertical lines every 0.8s
interval = 0.8
max_time = time.max()
for t in range(int(max_time // interval) + 1):
    plt.axvline(x=t * interval, color='gray', linestyle='--', linewidth=0.8)

plt.xlabel('Time (s)')
plt.ylabel('Joint Velocity (rad/s)')
plt.title('Robot0 Joint Velocities Over Time')
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.show()
