import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "/home/hisham246/uwaterloo/vidp_peg_in_hole_test_4/robot_state_2_episode_0.csv"

action = pd.read_csv(file_path)
action = action.iloc[:, :7]
action = action.dropna()


state = pd.read_csv(file_path)
cols = state.columns[:19]
state = state.drop(columns=cols)
state = state.dropna()

# time_data = pd.read_csv(file_path_time)

# Time (shifted to start at 0)
time = action['timestamp'].to_numpy()
# time = time[520:]
time = time - time[0]

# Commanded positions
cmd_pos = action[['commanded_ee_pose_0', 'commanded_ee_pose_1', 'commanded_ee_pose_2']].to_numpy()
# cmd_pos = cmd_pos[520:]
act_pos = state[['actual_ee_pose_0', 'actual_ee_pose_1', 'actual_ee_pose_2']].to_numpy()
print(act_pos)
# act_pos = act_pos[520:]


# Velocity with fixed dt

freq = 10.0          # your control frequency
dt = 1.0 / freq      # = 0.1 s

# central difference: v[i] = (x[i+1] - x[i-1]) / (2*dt)
cmd_vel = (cmd_pos[2:] - cmd_pos[:-2]) / (2.0 * dt)
act_vel = (act_pos[2:] - act_pos[:-2]) / (2.0 * dt)
time_vel = time[1:-1]

# optional light smoothing (3-sample moving average)
cmd_vel_smooth = pd.DataFrame(cmd_vel).rolling(3, center=True).mean().to_numpy()
act_vel_smooth = pd.DataFrame(act_vel).rolling(3, center=True).mean().to_numpy()
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
axes = axes.flatten()
labels = ['x', 'y', 'z']

# Position
for i in range(3):
    axes[i].plot(time, cmd_pos[:, i], label='Commanded Position')
    axes[i].plot(time, act_pos[:, i], label='Actual Position')
    axes[i].set_title(f'Position - {labels[i]}')
    axes[i].set_xlabel('Time [s]')
    axes[i].set_ylabel('Position [m]')
    axes[i].grid(True)
    axes[i].legend()

# # Velocity
# for i in range(3):
#     axes[i+3].plot(time_vel, cmd_vel_smooth[:, i], label='Commanded Velocity')
#     axes[i+3].plot(time_vel, act_vel_smooth[:, i], label='Actual Velocity')
#     axes[i+3].set_title(f'Velocity - {labels[i]}')
#     axes[i+3].set_xlabel('Time [s]')
#     axes[i+3].set_ylabel('Velocity [m/s]')
#     axes[i+3].grid(True)
#     axes[i+3].legend()

plt.tight_layout()
plt.show()


#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
file_path = "/home/hisham246/uwaterloo/vidp_peg_in_hole_test_4/robot_state_2_episode_0.csv"

freq = 10.0
dt = 1.0 / freq
labels = ["x", "y", "z"]

cmd_pos_cols = ["commanded_ee_pose_0", "commanded_ee_pose_1", "commanded_ee_pose_2"]
act_pos_cols = ["actual_ee_pose_0", "actual_ee_pose_1", "actual_ee_pose_2"]

k_cols = ["commanded_stiffness_0", "commanded_stiffness_1", "commanded_stiffness_2"]
d_cols = ["commanded_damping_0", "commanded_damping_1", "commanded_damping_2"]

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
df = pd.read_csv(file_path)

# ------------------------------------------------------------
# Build masks
#   - action-valid: timestamp + commanded pos + stiffness/damping
#   - state-valid : action-valid + actual pos
# ------------------------------------------------------------
required_action = ["timestamp"] + cmd_pos_cols + k_cols + d_cols
required_state  = required_action + act_pos_cols

missing_action = [c for c in required_action if c not in df.columns]
missing_state  = [c for c in required_state  if c not in df.columns]

if missing_action:
    raise KeyError(f"Missing required ACTION columns in CSV: {missing_action}")

if missing_state:
    # We'll still allow stiffness/damping plotting even if actual_* are absent,
    # but position/velocity (actual) will be skipped.
    print(f"Warning: missing STATE columns in CSV (actual pose). Will skip actual plots: {missing_state}")

mask_action = df[required_action].notna().all(axis=1)
mask_state = mask_action.copy()
if all(c in df.columns for c in act_pos_cols):
    mask_state &= df[act_pos_cols].notna().all(axis=1)
else:
    mask_state &= False

print("rows total:", len(df))
print("rows valid action (cmd+K+D):", int(mask_action.sum()))
print("rows valid state  (plus actual):", int(mask_state.sum()))

if mask_action.sum() == 0:
    raise RuntimeError(
        "No valid action rows after NaN filtering. "
        "Either required columns contain NaNs everywhere or column names differ."
    )

# ------------------------------------------------------------
# ACTION timeline (always available for K/D and commanded pos)
# ------------------------------------------------------------
dfa = df.loc[mask_action].reset_index(drop=True)

time_a = dfa["timestamp"].to_numpy(dtype=np.float64)
time_a = time_a - time_a[0]

cmd_pos_a = dfa[cmd_pos_cols].to_numpy(dtype=np.float64)
k_trans_a = dfa[k_cols].to_numpy(dtype=np.float64)
d_trans_a = dfa[d_cols].to_numpy(dtype=np.float64)

# ------------------------------------------------------------
# STATE timeline (only rows where actual pos exists)
# ------------------------------------------------------------
have_state = mask_state.sum() > 2  # need at least 3 samples for central diff

if have_state:
    dfs = df.loc[mask_state].reset_index(drop=True)

    time_s = dfs["timestamp"].to_numpy(dtype=np.float64)
    time_s = time_s - time_s[0]

    cmd_pos_s = dfs[cmd_pos_cols].to_numpy(dtype=np.float64)
    act_pos_s = dfs[act_pos_cols].to_numpy(dtype=np.float64)

    # Central difference velocities (fixed dt)
    cmd_vel = (cmd_pos_s[2:] - cmd_pos_s[:-2]) / (2.0 * dt)
    act_vel = (act_pos_s[2:] - act_pos_s[:-2]) / (2.0 * dt)
    time_vel = time_s[1:-1]

    # Light smoothing (3-sample moving average)
    cmd_vel_smooth = pd.DataFrame(cmd_vel).rolling(3, center=True).mean().to_numpy()
    act_vel_smooth = pd.DataFrame(act_vel).rolling(3, center=True).mean().to_numpy()
else:
    print("Warning: Not enough valid STATE rows to plot actual position/velocity (need >= 3).")

# ------------------------------------------------------------
# Figure 1: Position + Velocity (only if actual pose exists)
# ------------------------------------------------------------
if have_state:
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    # Position
    for i in range(3):
        axes[i].plot(time_s, cmd_pos_s[:, i], label="Commanded Position")
        axes[i].plot(time_s, act_pos_s[:, i], label="Actual Position")
        axes[i].set_title(f"Position - {labels[i]}")
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel("Position [m]")
        axes[i].grid(True)
        axes[i].legend()

    # Velocity
    for i in range(3):
        axes[i + 3].plot(time_vel, cmd_vel_smooth[:, i], label="Commanded Velocity")
        axes[i + 3].plot(time_vel, act_vel_smooth[:, i], label="Actual Velocity")
        axes[i + 3].set_title(f"Velocity - {labels[i]}")
        axes[i + 3].set_xlabel("Time [s]")
        axes[i + 3].set_ylabel("Velocity [m/s]")
        axes[i + 3].grid(True)
        axes[i + 3].legend()

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Figure 2: Stiffness + Damping (translation only) (always if action-valid)
# ------------------------------------------------------------
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 6), sharex=True)

# Stiffness
for i in range(3):
    axes2[0, i].plot(time_a, k_trans_a[:, i])
    axes2[0, i].set_title(f"Stiffness - {labels[i]}")
    axes2[0, i].set_ylabel("Stiffness")
    axes2[0, i].grid(True)

# Damping
for i in range(3):
    axes2[1, i].plot(time_a, d_trans_a[:, i])
    axes2[1, i].set_title(f"Damping - {labels[i]}")
    axes2[1, i].set_xlabel("Time [s]")
    axes2[1, i].set_ylabel("Damping")
    axes2[1, i].grid(True)

plt.tight_layout()
plt.show()
