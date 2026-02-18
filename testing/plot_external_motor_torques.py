#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "/home/hisham246/uwaterloo/vidp_peg_in_hole_v3_test_1/robot_state_1_episode_1.csv"

df = pd.read_csv(CSV_PATH)

# time (s), relative
t = df["timestamp"].to_numpy(dtype=np.float64)
t = t - t[0]

# motor external torques
cols = [f"motor_torques_external_{i}" for i in range(7)]
missing = [c for c in cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

tau_ext_m = df[cols].to_numpy(dtype=np.float64)  # (N,7)

# Figure 1: raw motor external torques
plt.figure()
for i in range(7):
    plt.plot(t, tau_ext_m[:, i], label=f"joint {i}")
plt.title("motor_torques_external (raw)")
plt.xlabel("time (s)")
plt.ylabel("motor external torque")
plt.grid(True)
plt.legend()

# Figure 2: baseline removed (optional but usually helpful)
# Choose a window you believe is "no-contact". Adjust these numbers if needed.
bias_t0, bias_t1 = 0.0, 2.0
idx = (t >= bias_t0) & (t <= bias_t1)
if idx.sum() > 10:
    bias = tau_ext_m[idx].mean(axis=0, keepdims=True)
    tau_ext_m_d = tau_ext_m - bias

    plt.figure()
    for i in range(7):
        plt.plot(t, tau_ext_m_d[:, i], label=f"joint {i}")
    plt.title(f"motor_torques_external (baseline removed using {bias_t0:.1f}-{bias_t1:.1f}s)")
    plt.xlabel("time (s)")
    plt.ylabel("motor external torque (offset removed)")
    plt.grid(True)
    plt.legend()

plt.show()
