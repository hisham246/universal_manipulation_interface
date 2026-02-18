#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "/home/hisham246/uwaterloo/vidp_peg_in_hole_v3_test_1/robot_state_1_episode_1.csv"
JAC_PATH = "/home/hisham246/uwaterloo/vidp_peg_in_hole_v3_test_1/jacobian_episode_1.dat"

def damped_ls(A, b, lam=1e-2):
    """Solve min ||A x - b||^2 + lam^2 ||x||^2"""
    ATA = A.T @ A
    return np.linalg.solve(ATA + (lam**2) * np.eye(ATA.shape[0]), A.T @ b)

# ---- Load robot state ----
df = pd.read_csv(CSV_PATH)

# time vector (seconds)
t = df["timestamp"].to_numpy(dtype=np.float64)
t = t - t[0]

# external joint torques (N,7)
tau_cols = [f"motor_torques_external_{i}" for i in range(7)]
missing = [c for c in tau_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing expected torque columns: {missing}")
tau_ext = df[tau_cols].to_numpy(dtype=np.float64)

# ---- Load Jacobian file ----
jac = np.fromfile(JAC_PATH, dtype=np.float32)

PER_STEP = 2240
N_jac = jac.size // PER_STEP
N = min(len(df), N_jac)

jac = jac[:N * PER_STEP].reshape(N, PER_STEP)

# Take the first 42 floats as the 6x7 EE Jacobian (row-major), per timestep.
J = jac[:, :42].reshape(N, 6, 7).astype(np.float64)

# Trim tau/time to match
t = t[:N]
tau_ext = tau_ext[:N]

# ---- Estimate external wrench F from J^T F = tau ----
# F: (N,6) = [Fx,Fy,Fz,Tx,Ty,Tz]
F = np.zeros((N, 6), dtype=np.float64)
for k in range(N):
    A = J[k].T          # (7,6)
    b = tau_ext[k]      # (7,)
    F[k] = damped_ls(A, b, lam=1e-2)

print("tau_ext std per joint:", tau_ext.std(axis=0))
print("F std:", F.std(axis=0))

# ---- Plot ONLY external forces ----
labels_F = ["Fx (N)", "Fy (N)", "Fz (N)"]

plt.figure()
plt.plot(t, F[:, 0], label=labels_F[0])
plt.plot(t, F[:, 1], label=labels_F[1])
plt.plot(t, F[:, 2], label=labels_F[2])
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.title("Estimated external Cartesian forces at end-effector (no bias/smoothing)")
plt.legend()
plt.grid(True)

plt.show()
