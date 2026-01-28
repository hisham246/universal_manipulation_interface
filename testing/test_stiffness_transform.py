import numpy as np

def make_diag_spd(kx, ky, kz):
    return np.diag([kx, ky, kz]).astype(float)

# Rotation mapping vectors from data frame D -> robot base frame B
R_BD = np.array([
    [ 0, -1,  0],
    [ 1,  0,  0],
    [ 0,  0,  1]
], dtype=float)

# Example SPD diagonal stiffness in D
K_D = make_diag_spd(kx=100.0, ky=400.0, kz=900.0)

# Transform into B: K_B = R * K_D * R^T
K_B = R_BD @ K_D @ R_BD.T

# Show results + sanity checks
eig_D = np.linalg.eigvalsh(K_D)
eig_B = np.linalg.eigvalsh(K_B)
is_spd_D = np.all(eig_D > 0)
is_spd_B = np.all(eig_B > 0)

print("R_BD (D -> B):\n", R_BD)
print("\nK_D (diag in data frame D):\n", K_D)
print("\nK_B = R_BD @ K_D @ R_BD.T (in robot base frame B):\n", K_B)

print("\nDiagonal entries:")
print("diag(K_D) =", np.diag(K_D))
print("diag(K_B) =", np.diag(K_B))

print("\nEigenvalues (should be > 0 for SPD):")
print("eig(K_D) =", eig_D, " SPD?", is_spd_D)
print("eig(K_B) =", eig_B, " SPD?", is_spd_B)

print("\nExpected (for this 90° z-rotation): diag(K_B) should be [ky, kx, kz].")
