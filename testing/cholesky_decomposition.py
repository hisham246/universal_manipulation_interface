import numpy as np

A = np.array([[4.0, 0.0, 0.0],
              [0.0, 3.0, 0.0],
              [0.0, 0.0, 2.0]])

L = np.linalg.cholesky(A)

chol_vec = np.array([
    L[0, 0],
    L[1, 0], L[1, 1],
    L[2, 0], L[2, 1], L[2, 2]
])

def vec_to_chol(chol_vec):
    L = np.zeros((3, 3))
    L[0, 0] = chol_vec[0]
    L[1, 0] = chol_vec[1]
    L[1, 1] = chol_vec[2]
    L[2, 0] = chol_vec[3]
    L[2, 1] = chol_vec[4]
    L[2, 2] = chol_vec[5]
    return L

lower = vec_to_chol(chol_vec)
Kx_pos = lower @ lower.T
print(Kx_pos)
