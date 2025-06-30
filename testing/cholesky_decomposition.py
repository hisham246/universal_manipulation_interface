import numpy as np

A = np.array([[4.0, 0.0, 0.0],
              [0.0, 3.0, 0.0],
              [0.0, 0.0, 2.0]])

L = np.linalg.cholesky(A)
U = L.T

chol_vec_upper = np.array([U[0, 0], U[0, 1], U[1, 1], U[0, 2], U[1, 2], U[2, 2]])

# Function to convert Cholesky vector back to upper triangular matrix
def vec_to_chol_upper(chol_vec):
    U = np.zeros((3, 3))
    U[0, 0] = chol_vec[0]
    U[0, 1] = chol_vec[1]
    U[1, 1] = chol_vec[2]
    U[0, 2] = chol_vec[3]
    U[1, 2] = chol_vec[4]
    U[2, 2] = chol_vec[5]
    return U

# Reconstruct the upper triangular matrix from the vector
upper = vec_to_chol_upper(chol_vec_upper)
Kx_pos_upper = upper.T @ upper
stiffness = np.diag(Kx_pos_upper)
# print("Reconstructed matrix from upper triangular Cholesky:")
print(stiffness)