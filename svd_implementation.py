import numpy as np

from objects import matrix as a


def custom_svd(a):
    aat = np.dot(a, a.T)
    ata = np.dot(a.T, a)

    eigenvalues_u, u = np.linalg.eigh(aat)  # eigenvectors are already orthonormal
    eigenvalues_v, v = np.linalg.eigh(ata)

    sorted_indices_u = np.argsort(eigenvalues_u)[::-1]  # get and sort indices of eigenvalues in descending order
    sorted_indices_v = np.argsort(eigenvalues_v)[::-1]

    eigenvalues_u = eigenvalues_u[sorted_indices_u]  # sort eigenvalues

    u = u[:, sorted_indices_u]  # sort eigenvectors to get u and v
    v = v[:, sorted_indices_v]

    sigma = np.zeros(a.shape)  # create m*n matrix sigma filled with 0
    min_dim = min(a.shape)
    sigma[:min_dim, :min_dim] = np.diag(
        np.sqrt(eigenvalues_u[:min_dim]))  # fill sigma's diagonal with min{m, n} singular values

    for i in range(min_dim):
        u[:, i] = np.dot(a, v[:, i]) / sigma[i, i]  # align left singular vectors U according to formula

    return u, sigma, v.T


def verify_svd(u, sigma, vt):
    return np.allclose(a, np.dot(u, np.dot(sigma, vt)), rtol=1e-05, atol=1e-08)


u, sigma, vt = custom_svd(a)
is_correct_svd = verify_svd(u, sigma, vt)

print(f"Matrix A:\n", a)
print("\nMatrix U:\n", u)
print("Matrix Sigma:\n", sigma)
print("Matrix V^T:\n", vt)
print("\nSVD is correct:\n", is_correct_svd)
