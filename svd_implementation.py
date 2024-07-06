import numpy as np


def custom_svd(a):
    aat = np.dot(a, a.T)
    ata = np.dot(a.T, a)

    eigenvalues_u, u = np.linalg.eigh(aat)
    eigenvalues_v, v = np.linalg.eigh(ata)

    sorted_indices_u = np.argsort(eigenvalues_u)[::-1]
    sorted_indices_v = np.argsort(eigenvalues_v)[::-1]

    eigenvalues_u = eigenvalues_u[sorted_indices_u]

    u = u[:, sorted_indices_u]
    v = v[:, sorted_indices_v]

    sigma = np.zeros((u.shape[0], v.shape[0]), dtype=float)
    min_dim = min(a.shape)
    sigma[:min_dim, :min_dim] = np.diag(np.sqrt(eigenvalues_u[:min_dim]))

    for i in range(min_dim):
        u[:, i] = np.dot(a, v[:, i]) / sigma[i, i]

    a_reconstructed = np.dot(u, np.dot(sigma, v.T))

    return u, sigma, v.T, a_reconstructed

