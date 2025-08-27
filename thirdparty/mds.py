# Taken from: https://github.com/danilomotta/LMDS/blob/84e933d42607d046f3cb0b7c36c44168e8d60345/mds.py
# Modifications:
#  - Avoid using mutable default arguments in `MDS`

# Author: Danilo Motta  -- <ddanilomotta@gmail.com>

# This is an implementation of the technique described in:
# Sparse multidimensional scaling using landmark points
# http://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf

# type: ignore
# fmt: ignore
import numpy as np
import scipy as sp
from torch.random import fork_rng


def MDS(D, dim=None):
    # Number of points
    n = len(D)

    # Centering matrix
    H = -np.ones((n, n)) / n
    np.fill_diagonal(H, 1 - 1 / n)
    # YY^T
    H = -H.dot(D**2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(H)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    (w,) = np.where(evals > 0)
    if dim is not None:
        arr = evals
        w = arr.argsort()[-dim:][::-1]
        if np.any(evals[w] < 0):
            print("Error: Not enough positive eigenvalues for the selected dim.")
            return []
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)
    return Y


def landmark_MDS(D, lands, dim):
    Dl = D[:, lands]
    n = len(Dl)

    # Centering matrix
    H = -np.ones((n, n)) / n
    np.fill_diagonal(H, 1 - 1 / n)
    # YY^T
    H = -H.dot(Dl**2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(H)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    (w,) = np.where(evals > 0)
    if dim:
        arr = evals
        w = arr.argsort()[-dim:][::-1]
        if np.any(evals[w] < 0):
            print("Error: Not enough positive eigenvalues for the selected dim.")
            return []
    if w.size == 0:
        print("Error: matrix is negative definite.")
        return []

    V = evecs[:, w]
    L = V.dot(np.diag(np.sqrt(evals[w]))).T
    N = D.shape[1]
    Lh = V.dot(np.diag(1.0 / np.sqrt(evals[w]))).T
    Dm = D - np.tile(np.mean(Dl, axis=1), (N, 1)).T
    dim = w.size
    X = -Lh.dot(Dm) / 2.0
    X -= np.tile(np.mean(X, axis=1), (N, 1)).T

    _, evecs = sp.linalg.eigh(X.dot(X.T))

    return (evecs[:, ::-1].T.dot(X)).T
