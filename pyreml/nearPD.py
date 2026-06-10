import torch
import numpy as np

def nearPD(S, eps: float = 1e-4):
    """
    Nearest positive-definite surrogate of a symmetric matrix.

    Symmetrize, clamp the eigenvalues at `eps`, rebuild, symmetrize again.
    Guarantees strictly positive eigenvalues, which is what the factor-analytic
    initialization relies on (the q dominant eigenvalues feed Lambda > 0).
    """
    was_numpy = isinstance(S, np.ndarray)
    S = torch.as_tensor(S, dtype=torch.double)
    S = (S + S.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(S)
    eigvals = torch.clamp(eigvals, min=eps)
    S = (eigvecs * eigvals) @ eigvecs.T
    S = (S + S.T) / 2
    return S.numpy() if was_numpy else S