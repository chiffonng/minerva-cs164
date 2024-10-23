import autograd.numpy as np
from autoograd import grad, hess

def is_positive_definite(matrix: np.ndarray) -> bool:
    if not is_symmetric(matrix):
        raise ValueError("Matrix must be symmetric")
    return np.all(np.linalg.eigvals(matrix) > 0)

def is_negative_definite(matrix: np.ndarray) -> bool:
    if not is_symmetric(matrix):
        raise ValueError("Matrix must be symmetric")
    return np.all(np.linalg.eigvals(matrix) < 0)

def is_indefinite(matrix: np.ndarray) -> bool:
    if not is_symmetric(matrix):
        raise ValueError("Matrix must be symmetric")
    return (not is_positive_definite(matrix)) and (not is_negative_definite(matrix))

def sylvester_criterion(matrix: np.ndarray) -> bool:
    """Check if a matrix is positive definite using Sylvester's criterion."""

    if not is_symmetric(matrix):
        raise ValueError("Matrix must be symmetric")

    for i in range(matrix.shape[0]):
        sub_matrix = matrix[:i, :i]
        if np.linalg.det(sub_matrix) <= 0:
            return False

    return True

def is_symmetric(matrix: np.ndarray) -> bool:
    """Check if a matrix is symmetric."""
    return np.allclose(matrix, matrix.T)

def is_square(matrix: np.ndarray) -> bool:
    """Check if a matrix is square."""
    return matrix.shape[0] == matrix.shape[1]

def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize a square matrix by averaging it with its transpose."""
    if not is_square(matrix):
        raise ValueError("Matrix must be square")
    elif is_symmetric(matrix):
        return matrix
    else:
        return (matrix + matrix.T) / 2