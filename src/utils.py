import numpy as np
from numpy.typing import ArrayLike


def _validate_nd_array(f_input: ArrayLike, dim: int = 2) -> None:
    """Validate that the input is an array with the correct number of dimensions."""
    if isinstance(f_input, np.ndarray) and f_input.ndim != dim:
        raise ValueError(f"Input must be a {dim}-D numpy array")
    elif isinstance(f_input, list) and len(f_input) != dim:
        raise ValueError(f"Input must be a list with {dim} elements")
    elif not isinstance(f_input, (np.ndarray, list)):
        raise ValueError(
            f"Input must be a numpy.ndarray or a list. Current type: {type(f_input)}"
        )


def rosenbrock(xy: ArrayLike) -> float:
    """Rosenbrock function for 2D input. Global minimum at (1, 1)."""
    _validate_nd_array(xy)
    x, y = xy
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def booth(xy: ArrayLike) -> float:
    """Booth function for 2D input. Global minimum at (1, 3)."""
    _validate_nd_array(xy)
    x, y = xy
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def wheeler_ridge(xy: ArrayLike) -> float:
    """Wheeler Ridge function for 2D input. Global minimum at (1, 1.5)."""
    _validate_nd_array(xy)
    x, y = xy
    return -np.exp(-((x - 1.5) ** 2) - (y - 1.5) ** 2)
